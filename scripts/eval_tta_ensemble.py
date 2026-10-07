#!/usr/bin/env python
"""TEMPO rung T6: seed-ensemble and flip-TTA evaluation. No retraining -- averages the softmax probabilities of
already-trained checkpoints (same architecture, different seeds) before Viterbi decoding, and/or averages the
probabilities of a clip with its horizontally+vertically flipped version (the official augmentation is H/V flip with
p=0.5, so a model should be roughly flip-invariant; averaging both views is free variance reduction at eval time).

Usage:
  PYTHONPATH=src python scripts/eval_tta_ensemble.py --runs runs/h7/resnet50_lstm_L8_split0_seed0 \
      runs/h7/resnet50_lstm_L8_split0_seed1 runs/h7/resnet50_lstm_L8_split0_seed2 --flip_tta
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from stseg.eval.kinetic_metrics import evaluate_videos
from stseg.kinetic.datasets import _frames
from stseg.kinetic.models import build_model
from stseg.kinetic.process import sequence_log_probs

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def frame_log_probs(model, frames: torch.Tensor, device, flip_tta: bool, eval_len: int, eval_stride: int) -> np.ndarray:
    """Whole-video log-probs for one checkpoint; averages with the H+V-flipped view if flip_tta."""
    def run(x):
        if model.is_sequence:
            return sequence_log_probs(model, x, eval_len=eval_len, eval_stride=eval_stride, device=device, amp=device.type == "cuda")
        lps = []
        for i in range(0, len(x), 256):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x[i:i + 256].to(device))["logits"].float()
            lps.append(torch.log_softmax(out, 1).cpu().numpy())
        return np.concatenate(lps)

    lp = run(frames)
    if flip_tta:
        flipped = torch.flip(frames, dims=[-1, -2])  # horizontal + vertical flip (matches the training augmentation)
        lp_f = run(flipped)
        lp = np.logaddexp(lp, lp_f) - np.log(2)  # average in probability space, log-domain
    return lp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="checkpoint dirs to ensemble (same architecture/data, different seeds)")
    ap.add_argument("--flip_tta", action="store_true")
    ap.add_argument("--partition", default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hsmm", action="store_true", help="also score the ensemble with the semi-Markov decoder (T7) for every duration-model/lambda setting; select on --partition val, report on test")
    a = ap.parse_args()

    device = torch.device(a.device)
    cfg = json.loads((ROOT / a.runs[0] / "config.resolved.json").read_text())
    ds = _frames(cfg["data"], a.partition, "eval", 0, None, None)
    log_trans = np.load(ROOT / a.runs[0] / "transition_log_matrix.npy")
    # Window = training clip length, 50% overlap stride (the T1 fixsec:windowbug): a fixed 150-frame
    # window silently collapses transformer-family heads with learned absolute positions. Recurrent/conv heads are
    # insensitive to this choice, so using the corrected rule uniformly is always safe, never just "safe for LSTM".
    clip_len = int(cfg["data"].get("clip_len", 150))
    eval_len, eval_stride = clip_len, max(1, clip_len // 2)

    per_video_lps: dict[str, list[np.ndarray]] = {}
    labels_times: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run in a.runs:
        run_dir = ROOT / run
        rcfg = json.loads((run_dir / "config.resolved.json").read_text())
        model = build_model(dict(rcfg["model"]), ds.in_channels)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["model"])
        model.to(device).eval()
        for vid, g in ds.rows.groupby("video", sort=False):
            loader = DataLoader(Subset(ds, g.index.to_numpy()), batch_size=150, shuffle=False, num_workers=a.num_workers)
            frames = torch.cat([b["image"] for b in loader])
            lp = frame_log_probs(model, frames, device, a.flip_tta, eval_len, eval_stride)
            per_video_lps.setdefault(vid, []).append(lp)
            labels_times[vid] = (g.label.to_numpy(), g.time_h.to_numpy(dtype=float))
        print(f"scored {run}")

    seqs = [{"video": v, "labels": labels_times[v][0], "log_probs": np.logaddexp.reduce(lps) - np.log(len(lps)),
             "times_h": labels_times[v][1]} for v, lps in per_video_lps.items()]
    r = evaluate_videos(seqs, log_trans)
    print(f"ensemble of {len(a.runs)} run(s), flip_tta={a.flip_tta}: p={r['p']:.3f} p_v={r['p_v']:.3f} "
          f"p_t={r['p_t']:.3f} MAE={r['mae_h_all']:.2f}h edit={r['edit']:.1f} F1@50={r['f1']['50']:.1f} ({r['n_videos']} videos)")
    if a.hsmm:
        from functools import partial
        from stseg.eval.kinetic_metrics import duration_log_probs, hsmm_viterbi, segment_transition_log_matrix
        labels = _frames(cfg["data"], "train", "eval", 0, None, None).label_sequences()
        lt_seg = segment_transition_log_matrix(labels)
        for kind, lam in [("flat", 0.0)] + [(k, l) for k in ("gamma", "empirical") for l in (0.5, 1.0, 2.0)]:
            dec = partial(hsmm_viterbi, log_trans_seg=lt_seg, log_dur=duration_log_probs(labels, 600, kind), lam=lam)
            h = evaluate_videos(seqs, log_trans, decode=dec)
            print(f"  hsmm {kind:9s} lam={lam:<3g} p_v={h['p_v']:.3f} p_t={h['p_t']:.3f} MAE={h['mae_h_all']:.2f}h edit={h['edit']:.1f} F1@50={h['f1']['50']:.1f}")
            r[f"hsmm_{kind}_lam{lam:g}"] = {k: v for k, v in h.items() if k != "per_video"}
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1, default=float))  # per_video kept for paired bootstrap between ensembles


if __name__ == "__main__":
    main()
