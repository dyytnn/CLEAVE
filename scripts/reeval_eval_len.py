#!/usr/bin/env python
"""Re-evaluate a finished H7 run on the Nantes test split with a different inference window (no retraining).

Why: sequence heads are trained on clips of ``clip_len`` frames but the paper protocol evaluates whole videos in chunks
of 150. LSTM/GRU/TCN extrapolate; a transformer with learned absolute positions does not (only positions < clip_len were
ever trained). ``sequence_log_probs`` (process.py) runs windows of ``--eval_len`` every ``--eval_stride`` frames and
averages log-probs where they overlap. Writes ``<run>/results_eval_L<len>_s<stride>.json`` (same keys as results.json) and
prints p / p_v / r / p_t. The backbone runs once per video, so this is cheap (CPU-feasible with --limit_videos).

  PYTHONPATH=src python scripts/reeval_eval_len.py --run runs/h7/resnet18_transformer_L16_adamw_split0_seed0 --eval_len 16 --eval_stride 8
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval_len", type=int, required=True)
    ap.add_argument("--eval_stride", type=int, default=None, help="default = eval_len (non-overlapping)")
    ap.add_argument("--partition", default="test")
    ap.add_argument("--limit_videos", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--decoder", default=None, help="argmax|viterbi|monotonic; default = the run's transition matrix")
    a = ap.parse_args()
    stride = a.eval_stride or a.eval_len
    run = ROOT / a.run
    cfg = json.loads((run / "config.resolved.json").read_text())
    ds = _frames(cfg["data"], a.partition, "eval", 0, None, a.limit_videos)
    model = build_model(dict(cfg["model"]), ds.in_channels)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    device = torch.device(a.device); model.to(device).eval()
    if not model.is_sequence:
        raise SystemExit("per-frame model: the window has no effect")
    lt = np.load(run / "transition_log_matrix.npy")
    if a.decoder:
        from stseg.kinetic.registry import DECODER_REGISTRY
        from stseg.data.nantes_kinetic import NUM_CLASSES
        train = _frames(cfg["data"], "train", "eval", 0, None, None)
        lt = DECODER_REGISTRY.build({"name": a.decoder}).log_transition(train.label_sequences(), NUM_CLASSES)
    seqs = []
    for vid, g in ds.rows.groupby("video", sort=False):
        loader = DataLoader(Subset(ds, g.index.to_numpy()), batch_size=150, shuffle=False, num_workers=a.num_workers)
        frames = torch.cat([b["image"] for b in loader])
        lp = sequence_log_probs(model, frames, a.eval_len, stride, device, device.type == "cuda")
        seqs.append({"video": vid, "labels": g.label.to_numpy(), "log_probs": lp, "times_h": g.time_h.to_numpy(dtype=float)})
    r = evaluate_videos(seqs, lt)
    r.update({"eval_len": a.eval_len, "eval_stride": stride, "partition": a.partition, "clip_len": cfg["data"].get("clip_len")})
    tag = f"results_eval_L{a.eval_len}_s{stride}" + (f"_{a.decoder}" if a.decoder else "") + (f"_first{a.limit_videos}" if a.limit_videos else "") + ("" if a.partition == "test" else f"_{a.partition}")
    (run / f"{tag}.json").write_text(json.dumps(r, indent=1, default=float))
    print(f"{run.name} L={a.eval_len} s={stride} {a.partition}: p={r['p']:.3f} p_v={r['p_v']:.3f} r={r['r']:.3f} p_t={r['p_t']:.3f} | MAE {r['mae_h_all']:.2f} h edit {r['edit']:.1f} F1@50 {r['f1']['50']:.1f} ({r['n_videos']} videos)")


if __name__ == "__main__":
    main()
