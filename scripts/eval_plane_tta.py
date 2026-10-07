#!/usr/bin/env python
"""Plane test-time augmentation for single-plane models (TEMPO v17): run a single-plane checkpoint on each of the seven
focal planes of every test video and average the log-probabilities before decoding. Companion to data.plane_mode=random_single
(planes as augmentation): if the 7-plane gain is information rather than fusion, a single-plane model trained on random
planes and averaged over planes at test should match the cross-focal model -- while remaining deployable on one plane.

  PYTHONPATH=src python scripts/eval_plane_tta.py --runs runs/h7/<run>[ ...] [--planes all|central] [--partition test]
Writes <run>/results_plane_tta.json with per-plane and averaged metrics.
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
PLANES = ["embryo_dataset_F-45", "embryo_dataset_F-30", "embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15", "embryo_dataset_F30", "embryo_dataset_F45"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--partition", default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--num_workers", type=int, default=6)
    a = ap.parse_args(); device = torch.device(a.device)
    for run in a.runs:
        rd = ROOT / run; cfg = json.loads((rd / "config.resolved.json").read_text()); t = cfg.get("training", {})
        L = int(cfg["data"].get("clip_len", 4)); ev, st = int(t.get("eval_len", 150)), int(t.get("eval_stride", t.get("eval_len", 150)))
        d = dict(cfg["data"]); d["planes"] = None; d["plane_mode"] = None
        ds = _frames(d, a.partition, "eval", 0, None, None)
        model = build_model(dict(cfg["model"]), 3)
        ck = rd / ("swa.pt" if json.loads((rd / "results.json").read_text()).get("final_weights") == "swa" else "best.pt")
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["model"]); model.to(device).eval()
        lt = np.load(rd / "transition_log_matrix.npy")
        per_plane: dict[str, list[dict]] = {}
        for pl in PLANES:
            ds.eval_plane = pl; seqs = []
            for vid, g in ds.rows.groupby("video", sort=False):
                loader = DataLoader(Subset(ds, g.index.to_numpy()), batch_size=150, shuffle=False, num_workers=a.num_workers)
                frames = torch.cat([b["image"] for b in loader])
                lp = sequence_log_probs(model, frames, ev, st, device, device.type == "cuda") if model.is_sequence else None
                seqs.append({"video": vid, "labels": g.label.to_numpy(), "log_probs": lp, "times_h": g.time_h.to_numpy(dtype=float)})
            per_plane[pl] = seqs
            r = evaluate_videos(seqs, lt); print(f"{Path(run).name} plane {pl[14:] or 'F0':>4}: p_t {r['p_t']:.3f} p_v {r['p_v']:.3f} F1@50 {r['f1']['50']:.1f}", flush=True)
        vids = [s["video"] for s in per_plane[PLANES[0]]]
        avg = []
        for i, v in enumerate(vids):
            lps = [per_plane[pl][i]["log_probs"] for pl in PLANES]
            avg.append({**per_plane[PLANES[0]][i], "log_probs": np.logaddexp.reduce(lps, axis=0) - np.log(len(lps))})
        r_avg = evaluate_videos(avg, lt); r_c = evaluate_videos(per_plane["embryo_dataset"], lt)
        out = {"checkpoint": ck.name, "central_plane": {k: v for k, v in r_c.items() if k != "per_video"},
               "per_plane_p_t": {pl: evaluate_videos(per_plane[pl], lt)["p_t"] for pl in PLANES},
               "plane_tta": {k: v for k, v in r_avg.items() if k != "per_video"}}
        (rd / f"results_plane_tta_{a.partition}.json").write_text(json.dumps(out, indent=1, default=float))
        print(f"{Path(run).name}: central p_t {r_c['p_t']:.3f} -> 7-plane TTA p_t {r_avg['p_t']:.3f} (p_v {r_c['p_v']:.3f}->{r_avg['p_v']:.3f}, F1@50 {r_c['f1']['50']:.1f}->{r_avg['f1']['50']:.1f})")


if __name__ == "__main__":
    main()
