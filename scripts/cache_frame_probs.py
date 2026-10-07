#!/usr/bin/env python
"""Cache per-frame log-probabilities (val + test) of every trained kinetic run so retrain-free decoders (HSMM, TTA,
calibration studies) can be scored on all checkpoints in seconds instead of re-running inference per experiment.
Uses each run's *own* eval window (training.eval_len/eval_stride, default 150/150 = the window its published numbers
were scored with). Output: results/frame_probs/<run>/{val,test}.npz with keys <vid>__lp (T,16 float16), __y, __t.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/cache_frame_probs.py            # all runs/h7/*_seed* missing a cache
  PYTHONPATH=src python scripts/cache_frame_probs.py --runs runs/h7/foo_seed0          # specific runs
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_run import collect  # noqa: E402

from stseg.kinetic import diagnostics as diag  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out_root", default="results/frame_probs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=6)
    a = ap.parse_args()
    device = torch.device(a.device)
    runs = [ROOT / r for r in a.runs] if a.runs else [Path(p) for p in sorted(glob.glob(str(ROOT / "runs/h7/*_seed*"))) if (Path(p) / "best.pt").exists()]
    for run in runs:
        out = ROOT / a.out_root / run.name; out.mkdir(parents=True, exist_ok=True)
        for part in ("val", "test"):
            dst = out / f"{part}.npz"
            if dst.exists():
                continue
            src = ROOT / "results/diagnostics" / run.name / f"frame_probs_{part}.npz"
            if src.exists():  # already computed by diagnose_run.py with the same window rule
                shutil.copy(src, dst); print(f"{run.name} {part}: copied from diagnostics"); continue
            _, seqs, _, win = collect(run, part, device, a.num_workers, want_feats=False)
            diag.dump_frame_probs(seqs, dst)
            print(f"{run.name} {part}: {len(seqs)} videos, window {win}", flush=True)


if __name__ == "__main__":
    main()
