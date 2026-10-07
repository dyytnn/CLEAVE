#!/usr/bin/env python3
"""CLEAVE v2 split: couple-level patient grouping of the 652 clean Nantes videos.

``nantes_grouped_v1`` groups by the video name up to the trailing embryo number, which is the
*treatment cycle* rather than the couple: ``GF667-1-1`` and ``GF667-2-6`` become two groups.  Four
couples therefore still straddle a partition in v1, one of them train/test.  v2 groups by
``stseg.data.patient.patient_of(..., level="couple")`` and has zero leaks at either level.

v1 is frozen and stays the split of every committed result; v2 is the split CLEAVE recommends going forward.  The residual bias
of v1 is bounded infrom the per-video effect measured by the sibling-leak experiment.

    PYTHONPATH=src python scripts/make_nantes_split_v2.py --out data/splits/nantes_grouped_v2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stseg.data.patient import group_videos, leaking_patients

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/derived/nantes_manifest_F0.csv")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/splits/nantes_grouped_v2.json")
    args = ap.parse_args()

    videos = sorted(pd.read_csv(ROOT / args.manifest).video.unique())
    by = group_videos(videos, "couple")
    patients = sorted(by)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(patients))
    n_train = int(round(len(patients) * args.train_frac))
    n_val = int(round(len(patients) * args.val_frac))
    parts = {
        "train": sorted(patients[i] for i in perm[:n_train]),
        "val": sorted(patients[i] for i in perm[n_train : n_train + n_val]),
        "test": sorted(patients[i] for i in perm[n_train + n_val :]),
    }
    split = {
        "version": "nantes_grouped_v2",
        "seed": args.seed,
        "grouping": "couple (stseg.data.patient.patient_of level='couple': embryo and cycle tokens stripped)",
        "source": "Gomez et al. 2022, Zenodo 7912264; 652/704 videos after excluding 52 JPEG-truncated videos",
        "supersedes": "nantes_grouped_v1 (cycle-level grouping; 4 couples straddle a partition, 1 of them train/test)",
        "n_patients": {k: len(v) for k, v in parts.items()},
        "patients": parts,
        "videos": {k: sorted(v for p in ps for v in by[p]) for k, ps in parts.items()},
    }
    part_of = {v: k for k, vs in split["videos"].items() for v in vs}
    for level in ("couple", "cycle"):
        leaks = leaking_patients(part_of, level)
        if leaks:
            raise SystemExit(f"refusing to write: {len(leaks)} {level}-level leaks {list(leaks)[:5]}")

    out = ROOT / args.out
    out.write_text(json.dumps(split, indent=2))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print("patients:", split["n_patients"], "| videos:", {k: len(v) for k, v in split["videos"].items()})
    print("zero leaks at couple and cycle level")
    print(f"wrote {out}\nsha256 {sha}")


if __name__ == "__main__":
    main()
