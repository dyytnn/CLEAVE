#!/usr/bin/env python3
"""WP9: patient-grouped train/val/test split for the Nantes dataset (652 clean videos).

Video names are `<patient_code>-<embryo_no>` (e.g. `AA83-7`, `ALR493-10`); grouping by patient_code
(everything before the last `-<digits>`) keeps all embryos of one patient in the same partition, so
no patient straddles train/val/test. 516 unique patients over 652 videos (~1.26 videos/patient).

    python scripts/make_nantes_split.py --out data/splits/nantes_grouped_v1.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PAT_RE = re.compile(r"^(.*)-_?(\d+[A-Za-z]?(?:-\d+)?)$")


def patient_of(video: str) -> str:
    m = PAT_RE.match(video)
    if not m:
        raise ValueError(f"unparseable video name: {video}")
    return m.group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/derived/nantes_manifest_F0.csv")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    videos = sorted(df.video.unique())
    by_patient: dict[str, list[str]] = {}
    for v in videos:
        by_patient.setdefault(patient_of(v), []).append(v)

    patients = sorted(by_patient)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(patients))
    n = len(patients)
    n_train = int(round(n * args.train_frac))
    n_val = int(round(n * args.val_frac))
    parts = {
        "train": [patients[i] for i in perm[:n_train]],
        "val": [patients[i] for i in perm[n_train : n_train + n_val]],
        "test": [patients[i] for i in perm[n_train + n_val :]],
    }

    split = {
        "version": "nantes_grouped_v1",
        "seed": args.seed,
        "grouping": "patient_code (video name before the trailing -<embryo_no>)",
        "source": "Gomez et al. 2022, Zenodo 7912264; 652/704 videos after quarantining 52 100%-corrupt videos",
        "n_patients": {k: len(v) for k, v in parts.items()},
        "patients": parts,
        "videos": {k: sorted(vid for p in v for vid in by_patient[p]) for k, v in parts.items()},
    }
    n_videos = {k: len(v) for k, v in split["videos"].items()}
    print("patients:", split["n_patients"], "| videos:", n_videos, "| total videos:", sum(n_videos.values()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(split, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
