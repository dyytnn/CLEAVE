#!/usr/bin/env python3
"""Five couple-grouped folds over all clean Nantes videos for an out-of-fold timeline audit.

Round-5 review found that the timeline audit could not see mismatches in the videos its classifier had been trained
on: a classifier fitted to a video's shifted labels reproduces them. Here every video is in the test partition of
exactly one fold, so each video is audited by a classifier that never saw it. Within a fold, 10 % of the remaining
couples form the validation partition used for checkpoint selection. Splits are for the audit only.

    PYTHONPATH=src python scripts/make_oof_audit_splits.py      # -> data/splits/nantes_oof_audit_v1_k{0..4}.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stseg.data.patient import patient_of

ROOT = Path(__file__).resolve().parents[1]
K, SEED = 5, 20261001


def main() -> None:
    v2 = json.loads((ROOT / "data/splits/nantes_grouped_v2.json").read_text())
    videos = sorted(v for part in ("train", "val", "test") for v in v2["videos"][part])
    couples = sorted({patient_of(v) for v in videos})
    rng = np.random.default_rng(SEED)
    order = rng.permutation(couples)
    fold_of = {c: i % K for i, c in enumerate(order)}
    for k in range(K):
        test_c = {c for c in couples if fold_of[c] == k}
        rest = [c for c in order if fold_of[c] != k]
        val_c = set(rest[: max(1, len(rest) // 10)])
        train_c = set(rest) - val_c
        parts = {"train": train_c, "val": val_c, "test": test_c}
        doc = {"version": f"nantes_oof_audit_v1_k{k}", "seed": SEED, "purpose": "out-of-fold timeline audit only",
               "grouping": v2["grouping"], "n_folds": K, "fold": k,
               "patients": {p: sorted(c) for p, c in parts.items()},
               "videos": {p: sorted(v for v in videos if patient_of(v) in c) for p, c in parts.items()}}
        out = ROOT / f"data/splits/nantes_oof_audit_v1_k{k}.json"
        if out.exists():
            raise SystemExit(f"{out.name} exists and splits are frozen once written")
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(k, {p: len(doc["videos"][p]) for p in parts})


if __name__ == "__main__":
    main()
