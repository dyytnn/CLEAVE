#!/usr/bin/env python
"""TEMPO protocol audit: a video-level random split matching EmbryoDiff's description ("randomly split into training
and test sets in a 7:3 ratio", Sun et al. 2025) — deliberately **not** patient-grouped, to run our own models under
their described protocol for a same-data, apples-to-apples comparison point (`track1_TEMPO/protocol_audit.md`).

We hold out a small validation slice from the 70% training share (model selection needs one; EmbryoDiff's paper does
not describe theirs either) so the file has the same {train,val,test} schema as every other split and needs no new
code path — just `data.split_mode: video` with this split file, same as always.

Usage:
  python scripts/make_embryodiff_split.py --seed 0 --out data/splits/nantes_random_73_v0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/derived/nantes_manifest_F0.csv")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--val_frac_of_train", type=float, default=0.10,
                    help="carve this fraction out of the 70% training share for validation/model selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    videos = sorted(pd.read_csv(a.manifest, usecols=["video"]).video.unique())
    n = len(videos)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(videos)
    n_test = int(round(n * a.test_frac))
    n_train_all = n - n_test
    n_val = int(round(n_train_all * a.val_frac_of_train))
    test, train_all = perm[:n_test], perm[n_test:]
    val, train = train_all[:n_val], train_all[n_val:]
    split = {
        "version": "nantes_random_73_v0", "seed": a.seed,
        "note": "video-level random split, NOT patient-grouped -- reproduces the protocol EmbryoDiff (Sun et al. 2025) "
                "describes, for direct comparison under the same (unsafe) protocol; see track1_TEMPO/protocol_audit.md. "
                "The val slice is our own addition (needed for model selection; not specified by EmbryoDiff).",
        "n_videos": {"train": len(train), "val": len(val), "test": len(test)},
        "videos": {"train": sorted(train.tolist()), "val": sorted(val.tolist()), "test": sorted(test.tolist())},
    }
    out = Path(a.out) if a.out else Path(f"data/splits/nantes_random_73_seed{a.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1))
    print(f"{out}: train {len(train)} / val {len(val)} / test {len(test)} (of {n} videos)")


if __name__ == "__main__":
    main()
