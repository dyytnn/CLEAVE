#!/usr/bin/env python3
"""Paired splits that isolate the causal effect of sibling-embryo leakage (CLEAVE defect 3).

The released folds confound two things: whether a test embryo has a sibling in train, and which
videos are in train at all.  This generator removes the confound with a difference-in-differences
design in which the test set, the validation set and the *size* of the training set are byte-identical
across the two arms; the only difference is the identity of 71 training videos.

    test T          one video from each multi-embryo couple (82 videos), split into
                      L  (41 couples) -- their remaining siblings go into the LEAKY arm's train set
                      C  (41 couples) -- their remaining siblings are used by neither arm
    val  V          singleton-couple videos, identical in both arms
    train           clean : P                      (all singleton couples not in V, minus nothing)
                    leaky : (P \\ R) u S_L          (|R| = |S_L|, R drawn from the same singleton pool)

Both arms therefore train on the same number of videos and are evaluated on the same test videos.

    Delta_L  = p_t(leaky) - p_t(clean)  on the L test videos   = leakage effect + swap effect
    Delta_C  = p_t(leaky) - p_t(clean)  on the C test videos   = swap effect alone
    DiD      = Delta_L - Delta_C                                = leakage effect

Multiplying the per-video DiD by the exposed fraction of a real split (31 % for the released folds,
1 % for nantes_grouped_v1; see results/protocol_audit.json) converts it into the inflation that
split carries.

    PYTHONPATH=src python scripts/make_sibling_leak_split.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stseg.data.patient import exposed_test_videos, group_videos, leaking_patients

ROOT = Path(__file__).resolve().parents[1]


def _balance(couples: list[tuple[str, int]], rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Split couples into two groups with equal counts and near-equal total sibling numbers."""
    order = sorted(couples, key=lambda kv: (-kv[1], kv[0]))
    left: list[str] = []
    right: list[str] = []
    load = [0, 0]
    half = len(order) // 2
    for name, n_sib in order:
        open_sides = [i for i, g in enumerate((left, right)) if len(g) < half or len(left) + len(right) >= 2 * half]
        if not open_sides:
            open_sides = [0, 1]
        i = min(open_sides, key=lambda i: (load[i], len((left, right)[i])))
        (left if i == 0 else right).append(name)
        load[i] += n_sib
    if len(left) > half:  # odd number of couples: drop the extra to keep the groups equal
        right += []
        left = left[:half]
    rng.shuffle(left)
    rng.shuffle(right)
    return sorted(left), sorted(right)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/derived/nantes_manifest_F0.csv")
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="data/splits")
    args = ap.parse_args()

    videos = sorted(pd.read_csv(ROOT / args.manifest).video.unique())
    by = group_videos(videos, "couple")
    multi = {c: sorted(v) for c, v in by.items() if len(v) > 1}
    singles = sorted(v for c, vs in by.items() if len(vs) == 1 for v in vs)
    rng = np.random.default_rng(args.seed)

    # test: the first video of every multi-embryo couple; the rest of that couple is its sibling pool
    test_of = {c: vs[0] for c, vs in multi.items()}
    sibs_of = {c: vs[1:] for c, vs in multi.items()}
    group_l, group_c = _balance([(c, len(s)) for c, s in sibs_of.items()], rng)
    used_couples = group_l + group_c
    test = sorted(test_of[c] for c in used_couples)
    sib_l = sorted(v for c in group_l for v in sibs_of[c])

    perm = list(rng.permutation(singles))
    val = sorted(perm[: args.n_val])
    pool = sorted(perm[args.n_val :])
    if len(sib_l) > len(pool):
        raise SystemExit(f"sibling pool {len(sib_l)} exceeds singleton pool {len(pool)}")
    reserve = sorted(rng.choice(pool, size=len(sib_l), replace=False).tolist())
    train_clean = sorted(pool)
    train_leaky = sorted([v for v in pool if v not in set(reserve)] + sib_l)
    assert len(train_clean) == len(train_leaky), (len(train_clean), len(train_leaky))

    meta = {
        "design": "difference-in-differences; identical test and val, identical train size",
        "seed": args.seed,
        "grouping": "couple (stseg.data.patient.patient_of level='couple')",
        "source": "Gomez et al. 2022, Zenodo 7912264; 652/704 videos after excluding 52 JPEG-truncated videos",
        "groups": {
            "L_leaked_couples": group_l, "C_control_couples": group_c,
            "L_test_videos": sorted(test_of[c] for c in group_l),
            "C_test_videos": sorted(test_of[c] for c in group_c),
            "siblings_in_leaky_train": sib_l,
            "reserve_dropped_from_leaky_train": reserve,
        },
    }
    written = {}
    for arm, train in (("clean", train_clean), ("leaky", train_leaky)):
        doc = dict(meta)
        doc = {"version": f"nantes_siblingleak_{arm}_v1", "arm": arm, **meta,
               "n_videos": {"train": len(train), "val": len(val), "test": len(test)},
               "videos": {"train": train, "val": val, "test": test}}
        part_of = {v: p for p, vs in doc["videos"].items() for v in vs}
        doc["audit"] = {
            "n_leaking_couples": len(leaking_patients(part_of, "couple")),
            "n_exposed_test_videos": len(exposed_test_videos(part_of, "couple")),
        }
        if arm == "clean" and doc["audit"]["n_exposed_test_videos"]:
            raise SystemExit("clean arm must have zero exposed test videos")
        if arm == "leaky" and doc["audit"]["n_exposed_test_videos"] != len(group_l):
            raise SystemExit(f"leaky arm exposure {doc['audit']['n_exposed_test_videos']} != {len(group_l)}")
        p = ROOT / args.out_dir / f"nantes_siblingleak_{arm}_v1.json"
        p.write_text(json.dumps(doc, indent=2))
        written[arm] = (p, hashlib.sha256(p.read_bytes()).hexdigest(), doc)

    for arm, (p, sha, doc) in written.items():
        print(f"{arm:5s} train {doc['n_videos']['train']} val {doc['n_videos']['val']} "
              f"test {doc['n_videos']['test']} | exposed test videos {doc['audit']['n_exposed_test_videos']} "
              f"| {p.name} sha256 {sha[:16]}")
    print(f"L couples {len(group_l)} ({len(sib_l)} siblings moved into the leaky train set), "
          f"C couples {len(group_c)}")
    a, b = written["clean"][2]["videos"], written["leaky"][2]["videos"]
    print(f"train sets differ in {len(set(a['train']) ^ set(b['train']))} videos; "
          f"test and val identical: {a['test'] == b['test'] and a['val'] == b['val']}")


if __name__ == "__main__":
    main()
