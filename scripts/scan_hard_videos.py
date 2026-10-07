#!/usr/bin/env python3
"""Systematic scan for videos that every model decodes badly.

The manuscript names three label-quality outliers and one genuinely hard video. Those were found by
inspection. This script replaces the eyeball with a stated criterion applied to every released run,
so the list is reproducible and so that nothing comparable was missed.

Criterion. For each test video, take the frame accuracy of every configuration that scored it, on
its own split. A video is flagged when the *best* configuration that ever saw it still fails, that
is when max frame accuracy over all configurations is below ``--threshold``. Using the maximum
rather than the mean is deliberate: it asks whether any model in the study can decode the video, so
a weak model cannot put a video on the list.

    PYTHONPATH=src python scripts/scan_hard_videos.py --threshold 0.45
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--min-configs", type=int, default=5, help="ignore videos scored by fewer configurations")
    ap.add_argument("--out", default="results/hard_videos.json")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / "results/runs_master.csv")
    # the image-level split puts frames of every video in training as well as in test, so a video's
    # accuracy there is not a held-out measurement and must not count towards "the best model can
    # decode it". Only video-level and patient-grouped splits are genuine held-out evaluations.
    df = df[df.split_key != "image"]
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in df.itertuples():
        f = ROOT / "runs/h7" / r.run / "per_video_test.json"
        if not f.exists():
            continue
        for v in json.loads(f.read_text()):
            acc[v["video"]][r.experiment].append(float(v["acc"]))

    rows = []
    for vid, per_exp in acc.items():
        means = {e: float(np.mean(x)) for e, x in per_exp.items()}
        if len(means) < a.min_configs:
            continue
        best_exp = max(means, key=lambda e: means[e])
        rows.append({"video": vid, "n_configs": len(means), "best_acc": means[best_exp],
                     "best_config": best_exp, "median_acc": float(np.median(list(means.values()))),
                     "worst_acc": float(min(means.values()))})
    rows.sort(key=lambda r: r["best_acc"])
    flagged = [r for r in rows if r["best_acc"] < a.threshold]

    known = {"LV683-2-8", "DRL1048-1", "CC938-4"}
    out = {"threshold": a.threshold, "min_configs": a.min_configs,
           "n_videos_scanned": len(rows), "n_flagged": len(flagged),
           "flagged": flagged, "known_outliers": sorted(known),
           "known_recovered": sorted(known & {r["video"] for r in flagged}),
           "known_missed": sorted(known - {r["video"] for r in flagged}),
           "new": sorted({r["video"] for r in flagged} - known)}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))

    print("image-level runs excluded: a video's frames are in training there")
    print(f"scanned {len(rows)} test videos, each scored by >= {a.min_configs} configurations")
    print(f"flagged at best-model frame accuracy < {a.threshold}: {len(flagged)}")
    for r in flagged:
        mark = "known" if r["video"] in known else "NEW"
        print(f"  {r['video']:14} best {r['best_acc']:.3f}  median {r['median_acc']:.3f}  "
              f"over {r['n_configs']:3d} configs   [{mark}]")
    print(f"known outliers recovered: {out['known_recovered']}")
    if out["known_missed"]:
        print(f"known outliers NOT flagged: {out['known_missed']}")
    print("\nnext ten, for the boundary:")
    for r in rows[len(flagged):len(flagged) + 10]:
        print(f"  {r['video']:14} best {r['best_acc']:.3f}  median {r['median_acc']:.3f}  over {r['n_configs']:3d} configs")
    print("wrote", ROOT / a.out)


if __name__ == "__main__":
    main()
