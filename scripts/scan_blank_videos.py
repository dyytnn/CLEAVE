#!/usr/bin/env python3
"""Second image-quality failure mode: annotated videos whose frames carry no embryo.

The release already excludes 52 videos that are JPEG-truncated in every focal plane. A systematic
scan for videos that every model decodes badly (scripts/scan_hard_videos.py) surfaced a different
defect: a video whose frames decode cleanly but are saturated to a featureless white disc, so there
is nothing to classify, while the annotation runs from pronuclear fade to the 8-cell stage. The
manifest's existing ``is_blank`` flag does not mark a single frame of it.

This script measures, on a sample of frames per video, the fraction of the well that is saturated
and the spatial contrast inside it, and lists the videos where an embryo is not visible.

    PYTHONPATH=src python scripts/scan_blank_videos.py --sample 6
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def frame_stats(path: str) -> tuple[float, float] | None:
    """(saturated fraction, contrast) of the bright well interior."""
    try:
        a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    except Exception:
        return None
    well = a[a > np.percentile(a, 60)]            # the lit disc, excluding the dark surround
    if well.size < 100:
        return None
    return float((well >= 250).mean()), float(well.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=6, help="frames sampled per video")
    ap.add_argument("--sat", type=float, default=0.50, help="saturated fraction above which a frame is featureless")
    ap.add_argument("--contrast", type=float, default=6.0, help="well contrast below which a frame is featureless")
    ap.add_argument("--out", default="results/blank_videos.json")
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    man = man[man.phase.notna()]
    jobs = []
    for vid, g in man.groupby("video"):
        g = g.sort_values("frame_index")
        idx = np.linspace(0, len(g) - 1, min(a.sample, len(g))).round().astype(int)
        jobs.append((vid, [g.path.iloc[i] for i in idx]))

    def one(job):
        vid, paths = job
        st = [s for s in (frame_stats(p) for p in paths) if s is not None]
        if not st:
            return vid, None
        sat = float(np.mean([x[0] for x in st]))
        con = float(np.mean([x[1] for x in st]))
        bad = float(np.mean([(x[0] >= a.sat) or (x[1] <= a.contrast) for x in st]))
        return vid, {"sat": sat, "contrast": con, "frac_featureless": bad, "n": len(st)}

    with ThreadPoolExecutor(max_workers=16) as ex:
        res = dict(ex.map(one, jobs))

    rows = [{"video": v, **s} for v, s in res.items() if s]
    rows.sort(key=lambda r: (-r["frac_featureless"], r["contrast"]))
    flagged = [r for r in rows if r["frac_featureless"] >= 0.5]
    excluded = set(json.loads((ROOT / "results/tempo_bench_release.json").read_text())["excluded_videos"]["video_ids"])
    out = {"sample": a.sample, "sat_threshold": a.sat, "contrast_threshold": a.contrast,
           "n_videos": len(rows), "n_flagged": len(flagged),
           "flagged": flagged,
           "flagged_already_excluded": sorted({r["video"] for r in flagged} & excluded),
           "flagged_still_in_use": sorted({r["video"] for r in flagged} - excluded)}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(f"{len(rows)} videos sampled at {a.sample} frames each")
    print(f"featureless in at least half the sampled frames: {len(flagged)}")
    for r in flagged[:15]:
        mark = "already excluded" if r["video"] in excluded else "STILL IN USE"
        print(f"  {r['video']:14} featureless {r['frac_featureless']:.2f}  saturated {r['sat']:.2f}  "
              f"contrast {r['contrast']:5.1f}   [{mark}]")
    print(f"still in the evaluation set: {out['flagged_still_in_use']}")
    print("wrote", ROOT / a.out)


if __name__ == "__main__":
    main()
