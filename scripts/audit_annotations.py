#!/usr/bin/env python3
"""Model-free anomaly audit of every annotated video.

Model-based scans can only reach the videos some model has evaluated held out, and on the released
protocol that is barely half the dataset. These checks need no model and therefore cover all of it.
Each is a property of the annotation or of the images alone, so a finding here is a defect in the
record rather than a failure of a model.

Checks
  order          an onset that precedes the onset of an earlier phase
  outside        an onset outside the video's own time range
  collapsed      three or more phases sharing one onset time
  sparse         fewer than 30 labelled frames
  short          annotated span under 24 h
  late_start     the first annotation after 24 h. This is a property of the record, not a defect:
                 the median first annotation is at 3.7 h but 5 % of recordings begin after 26 h, one of
                 them at 76.5 h with the embryo already at five cells. Reported so that a model is not
                 blamed for phases that never occur in a video.
  saturated      sampled frames are a featureless bright well (scripts/scan_blank_videos.py)

    PYTHONPATH=src python scripts/audit_annotations.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB", "tHB"]
RANK = {p: i for i, p in enumerate(ORDER)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/annotation_audit.json")
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    blank = set()
    bp = ROOT / "results/blank_videos.json"
    if bp.exists():
        blank = {r["video"] for r in json.loads(bp.read_text())["flagged"]}

    findings: dict[str, list[str]] = {k: [] for k in
                                      ("order", "outside", "collapsed", "sparse", "short", "late_start", "saturated")}
    detail = {}
    for vid, g in man.groupby("video"):
        g = g.sort_values("frame_index")
        lab = g.dropna(subset=["phase"])
        onset = lab.drop_duplicates("phase").set_index("phase").time_h.to_dict()
        if not onset:
            continue
        ordered = sorted(onset, key=lambda p: RANK.get(p, 99))
        times = [onset[p] for p in ordered]
        d = {"n_labelled": int(len(lab)), "n_phases": len(onset),
             "t_first": float(min(times)), "t_last": float(max(times)),
             "span_h": float(max(times) - min(times))}
        if any(b < a_ for a_, b in zip(times, times[1:])):
            findings["order"].append(vid)
        lo, hi = float(g.time_h.min()), float(g.time_h.max())
        if min(times) < lo - 1e-6 or max(times) > hi + 1e-6:
            findings["outside"].append(vid)
        if max(Counter(times).values()) >= 3:
            findings["collapsed"].append(vid)
        if len(lab) < 30:
            findings["sparse"].append(vid)
        if d["span_h"] < 24:
            findings["short"].append(vid)
        if d["t_first"] > 24:
            findings["late_start"].append(vid)
        if vid in blank:
            findings["saturated"].append(vid)
        detail[vid] = d

    excluded = set(json.loads((ROOT / "results/tempo_bench_release.json").read_text())["excluded_videos"]["video_ids"])
    # late_start is a documented property of the record rather than a defect, so it is reported but
    # does not enter the flagged union.
    flagged = sorted({v for k, vs in findings.items() if k != "late_start" for v in vs})
    out = {"n_videos": len(detail), "checks": {k: sorted(v) for k, v in findings.items()},
           "n_flagged": len(flagged),
           "flagged_already_excluded": sorted(set(flagged) & excluded),
           "flagged_still_in_use": sorted(set(flagged) - excluded),
           "detail": detail}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))

    print(f"audited {len(detail)} annotated videos, no model involved")
    for k, v in findings.items():
        still = sorted(set(v) - excluded)
        print(f"  {k:11} {len(v):3d} flagged, {len(still):3d} still in the evaluation set"
              + (f"   {still[:6]}" if still else ""))
    print(f"union still in use: {len(out['flagged_still_in_use'])}")
    print("wrote", ROOT / a.out)


if __name__ == "__main__":
    main()
