#!/usr/bin/env python3
"""Defect log v2: three tiers, and the image--annotation timeline audit.

v1 (data/qc/nantes_video_defects_v1.json, kept frozen) listed 41 videos found by two scans and inspected by eye. The
round-4 domain review showed that it mixed three different things, and that most of its "annotated too late"
videos are a release defect rather than annotator error. v2 is generated from:

* v1, for what was seen by eye;
* results/frame_timing/alignment.csv (scripts/audit_frame_timing.py), for videos whose images run on a different
  clock from their annotation: the fitted map image k -> annotation frame a*k + b lowers the mean phase error by at
  least 0.3 phase steps and to at most 60 % of the identity pairing, with a slope inside the search grid;
* the reclassifications below.

Tiers and their treatment in nantes_grouped_v2_trainclean_v2:
  A  label-image mismatch -- the label is not about the image. Removed from training.
  B  image artefact, label correct -- kept, except that for an embryo removed from the well the annotation, which the
     release extends to the end of the recording, is cut at the hour the well empties (``cutoff_h``).
  C  atypical biology -- an embryo a benchmark should keep. Kept everywhere.

    PYTHONPATH=src python scripts/make_defect_log_v2.py    # -> data/qc/nantes_video_defects_v2.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data/qc/nantes_video_defects_v1.json"
ALIGN = ROOT / "results/frame_timing/alignment.csv"
OUT = ROOT / "data/qc/nantes_video_defects_v2.json"
SLOPE_EDGE = (0.605, 1.595)

CATEGORIES = {
    "timeline_mismatch": ("A", "The images run on a different clock from the annotation: pairing the k-th image with "
                               "annotation frame k lets the label drift, by up to tens of hours by the end of the video."),
    "no_division": ("A", "The embryo stays a single cell and never divides, while the annotation records divisions."),
    "blank_throughout": ("A", "Every frame is a featureless bright well; no embryo is visible."),
    "annotation_ahead": ("A", "The annotated onsets run ahead of what the images show, by one to two division stages."),
    "embryo_removed": ("B", "The well empties partway through the recording (removal for transfer or freezing, or "
                            "hatching out of view) and the release extends the last annotated phase to the end."),
    "dark_frames": ("B", "Frames in the middle of the recording are under-exposed to near black."),
    "out_of_focus": ("B", "Frames are an out-of-focus blur."),
    "abrupt_cleavage": ("C", "Direct cleavage from one cell to three or more, or to a fragmented mass, within minutes."),
    "heavy_fragmentation": ("C", "Heavy fragmentation makes the cell count, and so the stage, hard to read."),
}
#: v1 category -> v2 category; "annotation_behind" is resolved by the timeline audit, not mapped
V1_MAP = {"no_division": "no_division", "blank_throughout": "blank_throughout", "annotation_ahead": "annotation_ahead",
          "embryo_absent_late": "embryo_removed", "dark_frames": "dark_frames", "out_of_focus_late": "out_of_focus",
          "out_of_focus_mid": "out_of_focus", "abrupt_cleavage": "abrupt_cleavage",
          "heavy_fragmentation": "heavy_fragmentation"}
#: hour at which the well empties, from the v1 inspection notes (first frame with no embryo visible)
CUTOFF_H = {"OC110-5": 75.0, "DV116-3": 72.0, "QC267-8": 94.0, "FV709-11": 96.0, "PO13-3": 99.0, "PMDPI029-1-8": 124.0,
            "ST586-7": 118.5, "RM126-4": 122.0, "RM126-8": 122.0, "RM126-9": 122.0, "LM184-4": 122.0, "BE327-2": 120.0,
            "GM456-3": 53.0, "AAL839-6": 107.0, "DHDPI042-6": 89.0, "RM126-6": 99.0, "GC340-1": 97.0}


def main() -> None:
    v1 = json.loads(V1.read_text())
    al = pd.read_csv(ALIGN)
    al["gain"] = al.cost_identity - al.cost_fit
    edge = (al.slope_fit <= SLOPE_EDGE[0]) | (al.slope_fit >= SLOPE_EDGE[1])
    mis = al[(al.gain >= 0.3) & (al.cost_fit <= 0.6 * al.cost_identity) & ~edge].set_index("video")

    videos: dict[str, dict] = {}
    for v, rec in v1["videos"].items():
        cats = sorted({V1_MAP[c] for c in rec["defects"] if c in V1_MAP})
        videos[v] = {"defects": cats, "note": rec["note"], "source": "inspection (v1)"}
    for v, r in mis.iterrows():
        e = videos.setdefault(v, {"defects": [], "note": "", "source": "timeline audit"})
        e["defects"] = sorted(set(e["defects"]) | {"timeline_mismatch"})
        e["timeline"] = {"slope": float(r.slope_fit), "offset": float(r.offset_fit),
                         "phase_error_identity": round(float(r.cost_identity), 3), "phase_error_fit": round(float(r.cost_fit), 3)}
        if e["source"] != "timeline audit":
            e["source"] += " + timeline audit"
    for v, h in CUTOFF_H.items():
        if v in videos and "embryo_removed" in videos[v]["defects"]:
            videos[v]["cutoff_h"] = h
    for v, e in videos.items():
        e["tier"] = min((CATEGORIES[c][0] for c in e["defects"]), default="none")
    unresolved = sorted(v for v, e in videos.items() if not e["defects"])
    doc = {
        "version": "nantes_video_defects_v2",
        "derived_from": {"v1": "data/qc/nantes_video_defects_v1.json", "v1_sha256": hashlib.sha256(V1.read_bytes()).hexdigest(),
                         "timeline_audit": "results/frame_timing/alignment.csv",
                         "timeline_audit_sha256": hashlib.sha256(ALIGN.read_bytes()).hexdigest()},
        "tiers": {"A": "label-image mismatch: removed from training", "B": "image artefact, label correct: kept (labels cut at cutoff_h)",
                  "C": "atypical biology: kept everywhere"},
        "categories": {k: {"tier": t, "description": d} for k, (t, d) in CATEGORIES.items()},
        "inspected_by": v1["inspected_by"],
        "unresolved": unresolved,
        "videos": dict(sorted(videos.items())),
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    cut = {v: e["cutoff_h"] for v, e in videos.items() if "cutoff_h" in e and e["tier"] != "A"}
    (ROOT / "data/qc/nantes_label_cutoff_v2.json").write_text(json.dumps({"source": "nantes_video_defects_v2", "cutoff_h": cut}, indent=1) + "\n")
    by = pd.Series({v: e["tier"] for v, e in videos.items()}).value_counts().to_dict()
    cats = pd.Series([c for e in videos.values() for c in e["defects"]]).value_counts().to_dict()
    print(f"{len(videos)} videos; tiers {by}; categories {cats}; cut labels {len(cut)}; unresolved {unresolved}")


if __name__ == "__main__":
    main()
