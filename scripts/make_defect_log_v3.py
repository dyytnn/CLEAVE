#!/usr/bin/env python3
"""Defect log v3: the timeline audit out of fold, cross-checked against the release's own files.

v2 took its timeline mismatches from an audit whose classifier had been trained on 436 of the videos it audited; a
classifier fitted to a video's shifted labels reproduces them, so that audit found 23 mismatches among the 216 videos
it had not seen and none among the 436 it had (round-5 review, R1 and R2 independently). v3 changes three things:

* timeline mismatches come from the out-of-fold audit (results/frame_timing/alignment_oof.csv): every video is
  audited by a classifier that never saw it. A video is a mismatch when the fitted map lowers the phase error by at
  least 0.3 steps and to at most 60 % of the released pairing, with a slope inside the grid -- the v2 rule unchanged --
  and the model-free file signature (a time file on an exactly uniform 0.2-h grid, or annotation frames beyond the last
  image) is reported next to it; ``CONFIRMED_BY_EYE`` lists the videos where the two disagree and inspection decided;
* inspection additions: DHDPI042-7 and DHDPI042-8 never divide (round-5 R2, confirmed on the raw frames under both
  pairings); EH315-3 is direct cleavage, biology rather than a label error (tier C);
* label cut-offs for embryos removed from the well are measured -- the first sampled frame after which the well stays
  featureless, by the structure measure of scripts/scan_featureless_frames.py -- and the earlier of that and the
  inspection note is used.

    PYTHONPATH=src:scripts python scripts/make_defect_log_v3.py   # -> data/qc/nantes_video_defects_v3.json,
                                                                   #    data/qc/nantes_label_cutoff_v3.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_defect_log_v2 import CATEGORIES, CUTOFF_H, V1_MAP
from scan_featureless_frames import structure

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data/qc/nantes_video_defects_v1.json"
ALIGN = ROOT / "results/frame_timing/alignment_oof.csv"
OUT = ROOT / "data/qc/nantes_video_defects_v3.json"
CUT_OUT = ROOT / "data/qc/nantes_label_cutoff_v3.json"
SLOPE_EDGE = (0.605, 1.595)

#: inspection results added in round 5 (raw frames, reference plane, index and 1.25x pairing both checked)
ADDED = {
    "DHDPI042-7": (["no_division"], "A single cell in every frame from fertilisation to the end of the recording, under "
                                    "both the released and the 1.25x pairing; annotated through t8, tM and tSB."),
    "DHDPI042-8": (["no_division"], "A single cell in every frame to the end of the recording, under both pairings; "
                                    "annotated through t8, tM and tSB."),
}
#: v1 entries reclassified on inspection
RECLASSIFY = {"EH315-3": ["abrupt_cleavage", "heavy_fragmentation"]}
#: decisions where the out-of-fold audit and the file signature disagree, filled in after inspection
#: (video -> True: timeline mismatch, False: not one), from the contact sheets of the round-5 session
CONFIRMED_BY_EYE: dict[str, bool] = {
    # proposed by the audit only
    "LP284-3": True,         # index pairing ~6 cells at annotated t4; 1.25x pairing ~4
    "PC758-2": True,         # index pairing 3-4 cells at annotated t2
    "GF083-5": False,        # index pairing already shows 2 / 4 / 8 cells at t2 / t4 / t8
    "FM864-7": False,        # dark frames; the cell count cannot be read under either pairing
    "PMDPI029-1-11": False,  # embryo at the edge of the well, out of focus; undecidable
    # proposed by the file signature only
    "GC340-10": True,        # index pairing 2+ cells at annotated tPNf, 1.25x pairing one cell with pronuclei
    "RC545-2-8": True,       # index pairing ~8 cells at annotated t2 (siblings -2-5, -2-9 mismatched too)
    "EH315-3": True,         # index pairing many cells at annotated t2; also direct cleavage (kept as a defect)
    "FE14-020": False,       # recording starts at t5: no early onset to check
    "DHDPI042-7": False,     # never divides (no_division); no clock to fit
    "DHDPI042-8": False,     # never divides (no_division); no clock to fit
    "EJ393-3": False,        # blank throughout; no clock to fit
}


def measured_cutoff(video: str, thr: float) -> float | None:
    m = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv", usecols=["video", "plane", "frame_index", "path", "time_h"])
    x = m[(m.video == video) & (m.plane == "embryo_dataset")].sort_values("frame_index")
    ok = np.array([(structure(p) or 0.0) >= thr for p in x.path])
    t = x.time_h.to_numpy()
    for i in range(len(ok) - 5):
        if not ok[i:i + 6].any() and ok[i:].mean() < 0.2:
            return float(t[i])
    return None


def main() -> None:
    v1 = json.loads(V1.read_text())
    al = pd.read_csv(ALIGN)
    al["gain"] = al.cost_identity - al.cost_fit
    edge = (al.slope_fit <= SLOPE_EDGE[0]) | (al.slope_fit >= SLOPE_EDGE[1])
    al["audit_flag"] = (al.gain >= 0.3) & (al.cost_fit <= 0.6 * al.cost_identity) & ~edge
    disagree = sorted(al[al.audit_flag != al.file_signature].video)
    undecided = [v for v in disagree if v not in CONFIRMED_BY_EYE]
    if undecided:
        raise SystemExit(f"audit and file signature disagree on {undecided}: inspect them and fill CONFIRMED_BY_EYE")
    mis = {v for v, f in zip(al.video, al.audit_flag) if f and CONFIRMED_BY_EYE.get(v, True)}
    mis |= {v for v, ok in CONFIRMED_BY_EYE.items() if ok}
    by = al.set_index("video")

    videos: dict[str, dict] = {}
    for v, rec in v1["videos"].items():
        cats = RECLASSIFY.get(v) or sorted({V1_MAP[c] for c in rec["defects"] if c in V1_MAP})
        videos[v] = {"defects": sorted(cats), "note": rec["note"], "source": "inspection (v1)"}
    for v, (cats, note) in ADDED.items():
        videos[v] = {"defects": cats, "note": note, "source": "inspection (round 5)"}
    for v in sorted(mis):
        r = by.loc[v]
        e = videos.setdefault(v, {"defects": [], "note": "", "source": "timeline audit (out of fold)"})
        e["defects"] = sorted(set(e["defects"]) | {"timeline_mismatch"})
        e["timeline"] = {"slope": float(r.slope_fit), "offset": float(r.offset_fit), "file_signature": bool(r.file_signature),
                         "phase_error_identity": round(float(r.cost_identity), 3), "phase_error_fit": round(float(r.cost_fit), 3)}
        if v in CONFIRMED_BY_EYE:
            e["timeline"]["decided_by_inspection"] = True
    # a v1 "annotation behind" entry that the out-of-fold audit does not confirm keeps its inspection note but no
    # timeline category; it is left without a defect only if inspection found nothing else
    thr = json.loads((ROOT / "results/featureless_frames.json").read_text())["threshold_structure"]
    for v, e in videos.items():
        if "embryo_removed" in e["defects"]:
            meas = measured_cutoff(v, thr)
            note_h = CUTOFF_H.get(v)
            cands = [h for h in (meas, note_h) if h is not None]
            if cands:
                e["cutoff_h"] = round(min(cands), 1)
                e["cutoff_source"] = {"measured": meas, "inspection_note": note_h}
    for e in videos.values():
        e["tier"] = min((CATEGORIES[c][0] for c in e["defects"]), default="none")
    dropped = sorted(v for v, e in videos.items() if e["tier"] == "none")
    for v in dropped:
        del videos[v]
    doc = {
        "version": "nantes_video_defects_v3",
        "derived_from": {"v1": str(V1.relative_to(ROOT)), "v1_sha256": hashlib.sha256(V1.read_bytes()).hexdigest(),
                         "timeline_audit": str(ALIGN.relative_to(ROOT)),
                         "timeline_audit_sha256": hashlib.sha256(ALIGN.read_bytes()).hexdigest(),
                         "supersedes": "data/qc/nantes_video_defects_v2.json (in-sample timeline audit)"},
        "tiers": {"A": "label-image mismatch: removed from training", "B": "image artefact, label correct: kept (labels cut at cutoff_h)",
                  "C": "atypical biology: kept everywhere"},
        "categories": {k: {"tier": t, "description": d} for k, (t, d) in CATEGORIES.items()},
        "timeline_rule": "out-of-fold fit lowers phase error by >= 0.3 and to <= 60 %, slope inside grid; disagreements with "
                         "the file signature decided by inspection",
        "audit_vs_signature": {"agree": int((al.audit_flag == al.file_signature).sum()), "disagree": disagree},
        "inspected_by": v1["inspected_by"],
        "dropped_without_defect": dropped,
        "videos": dict(sorted(videos.items())),
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    cut = {v: e["cutoff_h"] for v, e in videos.items() if "cutoff_h" in e and e["tier"] != "A"}
    CUT_OUT.write_text(json.dumps({"source": "nantes_video_defects_v3", "cutoff_h": cut}, indent=1) + "\n")
    tiers = pd.Series({v: e["tier"] for v, e in videos.items()}).value_counts().to_dict()
    cats = pd.Series([c for e in videos.values() for c in e["defects"]]).value_counts().to_dict()
    print(f"{len(videos)} videos; tiers {tiers}; categories {cats}; cut labels {len(cut)}; dropped {dropped}; "
          f"audit/signature disagree {disagree}")


if __name__ == "__main__":
    main()
