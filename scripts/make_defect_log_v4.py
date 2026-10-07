#!/usr/bin/env python3
"""Defect log v4: v3 with the round-6 corrections (results/qc_inspection_round6.md, AI-assisted inspection of the raw
frames).

* three "timeline mismatches" are not the clock defect: LP284-3 is an embryo removed on day 3 (tier B, cut 71.2 h),
  EH315-3 and EH315-8 are abrupt cleavages annotated as normal development (tier A, annotation_ahead);
* PC758-2 is not a clock mismatch either: its early onsets fit the released pairing, its t5-t8 labels run 6-12 h behind
  the images (tier A, annotation_behind);
* GC340-10 and RC545-2-8 keep the mismatch, now with the inspection note the v3 log left empty;
* the five strongest unlogged disagreements of the out-of-fold audit and the edge-fitted RM126-5 were inspected: AMT360-1-9
  (embryo removed, tier B), OJ319-10 (annotation stops at t6 while the embryo cavitates, tier B), CAV074-9 (arrest,
  tier C); LA367-4 and OJ319-7 have correct labels; RM126-5 is a direct cleavage, not a mismatch (tier C);
* cut-offs of PMDPI029-1-8 and RM126-8 move to the first empty-well image (the structure detector of v3 fired late);
* every entry states its basis: inspection, or agreement of the out-of-fold alignment and the time-file signature.

The cleaned training split nantes_grouped_v2_trainclean_v3 (frozen, used for v38) was built from v3; v4 changes its
tier-A set by one video (LP284-3 A -> B, PC758-2 stays A) and adds no tier-A video.

    PYTHONPATH=src:scripts python scripts/make_defect_log_v4.py   # -> data/qc/nantes_video_defects_v4.json, ..._cutoff_v4.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from make_defect_log_v2 import CATEGORIES as CAT_V2

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data/qc/nantes_video_defects_v3.json"
OUT = ROOT / "data/qc/nantes_video_defects_v4.json"
CUT_OUT = ROOT / "data/qc/nantes_label_cutoff_v4.json"
CATEGORIES = dict(CAT_V2) | {
    "annotation_behind": ("A", "Annotated onsets run several hours behind what the images show."),
    "annotation_truncated": ("B", "The annotation stops at one phase and holds it to the end while the embryo develops further; labels are cut where the images leave that phase."),
    "developmental_arrest": ("C", "The embryo arrests and degenerates; the last annotated phase is held to the end."),
}
SRC = "inspection (round 6)"
#: video -> (defects, note, cutoff_h or None); replaces the v3 entry, or adds one
ROUND6 = {
    "LP284-3": (["embryo_removed"], "Index pairing correct: 1 cell at tPNf, 2 cells plus fragments at t2, ~4 at t4, ~8 at t8 (RUN247). 8-cell embryo at RUN280 (70.9 h); well and zona gone from RUN281 (71.2 h) to the end.", 71.2),
    "PC758-2": (["annotation_behind"], "Index pairing fits tPNf and t2 (one cell at RUN86, cleavage at RUN91); ~8 cells by RUN172 (58 h) but t5-t8 annotated at RUN175-190: labels 6-12 h late.", None),
    "EH315-8": (["abrupt_cleavage", "annotation_ahead", "heavy_fragmentation"], "One cell with inclusions through annotated t2/t3/t4 (to RUN171, 43.7 h); abrupt cleavage to one large blastomere plus fragments at RUN172-173. No blastocoel at annotated tSB/tB/tEB in any plane.", None),
    "EH315-3": (["abrupt_cleavage", "annotation_ahead", "heavy_fragmentation"], "One cell to RUN108 (27.8 h), direct cleavage to a fragmented many-cell mass by RUN110-113 (annotated tPNf). Compact mass to the end; no blastocoel at annotated tSB/tB/tEB in any plane.", None),
    "RM126-5": (["abrupt_cleavage", "heavy_fragmentation"], "One cell to RUN135 (35.0 h), three cells plus fragments at RUN138 (annotated t2, 35.8 h); the released pairing fits, the edge fit (a=1.595) does not. Arrests near 8 cells.", None),
    "AMT360-1-9": (["embryo_removed", "heavy_fragmentation"], "One cell to RUN35 (27.6 h); two blastomeres in heavy fragments at annotated t2 (RUN38), released pairing correct. Cavity from ~111 h, tB never annotated; well empty from RUN390 (116.9 h).", 116.9),
    "OJ319-10": (["annotation_truncated", "heavy_fragmentation"], "t2/t3 match the released pairing (RUN95/103). The annotation ends with t6 (from RUN212, 53.6 h) held to the last frame, but cavitation is visible from RUN399 (100.8 h).", 100.8),
    "CAV074-9": (["developmental_arrest"], "t2 (RUN154, 39.7 h) and t3 (RUN227, 58.1 h) match the released pairing. Arrests with one large and ~4-6 small blastomeres and degenerates from ~127 h; t4 held to 140 h.", None),
}
NOTES = {
    "GC340-10": "Pronuclei to RUN94, first cleavage RUN96; the released image at annotated tPNf (RUN109) shows ~4 cells. Under the 1.25x pairing tPNf/t2/tM/tB fall on RUN87/95/358/454: pronuclei, furrow, compaction, cavity.",
    "RC545-2-8": "Pronuclei to RUN144, direct cleavage to a fragmented mass at RUN146; released tPNf/t2/t4 (173/184/189) all on that mass, the 1.25x pairing gives RUN138/147/151. tM/tSB lie beyond the last image.",
}
CUTOFF = {"PMDPI029-1-8": 119.8, "RM126-8": 119.0}   # first empty-well image (time file)
#: v1 notes that contradict the measured cut-offs or the zona argument: the well, zona included, empties
#: between two frames, as in removal; the cut-off is the first empty-well image
NOTE_FIX = {
    "LM184-4": "Hatching blastocyst to 119.6 h; the well, zona included, is empty from 120.0 h to the end, annotated tEB.",
    "RM126-4": "Expanded blastocyst at 114 h; the well, zona included, is empty from 119.0 h to the end, annotated tEB.",
    "RM126-8": "Hatching blastocyst at RUN465; the well, zona included, is empty from RUN466 (119.0 h) to the end, annotated tEB.",
    "RM126-9": "Blastocyst to 119 h; the well, zona included, is empty from 119.0 h to the end, annotated tEB.",
    "OC110-5": "Four cells visible at 53 h; from about 69.6 h to the end the well is featureless, all of it annotated t8: about 70 hours of 8-cell labels on frames with no embryo.",
    "DV116-3": "Embryo visible to t5 (48 h); from about 68.5 h to the end the well is featureless, annotated t8 throughout.",
    "PMDPI029-1-8": "Expanded blastocyst at RUN468 (119.6 h); the well is empty from RUN469 (119.8 h) to the end, annotated tB.",
    "CC938-4": "At annotated t2 (22 h) still one cell with pronuclei; at annotated t4 two cells and at annotated t9+ four. Runs one to two division stages ahead.",
}
#: inspected and found correct: recorded so the audit's strongest disagreements are accounted for
CLEARED = {"LA367-4": "labels correct; off-centre, dim embryo", "OJ319-7": "labels correct; low-contrast blastomeres"}


def main() -> None:
    v3 = json.loads(V3.read_text())
    vids = {v: dict(e) for v, e in v3["videos"].items()}
    for v, (cats, note, cut) in ROUND6.items():
        e = {"defects": sorted(cats), "note": note, "source": SRC}
        if v in vids and "timeline" in vids[v]:
            e["timeline_fit_v3"] = vids[v]["timeline"]
        if cut is not None:
            e["cutoff_h"] = cut
            e["cutoff_source"] = {"inspection": cut}
        vids[v] = e
    for v, note in NOTES.items():
        vids[v]["note"] = note
        vids[v]["source"] = vids[v].get("source", "") + "; " + SRC
    for v, note in NOTE_FIX.items():
        vids[v]["note"] = note
    for v, cut in CUTOFF.items():
        vids[v]["cutoff_h"] = cut
        vids[v]["cutoff_source"] = {"inspection (first empty-well image)": cut, "v3": vids[v].get("cutoff_h")}
    for v, e in vids.items():
        e["tier"] = min(CATEGORIES[c][0] for c in e["defects"])
        if "timeline_mismatch" in e["defects"] and not e.get("note"):
            e["basis"] = "out-of-fold alignment and time-file signature (not inspected)"
        elif "timeline_mismatch" in e["defects"]:
            e["basis"] = "out-of-fold alignment and inspection"
        else:
            e["basis"] = "inspection"
    doc = {k: v for k, v in v3.items() if k not in ("videos", "version", "categories")}
    doc.update({"version": "nantes_video_defects_v4",
                "derived_from": {"v3": str(V3.relative_to(ROOT)), "v3_sha256": hashlib.sha256(V3.read_bytes()).hexdigest(),
                                 "round6_inspection": "results/qc_inspection_round6.md"},
                "categories": {k: {"tier": t, "description": d} for k, (t, d) in CATEGORIES.items()},
                "inspected_and_cleared": CLEARED,
                "inspected_by": "AI-assisted inspection: a vision-capable language model under the authors' direction; not an embryologist",
                "videos": dict(sorted(vids.items()))})
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    cut = {v: e["cutoff_h"] for v, e in vids.items() if "cutoff_h" in e and e["tier"] != "A"}
    CUT_OUT.write_text(json.dumps({"source": "nantes_video_defects_v4", "cutoff_h": cut}, indent=1) + "\n")
    s = pd.Series({v: e["tier"] for v, e in vids.items()})
    tl = sum("timeline_mismatch" in e["defects"] for e in vids.values())
    nb = sum(e["basis"].startswith("out-of-fold alignment and time-file") for e in vids.values())
    print(f"{len(vids)} videos; tiers {s.value_counts().to_dict()}; timeline mismatches {tl} ({nb} not inspected); cut labels {len(cut)}")


if __name__ == "__main__":
    main()
