#!/usr/bin/env python3
"""Emit the CLEAVE release manifest: excluded videos, the per-video defect log, split checksums.

Closes the reviewer request (review_round1 RM-46, RM-34) that the exclusion list, the outlier list and
the grouped split be released as named, checksummed artifacts rather than described in prose.

The 52 excluded videos are the F0 video folders that the corruption audit quarantined out of the
extracted archive; they are recovered here from the quarantine directory when it is reachable, and
otherwise from the complement of the per-frame manifest against the released 704-video list.

  python scripts/make_release_manifest.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("data/raw/nantes_embryo_dataset")
OUT = ROOT / "results/tempo_bench_release.json"
QC = ROOT / "data/qc/nantes_video_defects_v3.json"
QC_V1 = ROOT / "data/qc/nantes_video_defects_v1.json"
ALIGN = ROOT / "results/frame_timing/alignment_oof.csv"
CUTOFF = ROOT / "data/qc/nantes_label_cutoff_v3.json"
AUDIT = ROOT / "results/protocol_audit.json"
#: released split files and what each is for; exposure numbers are read from the audit, never typed here
SPLITS = {
    "nantes_grouped_v1": "cycle-level grouping; the split every model in the paper is evaluated on",
    "nantes_grouped_v2": "couple-level grouping; the split CLEAVE recommends",
    "nantes_grouped_v2_trainclean_v1": "superseded: nantes_grouped_v2 minus every video of the first, untiered log; kept "
                                       "for the v36 retraining it was used in",
    "nantes_grouped_v2_trainclean_v3": "nantes_grouped_v2 with tier A of defect log v3 removed from train only (use with "
                                       "data/qc/nantes_label_cutoff_v3.json); val and test identical; the cleaned split CLEAVE recommends",
    "nantes_grouped_v2_trainclean_v2": "superseded: built from an in-sample timeline audit that could not see mismatches among "
                                       "its own classifier's training videos",
    "nantes_siblingleak_clean_v1": "paired sibling-leak design, clean arm",
    "nantes_siblingleak_leaky_v1": "paired sibling-leak design, leaky arm; identical val/test and training-set size to the clean arm",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def excluded_videos() -> tuple[list[str], str]:
    q = RAW / "_quarantine_corrupt/embryo_dataset"
    if q.is_dir():
        return sorted(p.name for p in q.iterdir() if p.is_dir()), str(q)
    man = ROOT / "data/derived/nantes_manifest_F0.csv"
    kept = set()
    with man.open() as fh:
        for row in csv.DictReader(fh):
            kept.add(row["video"])
    allv = {line.strip() for line in (RAW / "zenodo_manifest.txt").read_text().splitlines() if line.strip()}
    return sorted(allv - kept), str(man)


def main() -> None:
    videos, _ = excluded_videos()
    audit = json.loads(AUDIT.read_text())["own_splits"]
    qc = json.loads(QC.read_text())
    splits = {}
    for name, note in SPLITS.items():
        path = ROOT / f"data/splits/{name}.json"
        c = audit[name]["couple"]
        splits[f"data/splits/{name}.json"] = {
            "sha256": _sha256(path), "bytes": path.stat().st_size, "note": note,
            "test_videos": c["n_test_videos"], "test_videos_with_couple_sibling_in_train": c["n_exposed_test_videos"]}
    doc = {
        "version": "tempo_bench_release_v3",
        "dataset": "Nantes morphokinetic time-lapse (Zenodo record 7912264)",
        "dataset_doi": "10.5281/zenodo.7912264",
        "producer": "scripts/make_release_manifest.py (sole writer of this file)",
        "excluded_videos": {
            "reason": "100 % JPEG-truncated in every focal plane in the released archive",
            "n": len(videos),
            "source": "Zenodo record 7912264; identifiers of the video folders that fail to decode",
            "video_ids": videos,
        },
        "defect_log": {
            "file": "data/qc/nantes_video_defects_v3.json",
            "sha256": _sha256(QC),
            "n": len(qc["videos"]),
            "tiers": qc["tiers"],
            "status": "every video is kept in the released splits and the headline numbers; tier A is removed from training "
                      "in nantes_grouped_v2_trainclean_v3; report results with and without tier A",
            "categories": {k: v["tier"] for k, v in qc["categories"].items()},
            "videos": {v: {"tier": r["tier"], "defects": r["defects"]} for v, r in sorted(qc["videos"].items())},
            "label_cutoff": {"file": str(CUTOFF.relative_to(ROOT)), "sha256": _sha256(CUTOFF)},
            "timeline_audit": {"file": str(ALIGN.relative_to(ROOT)), "sha256": _sha256(ALIGN),
                               "producer": "scripts/audit_frame_timing.py"},
            "superseded": {"file": str(QC_V1.relative_to(ROOT)), "sha256": _sha256(QC_V1)},
            "inspected_by": qc["inspected_by"],
        },
        "grouping": {
            "rule": "stseg.data.patient.patient_of; video names are <code>[-<cycle>]-<embryo>",
            "levels": {"couple": "embryo and cycle tokens stripped (the CLEAVE unit)",
                       "cycle": "embryo token only (what nantes_grouped_v1 used)"},
            "audit": "results/protocol_audit.json, produced by scripts/audit_protocol_splits.py",
        },
        "splits": splits,
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{OUT.relative_to(ROOT)}: {len(videos)} excluded, {len(qc['videos'])} logged defects, {len(splits)} splits")


if __name__ == "__main__":
    main()
