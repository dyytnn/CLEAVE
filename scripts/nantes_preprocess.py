#!/usr/bin/env python3
"""WP9 prep: build a clean, unified per-frame manifest for the Nantes morphokinetic dataset.

Reads the extracted, quarantine-cleaned dataset at ``--root`` (default
data/raw/nantes_embryo_dataset, see /) and joins, per
frame: the RUN-numbered image path, its time_elapsed hour (frame_index matched directly -- verified
2026-09-05 that time_elapsed's frame_index is the same integer space as the RUN filename number and
the phases.csv frame range, not an independent offset; frame_index 0 in time_elapsed has no RUN0
image and is simply dropped, by design), its kinetic-phase label (from *_phases.csv, or None if
before the first / after the last labelled phase), and its TE/ICM grade (grades.csv, mostly NA).

52/704 F0 videos were found 100% JPEG-truncated (audit_full.py, 2026-09-05) and must already be
quarantined out of ``--root``/embryo_dataset before running this (see find_corrupt_videos.py); this
script assumes that is done and does not re-check per-frame corruption (none of the surviving videos
had partial corruption in the audit). The 211 duplicated ``<video>/F0/`` subfolders must also already
be removed (bit-identical to the parent, confirmed 2026-09-05).

Frames after the last annotated phase end (or after time_elapsed's last row) are exactly where our
own time-lapse videos showed an empty well (embryo removed for transfer/freeze, camera left running,
scripts/extract_n5_ssl_subset.py). Only for those "tail" frames, a Laplacian-variance focus score on
the central quarter is computed and a frame flagged blank below --min-focus (same heuristic and
threshold rationale as extract_n5_ssl_subset.py); in-annotation frames are trusted and not scored.

    python scripts/nantes_preprocess.py --root data/raw/nantes_embryo_dataset --plane embryo_dataset --out data/derived/nantes_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

RUN_RE = re.compile(r"RUN(\d+)\.jpeg$")


def load_time_elapsed(path: Path) -> dict[int, float]:
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[int(r["frame_index"])] = float(r["time"])
    return out


def load_phases(path: Path) -> list[tuple[str, int, int]]:
    out = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            out.append((row[0], int(row[1]), int(row[2])))
    return out


def phase_for(frame_idx: int, phases: list[tuple[str, int, int]]) -> str | None:
    for name, start, end in phases:
        if start <= frame_idx <= end:
            return name
    return None


def focus_score(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    central = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return float(cv2.Laplacian(central, cv2.CV_64F).var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/nantes_embryo_dataset")
    ap.add_argument("--plane", default="embryo_dataset", help="folder name under --root holding per-video image dirs")
    ap.add_argument("--min-focus", type=float, default=15.0, help="see extract_n5_ssl_subset.py for the derivation")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    img_root = root / args.plane
    ann_dir = root / "ann/embryo_dataset_annotations"
    te_dir = root / "time_elapsed/embryo_dataset_time_elapsed"
    grades_path = root / "embryo_dataset_grades.csv"

    with open(grades_path) as f:
        grades = {r["video_name"]: r for r in csv.DictReader(f)}

    video_dirs = sorted(d for d in img_root.iterdir() if d.is_dir())
    rows = []
    n_tail_scored = n_blank = 0
    for n_done, vd in enumerate(video_dirs, 1):
        name = vd.name
        images = {}
        for p in vd.iterdir():
            if p.is_dir():
                continue
            m = RUN_RE.search(p.name)
            if m:
                images[int(m.group(1))] = p
        te_path = te_dir / f"{name}_timeElapsed.csv"
        time_map = load_time_elapsed(te_path) if te_path.exists() else {}
        ann_path = ann_dir / f"{name}_phases.csv"
        phases = load_phases(ann_path) if ann_path.exists() else []
        last_phase_end = max((e for _, _, e in phases), default=None)
        grade = grades.get(name, {})

        for frame_idx, path in sorted(images.items()):
            phase = phase_for(frame_idx, phases)
            is_tail = last_phase_end is not None and frame_idx > last_phase_end
            focus = blank = None
            if is_tail:
                focus = focus_score(path)
                blank = focus < args.min_focus
                n_tail_scored += 1
                n_blank += int(blank)
            rows.append(
                {
                    "video": name,
                    "plane": args.plane,
                    "frame_index": frame_idx,
                    "path": str(path),
                    "time_h": time_map.get(frame_idx),
                    "phase": phase,
                    "is_tail": is_tail,
                    "focus": focus,
                    "is_blank": blank,
                    "grade_TE": grade.get("TE"),
                    "grade_ICM": grade.get("ICM"),
                }
            )
        if n_done % 100 == 0:
            print(f"{n_done}/{len(video_dirs)} videos processed, {len(rows)} frames so far", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_videos = len({r["video"] for r in rows})
    n_usable = sum(1 for r in rows if not r["is_blank"])
    print(f"\ndone: {len(rows)} frames, {n_videos} videos -> {out}")
    print(f"tail frames scored for blank: {n_tail_scored}, flagged blank: {n_blank}")
    print(f"usable (non-blank) frames: {n_usable}/{len(rows)}")


if __name__ == "__main__":
    main()
