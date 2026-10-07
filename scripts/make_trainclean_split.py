#!/usr/bin/env python3
"""nantes_grouped_v2_trainclean_v1: nantes_grouped_v2 with the inspected defective videos removed from
*train only*.

The question is what the defective training videos did to the models, so this is the single change:
val and test are copied verbatim, so a run on this split is paired, video for video, with the committed
run of the same recipe and seed on nantes_grouped_v2. Val keeps its defective videos on purpose:
cleaning it too would change checkpoint selection and confound the comparison.

    PYTHONPATH=src python scripts/make_trainclean_split.py              # v1: the 41 inspected videos out of train
    PYTHONPATH=src python scripts/make_trainclean_split.py --version 3  # v3: the same with defect log v3 (out-of-fold
                                                                        # audit); supersedes v2, which kept mismatches
    PYTHONPATH=src python scripts/make_trainclean_split.py --version 2  # v2: tier-A videos of defect log v2 out of
                                                                        # train (tier-B label tails are cut by the
                                                                        # data.train_label_cutoff file instead)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import argparse

from stseg.data.patient import patient_of

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/splits/nantes_grouped_v2.json"
VERSIONS = {1: (ROOT / "data/qc/nantes_video_defects_v1.json", None),
            2: (ROOT / "data/qc/nantes_video_defects_v2.json", "A"),
            3: (ROOT / "data/qc/nantes_video_defects_v3.json", "A")}   # v3: out-of-fold timeline audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1, choices=sorted(VERSIONS))
    ver = ap.parse_args().version
    QC, tier = VERSIONS[ver]
    OUT = ROOT / f"data/splits/nantes_grouped_v2_trainclean_v{ver}.json"
    if OUT.exists():
        raise SystemExit(f"{OUT.name} exists and splits are frozen once written")
    src = json.loads(SRC.read_text())
    log = json.loads(QC.read_text())["videos"]
    defective = {v for v, e in log.items() if tier is None or e.get("tier") == tier}
    train = sorted(v for v in src["videos"]["train"] if v not in defective)
    removed = sorted(set(src["videos"]["train"]) & defective)
    doc = {
        "version": f"nantes_grouped_v2_trainclean_v{ver}",
        "seed": src["seed"],
        "grouping": src["grouping"],
        "source": src["source"],
        "derived_from": {"split": "nantes_grouped_v2", "sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
                         "defect_list": str(QC.relative_to(ROOT)), "tier_removed": tier or "all",
                         "defect_list_sha256": hashlib.sha256(QC.read_bytes()).hexdigest()},
        "change": "defective videos removed from train; val and test identical to nantes_grouped_v2",
        "removed_from_train": removed,
        "n_patients": {"train": len({patient_of(v) for v in train}), **{k: src["n_patients"][k] for k in ("val", "test")}},
        "patients": {"train": sorted({patient_of(v) for v in train}), "val": src["patients"]["val"], "test": src["patients"]["test"]},
        "videos": {"train": train, "val": src["videos"]["val"], "test": src["videos"]["test"]},
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"train {len(src['videos']['train'])} -> {len(train)} (removed {len(removed)}); val/test unchanged -> {OUT.name}")


if __name__ == "__main__":
    main()
