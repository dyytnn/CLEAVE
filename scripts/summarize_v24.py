#!/usr/bin/env python3
"""Aggregate the preregistered v24 matched-target validation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]


def load_arm(root: Path, arm: str) -> list[dict]:
    """Load exactly three completed non-smoke seed reports."""
    reports = []
    for seed in range(3):
        path = root / f"v24_{arm}_seed{seed}" / "results.json"
        report = json.loads(path.read_text())
        if report["test_evaluated"] is not False or report["seed"] != seed:
            raise ValueError(f"invalid v24 result provenance: {path}")
        reports.append(report)
    return reports


def vector(reports: list[dict], *keys: str) -> np.ndarray:
    """Extract one scalar from each seed's best validation record."""
    values = []
    for report in reports:
        value = report["best"]
        for key in keys:
            value = value[key]
        values.append(float(value))
    return np.asarray(values)


def summary(values: np.ndarray) -> dict:
    """Three-seed descriptive summary with sample SD."""
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "n_seeds": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/h7/v24")
    parser.add_argument("--out", default="runs/h7/v24/summary.json")
    args = parser.parse_args()
    root = ROOT / args.root
    onset, window = load_arm(root, "dense_onset"), load_arm(root, "dense_window")
    metrics = {
        "event_ap_at_2": ("event_at_2", "event_ap"),
        "event_ap_at_1": ("event_at_1", "event_ap"),
        "event_ap_exact": ("event_at_0", "event_ap"),
        "event_f1_at_2": ("event_at_2", "best_f1"),
        "dense_onset_ap": ("dense_onset", "onset_ap"),
        "dense_brier": ("dense_onset", "brier"),
    }
    report = {
        "kind": "matched_validation_target_ablation",
        "split": "data/splits/nantes_grouped_v1.json",
        "test_evaluated": False,
        "selection_note": "best epoch and metrics use the same validation partition",
        "metrics": {},
    }
    for name, keys in metrics.items():
        a, b = vector(onset, *keys), vector(window, *keys)
        delta = a - b
        paired = ttest_rel(a, b)
        report["metrics"][name] = {
            "onset": summary(a),
            "window": summary(b),
            "paired_delta": summary(delta),
            "wins": int((delta > 0).sum()),
            "paired_t_p_two_sided_descriptive_n3": float(paired.pvalue),
        }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
