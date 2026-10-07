#!/usr/bin/env python3
"""Aggregate three seed-matched v25 validation decoder results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]


def describe(values: np.ndarray) -> dict:
    """Return values, mean and sample SD without implying confirmation."""
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "n_seeds": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/h7/v25")
    parser.add_argument("--out", default="runs/h7/v25/decoder_summary.json")
    args = parser.parse_args()
    root = ROOT / args.root
    reports = []
    for seed in range(3):
        path = root / f"v25_onset_decoder_seed{seed}" / "results.json"
        report = json.loads(path.read_text())
        if report["source_seed"] != seed or report["test_evaluated"] is not False:
            raise ValueError(f"bad integration provenance: {path}")
        reports.append(report)
    metrics = {
        "p_t": ("p_t",),
        "p_v": ("p_v",),
        "f1_50": ("f1", "50"),
        "mae_h_all": ("mae_h_all",),
    }

    def values(section: str, keys: tuple[str, ...]) -> np.ndarray:
        output = []
        for report in reports:
            value = report[section]
            for key in keys:
                value = value[key]
            output.append(float(value))
        return np.asarray(output)

    summary = {
        "kind": "three_seed_matched_grouped_validation_event_integration",
        "test_evaluated": False,
        "gate": {
            "required_mean_delta_p_t": 0.01,
            "required_wins": 3,
            "requires_no_material_f1_or_timing_regression": True,
        },
        "metrics": {},
    }
    for name, keys in metrics.items():
        baseline = values("baseline", keys)
        predicted = values("predicted", keys)
        delta = predicted - baseline
        paired = ttest_rel(predicted, baseline)
        summary["metrics"][name] = {
            "baseline": describe(baseline),
            "predicted": describe(predicted),
            "paired_delta": describe(delta),
            "wins": int((delta > 0).sum()),
            "paired_t_p_two_sided_descriptive_n3": float(paired.pvalue)
            if np.any(delta != 0)
            else 1.0,
        }
    p_t = summary["metrics"]["p_t"]
    f1 = summary["metrics"]["f1_50"]["paired_delta"]["mean"]
    mae = summary["metrics"]["mae_h_all"]["paired_delta"]["mean"]
    summary["gate"]["passed"] = bool(
        p_t["paired_delta"]["mean"] >= 0.01
        and p_t["wins"] == 3
        and f1 >= -0.5
        and mae <= 0.1
    )
    summary["gate"]["secondary_thresholds"] = {
        "f1_50_delta_min": -0.5,
        "mae_h_delta_max": 0.1,
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
