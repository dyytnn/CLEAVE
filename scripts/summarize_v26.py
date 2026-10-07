#!/usr/bin/env python3
"""Aggregate the preregistered v26 validation-only comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/h7/v26"
ARMS = ("clip16_visual", "full_visual", "full_clock", "full_visual_clock")


def sample_sd(values: list[float]) -> float:
    """Sample standard deviation, or NaN for fewer than two observations."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def load_arm(arm: str) -> list[dict]:
    """Load all three matched-seed validation results for one arm."""
    records = []
    for seed in range(3):
        path = RUN_ROOT / f"v26_{arm}_src{seed}_seed{seed}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text())
        if result.get("test_evaluated") is not False or "test" in result:
            raise ValueError(f"{path}: v26 must remain validation-only")
        val = result["val"]
        records.append(
            {
                "seed": seed,
                "p_t": float(val["p_t"]),
                "p_v": float(val["p_v"]),
                "f1_50": float(val["f1"]["50"]),
                "mae_h_all": float(val["mae_h_all"]),
                "edit": float(val["edit"]),
                "clock_stress_val": result.get("clock_stress_val"),
            }
        )
    return records


def aggregate(records: list[dict]) -> dict:
    """Mean/sample-SD summary for headline validation metrics."""
    out = {"seeds": [record["seed"] for record in records], "n_seeds": len(records)}
    for metric in ("p_t", "p_v", "f1_50", "mae_h_all", "edit"):
        values = [record[metric] for record in records]
        out[metric] = {"values": values, "mean": float(np.mean(values)), "sd": sample_sd(values)}
    return out


def paired(treatment: list[dict], control: list[dict]) -> dict:
    """Seed-matched treatment-minus-control deltas and preregistered bars."""
    if [r["seed"] for r in treatment] != [r["seed"] for r in control]:
        raise ValueError("seed order differs between paired arms")
    out = {}
    for metric in ("p_t", "p_v", "f1_50", "mae_h_all", "edit"):
        delta = [a[metric] - b[metric] for a, b in zip(treatment, control)]
        wins = (
            sum(value < 0 for value in delta)
            if metric == "mae_h_all"
            else sum(value > 0 for value in delta)
        )
        out[metric] = {
            "values": delta,
            "mean": float(np.mean(delta)),
            "sd": sample_sd(delta),
            "wins": int(wins),
        }
    out["gate"] = {
        "p_t_mean_ge_0p01": out["p_t"]["mean"] >= 0.01,
        "p_t_wins_3_of_3": out["p_t"]["wins"] == 3,
        "f1_50_mean_ge_minus_0p5": out["f1_50"]["mean"] >= -0.5,
        "mae_mean_le_plus_0p1h": out["mae_h_all"]["mean"] <= 0.1,
    }
    out["gate"]["passed"] = all(out["gate"].values())
    return out


def stress_directionality(stressed: list[dict], reference: list[dict]) -> dict:
    """Compare each prespecified stressed clock input with visual-only p_t."""
    reports = {}
    for stress in ("plus6h", "minus6h", "rate1p10"):
        values = [float(record["clock_stress_val"][stress]["p_t"]) for record in stressed]
        deltas = [value - base["p_t"] for value, base in zip(values, reference)]
        reports[stress] = {
            "p_t_values": values,
            "delta_vs_full_visual_values": deltas,
            "delta_vs_full_visual_mean": float(np.mean(deltas)),
            "delta_vs_full_visual_sd": sample_sd(deltas),
            "wins": int(sum(delta > 0 for delta in deltas)),
            # "Directionally useful" is deliberately the lenient mean-positive
            # interpretation; a stress does not need to clear +0.01 again.
            "directionally_useful": bool(np.mean(deltas) > 0),
        }
    reports["all_stresses_directionally_useful"] = all(
        report["directionally_useful"] for report in reports.values() if isinstance(report, dict)
    )
    return reports


def main() -> None:
    records = {arm: load_arm(arm) for arm in ARMS}
    clock_increment = paired(records["full_visual_clock"], records["full_visual"])
    clock_stress_directionality = stress_directionality(
        records["full_visual_clock"], records["full_visual"]
    )
    summary = {
        "version": "v26_summary_v1",
        "split": "data/splits/nantes_grouped_v1.json",
        "test_evaluated": False,
        "arms": {arm: aggregate(value) for arm, value in records.items()},
        "primary_full_visual_vs_clip16_visual": paired(
            records["full_visual"], records["clip16_visual"]
        ),
        "clock_increment_visual_clock_vs_visual": clock_increment,
        "clock_only_within_0p01_of_full_visual": bool(
            abs(
                np.mean([r["p_t"] for r in records["full_clock"]])
                - np.mean([r["p_t"] for r in records["full_visual"]])
            )
            <= 0.01
        ),
        "clock_stress": {
            arm: [record["clock_stress_val"] for record in records[arm]]
            for arm in ("full_clock", "full_visual_clock")
        },
        "clock_stress_directionality": clock_stress_directionality,
        "clock_promotion": {
            "nominal_gate_passed": clock_increment["gate"]["passed"],
            "all_stresses_directionally_useful": clock_stress_directionality[
                "all_stresses_directionally_useful"
            ],
            "passed": bool(
                clock_increment["gate"]["passed"]
                and clock_stress_directionality["all_stresses_directionally_useful"]
            ),
        },
    }
    if not all(
        math.isfinite(v)
        for arm in summary["arms"].values()
        for metric in ("p_t", "p_v", "f1_50", "mae_h_all", "edit")
        for v in arm[metric]["values"]
    ):
        raise ValueError("non-finite v26 metric")
    path = RUN_ROOT / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
