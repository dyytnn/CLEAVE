#!/usr/bin/env python3
"""Aggregate the preregistered v27 EmbryoDiff validation comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/h7/v27"
V26_ROOT = ROOT / "runs/h7/v26"
ARMS = ("sce", "sce_diffusion", "sce_boundary_diffusion")


def sample_sd(values: list[float]) -> float:
    """Return sample SD or NaN when fewer than two values exist."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def metrics(val: dict) -> dict:
    """Select the locked headline and guardrail metrics."""
    return {
        "p": float(val["p"]),
        "p_t": float(val["p_t"]),
        "p_v": float(val["p_v"]),
        "edit": float(val["edit"]),
        "f1_10": float(val["f1"]["10"]),
        "f1_25": float(val["f1"]["25"]),
        "f1_50": float(val["f1"]["50"]),
        "mae_h_all": float(val["mae_h_all"]),
        "timing_error_h": val["timing_error_h"],
    }


def load_v27(arm: str) -> list[dict]:
    """Load the fixed-checkpoint result for every matched source seed."""
    rows = []
    for seed in range(3):
        path = RUN_ROOT / f"v27_{arm}_src{seed}_seed{seed}" / "results.json"
        result = json.loads(path.read_text())
        if result.get("test_evaluated") is not False or "test" in result:
            raise ValueError(f"{path}: v27 must remain validation-only")
        if arm == "sce":
            value = result["val"]
            ddim = None
        else:
            ddim = result.get("diffusion_steps_val")
            if set(ddim or {}) != {"1", "15", "25"}:
                raise ValueError(f"{path}: missing fixed-checkpoint DDIM diagnostics")
            value = ddim["25"]
        rows.append({"seed": seed, **metrics(value), "ddim": ddim})
    return rows


def load_v26() -> list[dict]:
    """Load the promoted seed-matched full-video visual parent."""
    rows = []
    for seed in range(3):
        path = V26_ROOT / f"v26_full_visual_src{seed}_seed{seed}" / "results.json"
        result = json.loads(path.read_text())
        if result.get("test_evaluated") is not False or "test" in result:
            raise ValueError(f"{path}: v26 parent must be validation-only")
        rows.append({"seed": seed, **metrics(result["val"])})
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Aggregate scalar metrics across seeds."""
    output = {"seeds": [row["seed"] for row in rows], "n_seeds": len(rows)}
    for metric in (
        "p",
        "p_t",
        "p_v",
        "edit",
        "f1_10",
        "f1_25",
        "f1_50",
        "mae_h_all",
    ):
        values = [row[metric] for row in rows]
        output[metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "sd": sample_sd(values),
        }
    return output


def paired(treatment: list[dict], control: list[dict]) -> dict:
    """Compute seed-matched treatment-minus-control deltas."""
    if [row["seed"] for row in treatment] != [row["seed"] for row in control]:
        raise ValueError("seed order differs between paired arms")
    output = {}
    for metric in (
        "p",
        "p_t",
        "p_v",
        "edit",
        "f1_10",
        "f1_25",
        "f1_50",
        "mae_h_all",
    ):
        values = [a[metric] - b[metric] for a, b in zip(treatment, control)]
        wins = (
            sum(value < 0 for value in values)
            if metric == "mae_h_all"
            else sum(value > 0 for value in values)
        )
        output[metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "sd": sample_sd(values),
            "wins": int(wins),
        }
    output["gate"] = {
        "p_t_mean_ge_0p01": output["p_t"]["mean"] >= 0.01,
        "p_t_wins_3_of_3": output["p_t"]["wins"] == 3,
        "f1_50_mean_ge_minus_0p5": output["f1_50"]["mean"] >= -0.5,
        "mae_mean_le_plus_0p1h": output["mae_h_all"]["mean"] <= 0.1,
    }
    output["gate"]["passed"] = all(output["gate"].values())
    return output


def ddim_aggregate(rows: list[dict]) -> dict:
    """Aggregate 1/15/25-step metrics from one selected checkpoint per seed."""
    output = {}
    for step in ("1", "15", "25"):
        step_rows = [{"seed": row["seed"], **metrics(row["ddim"][step])} for row in rows]
        output[step] = aggregate(step_rows)
    return output


def ddim_rows(rows: list[dict], step: str) -> list[dict]:
    """Return scalar records for one fixed-checkpoint DDIM step count."""
    return [{"seed": row["seed"], **metrics(row["ddim"][step])} for row in rows]


def main() -> None:
    """Write the machine-readable v27 summary after all nine runs finish."""
    records = {arm: load_v27(arm) for arm in ARMS}
    parent = load_v26()
    summary = {
        "version": "v27_summary_v1",
        "split": "data/splits/nantes_grouped_v1.json",
        "test_evaluated": False,
        "protocol": "transparent EmbryoDiff architecture adaptation;",
        "arms": {arm: aggregate(rows) for arm, rows in records.items()},
        "ddim_steps": {
            arm: ddim_aggregate(records[arm]) for arm in ("sce_diffusion", "sce_boundary_diffusion")
        },
        "component_diffusion_vs_sce": paired(records["sce_diffusion"], records["sce"]),
        "component_boundary_vs_diffusion": paired(
            records["sce_boundary_diffusion"], records["sce_diffusion"]
        ),
        "primary_full_vs_v26_full_visual": paired(records["sce_boundary_diffusion"], parent),
        "sce_vs_v26_full_visual": paired(records["sce"], parent),
        "diffusion_1step_vs_sce": paired(ddim_rows(records["sce_diffusion"], "1"), records["sce"]),
        "boundary_1step_vs_diffusion_1step": paired(
            ddim_rows(records["sce_boundary_diffusion"], "1"),
            ddim_rows(records["sce_diffusion"], "1"),
        ),
        "v26_full_visual": aggregate(parent),
    }
    scalars = [
        value
        for arm in summary["arms"].values()
        for metric in (
            "p",
            "p_t",
            "p_v",
            "edit",
            "f1_10",
            "f1_25",
            "f1_50",
            "mae_h_all",
        )
        for value in arm[metric]["values"]
    ]
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("non-finite v27 metric")
    path = RUN_ROOT / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
