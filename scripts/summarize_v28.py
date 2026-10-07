#!/usr/bin/env python3
"""Aggregate the preregistered v28 EMFiT common-protocol comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/h7/v28"


def scalar_metrics(value: dict) -> dict[str, float]:
    return {
        "p": float(value["p"]),
        "p_v": float(value["p_v"]),
        "p_t": float(value["p_t"]),
        "edit": float(value["edit"]),
        "f1_10": float(value["f1"]["10"]),
        "f1_25": float(value["f1"]["25"]),
        "f1_50": float(value["f1"]["50"]),
        "mae_h_all": float(value["mae_h_all"]),
    }


def load_emfit() -> list[dict]:
    rows = []
    for seed in range(3):
        path = RUN_ROOT / f"v28_emfit_src{seed}_seed{seed}" / "results.json"
        result = json.loads(path.read_text())
        if result.get("test_evaluated") is not False or "test" in result:
            raise ValueError(f"{path}: EMFiT comparison must remain validation-only")
        rows.append(
            {
                "seed": seed,
                **scalar_metrics(result["val"]),
                "training_accounting": result["training_accounting"],
            }
        )
    return rows


def load_control(kind: str) -> list[dict]:
    rows = []
    for seed in range(3):
        if kind == "sce":
            path = ROOT / "runs/h7/v27" / f"v27_sce_src{seed}_seed{seed}" / "results.json"
        elif kind == "v26_full_visual":
            path = ROOT / "runs/h7/v26" / f"v26_full_visual_src{seed}_seed{seed}" / "results.json"
        else:
            raise ValueError(f"unknown control {kind}")
        result = json.loads(path.read_text())
        if result.get("test_evaluated") is not False or "test" in result:
            raise ValueError(f"{path}: control must remain validation-only")
        rows.append({"seed": seed, **scalar_metrics(result["val"])})
    return rows


def aggregate(rows: list[dict]) -> dict:
    output = {"seeds": [row["seed"] for row in rows], "n_seeds": len(rows)}
    for metric in ("p", "p_v", "p_t", "edit", "f1_10", "f1_25", "f1_50", "mae_h_all"):
        values = [row[metric] for row in rows]
        output[metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
        }
    return output


def paired(treatment: list[dict], control: list[dict]) -> dict:
    if [row["seed"] for row in treatment] != [row["seed"] for row in control]:
        raise ValueError("seed order differs")
    output = {}
    for metric in ("p", "p_v", "p_t", "edit", "f1_10", "f1_25", "f1_50", "mae_h_all"):
        values = [a[metric] - b[metric] for a, b in zip(treatment, control)]
        wins = sum(value < 0 for value in values) if metric == "mae_h_all" else sum(value > 0 for value in values)
        output[metric] = {
            "values": values,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
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


def main() -> None:
    emfit = load_emfit()
    sce = load_control("sce")
    parent = load_control("v26_full_visual")
    summary = {
        "version": "v28_summary_v1",
        "split": "data/splits/nantes_grouped_v1.json",
        "test_evaluated": False,
        "protocol": "official-code EMFiT common-protocol adaptation;",
        "emfit": aggregate(emfit),
        "sce": aggregate(sce),
        "v26_full_visual": aggregate(parent),
        "primary_emfit_vs_sce": paired(emfit, sce),
        "emfit_vs_v26_full_visual": paired(emfit, parent),
        "training_accounting": [row["training_accounting"] for row in emfit],
    }
    path = RUN_ROOT / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
