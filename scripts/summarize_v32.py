#!/usr/bin/env python3
"""Aggregate the v32 replication of v29 ``onset_linear`` with fresh training seeds.

Nine runs: caches 0/1/2 x training seeds 3/4/5. Reports the training-seed spread of the onset head per cache and
pooled, the replication mean against the original v29 value (0.7657, one training seed per cache) and against the
v27 SCE parent (matched by cache seed), so the base's +0.0140 can be judged against its own noise.

  PYTHONPATH=src python scripts/summarize_v32.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import summarize_v29 as V29  # noqa: E402
from summarize_v27 import metrics  # noqa: E402

RUN_ROOT = ROOT / "runs/h7/v32"
METRICS = ("p", "p_t", "p_v", "edit", "f1_50", "mae_h_all")


def _sd(v: list[float]) -> float:
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def main() -> None:
    reps = {k: {s: metrics(V29._result(RUN_ROOT / f"v32_onset_linear_rep_src{k}_seed{s}" / "results.json")["val"])
                for s in (3, 4, 5)} for k in range(3)}
    v29 = {k: metrics(V29._result(V29.RUN_ROOT / f"v29_onset_linear_src{k}_seed{k}" / "results.json")["val"]) for k in range(3)}
    sce = {k: metrics(V29._result(V29.V27_ROOT / f"v27_sce_src{k}_seed{k}" / "results.json")["val"]) for k in range(3)}
    summary: dict = {"version": "v32_summary_v1", "split": "data/splits/nantes_grouped_v1.json", "test_evaluated": False,
                     "protocol": "pure training-seed replication of v29 onset_linear;",
                     "per_cache": {}, "pooled": {}, "v29_original": {}, "sce_parent": {}}
    for m in METRICS:
        allv = [reps[k][s][m] for k in range(3) for s in (3, 4, 5)]
        summary["pooled"][m] = {"values": allv, "mean": float(np.mean(allv)), "sd": _sd(allv)}
        summary["v29_original"][m] = {"values": [v29[k][m] for k in range(3)], "mean": float(np.mean([v29[k][m] for k in range(3)]))}
        summary["sce_parent"][m] = {"values": [sce[k][m] for k in range(3)], "mean": float(np.mean([sce[k][m] for k in range(3)]))}
    for k in range(3):
        summary["per_cache"][str(k)] = {m: {"values": [reps[k][s][m] for s in (3, 4, 5)],
                                            "mean": float(np.mean([reps[k][s][m] for s in (3, 4, 5)])),
                                            "sd": _sd([reps[k][s][m] for s in (3, 4, 5)]),
                                            "v29_original": v29[k][m], "sce_parent": sce[k][m]} for m in METRICS}
    pooled = summary["pooled"]["p_t"]
    # all 12 onset_linear runs (3 original + 9 replication) as the best estimate of the head's p_t and its seed SD
    twelve = pooled["values"] + summary["v29_original"]["p_t"]["values"]
    summary["onset_linear_all_12_runs"] = {"p_t_mean": float(np.mean(twelve)), "p_t_sd": _sd(twelve), "n": len(twelve)}
    summary["verdict"] = {
        "replication_mean_p_t": pooled["mean"],
        "v29_original_mean_p_t": summary["v29_original"]["p_t"]["mean"],
        "replication_minus_v29": pooled["mean"] - summary["v29_original"]["p_t"]["mean"],
        "replication_minus_sce_paired_by_cache": float(np.mean([reps[k][s]["p_t"] - sce[k]["p_t"] for k in range(3) for s in (3, 4, 5)])),
        "replication_runs_above_sce": int(sum(reps[k][s]["p_t"] > sce[k]["p_t"] for k in range(3) for s in (3, 4, 5))),
        "holds_within_0p005_of_v29": abs(pooled["mean"] - summary["v29_original"]["p_t"]["mean"]) <= 0.005,
    }
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"verdict": summary["verdict"], "pooled_p_t": pooled, "all_12": summary["onset_linear_all_12_runs"],
                      "per_cache_p_t": {k: v["p_t"] for k, v in summary["per_cache"].items()}}, indent=2))


if __name__ == "__main__":
    main()
