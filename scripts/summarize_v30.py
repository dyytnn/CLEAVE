#!/usr/bin/env python3
"""Aggregate the preregistered v30 SCE + auxiliary-branch validation comparisons.

Every arm is compared seed-matched against the v27 ``sce`` parent (same encoder/head/recipe/cache; only the
training-time auxiliary term differs) with the standard gate, and against v26 ``full_visual`` for context. Event
recovery vs the parent reuses scripts/summarize_v29.py (hit = decoded onset within THETA_H[e]).

  PYTHONPATH=src python scripts/summarize_v30.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import summarize_v29 as V29  # noqa: E402
from summarize_v27 import aggregate, load_v26, metrics, paired  # noqa: E402

RUN_ROOT = ROOT / "runs/h7/v30"
ARMS = ("change_contrastive", "peak_count", "time_to_event", "boundary")


def load_arm(arm: str) -> list[dict]:
    return [{"seed": s, **metrics(V29._result(RUN_ROOT / f"v30_{arm}_src{s}_seed{s}" / "results.json")["val"])}
            for s in range(3)]


def main() -> None:
    V29.RUN_ROOT = RUN_ROOT  # event_recovery reads arm per-video files from the v30 run root
    records = {arm: load_arm(arm) for arm in ARMS}
    sce, v26 = V29.load_sce(), load_v26()
    summary = {
        "version": "v30_summary_v1",
        "split": "data/splits/nantes_grouped_v1.json",
        "test_evaluated": False,
        "protocol": "v27 SCE + one training-time auxiliary branch on frozen v26 features;",
        "arms": {arm: aggregate(rows) for arm, rows in records.items()},
        "parent_v27_sce": aggregate(sce),
        "v26_full_visual": aggregate(v26),
        "vs_v27_sce": {arm: paired(records[arm], sce) for arm in ARMS},
        "vs_v26_full_visual": {arm: paired(records[arm], v26) for arm in ARMS},
        "event_recovery_vs_v27_sce": {
            arm: V29.event_recovery(f"v30_{arm}_src{{s}}_seed{{s}}", "v27_sce_src{s}_seed{s}") for arm in ARMS
        },
    }
    scalars = [v for arm in summary["arms"].values() for m in ("p", "p_t", "f1_50", "mae_h_all") for v in arm[m]["values"]]
    if not all(math.isfinite(v) for v in scalars):
        raise ValueError("non-finite v30 metric")
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({arm: {"gate": summary["vs_v27_sce"][arm]["gate"], "delta_p_t": summary["vs_v27_sce"][arm]["p_t"],
                            "focus_recovery": summary["event_recovery_vs_v27_sce"][arm]["focus_t3_t5_t7"]}
                      for arm in ARMS}, indent=2))


if __name__ == "__main__":
    main()
