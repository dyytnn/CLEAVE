#!/usr/bin/env python3
"""Aggregate the preregistered v33 end-to-end comparisons.

Gate: ``e2e`` vs v27 ``sce`` (matched seed, standard rule). Pipeline check: ``frozen_control`` vs v27 ``sce``
(must be within noise). Attribution: ``e2e`` vs ``frozen_control`` (the one change). Event recovery vs SCE reuses
the 12.bbb/12.fff logic.

  PYTHONPATH=src python scripts/summarize_v33.py
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
from summarize_v31 import event_recovery  # noqa: E402

RUN_ROOT = ROOT / "runs/h7/v33"
V27_ROOT = ROOT / "runs/h7/v27"
ARMS = ("frozen_control", "e2e")


def _rows(root: Path, pattern: str) -> list[dict]:
    return [{"seed": s, **metrics(V29._result(root / pattern.format(s=s) / "results.json")["val"])} for s in range(3)]


def main() -> None:
    records = {arm: _rows(RUN_ROOT, f"v33_{arm}_src{{s}}_seed{{s}}") for arm in ARMS}
    sce = _rows(V27_ROOT, "v27_sce_src{s}_seed{s}")
    v26 = load_v26()
    summary = {
        "version": "v33_summary_v1",
        "split": "data/splits/nantes_grouped_v1.json",
        "test_evaluated": False,
        "protocol": "end-to-end whole-video training, trainable cross-focal backbone + SCE head;",
        "arms": {arm: aggregate(rows) for arm, rows in records.items()},
        "parent_v27_sce": aggregate(sce),
        "v26_full_visual": aggregate(v26),
        "gate_e2e_vs_v27_sce": paired(records["e2e"], sce),
        "pipeline_check_frozen_vs_v27_sce": paired(records["frozen_control"], sce),
        "attribution_e2e_vs_frozen": paired(records["e2e"], records["frozen_control"]),
        "vs_v26_full_visual": {arm: paired(records[arm], v26) for arm in ARMS},
        "event_recovery_vs_v27_sce": {
            arm: event_recovery(RUN_ROOT, f"v33_{arm}_src{{s}}_seed{{s}}", V27_ROOT, "v27_sce_src{s}_seed{s}") for arm in ARMS},
    }
    pc = summary["pipeline_check_frozen_vs_v27_sce"]["p_t"]
    summary["pipeline_check_passed"] = bool(abs(pc["mean"]) <= 0.01)
    scalars = [v for a in summary["arms"].values() for k in ("p", "p_t", "f1_50", "mae_h_all") for v in a[k]["values"]]
    if not all(math.isfinite(v) for v in scalars):
        raise ValueError("non-finite v33 metric")
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"gate": summary["gate_e2e_vs_v27_sce"]["gate"], "dp_t_e2e_vs_sce": summary["gate_e2e_vs_v27_sce"]["p_t"],
                      "pipeline_check_dp_t": pc, "pipeline_check_passed": summary["pipeline_check_passed"],
                      "attribution_dp_t": summary["attribution_e2e_vs_frozen"]["p_t"],
                      "focus": {a: summary["event_recovery_vs_v27_sce"][a]["focus_t3_t5_t7"] for a in ARMS}}, indent=2))


if __name__ == "__main__":
    main()
