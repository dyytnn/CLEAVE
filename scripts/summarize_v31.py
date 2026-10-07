#!/usr/bin/env python3
"""Aggregate the preregistered v31 onset-head validation comparisons.

Gate: each arm vs v27 ``sce`` (matched seed) with the standard rule. Treatment baseline: each arm vs v29
``onset_linear``. Mechanism check (locked): pooled t3/t5/t7 net recovery vs SCE >= 0 AND t8 net >= 0 (v29
onset_linear gained short phases but lost t8).

  PYTHONPATH=src python scripts/summarize_v31.py
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

RUN_ROOT = ROOT / "runs/h7/v31"
V29_ROOT = ROOT / "runs/h7/v29"
V27_ROOT = ROOT / "runs/h7/v27"
ARMS = ("combined_smooth", "gated_fusion", "duration_hazard")


def _rows(root: Path, pattern: str) -> list[dict]:
    return [{"seed": s, **metrics(V29._result(root / pattern.format(s=s) / "results.json")["val"])} for s in range(3)]


def event_recovery(arm_root: Path, arm_pat: str, parent_root: Path, parent_pat: str) -> dict:
    per_event: dict[str, dict[str, int]] = {}
    for s in range(3):
        a = V29._hits(json.loads((arm_root / arm_pat.format(s=s) / "per_video_val.json").read_text()))
        p = V29._hits(json.loads((parent_root / parent_pat.format(s=s) / "per_video_val.json").read_text()))
        if set(a) != set(p):
            raise ValueError("arm and parent disagree on the GT event set")
        for (video, event), hit in a.items():
            row = per_event.setdefault(event, {"n_gt": 0, "parent_hits": 0, "arm_hits": 0, "recovered": 0, "lost": 0})
            row["n_gt"] += 1
            row["parent_hits"] += int(p[(video, event)])
            row["arm_hits"] += int(hit)
            row["recovered"] += int(hit and not p[(video, event)])
            row["lost"] += int(p[(video, event)] and not hit)
    focus = {k: sum(per_event.get(e, {}).get(k, 0) for e in V29.FOCUS)
             for k in ("n_gt", "parent_hits", "arm_hits", "recovered", "lost")}
    focus["net"] = focus["recovered"] - focus["lost"]
    t8 = per_event.get("t8", {"recovered": 0, "lost": 0})
    return {"per_event": per_event, "focus_t3_t5_t7": focus, "t8_net": t8["recovered"] - t8["lost"],
            "mechanism_check": {"focus_net_ge_0": focus["net"] >= 0, "t8_net_ge_0": t8["recovered"] - t8["lost"] >= 0}}


def main() -> None:
    records = {arm: _rows(RUN_ROOT, f"v31_{arm}_src{{s}}_seed{{s}}") for arm in ARMS}
    sce = _rows(V27_ROOT, "v27_sce_src{s}_seed{s}")
    onset = _rows(V29_ROOT, "v29_onset_linear_src{s}_seed{s}")
    v26 = load_v26()
    summary = {
        "version": "v31_summary_v1",
        "split": "data/splits/nantes_grouped_v1.json",
        "test_evaluated": False,
        "protocol": "event-shaped onset head as registered treatment, one change per arm;",
        "arms": {arm: aggregate(rows) for arm, rows in records.items()},
        "parent_v27_sce": aggregate(sce),
        "baseline_v29_onset_linear": aggregate(onset),
        "v26_full_visual": aggregate(v26),
        "gate_vs_v27_sce": {arm: paired(records[arm], sce) for arm in ARMS},
        "vs_v29_onset_linear": {arm: paired(records[arm], onset) for arm in ARMS},
        "vs_v26_full_visual": {arm: paired(records[arm], v26) for arm in ARMS},
        "event_recovery_vs_v27_sce": {
            arm: event_recovery(RUN_ROOT, f"v31_{arm}_src{{s}}_seed{{s}}", V27_ROOT, "v27_sce_src{s}_seed{s}") for arm in ARMS},
        "event_recovery_vs_v29_onset_linear": {
            arm: event_recovery(RUN_ROOT, f"v31_{arm}_src{{s}}_seed{{s}}", V29_ROOT, "v29_onset_linear_src{s}_seed{s}")
            for arm in ARMS},
    }
    for arm in ARMS:
        g = summary["gate_vs_v27_sce"][arm]["gate"]
        m = summary["event_recovery_vs_v27_sce"][arm]["mechanism_check"]
        summary["gate_vs_v27_sce"][arm]["gate_with_mechanism"] = bool(g["passed"] and all(m.values()))
    scalars = [v for a in summary["arms"].values() for k in ("p", "p_t", "f1_50", "mae_h_all") for v in a[k]["values"]]
    if not all(math.isfinite(v) for v in scalars):
        raise ValueError("non-finite v31 metric")
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({arm: {"gate": summary["gate_vs_v27_sce"][arm]["gate"],
                            "gate_with_mechanism": summary["gate_vs_v27_sce"][arm]["gate_with_mechanism"],
                            "dp_t_vs_sce": summary["gate_vs_v27_sce"][arm]["p_t"],
                            "dp_t_vs_onset_linear": summary["vs_v29_onset_linear"][arm]["p_t"],
                            "focus": summary["event_recovery_vs_v27_sce"][arm]["focus_t3_t5_t7"],
                            "t8_net": summary["event_recovery_vs_v27_sce"][arm]["t8_net"]} for arm in ARMS}, indent=2))


if __name__ == "__main__":
    main()
