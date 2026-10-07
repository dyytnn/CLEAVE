#!/usr/bin/env python3
"""Aggregate the preregistered v29 milestone-query validation comparisons.

Primary: query_refine - v27 sce (matched seed). Components: query_global - onset_linear (global queries),
query_refine - query_global (local refinement). Also every arm vs v26 full_visual, and event recovery vs the SCE
parent: a GT event is a hit when its decoded onset lies within the p_t tolerance THETA_H[e]; "recovered" = parent
miss & arm hit, "lost" = parent hit & arm miss, pooled over the three matched seeds.

  PYTHONPATH=src python scripts/summarize_v29.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from summarize_v27 import aggregate, load_v26, metrics, paired  # noqa: E402

from stseg.eval.kinetic_metrics import THETA_H  # noqa: E402

RUN_ROOT = ROOT / "runs/h7/v29"
V27_ROOT = ROOT / "runs/h7/v27"
ARMS = ("onset_linear", "query_global", "query_refine")
FOCUS = ("t3", "t5", "t7")


def _result(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result.get("test_evaluated") is not False or "test" in result:
        raise ValueError(f"{path}: v29 comparisons must remain validation-only")
    return result


def load_arm(arm: str) -> list[dict]:
    return [{"seed": s, **metrics(_result(RUN_ROOT / f"v29_{arm}_src{s}_seed{s}" / "results.json")["val"])}
            for s in range(3)]


def load_sce() -> list[dict]:
    return [{"seed": s, **metrics(_result(V27_ROOT / f"v27_sce_src{s}_seed{s}" / "results.json")["val"])}
            for s in range(3)]


def _hits(per_video: list[dict]) -> dict[tuple[str, str], bool]:
    out = {}
    for v in per_video:
        for event, gt in v["gt_times"].items():
            if event not in THETA_H:
                continue
            pred = v["pred_times"].get(event)
            out[(v["video"], event)] = pred is not None and abs(pred - gt) <= THETA_H[event]
    return out


def event_recovery(arm_dir: str, parent_dir: str) -> dict:
    """Pooled recovered/lost GT events vs the parent, per event and for the t3/t5/t7 focus set."""
    per_event: dict[str, dict[str, int]] = {}
    for s in range(3):
        a = _hits(json.loads((RUN_ROOT / arm_dir.format(s=s) / "per_video_val.json").read_text()))
        p = _hits(json.loads((V27_ROOT / parent_dir.format(s=s) / "per_video_val.json").read_text()))
        if set(a) != set(p):
            raise ValueError("arm and parent disagree on the GT event set")
        for (video, event), hit in a.items():
            row = per_event.setdefault(event, {"n_gt": 0, "parent_hits": 0, "arm_hits": 0, "recovered": 0, "lost": 0})
            row["n_gt"] += 1
            row["parent_hits"] += int(p[(video, event)])
            row["arm_hits"] += int(hit)
            row["recovered"] += int(hit and not p[(video, event)])
            row["lost"] += int(p[(video, event)] and not hit)
    focus = {k: sum(per_event.get(e, {}).get(k, 0) for e in FOCUS)
             for k in ("n_gt", "parent_hits", "arm_hits", "recovered", "lost")}
    focus["net"] = focus["recovered"] - focus["lost"]
    return {"per_event": per_event, "focus_t3_t5_t7": focus}


def main() -> None:
    records = {arm: load_arm(arm) for arm in ARMS}
    sce, v26 = load_sce(), load_v26()
    summary = {
        "version": "v29_summary_v1",
        "split": "data/splits/nantes_grouped_v1.json",
        "test_evaluated": False,
        "protocol": "milestone queries + local refinement on frozen v26 features;",
        "arms": {arm: aggregate(rows) for arm, rows in records.items()},
        "parent_v27_sce": aggregate(sce),
        "v26_full_visual": aggregate(v26),
        "primary_query_refine_vs_v27_sce": paired(records["query_refine"], sce),
        "component_query_global_vs_onset_linear": paired(records["query_global"], records["onset_linear"]),
        "component_query_refine_vs_query_global": paired(records["query_refine"], records["query_global"]),
        "vs_v27_sce": {arm: paired(records[arm], sce) for arm in ARMS},
        "vs_v26_full_visual": {arm: paired(records[arm], v26) for arm in ARMS},
        "event_recovery_vs_v27_sce": {
            arm: event_recovery(f"v29_{arm}_src{{s}}_seed{{s}}", "v27_sce_src{s}_seed{s}") for arm in ARMS
        },
    }
    scalars = [v for arm in summary["arms"].values() for m in ("p", "p_t", "f1_50", "mae_h_all") for v in arm[m]["values"]]
    if not all(math.isfinite(v) for v in scalars):
        raise ValueError("non-finite v29 metric")
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    prim = summary["primary_query_refine_vs_v27_sce"]
    print(json.dumps({"primary_gate": prim["gate"], "primary_delta_p_t": prim["p_t"],
                      "focus_recovery": {a: summary["event_recovery_vs_v27_sce"][a]["focus_t3_t5_t7"] for a in ARMS}},
                     indent=2))


if __name__ == "__main__":
    main()
