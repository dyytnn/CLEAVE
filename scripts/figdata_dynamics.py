#!/usr/bin/env python3
"""Training dynamics, loss bottlenecks and phase drift for the "how the task is learnt" figure.

* ``loss``: validation cross-entropy per phase and epoch (runs/h7/*/diagnostics/epochs.json), for every fold-0 run at
  the common 10-epoch recipe, grouped by temporal-head family; boundary vs interior loss per epoch.
* ``curves``: validation p_t per epoch (history.json) for every fold-0 configuration at the common recipe.
* ``drift``: per test embryo, onset errors e_j = predicted - annotated (hours, seed-averaged) are split into one
  offset per embryo (mean over events) and a residual. The offset is regressed on the embryo's own tempo, the mean
  deviation of its annotated onsets from the cohort median: slope 0 means the model follows the embryo's tempo, slope
  -1 means it predicts the cohort's average embryo whatever it sees.

    PYTHONPATH=src python scripts/figdata_dynamics.py      # -> results/figdata/dynamics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stseg.data.nantes_kinetic import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
H7 = ROOT / "runs/h7"
EVENTS = ["t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB"]
DRIFT_MODELS = {"reference": "resnet18_lstm_L4_split{k}", "cross-focal": "resnet18_transformer_L16_crossfocal7_evalfix_split{k}"}


def family(cfg: dict) -> str:
    h = cfg["model"].get("head", {}).get("name", "none")
    return {"lstm": "LSTM", "transformer": "transformer", "none": "per-frame"}.get(h, "other heads")


def common_recipe(cfg: dict) -> bool:
    eid = cfg["experiment"]["id"]
    return int(cfg["training"].get("epochs", 10)) == 10 and "recipe" not in eid and "e20" not in eid


def main() -> None:
    df = pd.read_csv(ROOT / "results/runs_master.csv")
    f0 = df[df.split_key == "fold0"]
    loss, curves = {}, {}
    for run in f0.run:
        rd = H7 / run
        cfg = json.loads((rd / "config.resolved.json").read_text())
        if not common_recipe(cfg):
            continue
        fam = family(cfg)
        ep = rd / "diagnostics/epochs.json"
        if ep.exists():
            E = json.loads(ep.read_text())
            if len(E) == 10:
                loss.setdefault(fam, []).append({
                    "run": run,
                    "per_phase": [[e["loss_per_phase"].get(p, np.nan) for p in CLASS_NAMES] for e in E],
                    "boundary": [e["loss_boundary"] for e in E], "interior": [e["loss_interior"] for e in E]})
        h = json.loads((rd / "history.json").read_text())
        curves.setdefault(cfg["experiment"]["id"], {"family": fam, "runs": []})["runs"].append([x["val_p_t"] for x in h])
    print({k: len(v) for k, v in loss.items()}, "configs with curves:", len(curves))

    drift = {}
    for name, pat in DRIFT_MODELS.items():
        per_video: dict[str, dict] = {}
        for k in range(5):
            for run in df[df.experiment == pat.format(k=k)].run:
                for v in json.loads((H7 / run / "per_video_test.json").read_text()):
                    d = per_video.setdefault(v["video"], {"gt": v["gt_times"], "pred": []})
                    d["pred"].append(v["pred_times"])
        gt = pd.DataFrame({v: {e: float(d["gt"][e]) for e in EVENTS if e in d["gt"]} for v, d in per_video.items()}).T
        pr = pd.DataFrame({v: {e: np.mean([float(p[e]) for p in d["pred"] if e in p]) for e in EVENTS
                               if e in d["gt"] and all(e in p for p in d["pred"])} for v, d in per_video.items()}).T
        err = (pr - gt).reindex(columns=EVENTS)
        tempo = (gt - gt.median()).reindex(columns=EVENTS)
        keep = err.notna().sum(1) >= 6
        err, tempo = err[keep], tempo[keep]
        off = err.mean(1)
        tmp = tempo.where(err.notna()).mean(1)
        resid = err.sub(off, axis=0)
        slope, icpt = np.polyfit(tmp, off, 1)
        share = 1 - np.nansum(resid.values ** 2) / np.nansum(err.values ** 2)
        lag1 = [float(pd.concat([resid[a], resid[b]], axis=1).dropna().corr().iloc[0, 1]) for a, b in zip(EVENTS, EVENTS[1:])]
        drift[name] = {"n_embryos": int(keep.sum()), "offset_share_of_sq_error": float(share),
                       "offset_on_tempo_slope": float(slope), "offset_on_tempo_icpt": float(icpt),
                       "offset_tempo_r": float(np.corrcoef(tmp, off)[0, 1]),
                       "residual_lag1_corr": dict(zip([f"{a}->{b}" for a, b in zip(EVENTS, EVENTS[1:])], lag1)),
                       "tempo": tmp.tolist(), "offset": off.tolist(),
                       "err": err.values.tolist(), "videos": list(err.index)}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in drift[name].items()
                     if k in ("n_embryos", "offset_share_of_sq_error", "offset_on_tempo_slope", "offset_tempo_r")},
              "lag1", [round(x, 2) for x in lag1])
    out = {"phases": CLASS_NAMES, "loss": loss, "curves": curves, "drift": drift, "events": EVENTS}
    (ROOT / "results/figdata/dynamics.json").write_text(json.dumps(out))
    print("wrote results/figdata/dynamics.json")


if __name__ == "__main__":
    main()
