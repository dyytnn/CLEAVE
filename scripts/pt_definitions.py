#!/usr/bin/env python3
"""Can any reasonable definition of temporal accuracy reproduce the published 0.559?

The source publication reports p_t = 0.559 for the ResNet-LSTM reference, which our implementation does not reproduce
(p and p_v do). The released code contains no p_t script, so the gap could be the metric's definition. This script
recomputes p_t on the stored fold-0 test predictions of the reference under every combination of the choices the
definition leaves open, and reports the range.

Choices: decoding (Viterbi with the train transition matrix | raw argmax); onset of a predicted phase (first frame |
first frame of the longest run); a phase the model skips (imputed from the next predicted phase | from the previous
one | counted as a miss); aggregation (mean over videos | pooled over transitions); tolerance (theta | theta/2 | 2 theta,
theta = the paper's per-event inter-operator SD); events scored (with tolerance | also tPB2 and tHB at the largest
theta); frame-label alignment (direct | the released one-frame offset).

    PYTHONPATH=src python scripts/pt_definitions.py     # -> results/pt_definitions.json
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from stseg.eval.kinetic_metrics import THETA_BY_CLASS, viterbi

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = 0.559
#: every stored set of test predictions of the reference recipe, all five released folds
RUNS = sorted([(str(f.relative_to(ROOT)), f"runs/h7/{f.parent.parent.name}")
               for f in (ROOT / "runs/h7").glob("resnet18_lstm_L4_split?_seed?/diagnostics/frame_probs_test.npz")]
              + [(str(f.relative_to(ROOT)), f"runs/h7/{f.parent.name}")
                 for f in (ROOT / "results/diagnostics").glob("resnet18_lstm_L4_split?_seed?/frame_probs_test.npz")])


def onsets(path: np.ndarray, t: np.ndarray, rule: str) -> dict[int, float]:
    out: dict[int, float] = {}
    if rule == "first":
        for c, tt in zip(path, t):
            out.setdefault(int(c), float(tt))
        return out
    best: dict[int, tuple[int, float]] = {}
    i = 0
    while i < len(path):
        j = i
        while j + 1 < len(path) and path[j + 1] == path[i]:
            j += 1
        c = int(path[i])
        if c not in best or j - i + 1 > best[c][0]:
            best[c] = (j - i + 1, float(t[i]))
        i = j + 1
    return {c: v[1] for c, v in best.items()}


def score(videos, decode, onset, skip, agg, tol, events, offset) -> float:
    hits = total = 0
    per = []
    for lp, y, t, lt in videos:
        if offset:
            y = np.r_[y[1:], y[-1]]
        path = viterbi(lp, lt) if decode == "viterbi" else lp.argmax(1)
        gt, pr = onsets(y, t, "first"), onsets(path, t, onset)
        classes = sorted(gt)
        theta = dict(THETA_BY_CLASS)
        if events == "all":
            for c in classes:
                theta.setdefault(c, max(THETA_BY_CLASS.values()))
        trans = [c for c in classes[1:] if c in theta]
        h = 0
        for c in trans:
            p = pr.get(c)
            if p is None and skip != "miss":
                nb = [k for k in sorted(pr) if (k > c if skip == "next" else k < c)]
                p = pr[nb[0] if skip == "next" else nb[-1]] if nb else None
            h += int(p is not None and abs(p - gt[c]) <= tol * theta[c])
        if trans:
            per.append(h / len(trans))
        hits, total = hits + h, total + len(trans)
    return float(np.mean(per)) if agg == "video" else hits / max(total, 1)


def main() -> None:
    by_seed = []
    for npz, run in RUNS:
        z = np.load(ROOT / npz, allow_pickle=True)   # written by this repo's diagnostics
        lt = np.load(ROOT / run / "transition_log_matrix.npy")
        vids = sorted({k.split("__")[0] for k in z.files})
        by_seed.append([(z[f"{v}__lp"], z[f"{v}__y"], z[f"{v}__t"], lt) for v in vids])
    grid = list(itertools.product(("viterbi", "argmax"), ("first", "longest"), ("next", "previous", "miss"),
                                  ("video", "pooled"), (0.5, 1.0, 2.0), ("tolerance", "all"), (False, True)))
    rows = []
    folds = [r[1].rsplit("_split", 1)[1][0] for r in RUNS]
    for g in grid:
        per_run = [score(v, *g) for v in by_seed]
        vals = [np.mean([x for x, f in zip(per_run, folds) if f == k]) for k in sorted(set(folds))]   # fold means
        rows.append({"decode": g[0], "onset": g[1], "skip": g[2], "aggregate": g[3], "tolerance_x_theta": g[4],
                     "events": g[5], "released_offset": g[6], "p_t": float(np.mean(vals))})
    rows.sort(key=lambda r: abs(r["p_t"] - PUBLISHED))
    ours = next(r for r in rows if (r["decode"], r["onset"], r["skip"], r["aggregate"], r["tolerance_x_theta"], r["events"],
                                    r["released_offset"]) == ("viterbi", "first", "next", "video", 1.0, "tolerance", False))
    th1 = [r["p_t"] for r in rows if r["tolerance_x_theta"] == 1.0]
    out = {"published": PUBLISHED, "runs": [r[1] for r in RUNS], "n_definitions": len(rows), "ours": ours["p_t"],
           "range_all": [min(r["p_t"] for r in rows), max(r["p_t"] for r in rows)],
           "range_theta_as_published": [min(th1), max(th1)],
           "closest": rows[:5], "n_within_0_02_at_theta": sum(abs(x - PUBLISHED) <= 0.02 for x in th1)}
    def pick(**kw):
        base = {"decode": "viterbi", "onset": "first", "tolerance_x_theta": 1.0, "events": "tolerance", "released_offset": False}
        base.update(kw)
        return next(r["p_t"] for r in rows if all(r[k] == v for k, v in base.items()))
    out["named"] = {"impute_next_video": pick(skip="next", aggregate="video"),
                    "impute_next_pooled": pick(skip="next", aggregate="pooled"),
                    "skip_as_miss_video": pick(skip="miss", aggregate="video"),
                    "skip_as_miss_pooled": pick(skip="miss", aggregate="pooled")}
    out["n_folds"] = len(set(folds))
    out["all"] = rows                                   # every definition, for Fig. 2d
    (ROOT / "results/pt_definitions.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "closest"}, indent=1))
    for r in rows[:5]:
        print(r)


if __name__ == "__main__":
    main()
