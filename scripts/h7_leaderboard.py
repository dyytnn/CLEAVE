#!/usr/bin/env python3
"""Leaderboard for H7 runs: test p / p_v / r / p_t per experiment + paired bootstrap vs a reference on the common test videos.

    PYTHONPATH=src python scripts/h7_leaderboard.py --runs runs/h7 --ref resnet18_lstm_L4_split0 --out results/h7_leaderboard.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def per_video_frame(run: Path) -> dict[str, tuple[float, float]]:
    """video -> (acc_viterbi, temporal_acc)"""
    out = {}
    for v in json.loads((run / "per_video_test.json").read_text()):
        pt = (v["n_transitions"] - v["n_far"]) / v["n_transitions"] if v["n_transitions"] else np.nan
        out[v["video"]] = (v["acc_viterbi"], pt)
    return out


def paired_boot(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed); d = a - b; d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/h7")
    ap.add_argument("--ref", default="resnet18_lstm_L4_split0")
    ap.add_argument("--out", default="results/h7_leaderboard.md")
    a = ap.parse_args()
    rows, pv = [], {}
    for run in sorted(Path(a.runs).glob("*_seed*")):
        if not (run / "results.json").exists():
            continue
        res, meta = json.loads((run / "results.json").read_text()), json.loads((run / "meta.json").read_text())
        exp = meta["experiment"]["id"]; split = Path(meta["data"]["split"]).stem
        t = res["test"]
        rows.append({"experiment": exp, "seed": meta["seed"], "split": split, "params_M": round(meta.get("n_params_M", np.nan), 1),
                     "best_epoch": res["best_epoch"], "p": t["p"], "p_v": t["p_v"], "r": t["r"], "p_t": t["p_t"], "n_test_videos": t["n_videos"]})
        pv[(exp, meta["seed"])] = per_video_frame(run)
    if not rows:
        print("no finished runs"); return
    df = pd.DataFrame(rows).sort_values("p_t", ascending=False)
    ref_key = next((k for k in pv if k[0] == a.ref), None)
    lines = ["# H7 leaderboard (Nantes test partition of each split; published ResNet 0.663/0.701/0.371, ResNet-LSTM 0.685/0.696/0.559, ResNet-3D 0.705/0.735/0.659 for p/p_v/p_t)", "",
             df.to_markdown(index=False, floatfmt=".3f"), ""]
    if ref_key:
        lines += [f"## Paired differences vs `{a.ref}` (same test videos; mean and 95 % bootstrap CI over videos)", "",
                  "| experiment | Δ acc_viterbi | Δ temporal acc |", "|---|---|---|"]
        ref = pv[ref_key]
        for key, cur in pv.items():
            if key == ref_key:
                continue
            common = sorted(set(ref) & set(cur))
            if not common:
                continue
            da = paired_boot(np.array([cur[v][0] for v in common]), np.array([ref[v][0] for v in common]))
            dt = paired_boot(np.array([cur[v][1] for v in common]), np.array([ref[v][1] for v in common]))
            lines.append(f"| {key[0]} (seed {key[1]}, n={len(common)}) | {da[0]:+.3f} [{da[1]:+.3f}, {da[2]:+.3f}] | {dt[0]:+.3f} [{dt[1]:+.3f}, {dt[2]:+.3f}] |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n"); df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
