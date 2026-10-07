#!/usr/bin/env python3
"""Difference-in-differences estimate of the sibling-leakage effect (CLEAVE defect 3).

Reads the per-video test scores of the paired runs written by ``configs/h7/siblingleak/`` and
reports

    delta(v)  = mean over seeds of  p_t_leaky(v) - p_t_clean(v)        (same video, same recipe)
    Delta_L   = mean of delta over the 41 test embryos whose siblings are in the leaky train set
    Delta_C   = mean of delta over the 41 control embryos (siblings in neither arm)
    DiD       = Delta_L - Delta_C     <- the leakage effect, with the training-set swap differenced out

and converts the per-video effect into the inflation carried by a real split by multiplying by that
split's exposed fraction from ``results/protocol_audit.json``.

Per-video p_t is ``1 - n_far / n_transitions``, which reproduces the reported p_t exactly (the
aggregate is the unweighted mean over videos).

    PYTHONPATH=src python scripts/summarize_siblingleak.py --backbone resnet18
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/h7/siblingleak"


def per_video_pt(run: Path) -> dict[str, float]:
    rows = json.loads((run / "per_video_test.json").read_text())
    return {r["video"]: 1.0 - r["n_far"] / r["n_transitions"] for r in rows if r["n_transitions"]}


def collect(backbone: str, seeds: list[int]) -> tuple[dict[str, list[float]], list[int]]:
    """{video: [delta per seed]} over the seeds where both arms finished."""
    deltas: dict[str, list[float]] = {}
    used = []
    for s in seeds:
        runs = {arm: RUNS / f"siblingleak_{arm}_{backbone}_seed{s}" for arm in ("clean", "leaky")}
        if not all((r / "per_video_test.json").exists() for r in runs.values()):
            continue
        a, b = per_video_pt(runs["clean"]), per_video_pt(runs["leaky"])
        for v in sorted(set(a) & set(b)):
            deltas.setdefault(v, []).append(b[v] - a[v])
        used.append(s)
    return deltas, used


def boot(lo_group: list[float], c: list[float], reps: int, rng: np.random.Generator) -> tuple[float, float]:
    d = np.array([rng.choice(lo_group, len(lo_group), replace=True).mean() - rng.choice(c, len(c), replace=True).mean()
                  for _ in range(reps)])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--out", default="results/siblingleak_did.json")
    args = ap.parse_args()

    groups = json.loads((ROOT / "data/splits/nantes_siblingleak_leaky_v1.json").read_text())["groups"]
    deltas, used = collect(args.backbone, args.seeds)
    if not used:
        raise SystemExit(f"no finished pair of runs for backbone {args.backbone} under {RUNS}; "
                         "run configs/h7/siblingleak/sweep_resnet18.yaml first")
    mean_d = {v: statistics.mean(x) for v, x in deltas.items()}
    L = [mean_d[v] for v in groups["L_test_videos"] if v in mean_d]
    C = [mean_d[v] for v in groups["C_test_videos"] if v in mean_d]
    did = statistics.mean(L) - statistics.mean(C)
    rng = np.random.default_rng(0)
    lo, hi = boot(L, C, args.reps, rng)
    pooled = np.array(L + C)
    nL = len(L)
    perm = np.array([(lambda p: p[:nL].mean() - p[nL:].mean())(rng.permutation(pooled)) for _ in range(args.reps)])
    pval = float((np.abs(perm) >= abs(did)).mean())

    audit = json.loads((ROOT / "results/protocol_audit.json").read_text())
    exposure = {
        "released_folds_pooled": audit["official_pooled"]["clean652.couple"]["pooled_frac_exposed"],
        "released_folds_pooled_all704": audit["official_pooled"]["all704.couple"]["pooled_frac_exposed"],
        "random_video_level_split": audit["simulation"]["clean652.couple"]["frac_exposed_test_mean"],
        **{k: v["couple"]["frac_exposed_test_videos"] for k, v in audit["own_splits"].items()},
    }
    out = {
        "backbone": args.backbone, "seeds_used": used, "n_videos": len(mean_d),
        "Delta_L": statistics.mean(L), "n_L": len(L),
        "Delta_C": statistics.mean(C), "n_C": len(C),
        "DiD": did, "ci95": [lo, hi], "permutation_p": pval,
        "implied_inflation": {k: did * f for k, f in exposure.items()},
        "exposure": exposure,
    }
    (ROOT / args.out).write_text(json.dumps(out, indent=1))
    print(f"{args.backbone}: seeds {used}, {len(mean_d)} test videos")
    print(f"  Delta_L (siblings in train)  = {out['Delta_L']:+.4f}  (n={len(L)})")
    print(f"  Delta_C (control)            = {out['Delta_C']:+.4f}  (n={len(C)})")
    print(f"  DiD  = {did:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   permutation p = {pval:.4f}")
    for k, f in exposure.items():
        print(f"  implied inflation of {k:28s} (exposure {f:5.1%}) = {did * f:+.4f}")
    print("wrote", ROOT / args.out)


if __name__ == "__main__":
    main()
