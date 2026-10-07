#!/usr/bin/env python3
"""Paired effect of removing the defective videos from *training* (v36).

For each recipe, per-video test p_t is averaged over the three seeds of each arm and the clean-minus-
noisy difference is bootstrapped over patients (the paper's cluster_boot_ci). Scored twice: on the
full nantes_grouped_v2 test set, and without its defective test videos, so a gain on clean test
embryos can be told apart from a model that has merely learnt to agree with the bad labels less.

    PYTHONPATH=src:scripts python scripts/summarize_v36.py              # v36 -> results/v36_trainclean.json
    PYTHONPATH=src:scripts python scripts/summarize_v36.py --tag v37    # v37 -> results/v37_trainclean.json
                                                                        # (defect log v2; clean test = no tier A)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_results_master import cluster_boot_ci
from stseg.data.patient import patient_of

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "per_frame": ("resnet18_none_grouped_v2", "resnet18_none_grouped_v2_trainclean"),
    "reference_lstm": ("resnet18_lstm_L4_grouped_v2", "resnet18_lstm_L4_grouped_v2_trainclean"),
    "crossfocal": ("resnet18_transformer_L16_crossfocal7_evalfix_grouped_v2",
                   "resnet18_transformer_L16_crossfocal7_evalfix_grouped_v2_trainclean"),
}
SEEDS = (0, 1, 2)


def per_video(eid: str) -> dict[str, float] | None:
    runs = [ROOT / "runs/h7" / f"{eid}_seed{s}" / "per_video_test.json" for s in SEEDS]
    if not all(r.exists() for r in runs):
        return None
    acc: dict[str, list[float]] = {}
    for r in runs:
        for row in json.loads(r.read_text()):
            if row["n_transitions"]:
                acc.setdefault(row["video"], []).append(1 - row["n_far"] / row["n_transitions"])
    return {v: float(np.mean(x)) for v, x in acc.items() if len(x) == len(SEEDS)}


def per_video_seeds(eid: str) -> dict[str, np.ndarray] | None:
    """Per-video p_t of each seed separately (video -> array over SEEDS)."""
    runs = [ROOT / "runs/h7" / f"{eid}_seed{s}" / "per_video_test.json" for s in SEEDS]
    if not all(r.exists() for r in runs):
        return None
    acc: dict[str, list[float]] = {}
    for r in runs:
        for row in json.loads(r.read_text()):
            if row["n_transitions"]:
                acc.setdefault(row["video"], []).append(1 - row["n_far"] / row["n_transitions"])
    return {v: np.array(x) for v, x in acc.items() if len(x) == len(SEEDS)}


def hier_boot(pairs: list[tuple[dict, dict]], vids: list[str], weights: list[float], n: int = 10000) -> tuple[float, float, float]:
    """Two-level bootstrap of sum_i w_i * (mean(B_i) - mean(A_i)): patients are resampled with replacement, and within
    each replicate the seeds of every arm are resampled independently, so the interval carries seed variance as well as
    patient variance (round-4 review R1-M1). ``pairs`` holds one (A, B) per term; a single term is a plain delta, two
    terms with weights (+1, -1) the difference between two recipes' deltas."""
    rng = np.random.default_rng(0)
    pat = np.array([patient_of(v) for v in vids])
    groups = [np.where(pat == p)[0] for p in np.unique(pat)]
    arrs = [(np.stack([A[v] for v in vids]), np.stack([B[v] for v in vids])) for A, B in pairs]
    point = sum(w * (b.mean() - a.mean()) for w, (a, b) in zip(weights, arrs))
    bs = np.empty(n)
    for i in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        tot = 0.0
        for w, (a, b) in zip(weights, arrs):
            sa, sb = rng.integers(0, a.shape[1], a.shape[1]), rng.integers(0, b.shape[1], b.shape[1])
            tot += w * (b[idx][:, sb].mean() - a[idx][:, sa].mean())
        bs[i] = tot
    return float(point), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


TAGS = {"v36": ("_trainclean", "data/splits/nantes_grouped_v2_trainclean_v1.json", "data/qc/nantes_video_defects_v1.json", None),
        "v37": ("_trainclean2", "data/splits/nantes_grouped_v2_trainclean_v2.json", "data/qc/nantes_video_defects_v2.json", "A"),
        "v38": ("_trainclean3", "data/splits/nantes_grouped_v2_trainclean_v3.json", "data/qc/nantes_video_defects_v3.json", "A")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v36", choices=sorted(TAGS))
    tag = ap.parse_args().tag
    suffix, split, log, tier = TAGS[tag]
    entries = json.loads((ROOT / log).read_text())["videos"]
    bad = {v for v, e in entries.items() if tier is None or e.get("tier") == tier}
    recipes = {k: (a, a + suffix) for k, (a, _) in RECIPES.items()}
    out: dict = {"tag": tag, "split_clean": split, "n_seeds": len(SEEDS), "recipes": {}}
    for name, (noisy, clean) in recipes.items():
        a, b = per_video(noisy), per_video(clean)
        if a is None or b is None:
            out["recipes"][name] = {"status": "incomplete"}
            print(f"{name}: incomplete")
            continue
        row = {"status": "done"}
        for scope, keep in (("all_test", lambda v: True), ("clean_test", lambda v: v not in bad)):
            vids = sorted(v for v in a if v in b and keep(v))
            d = np.array([b[v] - a[v] for v in vids])
            m, lo, hi, k = cluster_boot_ci(d, vids)
            row[scope] = {"noisy": float(np.mean([a[v] for v in vids])), "clean": float(np.mean([b[v] for v in vids])),
                          "delta": m, "ci95": [lo, hi], "n_videos": len(vids), "n_patients": k}
        out["recipes"][name] = row
        print(f"{name}: " + "; ".join(f"{s} {row[s]['noisy']:.3f} -> {row[s]['clean']:.3f} "
                                      f"(delta {row[s]['delta']:+.3f} [{row[s]['ci95'][0]:+.3f}, {row[s]['ci95'][1]:+.3f}])"
                                      for s in ("all_test", "clean_test")))
    # seed-aware intervals, seed-level t-test, and the contrast between recipes (round-4 review R1-M1, R1-M2)
    seeds = {k: (per_video_seeds(a), per_video_seeds(b)) for k, (a, b) in recipes.items()}
    if all(x is not None and y is not None for x, y in seeds.values()):
        from scipy import stats
        common = sorted(set.intersection(*[set(x) & set(y) for x, y in seeds.values()]))
        for k, (A, B) in seeds.items():
            m, lo, hi = hier_boot([(A, B)], common, [1.0])
            sa = [np.mean([A[v][j] for v in common]) for j in range(len(SEEDS))]
            sb = [np.mean([B[v][j] for v in common]) for j in range(len(SEEDS))]
            t = stats.ttest_ind(sb, sa)
            out["recipes"][k]["seed_aware"] = {"delta": m, "ci95": [lo, hi], "seed_means_noisy": sa, "seed_means_clean": sb,
                                               "seed_ttest_p": float(t.pvalue), "n_videos": len(common)}
            print(f"{k}: seed-aware delta {m:+.3f} [{lo:+.3f}, {hi:+.3f}], seed t-test p={t.pvalue:.3f}")
        out["contrasts"] = {}
        for other in ("reference_lstm", "crossfocal"):
            m, lo, hi = hier_boot([seeds["per_frame"], seeds[other]], common, [1.0, -1.0])
            out["contrasts"][f"per_frame_minus_{other}"] = {"delta": m, "ci95": [lo, hi]}
            print(f"per-frame gain minus {other} gain: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for arm, idx in (("noisy", 0), ("trainclean", 1)):
            A, B = seeds["reference_lstm"][idx], seeds["crossfocal"][idx]
            m, lo, hi = hier_boot([(A, B)], common, [1.0])
            out.setdefault("headline_gap_seed_aware", {})[arm] = {"gap": m, "ci95": [lo, hi]}
            print(f"gap, {arm}, seed-aware: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # does the headline comparison survive training on clean labels? cross-focal minus reference, per arm
    gaps = {}
    for arm, idx in (("noisy", 0), ("trainclean", 1)):
        ref, cf = per_video(recipes["reference_lstm"][idx]), per_video(recipes["crossfocal"][idx])
        if ref is None or cf is None:
            continue
        vids = sorted(set(ref) & set(cf))
        m, lo, hi, k = cluster_boot_ci(np.array([cf[v] - ref[v] for v in vids]), vids)
        gaps[arm] = {"gap": m, "ci95": [lo, hi], "n_videos": len(vids), "n_patients": k}
        print(f"gap cross-focal - reference, {arm} training: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out["headline_gap"] = gaps
    (ROOT / f"results/{tag}_trainclean.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
