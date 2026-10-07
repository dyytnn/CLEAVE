#!/usr/bin/env python3
"""Error on the clinically used derived intervals cc2 and s2.

The manuscript motivates the whole task with cc2 $=t3-t2$ and s2 $=t4-t3$, the two intervals that
feed published morphokinetic selection algorithms, and then reports only per-event onset error.
Those are different quantities: the onset errors of t2, t3 and t4 within one video are correlated,
so they can cancel or compound, and an interval can be far more or far less accurate than its two
endpoints. This script measures the intervals directly.

It also reports the two failure modes that only exist for a derived interval:

* **undefined in the reference.** 3-cell is skipped in a large minority of embryos, so cc2 and s2 are
  undefined for them. A model that invents a t3 there produces a clinical number out of nothing.
* **spurious or missing interval.** GT defines the interval and the model does not, or the reverse.

Per-video errors are averaged over seeds first, then aggregated with a patient-clustered bootstrap,
matching the unit of analysis used everywhere else in the paper.

    PYTHONPATH=src python scripts/derived_intervals.py --split v2
    PYTHONPATH=src python scripts/derived_intervals.py --split v2 --exclude-defective \
        --out results/derived_intervals_nodefect.json     # sensitivity: without the inspected defective videos
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from stseg.data.patient import patient_of

ROOT = Path(__file__).resolve().parents[1]
NBOOT = 10000
#: interval name -> (start event, end event)
INTERVALS = {"cc2": ("t2", "t3"), "s2": ("t3", "t4")}
#: published selection cut-offs (Meseguer et al. 2011): cc2 <= 11.9 h and s2 <= 0.76 h are the favourable categories
CUTOFF_H = {"cc2": 11.9, "s2": 0.76}
MODELS = {"reference": "resnet18_lstm_L4_grouped_{v}", "crossfocal": "resnet18_transformer_L16_crossfocal7_evalfix_grouped_{v}"}


def per_video(run: Path) -> list[dict]:
    return json.loads((run / "per_video_test.json").read_text())


def interval(times: dict, a: str, b: str) -> float | None:
    if a in times and b in times:
        return float(times[b]) - float(times[a])
    return None


def cluster_boot(values: list[float], videos: list[str], rng: np.random.Generator) -> tuple[float, float, float, int]:
    """Mean with a patient-clustered percentile interval; patients are the resampling unit."""
    arr = np.asarray(values, dtype=float)
    pat = np.array([patient_of(v) for v in videos])
    groups = [arr[pat == p] for p in np.unique(pat)]
    bs = np.empty(NBOOT)
    for i in range(NBOOT):
        bs[i] = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean()
    return float(arr.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(groups)


def analyse(df: pd.DataFrame, exp: str, exclude: frozenset[str] = frozenset()) -> dict:
    runs = sorted(df[df.experiment == exp].run)
    if not runs:
        return {}
    rows: dict[str, list[dict]] = {}
    for r in runs:
        for v in per_video(ROOT / "runs/h7" / r):
            if v["video"] in exclude:
                continue
            rows.setdefault(v["video"], []).append(v)
    rng = np.random.default_rng(0)
    out: dict = {"experiment": exp, "n_seeds": len(runs), "n_videos": len(rows)}

    for name, (a, b) in INTERVALS.items():
        err, absr, vids = [], [], []
        gt_defined = model_defined = spurious = missing = 0
        for vid, seeds in rows.items():
            gt = interval(seeds[0]["gt_times"], a, b)
            preds = [interval(s["pred_times"], a, b) for s in seeds]
            if gt is None:
                if all(p is not None for p in preds):
                    spurious += 1
                continue
            gt_defined += 1
            if any(p is None for p in preds):
                missing += 1
                continue
            model_defined += 1
            d = statistics.mean(p - gt for p in preds)          # type: ignore[operator]
            err.append(d)
            absr.append(abs(d))
            vids.append(vid)
        if not err:
            continue
        m, lo, hi, npat = cluster_boot(err, vids, rng)
        am, alo, ahi, _ = cluster_boot(absr, vids, rng)
        gtv = [interval(seeds[0]["gt_times"], a, b) for vid, seeds in rows.items()]
        gtv = [g for g in gtv if g is not None]
        q1, med, q3 = (float(np.percentile(gtv, q)) for q in (25, 50, 75))
        out[name] = {
            "gt_median": med, "gt_iqr_lo": q1, "gt_iqr_hi": q3, "gt_iqr": q3 - q1,
            "n_gt_defined": gt_defined, "n_scored": model_defined,
            "n_spurious": spurious, "n_missing": missing,
            "signed_mean": m, "signed_lo": lo, "signed_hi": hi,
            "abs_mean": am, "abs_lo": alo, "abs_hi": ahi,
            "median_abs": float(np.median(absr)), "n_patients": npat,
            "within_1h": float(np.mean(np.array(absr) <= 1.0)),
        }

    # marginal onset errors of the three endpoints, for the cancellation comparison
    marg = {}
    for ev in ("t2", "t3", "t4"):
        e, vids = [], []
        for vid, seeds in rows.items():
            if ev not in seeds[0]["gt_times"] or any(ev not in s["pred_times"] for s in seeds):
                continue
            g = float(seeds[0]["gt_times"][ev])
            e.append(abs(statistics.mean(float(s["pred_times"][ev]) - g for s in seeds)))
            vids.append(vid)
        if e:
            m, lo, hi, _ = cluster_boot(e, vids, rng)
            marg[ev] = {"abs_mean": m, "abs_lo": lo, "abs_hi": hi, "n": len(e)}
    out["marginal_abs"] = marg
    return out


def train_medians(split: str) -> dict[str, float]:
    """Median of each interval over the TRAINING embryos of the split: the constant a model-free baseline predicts."""
    sp = json.loads((ROOT / f"data/splits/nantes_grouped_{split}.json").read_text())["videos"]["train"]
    m = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv", usecols=["video", "plane", "phase", "time_h"])
    m = m[(m.plane == "embryo_dataset") & m.video.isin(sp) & m.phase.notna()]
    on = m.groupby(["video", "phase"]).time_h.min().unstack()
    return {name: float((on[b] - on[a]).dropna().median()) for name, (a, b) in INTERVALS.items()}


def comparisons(df: pd.DataFrame, exp: str, const: dict[str, float], exclude: frozenset[str]) -> dict:
    """On exactly the videos analyse() scores: the constant baseline, the error of single seeds (not the seed
    average), and the variant in which a skipped 3-cell stage means t3 = t4 (s2 = 0) for annotation and model alike."""
    rows: dict[str, list[dict]] = {}
    for r in sorted(df[df.experiment == exp].run):
        for v in per_video(ROOT / "runs/h7" / r):
            if v["video"] not in exclude:
                rows.setdefault(v["video"], []).append(v)

    def iv(times: dict, a: str, b: str, fill: bool) -> float | None:
        t = dict(times)
        if fill and "t3" not in t and "t2" in t and "t4" in t:
            t["t3"] = t["t4"]
        return interval(t, a, b)

    out = {}
    rng = np.random.default_rng(1)
    for fill in (False, True):
        for name, (a, b) in INTERVALS.items():
            mod, seed, base, vids, cat_gt, cat_m = [], [], [], [], [], []
            for vid, seeds in rows.items():
                gt = iv(seeds[0]["gt_times"], a, b, fill)
                preds = [iv(x["pred_times"], a, b, fill) for x in seeds]
                if gt is None or any(p is None for p in preds):
                    continue
                mod.append(abs(statistics.mean(preds) - gt))
                seed.append(statistics.mean(abs(p - gt) for p in preds))
                base.append(abs(const[name] - gt))
                vids.append(vid)
                cat_gt.append(gt <= CUTOFF_H[name])
                cat_m.append(statistics.mean(preds) <= CUTOFF_H[name])
            if not mod:
                continue
            # within-1 h, model minus constant: patients resampled with replacement (videos of a patient stay together)
            d = (np.array(mod) <= 1.0).astype(float) - (np.array(base) <= 1.0).astype(float)
            pat = np.array([patient_of(v) for v in vids])
            groups = [d[pat == q] for q in np.unique(pat)]
            bs = [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(NBOOT)]
            g, m_ = np.array(cat_gt), np.array(cat_m)
            po, pe = np.mean(g == m_), g.mean() * m_.mean() + (1 - g.mean()) * (1 - m_.mean())
            key = f"{name}{'_skip3_as_zero' if fill else ''}"
            out[key] = {"n": len(mod), "const_value_h": const[name],
                        "within_1h_diff": float(d.mean()), "within_1h_diff_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        "category_cutoff_h": CUTOFF_H[name], "category_kappa": float((po - pe) / (1 - pe)) if pe < 1 else float("nan"),
                        "favourable_share_gt": float(g.mean()), "favourable_share_model": float(m_.mean()),
                        **{f"{tag}_{stat}": float(f(np.array(x))) for tag, x in (("model", mod), ("single_seed", seed), ("const", base))
                           for stat, f in (("mae", np.mean), ("median", np.median), ("within_1h", lambda z: np.mean(z <= 1.0)))}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v2", choices=["v1", "v2"])
    ap.add_argument("--out", default="results/derived_intervals.json")
    ap.add_argument("--exclude-defective", action="store_true",
                    help="drop the tier-A test videos of data/qc/nantes_video_defects_v4.json")
    a = ap.parse_args()
    log = json.loads((ROOT / "data/qc/nantes_video_defects_v4.json").read_text())["videos"]
    exclude = frozenset(v for v, e in log.items() if e["tier"] == "A") if a.exclude_defective else frozenset()
    df = pd.read_csv(ROOT / "results/runs_master.csv")
    report = {"split": f"nantes_grouped_{a.split}", "excluded_defective": bool(exclude)}
    const = train_medians(a.split)
    report["train_median_h"] = const
    for label, pat in MODELS.items():
        r = analyse(df, pat.format(v=a.split), exclude)
        if r:
            r["comparisons"] = comparisons(df, pat.format(v=a.split), const, exclude)
            report[label] = r
            for k, c in r["comparisons"].items():
                print(f"{label:12} {k:16} n={c['n']:3d}  MAE model {c['model_mae']:.2f} / single seed {c['single_seed_mae']:.2f} / "
                      f"constant {c['const_mae']:.2f} h;  median {c['model_median']:.2f} vs {c['const_median']:.2f};  "
                      f"within 1 h {100 * c['model_within_1h']:.0f} vs {100 * c['const_within_1h']:.0f} % "
                      f"(diff {c['within_1h_diff']:+.2f} [{c['within_1h_diff_ci'][0]:+.2f}, {c['within_1h_diff_ci'][1]:+.2f}]); "
                      f"kappa {c['category_kappa']:.2f}, favourable {100 * c['favourable_share_model']:.0f} vs {100 * c['favourable_share_gt']:.0f} %")
    (ROOT / a.out).write_text(json.dumps(report, indent=1))

    for label in MODELS:
        if label not in report:
            continue
        r = report[label]
        print(f"--- {label} ({r['n_seeds']} seeds, {r['n_videos']} test videos)")
        for name in INTERVALS:
            if name not in r:
                continue
            x = r[name]
            print(f"  {name}: defined in GT for {x['n_gt_defined']}, scored {x['n_scored']} "
                  f"(spurious {x['n_spurious']}, missing {x['n_missing']})")
            print(f"       signed {x['signed_mean']:+.2f} h [{x['signed_lo']:+.2f}, {x['signed_hi']:+.2f}]  "
                  f"absolute {x['abs_mean']:.2f} h [{x['abs_lo']:.2f}, {x['abs_hi']:.2f}]  "
                  f"median {x['median_abs']:.2f} h  within 1 h: {100 * x['within_1h']:.0f}%")
        m = r["marginal_abs"]
        print("       endpoint absolute onset error: " + ", ".join(f"{k} {v['abs_mean']:.2f} h" for k, v in m.items()))
    print("wrote", ROOT / a.out)


if __name__ == "__main__":
    main()
