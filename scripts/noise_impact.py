#!/usr/bin/env python3
"""How much do the defective videos change what the paper concludes?

Evaluation-side only: every model here was trained on the released data, defects included, so this
measures what the defective *test* videos do to each result by re-scoring existing predictions
without them. What the defective *training* videos did can only be measured by retraining; the
training-set contamination is reported here so that experiment can be sized.

Two exclusion sets: all inspected defective videos, and the subset found by the image scan alone,
which no model selected and is therefore free of the circularity of dropping videos chosen for low
accuracy.

    PYTHONPATH=src:scripts python scripts/noise_impact.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import diagnose_run as D
from stseg.data.nantes_kinetic import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
MID = {CLASS_NAMES.index(c) for c in ("t3", "t4", "t5", "t6", "t7")}


def main() -> None:
    log = json.loads((ROOT / "data/qc/nantes_video_defects_v4.json").read_text())["videos"]
    qc = set(log)                                                      # every logged video, all tiers
    tier_a = {v for v, e in log.items() if e["tier"] == "A"}
    v1 = set(json.loads((ROOT / "data/qc/nantes_video_defects_v1.json").read_text())["videos"])
    hard = {r["video"] for r in json.loads((ROOT / "results/hard_videos.json").read_text())["flagged"]}
    # tier_A: labels not about the image; all_inspected: every logged video; image_scan_only: the v1 videos found by
    # the model-free image scan, which no model selected (the circularity control)
    drops = {"tier_A": tier_a, "all_inspected": qc, "image_scan_only": {v for v in v1 if v not in hard}}
    df = pd.read_csv(ROOT / "results/runs_master.csv")
    cache: dict[str, list] = {}

    def pv(run):
        if run not in cache:
            cache[run] = json.loads((ROOT / "runs/h7" / run / "per_video_test.json").read_text())
        return cache[run]

    def pt(run, drop):
        rows = [r for r in pv(run) if r["n_transitions"] and r["video"] not in drop]
        return st.mean(1 - r["n_far"] / r["n_transitions"] for r in rows)

    out: dict = {"defect_log": "nantes_video_defects_v4", "n_defective": len(qc), "n_tier_A": len(tier_a),
                 "n_image_scan_only": len(drops["image_scan_only"])}

    # training contamination per split
    man = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    nfr = man[man.phase.notna()].groupby("video").size()
    train = {}
    for name in ("nantes_grouped_v1", "nantes_grouped_v2", "nantes_siblingleak_clean_v1", "nantes_siblingleak_leaky_v1"):
        tr = set(json.loads((ROOT / f"data/splits/{name}.json").read_text())["videos"]["train"])
        train[name] = {"videos": len(tr), "defective": len(tr & qc), "tier_A": len(tr & tier_a),
                       "tier_A_frame_share": float(nfr.reindex(list(tr & tier_a)).sum() / nfr.reindex(list(tr)).sum()),
                       "frame_share_upper_bound": float(nfr.reindex(list(tr & qc)).sum() / nfr.reindex(list(tr)).sum())}
    for k in range(5):
        tr = {v for v, p in csv.reader(open(ROOT / f"data/splits/nantes_official/split{k}.csv")) if p == "train"} & set(nfr.index)
        train[f"official_split{k}"] = {"videos": len(tr), "defective": len(tr & qc), "tier_A": len(tr & tier_a),
                                       "tier_A_frame_share": float(nfr.reindex(list(tr & tier_a)).sum() / nfr.reindex(list(tr)).sum()),
                                       "frame_share_upper_bound": float(nfr.reindex(list(tr & qc)).sum() / nfr.reindex(list(tr)).sum())}
    out["training_contamination"] = train

    # the headline comparison on each protocol
    comp = {}
    for label, ref, cf in (("grouped_v1", "resnet18_lstm_L4_grouped_v1", "resnet18_transformer_L16_crossfocal7_evalfix_grouped_v1"),
                           ("grouped_v2", "resnet18_lstm_L4_grouped_v2", "resnet18_transformer_L16_crossfocal7_evalfix_grouped_v2")):
        R, C = sorted(df[df.experiment == ref].run), sorted(df[df.experiment == cf].run)
        row = {"n_defective_test": sum(1 for r in pv(R[0]) if r["video"] in qc)}
        for dn, drop in (("none", set()), *drops.items()):
            r_, c_ = st.mean(pt(x, drop) for x in R), st.mean(pt(x, drop) for x in C)
            row[dn] = {"reference": r_, "crossfocal": c_, "gap": c_ - r_}
        comp[label] = row
    out["headline_comparison"] = comp

    # released five-fold protocol, pooled as in the main table (mean over seeds per fold, then over folds)
    fivefold = {"n_defective_test_slots": sum(
        1 for k in range(5) for v, p in csv.reader(open(ROOT / f"data/splits/nantes_official/split{k}.csv"))
        if p == "test" and v in qc)}
    for tag, pat in (("reference", "resnet18_lstm_L4_split{k}"), ("crossfocal", "resnet18_transformer_L16_crossfocal7_evalfix_split{k}")):
        for dn, drop in (("none", set()), *drops.items()):
            folds = [np.mean([pt(r, drop) for r in df[df.experiment == pat.format(k=k)].run]) for k in range(5)]
            fivefold.setdefault(dn, {})[tag] = float(np.mean(folds))
    for dn in ("none", *drops):
        fivefold[dn]["gap"] = fivefold[dn]["crossfocal"] - fivefold[dn]["reference"]
    out["released_fivefold"] = fivefold

    # defect categories (a video may carry several)
    cats: dict[str, int] = {}
    for rec in log.values():
        for c in rec["defects"]:
            cats[c] = cats.get(c, 0) + 1
    out["categories"] = dict(sorted(cats.items(), key=lambda kv: -kv[1]))

    # does the fold-0 ranking of all configurations change?
    f0 = df[df.split_key == "fold0"]
    f0 = f0[(f0.epochs == 10) & ~f0.experiment.str.contains("recipe", regex=False)]   # common recipe (the sweep)
    exps = sorted(e for e, g in f0.groupby("experiment")
                  if all((ROOT / "runs/h7" / r / "per_video_test.json").exists() for r in g.run))
    base = np.array([np.mean([pt(r, set()) for r in f0[f0.experiment == e].run]) for e in exps])
    rank = {"n_configs": len(exps), "original_best": float(base.max()), "original_median": float(np.median(base))}
    for dn, drop in drops.items():
        c = np.array([np.mean([pt(r, drop) for r in f0[f0.experiment == e].run]) for e in exps])
        rank[dn] = {"best": float(c.max()), "median": float(np.median(c)), "mean_lift": float(np.mean(c - base)),
                    "spearman": float(stats.spearmanr(base, c).correlation),
                    "top5_kept": len(set(np.argsort(-base)[:5]) & set(np.argsort(-c)[:5]))}
    out["fold0_ranking"] = rank

    # share of fold-0 errors carried by defective videos, and the diagnosis claims
    diag = {}
    for label, run in (("reference", "resnet18_lstm_L4_split0_seed0"),
                       ("crossfocal", "resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0")):
        z = np.load(ROOT / "runs/h7" / run / "diagnostics/frame_probs_test.npz", allow_pickle=True)
        lt = np.load(ROOT / "runs/h7" / run / "transition_log_matrix.npy")
        vids = sorted({k.split("__")[0] for k in z.files})
        per = {}
        for v in vids:
            y = z[f"{v}__y"]
            e = D.viterbi(z[f"{v}__lp"], lt) != y
            d = D.dist_to_boundary(y)
            per[v] = (int(e.sum()), int((e & (d > 10)).sum()), int((e & np.isin(y, list(MID))).sum()), len(y))
        tot_err = sum(p[0] for p in per.values())
        row = {"share_of_errors_on_defective": sum(p[0] for v, p in per.items() if v in qc) / tot_err,
               "share_of_errors_on_tier_A": sum(p[0] for v, p in per.items() if v in tier_a) / tot_err,
               "n_test": len(per), "n_defective_test": sum(1 for v in per if v in qc),
               "n_tier_A_test": sum(1 for v in per if v in tier_a)}
        for dn, drop in (("none", set()), *drops.items()):
            keep = [p for v, p in per.items() if v not in drop]
            e = sum(p[0] for p in keep)
            row[dn] = {"frame_error": e / sum(p[3] for p in keep), "far_share": sum(p[1] for p in keep) / e,
                       "mid_cleavage_share": sum(p[2] for p in keep) / e}
        diag[label] = row
    out["diagnosis"] = diag

    (ROOT / "results/noise_impact.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("released_fivefold", "categories")}, indent=1))
    print("wrote results/noise_impact.json")


if __name__ == "__main__":
    main()
