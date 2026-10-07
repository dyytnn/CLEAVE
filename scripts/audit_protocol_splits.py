#!/usr/bin/env python3
"""CLEAVE protocol audit: every patient-leakage number quoted in the paper, from one producer.

Supersedes the ad-hoc counting in ``scripts/audit_followup_splits.py`` (which used a third parsing
rule and therefore a third set of counts).  All parsing goes through ``stseg.data.patient``.

Reports, for the released five folds and for this repository's own grouped splits:

* ``leaking_patients``   -- patients present in more than one partition (the weak summary; what the
  first draft of the paper quoted);
* ``exposed_test``       -- test videos that have a sibling embryo of the same patient in train.
  This is the quantity that actually contaminates a reported test score, and it is 4-6x larger;
* a simulation of unconstrained random video-level splits, to show the leak is a property of the
  splitting scheme rather than of one released file.

    PYTHONPATH=src python scripts/audit_protocol_splits.py --out results/protocol_audit.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from stseg.data.patient import exposed_test_videos, group_videos, leaking_patients, patient_of

ROOT = Path(__file__).resolve().parents[1]


def _official(fold: int) -> dict[str, str]:
    rows = [r for r in csv.reader(open(ROOT / f"data/splits/nantes_official/split{fold}.csv")) if r]
    return {v: p for v, p in rows}


def _json_split(path: Path) -> dict[str, str]:
    doc = json.loads(path.read_text())
    return {v: part for part, vids in doc["videos"].items() for v in vids}


def audit(partition_of: dict[str, str], level: str) -> dict:
    leaks = leaking_patients(partition_of, level)
    exposed = exposed_test_videos(partition_of, level)
    n_test = sum(1 for p in partition_of.values() if p == "test")
    n_pat = len({patient_of(v, level) for v in partition_of})
    return {
        "level": level,
        "n_videos": len(partition_of),
        "n_patients": n_pat,
        "n_leaking_patients": len(leaks),
        "frac_leaking_patients": len(leaks) / n_pat,
        "n_test_videos": n_test,
        "n_test_patients": len({patient_of(v, level) for v, p in partition_of.items() if p == "test"}),
        "n_exposed_test_videos": len(exposed),
        "frac_exposed_test_videos": len(exposed) / n_test if n_test else 0.0,
        "leaking_patients": leaks,
        "exposed_test_videos": exposed,
    }


def simulate(videos: list[str], level: str, train_frac: float, test_frac: float, seeds: int) -> dict:
    """Unconstrained random video-level splits: how many patients leak, by chance alone."""
    by = group_videos(videos, level)
    n = len(videos)
    n_tr, n_te = int(round(n * train_frac)), int(round(n * test_frac))
    leaked, exposed = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        perm = list(rng.permutation(videos))
        part = {v: "train" for v in perm[:n_tr]}
        part.update({v: "test" for v in perm[n_tr : n_tr + n_te]})
        part.update({v: "val" for v in perm[n_tr + n_te :]})
        leaked.append(len(leaking_patients(part, level)))
        exposed.append(len(exposed_test_videos(part, level)) / max(n_te, 1))
    return {
        "level": level, "seeds": seeds, "train_frac": train_frac, "test_frac": test_frac,
        "leaked_patients": leaked,
        "n_videos": n, "n_patients": len(by),
        "leaked_patients_mean": statistics.mean(leaked),
        "leaked_patients_min": min(leaked), "leaked_patients_max": max(leaked),
        "frac_exposed_test_mean": statistics.mean(exposed),
        "frac_exposed_test_min": min(exposed), "frac_exposed_test_max": max(exposed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--out", default="results/protocol_audit.json")
    args = ap.parse_args()

    release = json.loads((ROOT / "results/tempo_bench_release.json").read_text())
    excluded = set(release["excluded_videos"]["video_ids"])

    report: dict = {"excluded_videos": len(excluded), "levels": list(("couple", "cycle"))}

    # --- released five folds, all 704 videos and the 652 clean ones -------------------------------
    folds: dict = {}
    for fold in range(5):
        part = _official(fold)
        clean = {v: p for v, p in part.items() if v not in excluded}
        folds[f"split{fold}"] = {
            "all704": {lv: audit(part, lv) for lv in ("couple", "cycle")},
            "clean652": {lv: audit(clean, lv) for lv in ("couple", "cycle")},
        }
    report["official_folds"] = folds
    for scope in ("all704", "clean652"):
        for lv in ("couple", "cycle"):
            ex = [folds[f"split{k}"][scope][lv]["n_exposed_test_videos"] for k in range(5)]
            nt = [folds[f"split{k}"][scope][lv]["n_test_videos"] for k in range(5)]
            lk = [folds[f"split{k}"][scope][lv]["n_leaking_patients"] for k in range(5)]
            report.setdefault("official_pooled", {})[f"{scope}.{lv}"] = {
                "leaking_patients_per_fold": lk,
                "exposed_test_videos_per_fold": ex,
                "test_videos_per_fold": nt,
                "pooled_exposed": sum(ex), "pooled_test": sum(nt),
                "pooled_frac_exposed": sum(ex) / sum(nt),
            }

    # --- this repository's own splits -------------------------------------------------------------
    own = {}
    for name in ("nantes_grouped_v1", "nantes_grouped_v2", "nantes_grouped_v2_trainclean_v1", "nantes_grouped_v2_trainclean_v2", "nantes_grouped_v2_trainclean_v3", "nantes_siblingleak_clean_v1",
                 "nantes_siblingleak_leaky_v1", "nantes_random_73_seed0"):
        p = ROOT / f"data/splits/{name}.json"
        if p.exists():
            own[name] = {lv: audit(_json_split(p), lv) for lv in ("couple", "cycle")}
    report["own_splits"] = own

    # --- simulation -------------------------------------------------------------------------------
    # The simulation must be drawn from the SAME video population as the observed count it is
    # compared against; an earlier version placed the all-704 count inside a 652-video distribution,
    # which moved the released fold from the 1st percentile to the 31st. Both populations are now
    # simulated separately and each percentile is computed against its own.
    populations = {"all704": sorted(_official(0)), "clean652": sorted(set(_official(0)) - excluded)}
    # Two geometries. "70/30" is the shape the follow-up papers describe and is the reference for them.
    # "released" matches the released five-fold shape (564/70/70 of 704, scaled to the clean list), and
    # is the only one against which the released fold-0 count is a like-for-like comparison.
    geometries = {"7030": (0.70, 0.30), "released": (564 / 704, 70 / 704)}
    report["simulation"] = {}
    for scope, videos in populations.items():
        for lv in ("couple", "cycle"):
            for geo, (tr, te) in geometries.items():
                gsim = simulate(videos, lv, tr, te, args.seeds)
                gobs = report["official_folds"]["split0"][scope][lv]["n_leaking_patients"]
                gdraws = gsim["leaked_patients"]
                gsim["geometry"] = geo
                gsim["released_fold0_leaking_patients"] = gobs
                gsim["released_fold0_percentile"] = round(100 * sum(d <= gobs for d in gdraws) / len(gdraws))
                report["simulation"][f"{scope}.{lv}.{geo}"] = gsim
            sim = simulate(videos, lv, 0.70, 0.30, args.seeds)
            obs = report["official_folds"]["split0"][scope][lv]["n_leaking_patients"]
            draws = sim["leaked_patients"]   # kept: the Fig. 1b histogram is drawn from these
            sim["population"] = scope
            sim["released_fold0_leaking_patients"] = obs
            sim["released_fold0_percentile"] = round(100 * sum(d <= obs for d in draws) / len(draws))
            report["simulation"][f"{scope}.{lv}"] = sim
    # backwards-compatible aliases: the paper quotes the clean list, which is the one it evaluates on
    report["simulation"]["couple"] = report["simulation"]["clean652.couple"]
    report["simulation"]["cycle"] = report["simulation"]["clean652.cycle"]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))

    pool = report["official_pooled"]["all704.couple"]
    print(f"released folds (704 videos, couple level): leaking patients per fold "
          f"{pool['leaking_patients_per_fold']}")
    print(f"  test videos with a sibling in train: {pool['exposed_test_videos_per_fold']} "
          f"of {pool['test_videos_per_fold']} -> pooled {pool['pooled_frac_exposed']:.1%}")
    sim = report["simulation"]["clean652.couple"]
    print(f"simulation ({args.seeds} random 70/30 video-level splits): {sim['leaked_patients_mean']:.1f} "
          f"patients leak on average (range {sim['leaked_patients_min']}-{sim['leaked_patients_max']}), "
          f"{sim['frac_exposed_test_mean']:.1%} of test videos exposed")
    for name, a in own.items():
        c = a["couple"]
        print(f"{name}: {c['n_leaking_patients']} leaking patients (couple level), "
              f"{c['n_exposed_test_videos']}/{c['n_test_videos']} test videos exposed")
    print("wrote", out)


if __name__ == "__main__":
    main()
