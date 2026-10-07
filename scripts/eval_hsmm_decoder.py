#!/usr/bin/env python
"""TEMPO rung T7: retrain-free semi-Markov (HSMM) decoding on cached frame log-probs of every checkpoint.

Motivation (track1_TEMPO/diagnostics_findings.md): frame-level Viterbi skips 126-205 of 809 GT segments per test set --
short phases (t3, t5, t7, tB) with weak frame evidence are absorbed by neighbours, and the skip-fill rule then produces
the 13-26 h tails that dominate timing MAE. An HSMM adds an explicit per-phase duration model (fit on the run's own
training labels) and segment-level transitions, so a phase's existence and length are scored, not only its frames.

Protocol: duration model kind in {gamma, empirical} x lambda in {0.5, 1, 2, 4} selected on the run's *val* split by p_t
(tie-break p_v); the selected setting is applied once to test. Controls: frame-level Viterbi (official), flat-duration
HSMM with lambda=0 (segment-level Markov only). Writes <run>/results_eval_hsmm.json and results/hsmm_leaderboard.csv.

  PYTHONPATH=src python scripts/eval_hsmm_decoder.py                # every run with a cache in results/frame_probs
  PYTHONPATH=src python scripts/eval_hsmm_decoder.py --runs runs/h7/resnet50_lstm_L8_split0_seed1
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from functools import partial
from pathlib import Path

import numpy as np

from stseg.data.nantes_kinetic import NUM_CLASSES
from stseg.eval.kinetic_metrics import (duration_log_probs, evaluate_videos, hsmm_viterbi, segment_transition_log_matrix,
                                        transition_log_matrix)
from stseg.kinetic.datasets import _frames

ROOT = Path(__file__).resolve().parents[1]
GRID = [("flat", 0.0)] + [(kind, lam) for kind in ("gamma", "empirical") for lam in (0.5, 1.0, 2.0, 4.0)]  # flat/0 = segment-level Markov, no duration prior
MAX_DUR = 600
_train_cache: dict[str, list[np.ndarray]] = {}


def load_seqs(path: Path) -> list[dict]:
    z = np.load(path); vids = sorted({k.rsplit("__", 1)[0] for k in z.files})
    return [{"video": v, "log_probs": z[f"{v}__lp"].astype(np.float32), "labels": z[f"{v}__y"].astype(int), "times_h": z[f"{v}__t"].astype(float)} for v in vids]


def train_labels(cfg: dict) -> list[np.ndarray]:
    key = json.dumps({k: cfg["data"].get(k) for k in ("manifest", "split", "plane", "planes", "split_mode", "merge_last_class")}, sort_keys=True)
    if key not in _train_cache:
        _train_cache[key] = _frames(cfg["data"], "train", "eval", 0, None, None).label_sequences()
    return _train_cache[key]


def n_skipped(seqs, decode) -> int:
    return sum(len(set(np.asarray(s["labels"]).tolist()) - set(decode(np.asarray(s["log_probs"])).tolist())) for s in seqs)


def slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "per_video"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--cache_root", default="results/frame_probs")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    runs = [ROOT / r for r in a.runs] if a.runs else [ROOT / "runs/h7" / Path(p).name for p in sorted(glob.glob(str(ROOT / a.cache_root / "*")))]
    rows = []
    for run in runs:
        cache = ROOT / a.cache_root / run.name
        if not (cache / "val.npz").exists() or not (cache / "test.npz").exists() or not (run / "config.resolved.json").exists():
            print(f"skip {run.name} (no cache)"); continue
        out_f = run / "results_eval_hsmm.json"
        if out_f.exists() and not a.force:
            rows.append(json.loads(out_f.read_text())); print(f"{run.name}: cached"); continue
        cfg = json.loads((run / "config.resolved.json").read_text())
        labels = train_labels(cfg)
        lt_frame = transition_log_matrix(labels, NUM_CLASSES)
        lt_seg = segment_transition_log_matrix(labels)
        durs = {kind: duration_log_probs(labels, MAX_DUR, kind) for kind in ("gamma", "empirical", "flat")}
        val, test = load_seqs(cache / "val.npz"), load_seqs(cache / "test.npz")

        base_val = evaluate_videos(val, lt_frame); base_test = evaluate_videos(test, lt_frame)
        grid_val = {}
        for kind, lam in GRID:
            dec = partial(hsmm_viterbi, log_trans_seg=lt_seg, log_dur=durs[kind], lam=lam)
            r = evaluate_videos(val, lt_frame, decode=dec)
            grid_val[f"{kind}_lam{lam:g}"] = {"p_t": r["p_t"], "p_v": r["p_v"], "edit": r["edit"], "mae_h_all": r["mae_h_all"]}
        best_key = max(grid_val, key=lambda k: (grid_val[k]["p_t"], grid_val[k]["p_v"]))
        kind, lam = best_key.split("_lam"); lam = float(lam)
        dec_best = partial(hsmm_viterbi, log_trans_seg=lt_seg, log_dur=durs[kind], lam=lam)
        dec_flat = partial(hsmm_viterbi, log_trans_seg=lt_seg, log_dur=durs["flat"], lam=0.0)
        hs_test = evaluate_videos(test, lt_frame, decode=dec_best)
        flat_test = evaluate_videos(test, lt_frame, decode=dec_flat)
        rec = {"run": run.name, "selected": {"kind": kind, "lam": lam}, "grid_val": grid_val,
               "viterbi_val": slim(base_val), "viterbi_test": slim(base_test), "hsmm_test": slim(hs_test), "flat_seg_markov_test": slim(flat_test),
               "skipped_segments_test": {"viterbi": n_skipped(test, lambda lp: __import__("stseg.eval.kinetic_metrics", fromlist=["viterbi"]).viterbi(lp, lt_frame)),
                                         "hsmm": n_skipped(test, dec_best)},
               "hsmm_per_video_test": hs_test["per_video"]}
        out_f.write_text(json.dumps(rec, indent=1, default=float))
        rows.append(rec)
        b, h = base_test, hs_test
        print(f"{run.name}: [{best_key}] p_v {b['p_v']:.3f}->{h['p_v']:.3f} p_t {b['p_t']:.3f}->{h['p_t']:.3f} edit {b['edit']:.1f}->{h['edit']:.1f} "
              f"F1@50 {b['f1']['50']:.1f}->{h['f1']['50']:.1f} MAE {b['mae_h_all']:.2f}->{h['mae_h_all']:.2f}h skipped {rec['skipped_segments_test']['viterbi']}->{rec['skipped_segments_test']['hsmm']}", flush=True)

    out = ROOT / "results/hsmm_leaderboard.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "kind", "lam", "p_v_viterbi", "p_v_hsmm", "p_t_viterbi", "p_t_hsmm", "edit_viterbi", "edit_hsmm", "f1_50_viterbi", "f1_50_hsmm",
                    "mae_viterbi", "mae_hsmm", "skipped_viterbi", "skipped_hsmm", "p_t_flat_seg_markov"])
        for r in rows:
            b, h, fl = r["viterbi_test"], r["hsmm_test"], r["flat_seg_markov_test"]
            w.writerow([r["run"], r["selected"]["kind"], r["selected"]["lam"], f"{b['p_v']:.4f}", f"{h['p_v']:.4f}", f"{b['p_t']:.4f}", f"{h['p_t']:.4f}",
                        f"{b['edit']:.2f}", f"{h['edit']:.2f}", f"{b['f1']['50']:.2f}", f"{h['f1']['50']:.2f}", f"{b['mae_h_all']:.3f}", f"{h['mae_h_all']:.3f}",
                        r["skipped_segments_test"]["viterbi"], r["skipped_segments_test"]["hsmm"], f"{fl['p_t']:.4f}"])
    if rows:
        d = np.array([[r["hsmm_test"][k] - r["viterbi_test"][k] for k in ("p_v", "p_t", "edit", "mae_h_all")] + [r["hsmm_test"]["f1"]["50"] - r["viterbi_test"]["f1"]["50"]] for r in rows])
        print(f"\n{len(rows)} runs: mean delta (hsmm - viterbi) p_v {d[:,0].mean():+.4f} p_t {d[:,1].mean():+.4f} edit {d[:,2].mean():+.2f} MAE {d[:,3].mean():+.3f}h F1@50 {d[:,4].mean():+.2f}; "
              f"p_t improved in {(d[:,1] > 0).sum()}/{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
