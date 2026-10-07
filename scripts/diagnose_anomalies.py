#!/usr/bin/env python3
"""Per-video diagnosis of what is wrong with each anomalous video.

The scans in scan_hard_videos.py and scan_blank_videos.py say *which* videos no model can decode.
This script says *why*, one line per video, from measurements that can be checked:

image evidence (reference plane, frames sampled across the recording, inside the lit well)
  contrast     spatial std of the well; a featureless disc has almost none
  sharpness    variance of the Laplacian; low means out of focus
  change       mean absolute difference between the first and last sampled frame, relative to the
               dataset median; an embryo that develops from one cell to a blastocyst changes a lot

annotation-versus-image evidence (held-out predictions of every configuration that saw the video)
  lag_h        median over phases of (predicted onset - annotated onset); positive means the images
               reach each stage later than annotated, i.e. the annotation runs ahead of the embryo
  collapsed    share of configurations whose predicted onsets all fall at one time, i.e. the model
               reads a single stage for the whole video

Every category is decided by a threshold set at a dataset percentile, printed with its evidence,
and the rendered contact sheet (results/anomaly_sheet.png) lets each label be checked by eye.

    PYTHONPATH=src python scripts/diagnose_anomalies.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB", "tHB"]
DIVISIONS = {"t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+"}


def well_pixels(a: np.ndarray) -> np.ndarray:
    return a > np.percentile(a, 60)


def image_features(paths: list[str]) -> dict | None:
    imgs = []
    for p in paths:
        try:
            imgs.append(np.asarray(Image.open(p).convert("L").resize((250, 250)), dtype=np.float32))
        except Exception:
            continue
    if len(imgs) < 3:
        return None
    con, shp = [], []
    for a in imgs:
        m = well_pixels(a)
        con.append(float(a[m].std()))
        shp.append(float(ndimage.laplace(a)[m].var()))
    m0 = well_pixels(imgs[0]) & well_pixels(imgs[-1])
    change = float(np.abs(imgs[-1] - imgs[0])[m0].mean()) if m0.any() else 0.0
    return {"contrast": statistics.median(con), "sharpness": statistics.median(shp), "change": change}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--lag-h", type=float, default=4.0, help="|median onset lag| above which labels and images disagree")
    ap.add_argument("--out", default="results/anomaly_log")
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    lab = man[man.phase.notna()]
    hard = json.loads((ROOT / "results/hard_videos.json").read_text())
    targets = [r["video"] for r in hard["flagged"]]
    info = {r["video"]: r for r in hard["flagged"]}

    # --- image features for every video, so thresholds are dataset percentiles, not guesses ------
    jobs = []
    for vid, g in lab.groupby("video"):
        g = g.sort_values("frame_index")
        idx = np.linspace(0, len(g) - 1, min(a.sample, len(g))).round().astype(int)
        jobs.append((vid, [g.path.iloc[i] for i in idx]))
    with ThreadPoolExecutor(max_workers=16) as ex:
        feats = dict(zip([j[0] for j in jobs], ex.map(lambda j: image_features(j[1]), jobs)))
    F = pd.DataFrame({v: f for v, f in feats.items() if f}).T
    thr = {"contrast": float(F.contrast.quantile(0.01)), "sharpness": float(F.sharpness.quantile(0.02)),
           "change": float(F.change.quantile(0.02))}
    med_change = float(F.change.median())

    # --- held-out predictions ----------------------------------------------------------------
    df = pd.read_csv(ROOT / "results/runs_master.csv")
    df = df[df.split_key != "image"]
    preds: dict[str, list[dict]] = {v: [] for v in targets}
    for r in df.itertuples():
        f = ROOT / "runs/h7" / r.run / "per_video_test.json"
        if not f.exists():
            continue
        for v in json.loads(f.read_text()):
            if v["video"] in preds:
                preds[v["video"]].append(v)

    rows = []
    for vid in targets:
        g = lab[lab.video == vid]
        onset = g.drop_duplicates("phase").set_index("phase").time_h.to_dict()
        n_div = len(DIVISIONS & set(onset))
        f = feats.get(vid) or {}
        P = preds[vid]
        lags, collapsed = [], 0
        for p in P:
            common = [ph for ph in p["gt_times"] if ph in p["pred_times"]]
            if common:
                lags.append(statistics.median(p["pred_times"][ph] - p["gt_times"][ph] for ph in common))
            if len(set(p["pred_times"].values())) <= 1:
                collapsed += 1
        lag = statistics.median(lags) if lags else float("nan")
        coll = collapsed / len(P) if P else float("nan")

        # --- decide, most specific first --------------------------------------------------------
        cat, why = None, None
        if f and f["contrast"] <= thr["contrast"]:
            cat = "blank_well"
            why = (f"every sampled frame is a featureless bright well with no embryo visible "
                   f"(well contrast {f['contrast']:.1f}, dataset 1st percentile {thr['contrast']:.1f}), "
                   f"yet the annotation records {len(onset)} phases")
        elif f and f["change"] <= thr["change"] and n_div >= 2:
            cat = "embryo_static"
            why = (f"the embryo does not visibly change across the recording (first-to-last change "
                   f"{f['change']:.1f}, dataset median {med_change:.1f}) while the annotation records "
                   f"{n_div} divisions")
        elif f and f["sharpness"] <= thr["sharpness"]:
            cat = "out_of_focus"
            why = f"frames are out of focus (sharpness {f['sharpness']:.0f}, dataset 2nd percentile {thr['sharpness']:.0f})"
        elif not np.isnan(lag) and lag >= a.lag_h:
            cat = "label_ahead_of_images"
            why = (f"the annotation runs ahead of the embryo: models reach each annotated stage "
                   f"{lag:.1f} h later than it is labelled, across {len(lags)} configurations")
        elif not np.isnan(lag) and lag <= -a.lag_h:
            cat = "label_behind_images"
            why = (f"the embryo reaches each stage {-lag:.1f} h before it is labelled, "
                   f"across {len(lags)} configurations")
        elif not np.isnan(coll) and coll >= 0.5:
            cat = "single_stage_read"
            why = (f"{100 * coll:.0f}% of configurations read a single stage for the whole video; "
                   f"no image defect is measurable")
        else:
            cat = "hard_no_defect_found"
            why = (f"no image or annotation defect is measurable (lag {lag:+.1f} h, contrast "
                   f"{f.get('contrast', float('nan')):.1f}); the video is hard rather than broken")
        rows.append({"video": vid, "category": cat, "description": why,
                     "best_acc": round(info[vid]["best_acc"], 3), "median_acc": round(info[vid]["median_acc"], 3),
                     "n_configs": info[vid]["n_configs"], "n_phases": len(onset), "n_divisions": n_div,
                     "t_first_h": round(min(onset.values()), 1) if onset else None,
                     "t_last_h": round(max(onset.values()), 1) if onset else None,
                     "contrast": round(f.get("contrast", float("nan")), 1),
                     "sharpness": round(f.get("sharpness", float("nan")), 0),
                     "change": round(f.get("change", float("nan")), 1),
                     "median_lag_h": round(lag, 2), "collapsed_share": round(coll, 2)})

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"{a.out}.csv", index=False)
    (ROOT / f"{a.out}.json").write_text(json.dumps({"thresholds": thr, "dataset_median_change": med_change,
                                                     "lag_threshold_h": a.lag_h, "videos": rows}, indent=1))
    with open(ROOT / f"{a.out}.md", "w") as fh:
        fh.write("# Anomalous videos: what is wrong with each\n\n")
        fh.write(f"Generated by `scripts/diagnose_anomalies.py`. Thresholds are dataset percentiles: "
                 f"contrast <= {thr['contrast']:.1f}, sharpness <= {thr['sharpness']:.0f}, "
                 f"change <= {thr['change']:.1f}, |onset lag| >= {a.lag_h} h.\n\n")
        fh.write("| video | category | what is wrong | best acc | configs |\n|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['video']} | {r['category']} | {r['description']} | {r['best_acc']} | {r['n_configs']} |\n")
    print(f"thresholds (dataset percentiles): contrast<={thr['contrast']:.1f}  sharpness<={thr['sharpness']:.0f}  "
          f"change<={thr['change']:.1f} (median {med_change:.1f})  |lag|>={a.lag_h} h\n")
    for r in rows:
        print(f"{r['video']:12} [{r['category']}]")
        print(f"             {r['description']}")
    print(f"\nwrote {a.out}.csv / .json / .md")


if __name__ == "__main__":
    main()
