#!/usr/bin/env python3
"""Per-frame scan of every annotated video for labelled frames with no embryo in them.

Inspection of the model-flagged videos found wells that are empty, or an out-of-focus blur, late in
the recording while stages are still annotated there. That defect needs no model to detect, so this
scans all 652 annotated videos, not only the ones some model happened to evaluate held out.

Structure score: variance of the Laplacian inside the lit well. An embryo in focus scores high; an
empty well or a blur scores near the medium's own texture. A frame is featureless below the
threshold, set from the pooled distribution (see --q).

    PYTHONPATH=src python scripts/scan_featureless_frames.py
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]


def structure(path: str) -> float | None:
    try:
        a = np.asarray(Image.open(path).convert("L").resize((250, 250)), dtype=np.float32)
    except Exception:
        return None
    m = a > np.percentile(a, 60)
    return float(ndimage.laplace(a)[m].var()) if m.any() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=30)
    ap.add_argument("--min-frac", type=float, default=0.10, help="flag a video when this share of labelled frames is featureless")
    ap.add_argument("--out", default="results/featureless_frames.json")
    a = ap.parse_args()

    man = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    lab = man[man.phase.notna()].sort_values(["video", "frame_index"])
    jobs = []
    for vid, g in lab.groupby("video"):
        idx = np.linspace(0, len(g) - 1, min(a.per_video, len(g))).round().astype(int)
        for i in idx:
            r = g.iloc[i]
            jobs.append((vid, int(r.frame_index), float(r.time_h), r.phase, r.path))
    with ThreadPoolExecutor(max_workers=16) as ex:
        scores = list(ex.map(lambda j: structure(j[4]), jobs))
    F = pd.DataFrame([(*j[:4], s) for j, s in zip(jobs, scores)], columns=["video", "frame", "time_h", "phase", "s"]).dropna()

    # the pooled distribution is bimodal: embryo frames high, empty or blurred frames near zero.
    # Take the threshold at the valley between the two modes on a log scale.
    ls = np.log10(F.s.clip(lower=1e-3))
    hist, edges = np.histogram(ls, bins=80)
    lo_mode, hi_mode = np.argmax(hist[:40]), 40 + np.argmax(hist[40:])
    valley = lo_mode + int(np.argmin(hist[lo_mode:hi_mode + 1]))
    thr = float(10 ** edges[valley])
    F["featureless"] = F.s < thr

    rows = []
    for vid, g in F.groupby("video"):
        bad = g[g.featureless]
        if len(bad) / len(g) >= a.min_frac:
            rows.append({"video": vid, "frac_featureless": round(len(bad) / len(g), 2), "n_sampled": len(g),
                         "from_h": round(float(bad.time_h.min()), 1), "to_h": round(float(bad.time_h.max()), 1),
                         "phases_on_featureless": sorted(set(bad.phase)),
                         "pattern": "throughout" if len(bad) / len(g) > 0.9 else
                                    ("late" if bad.time_h.min() > g.time_h.quantile(0.5) else "scattered")})
    rows.sort(key=lambda r: -r["frac_featureless"])
    excluded = set(json.loads((ROOT / "results/tempo_bench_release.json").read_text())["excluded_videos"]["video_ids"])
    out = {"threshold_structure": thr, "per_video": a.per_video, "min_frac": a.min_frac,
           "n_videos": int(F.video.nunique()), "n_frames": len(F), "n_flagged": len(rows), "flagged": rows,
           "flagged_not_excluded": sorted({r["video"] for r in rows} - excluded)}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(f"{F.video.nunique()} videos, {len(F)} labelled frames sampled; featureless threshold {thr:.1f}")
    print(f"videos with >= {int(100 * a.min_frac)}% featureless labelled frames: {len(rows)}")
    for r in rows:
        print(f"  {r['video']:12} {100 * r['frac_featureless']:4.0f}%  {r['pattern']:10} {r['from_h']:6.1f}-{r['to_h']:6.1f} h  "
              f"annotated as {','.join(r['phases_on_featureless'])}")
    print("wrote", ROOT / a.out)


if __name__ == "__main__":
    main()
