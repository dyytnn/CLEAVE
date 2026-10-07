#!/usr/bin/env python3
"""Model-free test of the 0.25-h / 0.2-h explanation for the timeline mismatches.

If the images were acquired at the usual rate while the time and annotation files were written on a 0.2-h grid, then
for a mismatched video the recording span of its time file divided by its number of images should equal the usual
acquisition interval, even though the file itself steps by 0.2 h. Nothing here uses a model or a label.

    python scripts/timeline_mechanism.py       # -> results/frame_timing/mechanism.json, mechanism_videos.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("data/raw/nantes_embryo_dataset")


def main() -> None:
    al = pd.read_csv(ROOT / "results/frame_timing/alignment_oof.csv")
    log = json.loads((ROOT / "data/qc/nantes_video_defects_v4.json").read_text())["videos"]
    tl = {v for v, e in log.items() if "timeline_mismatch" in e["defects"]}
    rows = []
    for v, n in zip(al.video, al.images):
        t = pd.read_csv(RAW / "time_elapsed/embryo_dataset_time_elapsed" / f"{v}_timeElapsed.csv").time.to_numpy()
        d = np.diff(t)
        d = d[d > 0]
        rows.append({"video": v, "mismatch": v in tl, "file_step": float(d.mean()), "image_interval": float((t[-1] - t[0]) / (n - 1)),
                     "covers_all_images": bool(len(t) >= n), "time_rows": int(len(t)), "images": int(n)})
    d = pd.DataFrame(rows)
    out = {}
    out_extra = {"n_mismatch_step_0_2_interval_0_23_0_27": int((d.mismatch & d.file_step.between(0.19, 0.21) & d.image_interval.between(0.23, 0.27)).sum()),
                 "n_mismatch_step_0_2": int((d.mismatch & d.file_step.between(0.19, 0.21)).sum()),
                 "n_other_step_above_0_3": int((~d.mismatch & (d.file_step > 0.3)).sum()),
                 "n_other_covering": int((~d.mismatch & d.covers_all_images).sum())}
    for name, g in (("mismatch", d[d.mismatch]), ("other", d[~d.mismatch])):
        out[name] = {"n": len(g), "file_step_median": float(g.file_step.median()),
                     "image_interval_median": float(g.image_interval.median()),
                     "image_interval_iqr": [float(g.image_interval.quantile(0.25)), float(g.image_interval.quantile(0.75))],
                     "n_interval_0_23_0_27": int(g.image_interval.between(0.23, 0.27).sum())}
    out.update(out_extra)
    (ROOT / "results/frame_timing/mechanism.json").write_text(json.dumps(out, indent=1))
    d.to_csv(ROOT / "results/frame_timing/mechanism_videos.csv", index=False)   # per video, for Supplementary Fig. 12c
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
