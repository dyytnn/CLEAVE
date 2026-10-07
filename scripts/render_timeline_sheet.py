#!/usr/bin/env python3
"""Contact sheets for inspecting timeline mismatches.

For each video, the reference-plane image at the middle of its annotated t2, t4 and t8 under the released pairing
(image k <-> annotation frame k), under the out-of-fold fitted map and under a fixed 1.25 annotation frames per image
(the 0.25 h / 0.2 h hypothesis), side by side, so a reader can see which pairing
shows the annotated number of cells. Nantes is public; no identity overlay to crop.

    PYTHONPATH=src python scripts/render_timeline_sheet.py VIDEO [VIDEO ...] --out sheet.png
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("data/raw/nantes_embryo_dataset")
EVENTS = ("tPNf", "t2", "t4", "t8")
S = 140


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    al = pd.read_csv(ROOT / "results/frame_timing/alignment_oof.csv").set_index("video")
    W = 3 * S + 10
    sheet = Image.new("L", (W * len(EVENTS), S * len(a.videos)), 255)
    d = ImageDraw.Draw(sheet)
    for i, v in enumerate(a.videos):
        ann = pd.read_csv(RAW / "ann/embryo_dataset_annotations" / f"{v}_phases.csv", header=None,
                          names=["p", "s", "e"]).set_index("p")
        files = {int(re.search(r"RUN(\d+)", f).group(1)): f for f in glob.glob(str(RAW / "embryo_dataset" / v / "*.jp*g"))}
        sl, off = al.loc[v, "slope_fit"], al.loc[v, "offset_fit"]
        for j, ev in enumerate(EVENTS):
            if ev not in ann.index:
                continue
            jj = int((ann.loc[ev, "s"] + ann.loc[ev, "e"]) // 2)
            for q, k in enumerate((jj, int(round((jj - off) / sl)), int(round(jj / 1.25)))):
                if k in files:
                    sheet.paste(Image.open(files[k]).convert("L").resize((S, S)), (j * W + q * S, i * S))
            d.text((j * W + 3, i * S + 3), f"{v} {ev} | fit a={sl:.2f} | 1.25", fill=255)
    sheet.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
