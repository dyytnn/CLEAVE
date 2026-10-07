#!/usr/bin/env python3
"""Contact sheet of every anomalous video at its annotated onsets, so each diagnosis can be checked by eye.

One row per video, one column per annotated stage (tPNf, t2, t3, t4, t8, tM, tB): the frame shown is
the frame the annotation calls the onset of that stage. If the annotation is right, a t4 frame shows
four cells. Row titles carry the automated category so the two can be compared directly.

    PYTHONPATH=src python scripts/render_anomaly_sheet.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLS = ["tPNf", "t2", "t3", "t4", "t8", "tM", "tB"]


def main(per_sheet: int = 7) -> None:
    man = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    log = json.loads((ROOT / "results/anomaly_log.json").read_text())["videos"]
    for s in range(0, len(log), per_sheet):
        chunk = log[s:s + per_sheet]
        fig, axes = plt.subplots(len(chunk), len(COLS), figsize=(1.9 * len(COLS), 2.05 * len(chunk)))
        for i, r in enumerate(chunk):
            g = man[(man.video == r["video"]) & man.phase.notna()].drop_duplicates("phase").set_index("phase")
            for j, ph in enumerate(COLS):
                ax = axes[i][j]
                ax.set_xticks([])
                ax.set_yticks([])
                if ph in g.index:
                    try:
                        ax.imshow(Image.open(g.loc[ph, "path"]).convert("L"), cmap="gray")
                    except Exception:
                        pass
                    ax.set_title(f"{ph}  {g.loc[ph, 'time_h']:.0f} h", fontsize=8, pad=2)
                else:
                    ax.set_title(f"{ph}  (not annotated)", fontsize=7, pad=2, color="#999")
                if j == 0:
                    ax.set_ylabel(f"{r['video']}\n{r['category']}\nlag {r['median_lag_h']:+.1f} h",
                                  fontsize=7.5, rotation=0, ha="right", va="center", labelpad=8)
        fig.tight_layout()
        out = ROOT / f"results/anomaly_sheet_{s // per_sheet + 1}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print(out.name)


if __name__ == "__main__":
    main()
