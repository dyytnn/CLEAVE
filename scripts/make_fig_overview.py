#!/usr/bin/env python3
"""Figure 1: what the data, the task and the metric are, for a reader from either side.

Round-3 review (all three seats) found that the manuscript's first figure was an anatomy of protocol
defects expressed in a metric defined only in the Methods, so a reader who had not already trained a
model on this dataset had no entry point. This panel supplies one.

  a  one embryo at one instant through the seven focal planes
  b  the same embryo across development, with the annotated phase onsets on an hours axis
  c  what temporal accuracy counts: a predicted onset scores only if it falls inside that event's
     tolerance window around the annotated one

    PYTHONPATH=src python scripts/make_fig_overview.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RAW = Path("data/raw/nantes_embryo_dataset")
PLANES = ["embryo_dataset_F-45", "embryo_dataset_F-30", "embryo_dataset_F-15",
          "embryo_dataset", "embryo_dataset_F15", "embryo_dataset_F30", "embryo_dataset_F45"]
PLANE_LABEL = ["-45", "-30", "-15", "0", "+15", "+30", "+45"]
VIDEO = "BA958-2"
SHOW = ["t2", "t4", "t8", "tM", "tB"]          # phases to show as thumbnails in panel b
C_GT, C_PRED, C_OK = "#1f77b4", "#d62728", "#2ca02c"
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def crop_box(img: Image.Image, frac: float = 0.62) -> tuple[int, int, int, int]:
    """Square crop box centred on the most structured region, which is the embryo, not the dish."""
    a = np.asarray(img, dtype=float)
    g = np.abs(np.gradient(a)[0]) + np.abs(np.gradient(a)[1])
    k = 24
    box = g[: g.shape[0] // k * k, : g.shape[1] // k * k].reshape(g.shape[0] // k, k, -1, k).sum(axis=(1, 3))
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    w, h = img.size
    side = int(min(w, h) * frac)
    x = int(np.clip((cx + 0.5) * k - side / 2, 0, w - side))
    y = int(np.clip((cy + 0.5) * k - side / 2, 0, h - side))
    return (x, y, x + side, y + side)


def crop(img: Image.Image, frac: float = 0.62, box: tuple[int, int, int, int] | None = None) -> Image.Image:
    return img.crop(box or crop_box(img, frac))


def main() -> None:
    man = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    v = man[man.video == VIDEO].sort_values("frame_index").reset_index(drop=True)
    onsets = v.dropna(subset=["phase"]).drop_duplicates("phase")[["phase", "time_h", "path"]]
    fname = {r.phase: Path(r.path).name for r in onsets.itertuples()}
    t_of = {r.phase: float(r.time_h) for r in onsets.itertuples()}

    fig = plt.figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(3, 7, height_ratios=[1.05, 1.25, 1.0], hspace=0.55, wspace=0.08)

    # --- a: seven focal planes of one instant -------------------------------------------------
    ref = fname.get("t4") or next(iter(fname.values()))
    ref_img = RAW / "embryo_dataset" / VIDEO / ref
    box = crop_box(Image.open(ref_img).convert("L")) if ref_img.exists() else None
    for i, (pl, lab) in enumerate(zip(PLANES, PLANE_LABEL)):
        ax = fig.add_subplot(gs[0, i])
        f = RAW / pl / VIDEO / ref
        if f.exists():
            ax.imshow(crop(Image.open(f).convert("L"), box=box), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
        ax.set_title(f"{lab} \u00b5m" + (" (reference)" if i == 3 else ""), fontsize=7, pad=2)
        if i == 0:
            ax.text(-0.18, 0.5, "a", transform=ax.transAxes, fontsize=13, fontweight="bold", va="center", ha="right")
    fig.text(0.5, 0.965, f"one embryo ({VIDEO}) at one instant, through the seven focal planes the system records",
             ha="center", fontsize=8)

    # --- b: the same embryo across development -------------------------------------------------
    shown = [p for p in SHOW if p in fname]
    for i, ph in enumerate(shown):
        ax = fig.add_subplot(gs[1, i])
        f = RAW / "embryo_dataset" / VIDEO / fname[ph]
        if f.exists():
            ax.imshow(crop(Image.open(f).convert("L")), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
        ax.set_title(f"{ph}   {t_of[ph]:.0f} h", fontsize=7, pad=2)
        if i == 0:
            ax.text(-0.18, 0.5, "b", transform=ax.transAxes, fontsize=13, fontweight="bold", va="center", ha="right")

    axt = fig.add_subplot(gs[1, len(shown):])
    order = [p for p in ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB"] if p in t_of]
    axt.hlines(0, 0, max(t_of.values()) * 1.02, color="0.7", lw=1)
    axt.plot([t_of[p] for p in order], [0] * len(order), "|", color=C_GT, ms=11, mew=1.4)
    for j, p in enumerate(order):
        dy = (9, -14, 19)[j % 3]
        axt.annotate(p, (t_of[p], 0), textcoords="offset points", xytext=(0, dy),
                     ha="center", fontsize=5.6, color=C_GT)
    axt.set_yticks([])
    axt.set_xlabel("hours post insemination", fontsize=7)
    axt.set_ylim(-0.9, 0.9)
    axt.spines["left"].set_visible(False)
    axt.tick_params(labelsize=7)
    axt.set_title("the 15 annotated onsets of this embryo", fontsize=7, pad=3)

    # --- c: what temporal accuracy counts ------------------------------------------------------
    axm = fig.add_subplot(gs[2, :])
    axm.text(-0.035, 0.5, "c", transform=axm.transAxes, fontsize=13, fontweight="bold", va="center", ha="right")
    ev = ["t2", "t3", "t4", "t5"]
    gt = [t_of[e] for e in ev]
    tol = [1.0, 1.6, 1.0, 1.4]                      # illustrative, per-event tolerances are in Table 3
    pred = [g + d for g, d in zip(gt, [0.4, -2.2, -0.5, 0.7])]   # monotone, as a decoder's output must be
    assert all(a < b for a, b in zip(pred, pred[1:])), pred
    for i, e in enumerate(ev):
        axm.barh(i, 2 * tol[i], left=gt[i] - tol[i], height=0.5, color="0.88", zorder=1)
        axm.plot(gt[i], i, "|", color=C_GT, ms=14, mew=2, zorder=3)
        ok = abs(pred[i] - gt[i]) <= tol[i]
        axm.plot(pred[i], i, "o", color=C_OK if ok else C_PRED, ms=6, zorder=3)
        axm.text(gt[i] - tol[i] - 0.5, i, e, ha="right", va="center", fontsize=7)
        right = max(pred[i], gt[i] + tol[i])
        axm.text(right + 0.5, i, "counted" if ok else "missed", va="center", ha="left", fontsize=6.5,
                 color=C_OK if ok else C_PRED)
    axm.set_yticks([])
    axm.set_ylim(-0.8, len(ev) - 0.2)
    axm.set_xlim(min(g - t for g, t in zip(gt, tol)) - 3.0, max(max(pred), max(g + t for g, t in zip(gt, tol))) + 3.2)
    axm.set_xlabel("hours post insemination", fontsize=7)
    axm.tick_params(labelsize=7)
    axm.spines["left"].set_visible(False)
    axm.set_title("temporal accuracy $p_t$: the share of annotated onsets whose predicted onset falls inside that "
                  "event's tolerance window (grey); the window differs by event (Table 3)", fontsize=7.5, pad=4)
    axm.plot([], [], "|", color=C_GT, ms=10, mew=2, label="annotated onset")
    axm.plot([], [], "o", color=C_OK, ms=5, label="predicted, inside the window")
    axm.plot([], [], "o", color=C_PRED, ms=5, label="predicted, outside")
    axm.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.46))

    fig.savefig(FIG / "fig_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("fig_overview.png", json.dumps({"video": VIDEO, "planes": len(PLANES), "onsets": len(order)}))


if __name__ == "__main__":
    main()
