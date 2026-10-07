#!/usr/bin/env python3
"""Graphical abstract for the Medical Image Analysis submission. Data-driven, three panels:
the protocol leak, a timeline mismatch, and cell-count separability. Writes
figures/graphical_abstract.png (1700 x 680 px, above Elsevier's 1328 x 531 minimum).

    PYTHONPATH=src:scripts python scripts/make_graphical_abstract.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from audit_frame_timing import annotation  # noqa: E402
from stseg.data.nantes_kinetic import CLASS_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
C_REF, C_CF = "#7f7f7f", "#c0392b"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def main() -> None:
    nums = json.loads((ROOT / "results/numbers.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6), gridspec_kw={"wspace": 0.55})

    ax = axes[0]
    exposed = float(nums["leakcouple_pooled_pct"])          # same number as the abstract's macro
    ax.bar([0, 1], [exposed, 0.0], color=[C_REF, C_CF], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["released\nfolds", "patient-\ngrouped"])
    ax.set_ylabel("test embryos with a\nsibling in training (%)")
    ax.set_ylim(0, 40)
    ax.text(1, 1.5, "0", ha="center", color=C_CF, fontweight="bold")
    ax.set_title("1  Protocol", loc="left", fontweight="bold")

    ax = axes[1]
    v = "MM445-2-2"
    for f in sorted((ROOT / "results/frame_timing").glob("preds_oof_k*.npz")):
        z = np.load(f)
        if v in set(z["video"]):
            break
    sel = z["video"] == v
    k = z["frame_index"][sel]
    o = np.argsort(k)
    k, pred = k[o], z["log_probs"][sel][o].astype(np.float32).argmax(1)
    lab = annotation(v)
    al = pd.read_csv(ROOT / "results/frame_timing/alignment_oof.csv").set_index("video").loc[v]
    ok = (k < len(lab)) & (lab[np.clip(k, 0, len(lab) - 1)] >= 0)
    jf = np.rint(al.slope_fit * k + al.offset_fit).astype(int)
    okf = (jf >= 0) & (jf < len(lab))
    okf[okf] &= lab[jf[okf]] >= 0
    ax.plot(k, pred, ".", ms=2.5, color="black", label="what the images show")
    ax.step(k[ok], lab[k[ok]], where="post", color=C_REF, lw=2, label="released labels")
    ax.step(k[okf], lab[jf[okf]], where="post", color=C_CF, lw=2, label="re-timed labels")
    ax.set_yticks([CLASS_NAMES.index(p) for p in ("t2", "t4", "t8", "tB")])
    ax.set_yticklabels(["2-cell", "4-cell", "8-cell", "blast."])
    ax.set_xlabel("image")
    ax.set_ylim(-4, len(CLASS_NAMES))
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.02, -0.02))
    ax.set_title("2  Labels", loc="left", fontweight="bold")

    ax = axes[2]
    lat = json.loads((ROOT / "results/figdata/latent.json").read_text())["cross-focal"]["pairs_ci"]
    pairs = [("t4|t8", "4"), ("t5|t8", "5"), ("t6|t8", "6"), ("t7|t8", "7")]
    m = [lat[f"head|{p}"][0] for p, _ in pairs]
    lo = [lat[f"head|{p}"][1] for p, _ in pairs]
    hi = [lat[f"head|{p}"][2] for p, _ in pairs]
    x = np.arange(len(pairs))
    ax.errorbar(x, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="o-", color=C_CF, capsize=3)
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in pairs])
    ax.set_xlabel("cells (vs 8)")
    ax.set_ylabel("separability\n(balanced accuracy)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("3  Ceiling", loc="left", fontweight="bold")
    out = ROOT / "figures/graphical_abstract.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
