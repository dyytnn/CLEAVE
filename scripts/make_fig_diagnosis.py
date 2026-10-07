#!/usr/bin/env python
"""Fig. 2 (main) and Extended Data Figs. 1-3: the diagnosis panels for one checkpoint each, drawn from stored diagnostics
(results/diagnostics/<run>/summary.json) and per-video onset times (runs/h7/<run>/per_video_test.json). No model forward passes,
no run identifiers in the figure.

  PYTHONPATH=src python scripts/make_fig_diagnosis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stseg.data.nantes_kinetic import CLASS_NAMES  # noqa: E402
from stseg.eval.kinetic_metrics import THETA_H  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
PANELS = [("fig_diagnosis.png", "resnet50_lstm_L8_split0_seed1", "ResNet-50-LSTM"),
          ("ed_fig1_diagnosis_ref.png", "resnet18_lstm_L4_split0_seed1", "ResNet-18-LSTM (released reference)"),
          ("ed_fig2_diagnosis_tr1.png", "resnet18_transformer_L16_evalfix_split0_seed0", "ResNet-18 transformer, 1 plane"),
          ("ed_fig3_diagnosis_tr3.png", "resnet18_transformer_L16_multifocal3_evalfix_split0_seed0", "ResNet-18 transformer, 3 planes stacked")]
EVENTS = [c for c in CLASS_NAMES if c in THETA_H]


def panel_label(ax, s):
    ax.text(-0.1, 1.04, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def draw(out: str, run: str, title: str) -> None:
    S = json.load(open(ROOT / "results/diagnostics" / run / "summary.json"))
    pv = json.load(open(ROOT / "runs/h7" / run / "per_video_test.json"))
    fig = plt.figure(figsize=(11, 7.2)); gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 0.9, 1.15])
    # a: confusion (argmax | viterbi), row-normalised
    for j, (key, name) in enumerate((("argmax", "raw argmax"), ("viterbi", "Viterbi decoding"))):
        ax = fig.add_subplot(gs[0, j]); cm = np.array(S["confusion_test"][key], float); row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(row, cmap="Blues", vmin=0, vmax=1); ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=6); ax.set_yticklabels(CLASS_NAMES, fontsize=6); ax.set_xlabel("predicted"); ax.set_ylabel("annotated" if j == 0 else "")
        ax.set_title(name, fontsize=8); ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
        if j == 0: panel_label(ax, "a")
    # b: error vs distance
    ax = fig.add_subplot(gs[0, 2]); panel_label(ax, "b"); ev = S["error_vs_distance"]; keys = list(ev); x = np.arange(len(keys))
    ax.bar(x - 0.2, [ev[k]["err_argmax"] for k in keys], 0.4, color="#bbbbbb", label="argmax"); ax.bar(x + 0.2, [ev[k]["err_viterbi"] for k in keys], 0.4, color="#2c6fbb", label="Viterbi")
    ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_xlabel("frames to nearest annotated transition"); ax.set_ylabel("frame error rate"); ax.legend(frameon=False, fontsize=7)
    # c: signed onset error per event
    ax = fig.add_subplot(gs[1, :2]); panel_label(ax, "c")
    errs = {e: [] for e in EVENTS}
    for v in pv:
        for e, t in v["gt_times"].items():
            if e in errs and e in v["pred_times"]:
                errs[e].append(v["pred_times"][e] - t)
    data = [np.clip(errs[e], -12, 12) for e in EVENTS]
    ax.boxplot(data, whis=(5, 95), showfliers=False, tick_labels=EVENTS, medianprops={"color": "#c0392b"})
    for i, (e, d) in enumerate(zip(EVENTS, data), 1):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.15, 0.15, len(d)), d, s=3, alpha=0.3, color="#555")
        th = THETA_H[e]; ax.plot([i - 0.35, i + 0.35], [th, th], "--", color="#c0392b", lw=0.7); ax.plot([i - 0.35, i + 0.35], [-th, -th], "--", color="#c0392b", lw=0.7)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("predicted − annotated onset (h), clipped ±12"); ax.set_xlabel("event (dashed: ±θ$_p$)")
    # d: adjacent-pair probes
    ax = fig.add_subplot(gs[1, 2]); panel_label(ax, "d")
    pb, ph = S.get("probe_bb", {}).get("adjacent_pairs", {}), S.get("probe_hd", {}).get("adjacent_pairs", {})
    pairs = [k for k in pb if k in ph]; x = np.arange(len(pairs))
    ax.bar(x - 0.25, [pb[k]["acc"] for k in pairs], 0.25, color="#bbbbbb", label="frozen backbone"); ax.bar(x, [ph[k]["acc"] for k in pairs], 0.25, color="#2c6fbb", label="temporal head")
    ax.scatter(x + 0.25, [pb[k]["chance"] for k in pairs], marker="_", color="k", s=60, label="majority class", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([k.replace("|", "\n") for k in pairs], fontsize=5); ax.set_ylim(0.4, 1.08); ax.set_ylabel("linear-probe accuracy (val → test)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, columnspacing=1.0, handletextpad=0.4)
    fig.suptitle(title, fontsize=9, x=0.01, ha="left"); fig.tight_layout(); fig.savefig(FIG / out, dpi=200); plt.close(fig); print(out)


if __name__ == "__main__":
    for out, run, title in PANELS:
        draw(out, run, title)
