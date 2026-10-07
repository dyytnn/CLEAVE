#!/usr/bin/env python
"""Extra figures/analyses brainstormed for the CLEAVE paper (2026-09-16, see track1_TEMPO/EXTRA_FIGURES_TRACKING.md).
Everything here is CPU-only and reads already-cached artifacts (results/diagnostics/*/summary.json, frame_probs npz,
runs_master.csv, the public Nantes manifest) -- no new GPU inference or training. Writes to figures/.

  PYTHONPATH=src:scripts python scripts/make_fig_extra.py [--only N,N,...]   # 1-9, default all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DIAG = ROOT / "results/diagnostics"
sys.path.insert(0, str(ROOT / "scripts"))
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False})

from stseg.data.nantes_kinetic import CLASS_NAMES, PHASE_TO_CLASS  # noqa: E402
from stseg.eval.kinetic_metrics import THETA_H, viterbi  # noqa: E402

PHASE_COLORS = dict(zip(CLASS_NAMES, plt.cm.tab20(np.linspace(0, 1, 16))))


def panel_label(ax, s):
    ax.text(-0.08, 1.05, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def load_seqs(run: str, cache="frame_probs") -> dict[str, dict]:
    """video -> {log_probs, labels, times_h} from a cached npz (results/frame_probs/<run>/test.npz or
    results/diagnostics/<run>/frame_probs_test.npz)."""
    p1, p2 = ROOT / f"results/{cache}/{run}/test.npz", DIAG / run / "frame_probs_test.npz"
    z = np.load(p1 if p1.exists() else p2)
    vids = sorted({k.rsplit("__", 1)[0] for k in z.files})
    return {v: {"log_probs": z[f"{v}__lp"].astype(np.float64), "labels": z[f"{v}__y"].astype(int), "times_h": z[f"{v}__t"].astype(float)} for v in vids}


# =========================================================================================== 1. Timeline / Gantt
def fig_timeline():
    ref_run, cf_run = "resnet18_lstm_L4_split0_seed1", "resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0"
    ref = load_seqs(ref_run); cf = load_seqs(cf_run, cache="diagnostics")
    lt_ref = np.load(ROOT / f"runs/h7/{ref_run}/transition_log_matrix.npy")
    lt_cf = np.load(ROOT / f"runs/h7/{cf_run}/transition_log_matrix.npy")
    videos = [("PJ533-8", "well-decoded example (acc$_v$=0.96)"), ("DC307-1", "hard but plausible (acc$_v$=0.34)"),
              ("DRL1048-1", "label-quality outlier (acc$_v$=0.05)"), ("LV683-2-8", "label-quality outlier (acc$_v$=0.04)")]
    fig, axes = plt.subplots(len(videos), 1, figsize=(9, 1.35 * len(videos) + 0.6))
    for ax, (vid, note) in zip(axes, videos):
        rows = [("annotated", ref[vid]["labels"], ref[vid]["times_h"]),
                ("reference LSTM", viterbi(ref[vid]["log_probs"], lt_ref), ref[vid]["times_h"]),
                ("cross-focal transformer", viterbi(cf[vid]["log_probs"], lt_cf), cf[vid]["times_h"])]
        for row_i, (name, labels, t) in enumerate(rows):
            change = np.where(np.diff(labels) != 0)[0] + 1
            starts = np.concatenate([[0], change]); ends = np.concatenate([change, [len(labels)]])
            for s, e in zip(starts, ends):
                ax.barh(row_i, t[min(e, len(t) - 1)] - t[s], left=t[s], height=0.85, color=PHASE_COLORS[CLASS_NAMES[labels[s]]], edgecolor="white", linewidth=0.3)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["annotated", "reference\nLSTM", "cross-focal\ntransformer"], fontsize=6.5)
        ax.set_xlim(0, max(ref[vid]["times_h"].max(), cf[vid]["times_h"].max()))
        ax.set_title(vid, fontsize=7.5, loc="left", fontweight="bold"); ax.set_ylim(-0.6, 2.6)
    axes[-1].set_xlabel("hours post insemination")
    handles = [Patch(facecolor=PHASE_COLORS[c], label=c) for c in CLASS_NAMES if c != "tHB"]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(FIG / "fig_timeline.png", dpi=200); plt.close(fig); print("fig_timeline.png")


# =========================================================================================== 2. Outlier videos, 7 planes
def fig_outlier_planes():
    m = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    planes = ["embryo_dataset_F-45", "embryo_dataset_F-30", "embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15", "embryo_dataset_F30", "embryo_dataset_F45"]
    plane_labels = ["−45", "−30", "−15", "0", "+15", "+30", "+45"]
    videos = ["LV683-2-8", "DRL1048-1"]
    fig, axes = plt.subplots(2 * len(videos), 7, figsize=(11, 6.4))
    for vi, vid in enumerate(videos):
        g = m[(m.video == vid) & (m.plane == "embryo_dataset") & m.phase.notna() & (m.is_blank != True)].sort_values("frame_index")  # noqa: E712
        # two representative labelled timepoints: mid-cleavage and the latest annotated frame (where, if cleavage were
        # visible at all, the embryo should show the most cells of the whole video)
        for ti, frac in enumerate((0.5, 1.0)):
            row = g.iloc[min(int(len(g) * frac), len(g) - 1)]
            for pi, (pl, lab) in enumerate(zip(planes, plane_labels)):
                ax = axes[2 * vi + ti, pi]
                path = row.path.replace("/embryo_dataset/", f"/{pl}/", 1)
                try:
                    img = Image.open(path).convert("L")
                except Exception:
                    img = Image.new("L", (250, 250), 0)
                ax.imshow(img, cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
                if 2 * vi + ti == 0:
                    ax.set_title(f"{lab} µm", fontsize=7)
                if pi == 0:
                    ax.set_ylabel(f"{vid}\n{row.time_h:.0f} h\nannotated: {row.phase}", fontsize=6.5)
    fig.suptitle("Two videos every model decodes as a single phase, inspected across all seven focal planes", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(FIG / "ed_fig_outlier_planes.png", dpi=180); plt.close(fig); print("ed_fig_outlier_planes.png")


# =========================================================================================== 2b. CC938-4 mislabel
def fig_cc938_mislabel():
    """Third label-quality outlier: unlike DRL1048-1/LV683-2-8 (never
    visibly divide), CC938-4 divides normally but its annotated event times run 1-2 division stages ahead of what is
    visible -- shown here at its own annotated t2, t4 and t9+ frames."""
    m = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    g = m[(m.video == "CC938-4") & (m.plane == "embryo_dataset") & m.phase.notna() & (m.is_blank != True)].sort_values("time_h")  # noqa: E712
    targets = [("t2", 22.1), ("t4", 35.1), ("t9+", 48.4)]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, (phase, target_h) in zip(axes, targets):
        row = g.iloc[(g.time_h - target_h).abs().argmin()]
        img = Image.open(row.path).convert("L")
        ax.imshow(img, cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"annotated {phase}, {row.time_h:.1f} h", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "ed_fig_cc938_mislabel.png", dpi=180); plt.close(fig); print("ed_fig_cc938_mislabel.png")


# =========================================================================================== 3. Waterfall attribution
def fig_waterfall():
    """Main-text attribution figure: a, chain waterfall labelled with the pooled estimates; b, every plane
    contrast with the batch of each arm; c, fold-level p_t of the configurations. Source: results/attribution_v39.json."""
    n = json.loads((ROOT / "results/numbers.json").read_text())
    att = json.loads((ROOT / "results/attribution_v39.json").read_text())
    m = pd.read_csv(ROOT / "results/runs_master.csv")
    f = lambda k: float(n[k])  # noqa: E731
    fig = plt.figure(figsize=(13.0, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.35, 1.0], hspace=0.55, wspace=0.28)
    c_base, c_pos, c_null = "#7f7f7f", "#2c6fbb", "#b0b0b0"
    minus = lambda x: f"{x:+.3f}".replace("-", "\u2212")  # noqa: E731
    # a: waterfall
    ax = fig.add_subplot(gs[0, :]); panel_label(ax, "a")
    chain = ["ref", "lstmwin", "lstmwinadamw", "tr1", "mfSeven", "cf"]
    pts = [f(f"{t}_fold_pt") for t in chain]
    steps = ["stepclip", "stepopt", "stephead", "stepseven", "stepfuseseven"]
    labels = ["released\nreference\n(LSTM, 4 fr., SGD)", "16-fr. clips,\nbatch 4,\nwindow fix", "SGD \u2192 AdamW\nrecipe (LSTM)",
              "LSTM \u2192\ntransformer\nhead", "+6 planes stacked\n(batch 4 \u2192 2)", "stacking \u2192\ncross-focal\n(7 planes)", "cross-focal\ntransformer"]
    ax.bar(0, pts[0], color=c_base, width=0.6); ax.text(0, pts[0] + 0.003, f"{pts[0]:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    cum = pts[0]
    for i, tag in enumerate(steps, start=1):
        val = pts[i] - pts[i - 1]; a = att[tag]
        sig = a["lo"] > 0 or a["hi"] < 0
        col = c_pos if sig else c_null
        ax.bar(i, max(abs(val), 0.0008), bottom=cum if val >= 0 else cum + val, color=col, width=0.6)
        ax.errorbar(i, cum + a["d"], yerr=[[a["d"] - a["lo"]], [a["hi"] - a["d"]]], fmt="none", ecolor="#222", elinewidth=1, capsize=3)
        ax.plot([i - 1 + 0.3, i - 0.3], [cum, cum], color="#999", lw=0.8, ls="--")
        ax.text(i, max(cum, cum + val, cum + a["hi"]) + 0.003, minus(a["d"]), ha="center", va="bottom", fontsize=9,
                color=col if sig else "#666", fontweight="bold")
        cum += val
    k = len(steps) + 1
    ax.plot([k - 1 + 0.3, k - 0.3], [cum, cum], color="#999", lw=0.8, ls="--")
    ax.bar(k, pts[-1], color=c_base, width=0.6); ax.text(k, pts[-1] + 0.003, f"{pts[-1]:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("mean $p_t$, 5 official folds"); ax.set_ylim(0.66, 0.775)
    ax.legend(handles=[Patch(color=c_pos, label="interval excludes 0"), Patch(color=c_null, label="interval contains 0")],
              loc="upper left", fontsize=8.5, frameon=False)
    # b: plane contrasts
    ax = fig.add_subplot(gs[1, 0]); panel_label(ax, "b")
    rows = [("stepplanes", "3 stacked \u2212 1 plane", "4 vs 4"), ("stepseven", "7 stacked \u2212 1 plane", "2 vs 4"),
            ("stepcfone", "7 cross-focal \u2212 1 plane", "2 vs 4"), ("stepfusion", "7 cross-focal \u2212 3 stacked", "2 vs 4"),
            ("stepfuseseven", "7 cross-focal \u2212 7 stacked", "2 vs 2"), ("stepsevenvsthree", "7 stacked \u2212 3 stacked", "2 vs 4")]
    for y, (tag, lab, bat) in enumerate(rows[::-1]):
        a = att[tag]; sig = a["lo"] > 0 or a["hi"] < 0
        col = c_pos if sig else c_null
        ax.errorbar(a["d"], y, xerr=[[a["d"] - a["lo"]], [a["hi"] - a["d"]]], fmt="o", color=col, ecolor=col, capsize=3, ms=5)
        ax.text(0.047, y, f"{minus(a['d'])}  [{bat}]", va="center", fontsize=8.5)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows[::-1]], fontsize=8.5)
    ax.set_xlim(-0.035, 0.075); ax.set_xlabel("pooled paired $\\Delta p_t$ (95 % CI)  [batch]")
    # c: fold-level p_t
    ax = fig.add_subplot(gs[1, 1]); panel_label(ax, "c")
    arms = [("resnet18_lstm_L4_split{k}", "ref.", c_base), ("resnet18_lstm_L16_win_adamw_split{k}", "LSTM\nAdamW", "#8c6bb1"),
            ("resnet18_transformer_L16_evalfix_split{k}", "1\nplane", "#9ecae1"), ("resnet18_transformer_L16_multifocal3_evalfix_split{k}", "3\nstacked", "#4292c6"),
            ("resnet18_transformer_L16_multifocal7_evalfix_split{k}", "7\nstacked", "#08519c"), ("resnet18_transformer_L16_crossfocal7_evalfix_split{k}", "7 cross-\nfocal", "#c0392b")]
    rng = np.random.default_rng(0)
    fm = np.array([[m[m.experiment == e.format(k=kk)].p_t.mean() for e, _, _ in arms] for kk in range(5)])
    for kk in range(5):
        ax.plot(range(len(arms)), fm[kk], "-", color="#ccc", lw=0.8, zorder=1)
    for x, (e, _, col) in enumerate(arms):
        for kk in range(5):
            r = m[m.experiment == e.format(k=kk)].p_t
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(r)), r, color=col, s=9, alpha=0.8, zorder=2)
        ax.plot([x - 0.2, x + 0.2], [fm[:, x].mean()] * 2, color="k", lw=1.6, zorder=3)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([a_[1] for a_ in arms], fontsize=8.5)
    ax.set_ylabel("test $p_t$ (5 folds \u00d7 3 seeds)")
    fig.savefig(FIG / "fig_waterfall.png", dpi=200, bbox_inches="tight"); plt.close(fig); print("fig_waterfall.png")


# =========================================================================================== 4. Reliability diagram
def fig_calibration():
    runs = [("released reference (LSTM)", "resnet18_lstm_L4_split0_seed1", "#7f7f7f"),
            ("single-plane transformer", "resnet18_transformer_L16_evalfix_split0_seed0", "#2c6fbb"),
            ("cross-focal transformer", "resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0", "#c0392b")]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for label, run, col in runs:
        S = json.load(open(DIAG / run / "summary.json")); curve = S["calibration"]["curve"]
        curve = [c for c in curve if c[3] >= 10]  # drop near-empty bins (noisy tails, e.g. n=1)
        conf = [c[1] for c in curve]; acc = [c[2] for c in curve]; n = np.array([c[3] for c in curve], float)
        ax.plot(conf, acc, "-", color=col, lw=1.2, label=f"{label} (ECE={S['calibration']['ece']:.3f})")
        ax.scatter(conf, acc, s=8 + 40 * (n / n.max()), color=col, zorder=3)
    ax.set_xlabel("mean predicted confidence (bin)"); ax.set_ylabel("empirical accuracy (bin)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig_calibration.png", dpi=200); plt.close(fig); print("fig_calibration.png")


# =========================================================================================== 5. Dataset overview
def fig_dataset():
    m = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    m = m[m.plane == "embryo_dataset"]
    dur = m.groupby("video").time_h.max()
    labeled = m[m.phase.notna() & (m.is_blank != True)].copy()  # noqa: E712
    labeled["cls"] = labeled.phase.map(PHASE_TO_CLASS)
    onset = labeled.sort_values(["video", "frame_index"]).groupby(["video", "phase"]).time_h.min().reset_index()
    counts = labeled.phase.value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]; panel_label(ax, "a")
    ax.hist(dur, bins=30, color="#bbbbbb", edgecolor="white"); ax.set_xlabel("video duration (h)"); ax.set_ylabel("videos (n=652)")
    ax.axvline(dur.median(), color="#c0392b", lw=1.2); ax.text(dur.median() + 5, ax.get_ylim()[1] * 0.9, f"median {dur.median():.0f} h", color="#c0392b", fontsize=7)

    ax = axes[1]; panel_label(ax, "b")
    order = [c for c in CLASS_NAMES if c not in ("tPB2", "tHB") and c in onset.phase.values]
    data = [onset[onset.phase == c].time_h.to_numpy() for c in order]
    ax.boxplot(data, whis=(5, 95), showfliers=False, tick_labels=order, medianprops={"color": "#c0392b"})
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.15, 0.15, len(d)), d, s=1.5, alpha=0.15, color="#555")
    ax.set_xticklabels(order, rotation=90, fontsize=6.5); ax.set_ylabel("onset time (h post insemination)")

    ax = axes[2]; panel_label(ax, "c")
    order2 = [c for c in CLASS_NAMES if c in counts.index]
    vals = [counts.get(c, 0) for c in order2]
    thetas = [THETA_H.get(c, np.nan) for c in order2]
    ax2 = ax.twinx()
    ax.bar(range(len(order2)), vals, color="#bbbbbb"); ax.set_xticks(range(len(order2))); ax.set_xticklabels(order2, rotation=90, fontsize=6.5)
    ax.set_ylabel("annotated frames"); ax.set_yscale("log")
    ax2.plot(range(len(order2)), thetas, "o-", color="#c0392b", ms=3, lw=1); ax2.set_ylabel("tolerance $\\theta_p$ (h)", color="#c0392b")
    ax2.tick_params(axis="y", colors="#c0392b")
    fig.tight_layout(); fig.savefig(FIG / "fig_dataset.png", dpi=200); plt.close(fig); print("fig_dataset.png")


# =========================================================================================== 6. Forest plot
def fig_forest():
    import build_results_master as brm  # noqa
    df = pd.read_csv(ROOT / "results/runs_master.csv")
    ref = "resnet18_lstm_L4_split0"
    rows = [  # (label, experiment) -- a curated, readable subset of fold-0 configurations
        ("R(2+1)D-18 (3D baseline)", "r2plus1d_L8_split0"), ("ResNet-50 backbone", "resnet50_lstm_L8_split0"),
        ("ConvNeXt-Tiny backbone", "convnext_tiny_lstm_L8_split0"), ("EfficientNet-B0 backbone", "efficientnet_b0_lstm_L8_split0"),
        ("Swin-T backbone", "swin_t_lstm_L8_split0"), ("No ImageNet pretraining", "resnet18_lstm_L4_nopretrain_split0"),
        ("GRU head", "resnet18_gru_L8_split0"), ("TCN head", "resnet18_tcn_L16_split0"),
        ("Transformer, released window", "resnet18_transformer_L16_split0"), ("Transformer, window-fixed (E3)", "resnet18_transformer_L16_evalfix_split0"),
        ("Transformer, rotary, window-fixed", "resnet18_transformer_L16_focalattn_spatial_evalfix_split0" if False else "resnet18_transformer_relpos_L16_evalfix_split0"),
        ("Selective state-space head", "resnet18_mamba_L16_split0"), ("ODE-RNN head", "resnet18_node_L16_split0"),
        ("Diffusion-refinement head", "resnet18_diffactlite_L16_split0"), ("ASFormer-lite head", "resnet18_asformer_L16_evalfix_split0"),
        ("Class-weighted CE", "resnet18_lstm_L4_ceweighted_split0"), ("Ordinal CE", "resnet18_lstm_L4_ordinal_split0"),
        ("Boundary-weighted CE", "resnet18_lstm_L4_boundaryloss_split0"), ("3 planes stacked (LSTM)", "resnet18_lstm_L4_multifocal3_split0"),
        ("3 planes stacked (transformer, E4)", "resnet18_transformer_L16_multifocal3_evalfix_split0"),
        ("7 planes stacked (transformer, E8)", "resnet18_transformer_L16_multifocal7_evalfix_split0"),
        ("7 planes, cross-focal (transformer)", "resnet18_transformer_L16_crossfocal7_evalfix_split0"),
        ("Photometric aug + instance norm (R50)", "resnet50_lstm_L8_photo_split0"), ("Ordinal cell-count aux (R50)", "resnet50_lstm_L8_cellcount_split0"),
        ("Native 448px resolution (R50)", "resnet50_lstm_L8_hires448_split0"), ("Masked-frame pretraining (init)", "resnet18_lstm_L4_maeinit_split0"),
    ]
    out = []
    for label, exp in rows:
        if exp not in set(df.experiment):
            continue
        d = brm.paired_videos(exp, ref, df)
        if d is None or len(d) < 5:
            continue
        m, lo, hi = brm.boot_ci(d)
        out.append((label, m, lo, hi))
    out.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(out) + 1.2))
    for i, (label, m, lo, hi) in enumerate(out):
        sig = lo > 0 or hi < 0
        col = "#2c6fbb" if (sig and m > 0) else ("#c0392b" if (sig and m < 0) else "#999999")
        ax.plot([lo, hi], [i, i], color=col, lw=1.5); ax.scatter([m], [i], color=col, s=18, zorder=3)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(out))); ax.set_yticklabels([r[0] for r in out], fontsize=7); ax.set_ylim(-1, len(out))
    ax.set_xlabel("paired $\\Delta p_t$ vs. released reference (fold 0, 95% bootstrap CI over videos)")
    fig.tight_layout(); fig.savefig(FIG / "fig_forest.png", dpi=200); plt.close(fig); print(f"fig_forest.png ({len(out)} rows)")


# =========================================================================================== 7. Landscape scatter
def fig_landscape():
    df = pd.read_csv(ROOT / "results/runs_master.csv")
    df = df[df.split_key == "fold0"]
    meta_params = {}
    for run in df.run:
        f = ROOT / "runs/h7" / run / "meta.json"
        if f.exists():
            meta_params[run] = json.loads(f.read_text()).get("n_params_M")
    df["params_M"] = df.run.map(meta_params)
    df = df.dropna(subset=["params_M", "p_t"])
    heads = sorted(df["head"].unique())
    cmap = dict(zip(heads, plt.cm.tab20(np.linspace(0, 1, max(len(heads), 2)))))
    # bubble area proportional to parameter count (redundant with the x-position, but makes the size difference between
    # e.g. a 5 M-parameter and a 90 M-parameter model immediately visible without reading the axis)
    size = 15 + 260 * (df.params_M / df.params_M.max())
    fig, ax = plt.subplots(figsize=(7, 5))
    for h in heads:
        sub = df[df["head"] == h]
        ax.scatter(sub.params_M, sub.p_t, s=size.loc[sub.index], color=cmap[h], label=h, alpha=0.75, edgecolor="white", linewidth=0.3)
    best = df.loc[df.p_t.idxmax()]
    ax.annotate("best single run", (best.params_M, best.p_t), textcoords="offset points", xytext=(8, 6), fontsize=6.5)
    ax.set_xlabel("model size (M parameters) -- bubble area is also proportional to parameter count"); ax.set_ylabel("test $p_t$ (fold 0)"); ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig_landscape.png", dpi=200); plt.close(fig); print(f"fig_landscape.png ({len(df)} points)")


# =========================================================================================== 8. Multi-phase attention
def fig_attention_multiphase(device="cpu"):
    import torch
    from stseg.kinetic.datasets import _frames
    from stseg.kinetic.models import build_model
    rd = ROOT / "runs/h7/resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0"
    cfg = json.loads((rd / "config.resolved.json").read_text()); ds = _frames(cfg["data"], "test", "eval", 0, None, None)
    model = build_model(dict(cfg["model"]), ds.in_channels); model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu", weights_only=True)["model"]); model.to(device).eval()
    targets = [("t3", 4), ("t5", 6), ("t7", 8)]
    plane_labels = ["−45", "−30", "−15", "0", "+15", "+30", "+45"]
    fig, axes = plt.subplots(len(targets), 1, figsize=(7, 2.0 * len(targets)))
    for ax, (name, cls) in zip(axes, targets):
        rows = ds.rows[ds.rows.label == cls]
        r = rows.iloc[len(rows) // 3]
        x = ds[int(r.name)]["image"][None].to(device)
        bb = model.backbone
        with torch.no_grad():
            N, P, H, W = x.shape
            f = bb.cnn(x.reshape(N * P, 1, H, W).expand(-1, 3, -1, -1)).view(N, P, -1)
            tok = bb.norm(f + bb.plane_emb[None]); _, w = bb.attn(bb.query.expand(N, -1, -1), tok, tok, need_weights=True)
        w = w[0, 0].cpu().numpy()
        ax.bar(range(7), w, color="#c0392b"); ax.set_xticks(range(7)); ax.set_xticklabels(plane_labels, fontsize=7)
        ax.set_ylabel("attention", fontsize=7); ax.set_title(name, fontsize=7.5, loc="left", fontweight="bold")
    axes[-1].set_xlabel("focal plane (µm)")
    fig.tight_layout(); fig.savefig(FIG / "fig_attention_multiphase.png", dpi=200); plt.close(fig); print("fig_attention_multiphase.png")


# =========================================================================================== 9. Embedding merge
def fig_embeddings_merge():
    """Side-by-side PCA embedding comparison. The source PNGs (diagnose_run.py's embedding_plot, a general per-run
    diagnostic) carry a plot title with the raw run id -- not paper-appropriate -- so it is cropped off here; the
    single-plane-vs-cross-focal labelling lives in the LaTeX caption instead."""
    pairs = ["resnet18_transformer_L16_evalfix_split0_seed0", "resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0"]
    TITLE_CROP_PX = 76  # matplotlib title strip height at this figure's fixed figsize/dpi (measured, not computed)
    imgs = []
    for run in pairs:
        p = DIAG / run / "embeddings_bb_test.png"
        if p.exists():
            im = Image.open(p); imgs.append(im.crop((0, TITLE_CROP_PX, im.width, im.height)))
    if len(imgs) < 2:
        print("embeddings_bb_test.png missing for one side; skip fig_embeddings_merge"); return
    h = max(im.height for im in imgs); ws = [int(im.width * h / im.height) for im in imgs]
    imgs = [im.resize((w, h)) for im, w in zip(imgs, ws)]
    canvas = Image.new("RGB", (sum(ws) + 20, h), "white")
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0)); x += im.width + 20
    canvas.save(FIG / "fig_embeddings.png"); print("fig_embeddings.png")


# =========================================================================================== 10. probe vs p_t
def fig_probe_vs_pt():
    runs = ["resnet18_lstm_L4_split0_seed1", "resnet18_lstm_L8_hires448_split0_seed0", "resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0",
            "resnet18_transformer_L16_evalfix_split0_seed0", "resnet18_transformer_L16_multifocal3_evalfix_split0_seed0", "resnet50_lstm_L8_split0_seed1"]
    pts, probes_bb, probes_hd, labels = [], [], [], []
    for run in runs:
        Sf = DIAG / run / "summary.json"
        if not Sf.exists():
            continue
        S = json.load(open(Sf)); res = json.loads((ROOT / "runs/h7" / run / "results.json").read_text())
        pts.append(res["test"]["p_t"]); probes_bb.append(S["probe_bb"]["test_acc"]); probes_hd.append(S["probe_hd"]["test_acc"])
        labels.append(run.split("_split0")[0].replace("resnet18_", "").replace("resnet50_", "R50-").replace("transformer_L16_", "tr-").replace("lstm_L", "LSTM-L").replace("lstm_L8_hires448", "LSTM-L8-448px").replace("_evalfix", "").replace("crossfocal7", "crossfocal").replace("multifocal3", "3-plane"))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(probes_bb, pts, s=40, color="#7f7f7f", label="frozen backbone probe")
    ax.scatter(probes_hd, pts, s=40, color="#2c6fbb", label="temporal-head probe")
    order = np.argsort(pts)
    # five of the six checkpoints cluster within 0.02 of test p_t (0.712-0.735): stagger both the horizontal anchor
    # (left of the frozen-backbone probe dot vs. right of the temporal-head probe dot) and a wide vertical offset,
    # cycled by rank, so labels clear both the data points and each other without connecting leader lines.
    offsets = [("right", 8, 6), ("left", -8, -4), ("right", 8, 26), ("left", -8, 20), ("right", 8, -16), ("left", -8, 42)]
    for rank, idx in enumerate(order):
        x1, x2, y, lab = probes_bb[idx], probes_hd[idx], pts[idx], labels[idx]
        ax.plot([x1, x2], [y, y], color="#ccc", lw=0.8, zorder=0)
        side, dx, dy = offsets[rank % len(offsets)]
        anchor_x, ha = (x2, "left") if side == "right" else (x1, "right")
        ax.annotate(lab, (anchor_x, y), fontsize=6, ha=ha, textcoords="offset points", xytext=(dx, dy))
    lims = [min(probes_bb + [min(pts)]) - 0.02, max(probes_hd + [max(pts)]) + 0.02]
    ax.plot(lims, lims, "k--", lw=0.6, label="probe = model $p_t$")
    ax.set_xlabel("linear-probe frame accuracy (val→test)"); ax.set_ylabel("model test $p_t$"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_probe_vs_pt.png", dpi=200); plt.close(fig); print(f"fig_probe_vs_pt.png ({len(pts)} checkpoints)")


FIGS = {1: fig_timeline, 2: fig_outlier_planes, 3: fig_waterfall, 4: fig_calibration, 5: fig_dataset,
        6: fig_forest, 7: fig_landscape, 8: fig_attention_multiphase, 9: fig_embeddings_merge, 10: fig_probe_vs_pt}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="")
    a = ap.parse_args()
    which = [int(x) for x in a.only.split(",")] if a.only else list(FIGS)
    for k in which:
        try:
            FIGS[k]()
        except Exception as e:
            print(f"[{k}] FAILED: {type(e).__name__}: {e}")
