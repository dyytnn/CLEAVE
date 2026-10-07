#!/usr/bin/env python3
"""Four figures about the task itself rather than the protocol. All CPU, from stored results:

* fig_task.png     -- what the task is: one embryo through the phases, the phase chain, and how far time alone goes;
* fig_latent.png   -- what the representation encodes: trajectories, adjacent-phase separability, division events;
* fig_dynamics.png -- how the task is learnt: per-phase loss, boundary vs interior, convergence, and phase drift;
* fig_where.png    -- where the model looks: focal-plane attention by phase, attention vs focus, Grad-CAM.

Inputs: results/figdata/{latent.json, latent_pca.npz, dynamics.json, plane_attention_test.npz, gradcam_*.npz}
(scripts/figdata_*.py), the frame manifest and fold-0 diagnostics.

    PYTHONPATH=src:scripts python scripts/make_fig_problem.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from PIL import Image  # noqa: E402

from stseg.data.nantes_kinetic import CLASS_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FD = ROOT / "results/figdata"
CACHE = Path.home() / ".cache/stseg/nantes_250"
# every .npz read here (allow_pickle for string arrays) is written by this repo's own figdata/diagnostics scripts
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
C_REF, C_CF, C_TR, C_TIME = "#7f7f7f", "#c0392b", "#2c6fbb", "#e0a526"
MODEL_COL = {"reference": C_REF, "single-plane transformer": C_TR, "cross-focal": C_CF}
PH = CLASS_NAMES[:-1]                                   # tHB occurs in 0.8 % of videos; left out of every panel
TRANSIENT: set[str] = set()                            # phases with median duration < 2 h, set in fig_task()
CMAP = plt.get_cmap("viridis")
PCOL = {p: CMAP(i / (len(PH) - 1)) for i, p in enumerate(PH)}
EXAMPLE = "BL526-4"                                     # clean fold-0 training embryo, every phase present
EXAMPLE_BOX = (38, 60, 213, 235)                        # embryo-centred crop of the 250-px frame


def panel_label(ax, s, x=-0.12):
    ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def manifest() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data/derived/nantes_manifest_F0.csv")
    return m[(m.plane == "embryo_dataset") & m.phase.notna()]


def fold0() -> dict[str, str]:
    return {v: p for v, p in csv.reader(open(ROOT / "data/splits/nantes_official/split0.csv"))}


def per_phase_recall(run: str) -> np.ndarray:
    """Per-phase recall of the Viterbi-decoded fold-0 test predictions of one run."""
    import diagnose_run as D
    z = np.load(ROOT / "runs/h7" / run / "diagnostics/frame_probs_test.npz", allow_pickle=True)
    lt = np.load(ROOT / "runs/h7" / run / "transition_log_matrix.npy")
    hit, tot = np.zeros(len(CLASS_NAMES)), np.zeros(len(CLASS_NAMES))
    for v in {k.split("__")[0] for k in z.files}:
        y, p = z[f"{v}__y"], D.viterbi(z[f"{v}__lp"], lt)
        np.add.at(tot, y, 1)
        np.add.at(hit, y[p == y], 1)
    return np.where(tot > 0, hit / np.maximum(tot, 1), np.nan)


# ============================================================================================ A. the task
def fig_task() -> dict:
    m = manifest()
    sp = fold0()
    dur = m.groupby(["video", "phase"]).time_h.agg(lambda t: t.max() - t.min()).groupby("phase").median()
    TRANSIENT.update(p for p in PH if dur[p] < 2.0)
    fig = plt.figure(figsize=(10.5, 9.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.78, 1.75, 1.35], hspace=0.62, wspace=0.42)

    # a: one embryo through the phases
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    panel_label(ax, "a", -0.02)
    x = m[m.video == EXAMPLE]
    tiles = []
    for p in PH:
        r = x[x.phase == p]
        r = r.iloc[len(r) // 2]
        im = Image.open(CACHE / "embryo_dataset" / EXAMPLE / (Path(r.path).stem + ".jpg")).convert("L").crop(EXAMPLE_BOX)
        tiles.append((p, float(r.time_h), np.asarray(im)))
    n = len(tiles)
    for i, (p, t, im) in enumerate(tiles):
        sub = ax.inset_axes([i / n + 0.002, 0.16, 1 / n - 0.004, 0.84])
        sub.imshow(im, cmap="gray", vmin=0, vmax=255)
        sub.set_xticks([])
        sub.set_yticks([])
        for s in sub.spines.values():
            s.set_visible(True)
            s.set_edgecolor(C_CF if p in TRANSIENT else "#333333")
            s.set_linewidth(2.2 if p in TRANSIENT else 0.6)
        sub.set_title(p, fontsize=9.5, pad=5, color=C_CF if p in TRANSIENT else "black")
        sub.text(0.5, -0.1, f"{t:.0f} h", transform=sub.transAxes, ha="center", va="top", fontsize=8.5)
    ax.text(0, -0.05, "red frame = transient phase (median duration < 2 h); the frame shown is the middle of each "
            f"annotated phase of one embryo ({EXAMPLE}), reference focal plane", transform=ax.transAxes, fontsize=8.5,
            color="#444444", va="top")

    # b: the phase chain -- when each phase starts, how long it lasts, how often it is seen at all
    ax = fig.add_subplot(gs[1, :2])
    panel_label(ax, "b", -0.07)
    g = m.groupby(["video", "phase"]).time_h.agg(["min", "max"]).reset_index()
    nv = m.video.nunique()
    stats = {}
    for i, p in enumerate(PH):
        s = g[g.phase == p]
        on, dur = s["min"].to_numpy(), (s["max"] - s["min"]).to_numpy()
        stats[p] = {"present": len(s) / nv, "onset_med": float(np.median(on)), "onset_q": np.percentile(on, [25, 75]).tolist(),
                    "dur_med": float(np.median(dur))}
        y = len(PH) - 1 - i
        ax.barh(y, max(stats[p]["dur_med"], 0.4), left=stats[p]["onset_med"], height=0.62,
                color=PCOL[p], edgecolor="none")
        ax.plot(stats[p]["onset_q"], [y, y], color="black", lw=1.1)
        ax.text(163, y, f"{100 * stats[p]['present']:.0f} %", va="center", ha="right", fontsize=8.5,
                color="#333333")
    ax.set_yticks(range(len(PH)))
    ax.set_yticklabels(PH[::-1], fontsize=8.5)
    ax.set_xlim(0, 165)
    ax.set_xlabel("hours post insemination")
    ax.text(163, len(PH) - 0.2, "embryos\nwith phase", ha="right", va="bottom", fontsize=8.5)
    ax.set_title("median onset and duration (bar), inter-embryo onset IQR (line)", fontsize=9.5, loc="left")

    # c: the chain is almost strictly forward, with skips
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "c", -0.2)
    tr = m[m.video.map(sp) == "train"].sort_values(["video", "frame_index"])
    K = len(PH)
    T = np.zeros((K, K))
    for _, s in tr.groupby("video"):
        idx = [PH.index(p) for p in s.phase if p in PH]
        for a, b in zip(idx, idx[1:]):
            if a != b:
                T[a, b] += 1
    P = T / np.maximum(T.sum(1, keepdims=True), 1)
    im = ax.imshow(np.where(P > 0, P, np.nan), cmap="magma_r", norm=LogNorm(1e-3, 1))
    ax.set_xticks(range(K))
    ax.set_xticklabels(PH, rotation=90, fontsize=7.5)
    ax.set_yticks(range(K))
    ax.set_yticklabels(PH, fontsize=7.5)
    ax.set_xlabel("next phase")
    ax.set_ylabel("phase left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("P(next | leaving)", fontsize=9)
    skip3 = T[PH.index("t2"), PH.index("t4")] / max(T[PH.index("t2")].sum(), 1)
    ax.set_title(f"t2 goes straight to t4 in {100 * skip3:.0f} %", fontsize=9.5, loc="left")

    # d: how far time alone goes, per phase
    ax = fig.add_subplot(gs[2, :])
    panel_label(ax, "d", -0.05)
    trn, te = m[m.video.map(sp) == "train"], m[m.video.map(sp) == "test"]
    bins = np.arange(0, 200, 0.5)
    H = np.zeros((len(bins), len(CLASS_NAMES)))
    np.add.at(H, (np.clip(np.digitize(trn.time_h, bins) - 1, 0, len(bins) - 1), trn.phase.map(CLASS_NAMES.index).values), 1)
    pred = H.argmax(1)[np.clip(np.digitize(te.time_h, bins) - 1, 0, len(bins) - 1)]
    yte = te.phase.map(CLASS_NAMES.index).values
    rec_time = np.array([(pred[yte == c] == c).mean() if (yte == c).any() else np.nan for c in range(len(CLASS_NAMES))])
    acc_time = float((pred == yte).mean())
    rec_ref = per_phase_recall("resnet18_lstm_L4_split0_seed0")
    rec_cf = per_phase_recall("resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0")
    xs = np.arange(len(PH))
    for off, rec, col, lab in ((-0.26, rec_time, C_TIME, f"time only (frame accuracy {acc_time:.2f})"),
                               (0.0, rec_ref, C_REF, "reference LSTM"), (0.26, rec_cf, C_CF, "cross-focal transformer")):
        ax.bar(xs + off, rec[:len(PH)], width=0.25, color=col, label=lab)
    for i, p in enumerate(PH):
        if p in TRANSIENT:
            ax.axvspan(i - 0.45, i + 0.45, color=C_CF, alpha=0.07, lw=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(PH)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall, fold-0 test")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(FIG / "fig_task.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_task.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"time_only_acc": acc_time, "time_only_recall": dict(zip(CLASS_NAMES, map(float, rec_time))),
            "t2_to_t4_share": float(skip3), "phase_stats": stats}


# ============================================================================================ B. the representation
def fig_latent() -> dict:
    L = json.loads((FD / "latent.json").read_text())
    Z = np.load(FD / "latent_pca.npz", allow_pickle=True)
    fig = plt.figure(figsize=(10.5, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.45)
    example = None
    for j, name in enumerate(("single-plane transformer", "cross-focal")):
        ax = fig.add_subplot(gs[0, j])
        panel_label(ax, "ab"[j])
        xy, y, g, vids = Z[f"{name}|xy"], Z[f"{name}|y"], Z[f"{name}|g"], list(Z[f"{name}|videos"])
        var = Z[f"{name}|var"]
        keep = y < len(PH)
        ax.scatter(xy[keep, 0], xy[keep, 1], c=y[keep], cmap=CMAP, vmin=0, vmax=len(PH) - 1, s=1.2, alpha=0.25,
                   rasterized=True, linewidths=0)
        if example is None:
            example = max(vids, key=lambda v: len(set(y[g == vids.index(v)])))
        s = g == vids.index(example)
        ax.plot(xy[s, 0], xy[s, 1], color="black", lw=0.7, alpha=0.9)
        ax.scatter(xy[s, 0][[0, -1]], xy[s, 1][[0, -1]], c="black", s=18, marker="o", zorder=3)
        ax.set_title(name, fontsize=9.5, loc="left")
        ax.set_xlabel(f"PC1 ({100 * var[0]:.0f} %)")
        ax.set_ylabel(f"PC2 ({100 * var[1]:.0f} %)")
        ax.set_xticks([])
        ax.set_yticks([])
    cax = fig.add_subplot(gs[0, 2])
    cax.axis("off")
    for i, p in enumerate(PH):
        cax.scatter(0.05, 1 - i / len(PH), color=PCOL[p], s=40)
        cax.text(0.13, 1 - i / len(PH), p, va="center", fontsize=9)
    cax.plot([0.55, 0.7], [0.95, 0.95], color="black", lw=0.8)
    cax.text(0.73, 0.95, f"one test embryo\n({example})", va="center", fontsize=9)
    cax.set_xlim(0, 1.3)
    cax.set_ylim(-0.05, 1.05)
    cax.text(0.55, 0.62, "Every test frame of fold 0,\ncoloured by annotated phase;\nthe line follows one embryo\nfrom fertilisation (dot) to\nblastocyst (dot).",
             fontsize=9, va="top", color="#444444")

    ax = fig.add_subplot(gs[1, :2])
    panel_label(ax, "c", -0.07)
    pairs = list(L["cross-focal"]["adjacent"]["head"])
    xs = np.arange(len(pairs))
    for name, r in L.items():
        ax.plot(xs, [r["adjacent"]["head"].get(p, np.nan) for p in pairs], "-o", ms=3.5, color=MODEL_COL[name], label=name)
        ax.plot(xs, [r["adjacent"]["backbone"].get(p, np.nan) for p in pairs], "--", lw=0.9, color=MODEL_COL[name], alpha=0.7)
    ax.axhline(0.5, color="black", lw=0.7, ls=":")
    ax.text(len(pairs) - 0.6, 0.51, "chance", fontsize=8.5, ha="right", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([p.replace("|", "|\n") for p in pairs], fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("balanced accuracy,\nlinear probe")
    ax.set_title("separability of consecutive phases (solid: head, dashed: backbone features)", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.52, 1.0))

    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "d", -0.2)
    r = L["cross-focal"]["velocity"]
    kcol = {"synchronous division": "#1b7837", "transient division": C_CF, "morphological": "#5e3c99"}
    t = np.arange(-20, 21)
    for k, v in r.items():
        mu, se = np.array(v["mean"]), np.array(v["sem"])
        ax.plot(t, mu, color=kcol[k], label=f"{k} ({v['n_events']})")
        ax.fill_between(t, mu - se, mu + se, color=kcol[k], alpha=0.2, lw=0)
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("frames from annotated onset")
    ax.set_ylabel("latent speed\n(× per-video median)")
    ax.set_title("division events", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    fig.savefig(FIG / "fig_latent.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_latent.pdf", bbox_inches="tight")
    plt.close(fig)
    peak = {k: float(np.nanmax(np.array(v["mean"])[18:23]) / np.nanmean(np.array(v["mean"])[:10])) for k, v in r.items()}
    return {"adjacent_head": {n: rr["adjacent"]["head"] for n, rr in L.items()}, "velocity_peak_ratio": peak,
            "example": example}


# ============================================================================================ C. learning
def fig_dynamics() -> dict:
    D = json.loads((FD / "dynamics.json").read_text())
    phases = D["phases"]
    iph = [phases.index(p) for p in PH]
    fam_col = {"LSTM": C_REF, "transformer": C_TR, "other heads": "#8c6d31"}
    fig = plt.figure(figsize=(10.5, 7.9))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a", -0.22)
    runs = D["loss"]["transformer"]
    M = np.nanmean(np.array([r["per_phase"] for r in runs], dtype=float), 0)[:, iph].T        # phase x epoch
    im = ax.imshow(M, aspect="auto", cmap="rocket_r" if "rocket_r" in plt.colormaps() else "magma_r", vmin=0, vmax=3)
    ax.set_yticks(range(len(PH)))
    ax.set_yticklabels(PH, fontsize=8)
    ax.set_xticks(range(0, 10, 3))
    ax.set_xticklabels(range(1, 11, 3))
    ax.set_xlabel("epoch")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("training loss", fontsize=9, labelpad=2)
    ax.set_title(f"transformer heads ({len(runs)} runs)", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[0, 1:])
    panel_label(ax, "b", -0.08)
    xs = np.arange(len(PH))
    finals = {}
    for k, (fam, rs) in enumerate(sorted(D["loss"].items())):
        F = np.array([r["per_phase"][-1] for r in rs], dtype=float)[:, iph]
        finals[fam] = np.nanmedian(F, 0)
        jit = (k - 1) * 0.22
        ax.scatter(np.repeat(xs, len(F)).reshape(len(PH), -1).T + jit, F, s=7, color=fam_col[fam], alpha=0.5, lw=0)
        ax.plot(xs + jit, np.nanmedian(F, 0), "_", ms=11, mew=2, color=fam_col[fam], label=f"{fam} ({len(rs)} runs)")
    for i, p in enumerate(PH):
        if p in TRANSIENT:
            ax.axvspan(i - 0.45, i + 0.45, color=C_CF, alpha=0.07, lw=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(PH)
    ax.set_ylabel("training loss, epoch 10")
    ax.set_ylim(0, 3.2)
    ax.legend(frameon=False, fontsize=8.5, ncol=1, loc="upper right")
    ax.set_title("the same phases stay unfitted whatever the temporal head (optimiser follows the head family; shaded: transient)", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c", -0.22)
    for fam, rs in sorted(D["loss"].items()):
        ratio = np.array([np.array(r["boundary"], dtype=float) / np.array(r["interior"], dtype=float) for r in rs])
        mu, sd = ratio.mean(0), ratio.std(0)
        ax.plot(range(1, 11), mu, color=fam_col[fam], label=fam)
        ax.fill_between(range(1, 11), mu - sd, mu + sd, color=fam_col[fam], alpha=0.15, lw=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("boundary / interior loss")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("frames at a label change vs elsewhere", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d", -0.22)
    for eid, c in D["curves"].items():
        col = fam_col.get(c["family"], "#bbbbbb")
        for r in c["runs"]:
            ax.plot(range(1, len(r) + 1), r, color=col, lw=0.6, alpha=0.45)
    final = [r[-1] for c in D["curves"].values() for r in c["runs"] if len(r) == 10]
    ax.axhspan(np.percentile(final, 25), np.percentile(final, 75), color="black", alpha=0.08, lw=0)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation $p_t$")
    ax.set_title(f"{len(D['curves'])} configurations, common recipe", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "e", -0.22)
    dr = D["drift"]["cross-focal"]
    ax.scatter(dr["tempo"], dr["offset"], s=8, color=C_CF, alpha=0.5, lw=0, label="cross-focal")
    dr0 = D["drift"]["reference"]
    ax.scatter(dr0["tempo"], dr0["offset"], s=8, color=C_REF, alpha=0.35, lw=0, label="reference")
    lim = np.array([-25, 25])
    ax.plot(lim, -lim, color="black", lw=0.7, ls=":")
    ax.text(-24, -24, "dotted, slope −1:\npredicts the\naverage embryo", fontsize=8, va="bottom", ha="left")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlim(*lim)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("embryo tempo (h vs cohort median)")
    ax.set_ylabel("mean onset error (h)")
    ax.set_title(f"offsets carry {100 * dr['offset_share_of_sq_error']:.0f} % of squared error", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.savefig(FIG / "fig_dynamics.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_dynamics.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"final_loss_median": {f: dict(zip(PH, map(float, v))) for f, v in finals.items()},
            "drift": {n: {k: v for k, v in d.items() if k in ("n_embryos", "offset_share_of_sq_error",
                                                                "offset_on_tempo_slope", "offset_tempo_r", "residual_lag1_corr")}
                      for n, d in D["drift"].items()},
            "n_curves": len(D["curves"])}


# ============================================================================================ D. where it looks
def fig_where(video: str = "BM016-5") -> dict:
    Z = np.load(FD / "plane_attention_test.npz", allow_pickle=True)
    A, S, y = Z["attn"], Z["sharp"], Z["label"]
    planes = ["−45", "−30", "−15", "0", "+15", "+30", "+45"]
    G = np.load(FD / f"gradcam_{video}.npz", allow_pickle=True)
    phases = [p for p in ("t2", "t3", "t4", "t5", "t7", "t8", "tM", "tB") if f"single|{p}|cam" in G.files]
    fig = plt.figure(figsize=(10.5, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.28)
    top = gs[0].subgridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.6)

    ax = fig.add_subplot(top[0])
    panel_label(ax, "a", -0.18)
    rows = [p for p in PH if (y == CLASS_NAMES.index(p)).sum() >= 100]
    M = np.array([A[y == CLASS_NAMES.index(p)].mean(0) for p in rows])
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=0.35)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xticks(range(7))
    ax.set_xticklabels(planes, fontsize=8.5)
    ax.set_xlabel("focal plane (µm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("mean attention", fontsize=9)
    ax.set_title(f"{len(y):,} test frames", fontsize=9.5, loc="left")

    ax = fig.add_subplot(top[1])
    panel_label(ax, "b", -0.25)
    from scipy.stats import spearmanr
    rho = np.array([spearmanr(a, s).correlation for a, s in zip(A, S)])
    fixed = A.mean(0)                                   # one attention profile for every frame: the null
    rho_fix = np.array([spearmanr(fixed, s).correlation for s in S])
    bins = np.linspace(-1, 1, 29)
    ax.hist(rho_fix[np.isfinite(rho_fix)], bins=bins, color=C_REF, alpha=0.6, label="fixed mean profile")
    ax.hist(rho[np.isfinite(rho)], bins=bins, histtype="step", color=C_CF, lw=1.6, label="model's own attention")
    ax.set_xlabel("Spearman ρ, attention vs focus\n(per frame, over 7 planes)")
    ax.set_ylabel("frames")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    grp = pd.Series(list(zip(Z["video"], y))).factorize()[0]
    dev = lambda X: X - pd.DataFrame(X).groupby(grp).transform("mean").to_numpy()  # noqa: E731
    r_dev = float(np.corrcoef(dev(A).ravel(), dev(S).ravel())[0, 1])
    tot = A.var(0).sum()
    var_phase = float(pd.DataFrame(A).groupby(y).transform("mean").to_numpy().var(0).sum() / tot)
    var_vp = float(pd.DataFrame(A).groupby(grp).transform("mean").to_numpy().var(0).sum() / tot)
    ax.set_title(f"same median ({np.nanmedian(rho):.2f})", fontsize=9.5, loc="left")

    ax = fig.add_subplot(top[2])
    panel_label(ax, "c", -0.25)
    ent = -(A * np.log(A + 1e-9)).sum(1) / np.log(7)
    ents = [ent[y == CLASS_NAMES.index(p)] for p in rows]
    ax.boxplot(ents, vert=False, widths=0.6, showfliers=False, medianprops={"color": C_CF})
    ax.set_yticks(range(1, len(rows) + 1))
    ax.set_yticklabels(rows, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("attention entropy\n(1 = uniform over planes)")
    ax.set_title("how spread the attention is", fontsize=9.5, loc="left")

    bot = gs[1].subgridspec(3, len(phases), hspace=0.08, wspace=0.04)
    for j, p in enumerate(phases):
        img = G[f"single|{p}|img"]
        base = img[0] if img.ndim == 3 else img
        ref = G[f"crossfocal|{p}|img"][3]
        cam_s = G[f"single|{p}|cam"][0]
        w = G[f"crossfocal|{p}|attn"]
        cam_c = (G[f"crossfocal|{p}|cam"] * w[:, None, None]).sum(0)
        for i, (bg, cam, lab) in enumerate(((base, None, "image"), (base, cam_s, "single-plane"), (ref, cam_c, "cross-focal"))):
            a = fig.add_subplot(bot[i, j])
            a.set_xticks([])
            a.set_yticks([])
            a.imshow(bg, cmap="gray")
            if cam is not None:
                a.imshow(cam / (cam.max() + 1e-9), cmap="inferno", alpha=0.5, vmin=0, vmax=1)
                prob = G[f"{'single' if i == 1 else 'crossfocal'}|{p}|prob"]
                a.text(0.03, 0.03, f"p={float(prob):.2f}", transform=a.transAxes, color="white", fontsize=8)
            if i == 0:
                a.set_title(p, fontsize=10)
            if j == 0:
                a.set_ylabel(lab, fontsize=9)
                if i == 0:
                    panel_label(a, "d", -0.35)
    fig.savefig(FIG / "fig_where.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_where.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"attn_mean_by_phase": {p: M[i].tolist() for i, p in enumerate(rows)}, "rho_median": float(np.nanmedian(rho)), "rho_fixed_median": float(np.nanmedian(rho_fix)),
            "r_deviation": r_dev, "var_share_phase": var_phase, "var_share_video_phase": var_vp,
            "entropy_median_by_phase": {p: float(np.median(e)) for p, e in zip(rows, ents)}, "gradcam_video": video}



# ============================================================================================ E. timeline audit
def fig_timeline_audit(example: str = "MM445-2-2") -> dict:
    """Supplementary: the image--annotation timeline audit (scripts/audit_frame_timing.py)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from audit_frame_timing import STRIDE, annotation
    al = pd.read_csv(ROOT / "results/frame_timing/alignment_oof.csv")
    log = json.loads((ROOT / "data/qc/nantes_video_defects_v4.json").read_text())["videos"]
    mis = {v for v, e in log.items() if "timeline_mismatch" in e["defects"]}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.9), gridspec_kw={"width_ratios": [1, 1.5, 1], "wspace": 0.32})
    ax = axes[0]
    panel_label(ax, "a", -0.18)
    ratio = al.cost_identity - al.cost_fit
    m = al.video.isin(mis)
    ax.scatter(al.slope_fit[~m], ratio[~m], s=8, color=C_REF, alpha=0.5, lw=0, label=f"other videos ({(~m).sum()})")
    ax.scatter(al.slope_fit[m], ratio[m], s=16, color=C_CF, lw=0, label=f"timeline mismatch ({m.sum()})")
    ax.axvline(1.0, color="black", lw=0.6, ls=":")
    ax.axvline(1.25, color=C_CF, lw=0.6, ls=":")
    ax.text(1.26, ratio.max() * 1.02, "0.25 h / 0.2 h", color=C_CF, fontsize=8.5, va="bottom")
    ax.set_xlabel("fitted annotation frames per image ($a$)")
    ax.set_ylabel("phase error removed by\nthe fitted pairing")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax = axes[1]
    panel_label(ax, "b", -0.08)
    z = np.load(ROOT / "results/frame_timing/preds.npz")
    sel = z["video"] == example
    k = z["frame_index"][sel]
    order = np.argsort(k)
    k, pred = k[order], z["log_probs"][sel][order].astype(np.float32).argmax(1)
    lab = annotation(example)
    r = al.set_index("video").loc[example]
    j_id = k.astype(int)
    j_fit = np.rint(r.slope_fit * k + r.offset_fit).astype(int)
    ok_id = (j_id < len(lab)) & (lab[np.clip(j_id, 0, len(lab) - 1)] >= 0)
    ok_fit = (j_fit >= 0) & (j_fit < len(lab))
    ok_fit[ok_fit] &= lab[j_fit[ok_fit]] >= 0
    ax.plot(k, pred, ".", ms=3, color="black", label="classifier prediction")
    ax.step(k[ok_id], lab[j_id[ok_id]], where="post", color=C_REF, lw=1.6, label="label, image $k$ ↔ frame $k$ (released)")
    ax.step(k[ok_fit], lab[np.clip(j_fit[ok_fit], 0, len(lab) - 1)], where="post", color=C_CF, lw=1.6,
            label=f"label, image $k$ ↔ frame {r.slope_fit:.2f}$k${r.offset_fit:+.0f}")
    ax.set_yticks(range(0, len(PH), 2))
    ax.set_yticklabels(PH[::2])
    ax.set_xlabel(f"image index $k$ ({example}, every {STRIDE}rd image)")
    ax.set_ylabel("phase")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    # c: the mechanism without a model -- time-file step against recording span / images (scripts/timeline_mechanism.py)
    ax = axes[2]
    panel_label(ax, "c", -0.2)
    mv = pd.read_csv(ROOT / "results/frame_timing/mechanism_videos.csv")
    mm = mv.mismatch.astype(bool)
    cov = mv.covers_all_images.astype(bool)
    ax.scatter(mv.file_step[~mm & cov], mv.image_interval[~mm & cov], s=8, color=C_REF, alpha=0.5, lw=0, label="other videos, time file covers every image")
    ax.scatter(mv.file_step[mm], mv.image_interval[mm], s=16, color=C_CF, lw=0, label="timeline mismatch")
    ax.plot((0.18, 0.40), (0.18, 0.40), color="black", lw=0.6, ls=":")
    ax.text(0.398, 0.36, "$x=y$", fontsize=8.5, ha="right", va="top")
    ax.set_xlim(0.18, 0.40); ax.set_ylim(0.18, 0.40)
    ax.set_xlabel("step of the time file (h)")
    ax.set_ylabel("recording span / (images $-$ 1) (h)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(FIG / "fig_timeline_audit.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_timeline_audit.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"example": example, "slope": float(r.slope_fit), "offset": float(r.offset_fit)}

def main() -> None:
    out = {"task": fig_task(), "latent": fig_latent(), "dynamics": fig_dynamics(), "where": fig_where(),
           "timeline_audit": fig_timeline_audit()}
    (FD / "figure_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote fig_task, fig_latent, fig_dynamics, fig_where; numbers -> results/figdata/figure_numbers.json")


if __name__ == "__main__":
    main()
