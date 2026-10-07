#!/usr/bin/env python
"""Main figures for the CLEAVE paper. All panels are drawn from stored results; the only
model forward passes are (i) the window-length curve of Fig. 1a and (ii) the plane-attention example of Fig. 3b, both
inference only. Writes figures/fig_{defects,crossfocal,transfer}.png.

  PYTHONPATH=src:scripts python scripts/make_paper_figures.py [--device cuda:1] [--skip_gpu]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False})
C_REF, C_CF, C_TR = "#7f7f7f", "#c0392b", "#2c6fbb"


def panel_label(ax, s):
    ax.text(-0.12, 1.05, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


# ------------------------------------------------------------------------------------------------------ Fig. 1
def window_curve(device, windows=(8, 16, 32, 64, 150)):
    """Validation p_t vs evaluation window for the chunk-trained transformer and the LSTM reference (inference only)."""
    import torch
    from stseg.eval.kinetic_metrics import evaluate_videos
    from stseg.kinetic.datasets import _frames
    from stseg.kinetic.models import build_model
    from stseg.kinetic.process import sequence_log_probs
    from torch.utils.data import DataLoader, Subset
    out = {}
    for name, run in (("transformer (16-frame clips)", "resnet18_transformer_L16_split0_seed0"), ("LSTM reference (4-frame clips)", "resnet18_lstm_L4_split0_seed0")):
        rd = ROOT / "runs/h7" / run
        cfg = json.loads((rd / "config.resolved.json").read_text()); lt = np.load(rd / "transition_log_matrix.npy")
        ds = _frames(cfg["data"], "val", "eval", 0, None, None)
        model = build_model(dict(cfg["model"]), ds.in_channels); model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu", weights_only=True)["model"]); model.to(device).eval()
        frames_by_video = []
        for vid, g in ds.rows.groupby("video", sort=False):
            loader = DataLoader(Subset(ds, g.index.to_numpy()), batch_size=150, shuffle=False, num_workers=4)
            frames_by_video.append((vid, torch.cat([b["image"] for b in loader]), g.label.to_numpy(), g.time_h.to_numpy(dtype=float)))
        curve = []
        for w in windows:
            seqs = [{"video": v, "labels": y, "times_h": t, "log_probs": sequence_log_probs(model, fr, w, max(1, w // 2), device, True)} for v, fr, y, t in frames_by_video]
            curve.append(evaluate_videos(seqs, lt)["p_t"]); print(f"  {name} window {w}: p_t {curve[-1]:.3f}", flush=True)
        out[name] = curve
    json.dump({"windows": list(windows), **out}, open(ROOT / "results/window_curve_val.json", "w"), indent=1)
    return out


def fig_defects(device, skip_gpu):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0)); axes = axes.ravel()
    # a: window curve
    ax = axes[0]; panel_label(ax, "a")
    f = ROOT / "results/window_curve_val.json"
    if f.exists() and skip_gpu:
        d = json.load(open(f)); windows = d.pop("windows"); curves = d
    else:
        windows = (8, 16, 32, 64, 150); curves = window_curve(device, windows)
    short_label = {"transformer (16-frame clips)": "transformer", "LSTM reference (4-frame clips)": "LSTM reference"}
    for (name, c), col in zip(curves.items(), (C_TR, C_REF)):
        ax.plot(windows, c, "o-", color=col, label=short_label.get(name, name), lw=1.5, ms=4)
    ax.axvline(150, color="k", ls=":", lw=0.8); ax.text(148, 0.55, "released\nchunking ", fontsize=7, va="center", ha="right")
    ax.axvline(16, color=C_TR, ls=":", lw=0.8); ax.set_xscale("log", base=2); ax.set_xticks(windows); ax.set_xticklabels([str(w) for w in windows])
    ax.set_xlabel("evaluation window (frames)"); ax.set_ylabel("validation $p_t$"); ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.45, 0.02), handletextpad=2.2, handlelength=1.2)
    # b: leak histogram
    ax = axes[1]; panel_label(ax, "b")
    # drawn from results/protocol_audit.json (scripts/audit_protocol_splits.py), the single producer of
    # every leakage number in the text, so the panel and the paragraph cannot disagree.
    audit = json.load(open(ROOT / "results/protocol_audit.json"))
    # same video population AND the same three-way partition geometry as the observed count
    sim = audit["simulation"]["clean652.couple.released"]
    lp = np.array(sim["leaked_patients"])
    obs = sim["released_fold0_leaking_patients"]
    n_pat = audit["official_folds"]["split0"]["clean652"]["couple"]["n_patients"]
    ax.hist(lp, bins=np.arange(lp.min() - 0.5, lp.max() + 1.5), color="#bbbbbb", edgecolor="white")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)  # headroom so the annotations below don't sit on top of the tallest bars
    ax.axvline(obs, color=C_CF, lw=1.5); ax.text(obs + 2, ax.get_ylim()[1] * 0.94, f"released fold 0\n({obs} of {n_pat})", color=C_CF, fontsize=7, va="top")
    ax.axvline(0, color=C_TR, lw=1.5); ax.text(1.5, ax.get_ylim()[1] * 0.94, "patient-\ngrouped", color=C_TR, fontsize=7, va="top")
    ax.set_xlim(-2, lp.max() + 2)
    ax.set_xlabel(f"patients in >1 partition ({sim['seeds']} random video-level splits,\nreleased 564/70/70 shape)")
    ax.set_ylabel("splits")
    # c: split mechanisms
    ax = axes[2]; panel_label(ax, "c")
    m = pd.read_csv(ROOT / "results/runs_master.csv")
    def pt(exp):
        r = m[m.experiment == exp].p_t; return (r.mean(), r.std(ddof=1) if len(r) > 1 else 0.0) if len(r) else (np.nan, 0)
    groups = [("ResNet-18\nper-frame", [pt("resnet18_none_split0"), pt("resnet18_none_imagesplit")]),
              ("EfficientNet-V2-L\nper-frame", [pt("efficientnet_v2_l_none_split0"), pt("efficientnet_v2_l_none_imagesplit")]),
              ("ResNet-18-LSTM\n(reference)", [pt("resnet18_lstm_L4_grouped_v1"), pt("resnet18_lstm_L4_split0"), (np.nan, 0)])]
    x = 0
    for label, vals in groups:
        cols = [C_TR, C_REF, C_CF] if len(vals) == 3 else [C_REF, C_CF]
        names = ["patient-grouped", "video-level fold 0", "image-level"] if len(vals) == 3 else ["video-level fold 0", "image-level"]
        for (v, s), col, nm in zip(vals, cols, names):
            if not np.isnan(v):
                ax.bar(x, v, yerr=s if s else None, color=col, width=0.8, label=nm)
                ax.text(x, v + (s or 0) + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
            x += 1
        ax.text(x - len(vals) / 2 - 0.5, 0.53, label, ha="center", va="top", fontsize=8.5, clip_on=False); x += 0.8
    ax.set_ylim(0.55, 0.95); ax.set_xticks([]); ax.set_ylabel("test $p_t$"); ax.spines["bottom"].set_visible(False)
    h, l = ax.get_legend_handles_labels(); uniq = dict(zip(l, h)); ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8.5, loc="upper left")
    # d: the metric is underspecified -- p_t of the same reference predictions under every definition the text leaves open
    ax = axes[3]; panel_label(ax, "d")
    pdj = json.load(open(ROOT / "results/pt_definitions.json"))
    dd = pd.DataFrame(pdj["all"])
    skips = [("next", "impute from next", "#1b9e77"), ("previous", "impute from previous", "#7570b3"), ("miss", "count as miss", "#d95f02")]
    rng = np.random.default_rng(0)
    for gi, tol in enumerate((0.5, 1.0, 2.0)):
        for si, (sk, lab, col) in enumerate(skips):
            v = dd[(dd.tolerance_x_theta == tol) & (dd.skip == sk)].p_t.to_numpy()
            x = gi + (si - 1) * 0.26
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=7, color=col, alpha=0.7, label=lab if gi == 0 else None, lw=0)
    ax.axhline(pdj["published"], color="k", lw=1, ls="--")
    ax.text(2.45, pdj["published"] - 0.012, f"published {pdj['published']:.3f}", ha="right", va="top", fontsize=7.5)
    ax.scatter([1 - 0.26], [pdj["ours"]], marker="*", s=110, color="k", zorder=5, label="this paper's rule")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["$\\theta/2$", "$\\theta$", "$2\\theta$"])
    ax.set_xlabel("tolerance (per-event inter-operator s.d. $\\theta$)")
    ax.set_ylabel("$p_t$, reference, 5 folds"); ax.set_xlim(-0.5, 2.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", title="skipped phase", title_fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_defects.png", dpi=200); plt.close(fig); print("fig_defects.png")


# ------------------------------------------------------------------------------------------------------ Fig. 3
def plane_attention(device):
    """Attention weights over the seven planes for one 5-cell frame of a fold-0 test video (cross-focal transformer, seed 0)."""
    import torch
    from stseg.kinetic.datasets import _frames
    from stseg.kinetic.models import build_model
    rd = ROOT / "runs/h7/resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0"
    cfg = json.loads((rd / "config.resolved.json").read_text()); ds = _frames(cfg["data"], "test", "eval", 0, None, None)
    model = build_model(dict(cfg["model"]), ds.in_channels); model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu", weights_only=True)["model"]); model.to(device).eval()
    rows = ds.rows[ds.rows.label == 6]  # t5
    r = rows.iloc[len(rows) // 3]; x = ds[int(r.name)]["image"][None].to(device)  # (1, 7, H, W)
    bb = model.backbone
    with torch.no_grad():
        N, P, H, W = x.shape
        f = bb.cnn(x.reshape(N * P, 1, H, W).expand(-1, 3, -1, -1)).view(N, P, -1)
        tok = bb.norm(f + bb.plane_emb[None]); _, w = bb.attn(bb.query.expand(N, -1, -1), tok, tok, need_weights=True)
    return x[0].cpu().numpy(), w[0, 0].cpu().numpy(), str(r.video), float(r.time_h)


def fig_crossfocal(device, skip_gpu):
    fig = plt.figure(figsize=(12.6, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], width_ratios=[1.5, 1.0, 1.0], wspace=0.32)
    # a: schematic
    # a: schematic. Two branches on one aligned grid: same rows, same box widths, one row differs.
    ax = fig.add_subplot(gs[0, 0]); panel_label(ax, "a"); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(-0.9, 10.2)
    BW, XL, XR = 4.75, 2.55, 7.45                   # box width and the two column centres
    ROWS = {"stack": 7.55, "enc": 5.95, "fuse": 4.35, "feat": 2.95, "head": 1.55}
    GREY, BLUE, PINK = "#f0f0f0", "#dce9fb", "#fde3e3"

    def box(xc, yc, text, fc=GREY, h=0.95, w=BW, fs=7.2, lw=0.9, ec="#555"):
        ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(xc, yc, text, ha="center", va="center", fontsize=fs, linespacing=1.35)

    def down(xc, y_from, y_to):
        ax.annotate("", (xc, y_to), (xc, y_from), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#555",
                                                                  shrinkA=1, shrinkB=1))

    def planes(xc, y, n=7, w=0.30, h=0.78, dx=0.10, dy=0.055):
        """Seven plates drawn with depth, so the input reads as a focal stack, not a barcode."""
        for i in range(n - 1, -1, -1):
            ax.add_patch(FancyBboxPatch((xc - (n * dx) / 2 + i * dx - w / 2, y - h / 2 + i * dy), w, h,
                                        boxstyle="round,pad=0.01", fc="#ffffff", ec="#777", lw=0.8, zorder=3 + i))

    for xc, title in ((XL, "channel stacking"), (XR, "cross-focal attention")):
        ax.text(xc, 9.5, title, ha="center", fontsize=8.4, fontweight="bold")
        planes(xc, ROWS["stack"] + 0.75)

    ax.text(XL, ROWS["stack"] - 0.05, "7 planes as 7 channels", ha="center", va="center", fontsize=7.6)
    ax.text(XR, ROWS["stack"] - 0.05, "7 planes, encoded apart", ha="center", va="center", fontsize=7.6)
    for xc in (XL, XR):
        down(xc, ROWS["stack"] + 0.30, ROWS["stack"] + 0.18)
        down(xc, ROWS["stack"] - 0.30, ROWS["enc"] + 0.50)

    box(XL, ROWS["enc"], "one CNN; first convolution\nre-initialised (mean filter)", fs=7.6)
    box(XR, ROWS["enc"], "one shared pretrained CNN,\n+ learned plane embedding", fs=7.6)
    for xc in (XL, XR):
        down(xc, ROWS["enc"] - 0.50, ROWS["fuse"] + 0.50)

    box(XL, ROWS["fuse"], "fixed linear mixture\nof the planes", fc="#fafafa", ec="#bbb", lw=0.9, fs=7.6)
    box(XR, ROWS["fuse"], "attention from a learned query\n+ residual plane mean", fc=BLUE, lw=1.7, ec="#2b6cb0", fs=7.6)
    for xc in (XL, XR):
        down(xc, ROWS["fuse"] - 0.50, ROWS["feat"] + 0.40)

    for xc in (XL, XR):
        box(xc, ROWS["feat"], "one frame feature", fc=PINK, h=0.78, w=BW - 1.4, fs=7.6)
        down(xc, ROWS["feat"] - 0.42, ROWS["head"] + 0.48)
    box(XL, ROWS["head"], "temporal head (transformer)", fs=7.6)
    box(XR, ROWS["head"], "temporal head (transformer)", fs=7.6)
    for xc in (XL, XR):
        down(xc, ROWS["head"] - 0.48, 0.62)
        ax.text(xc, 0.35, "phase", ha="center", va="center", fontsize=7.6, fontweight="bold")

    ax.text(5.0, -0.75, "same temporal head; the branches differ in how planes are encoded and fused (blue)",
            ha="center", va="bottom", fontsize=7.4, color="#2b6cb0")


    # b: attention example
    ax = fig.add_subplot(gs[0, 1:]); panel_label(ax, "b"); ax.axis("off")
    f = ROOT / "results/plane_attention_example.npz"
    if f.exists() and skip_gpu:
        z = np.load(f, allow_pickle=True); planes, w, vid, th = z["planes"], z["w"], str(z["vid"]), float(z["th"])  # our own npz (string fields), not untrusted input
    else:
        planes, w, vid, th = plane_attention(device); np.savez(f, planes=planes, w=w, vid=vid, th=th)
    labels = ["-45", "-30", "-15", "0", "+15", "+30", "+45"]
    for i in range(7):
        sub = ax.inset_axes([i / 7 + 0.005, 0.28, 1 / 7 - 0.01, 0.7]); img = planes[i]; sub.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99)); sub.axis("off")
        sub.set_title(f"{labels[i]} µm", fontsize=7)
    sub = ax.inset_axes([0.005, 0.0, 0.99, 0.24]); sub.bar(np.arange(7), w, color=C_CF); sub.set_xticks(np.arange(7)); sub.set_xticklabels(labels, fontsize=6); sub.set_ylabel("attention", fontsize=6); sub.tick_params(labelsize=6)
    print(f"fig_crossfocal panel b: 5-cell frame, test video, {th:.0f} h post insemination (caption text in main.tex is hand-written to match)")
    # d: recall per phase for 1/3/cross-focal input
    ax = fig.add_subplot(gs[1, :]); panel_label(ax, "c")
    from stseg.data.nantes_kinetic import CLASS_NAMES
    cfgs = [("1 plane", "resnet18_transformer_L16_evalfix_split0_seed0", "#c7c7c7"), ("3 planes stacked", "resnet18_transformer_L16_multifocal3_evalfix_split0_seed0", "#9ecae1"), ("7 planes, cross-focal", "resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0", C_CF)]
    phases = [4, 6, 8, 13]; xs = np.arange(len(phases) + 1)
    for j, (lab, run, col) in enumerate(cfgs):
        S = json.load(open(ROOT / "results/diagnostics" / run / "summary.json")); cm = np.array(S["confusion_test"]["viterbi"])
        rec = [cm[k, k] / max(cm[k].sum(), 1) for k in phases]; skips = S["segments"]["phases_skipped_by_viterbi"]
        ax.bar(xs[:-1] + (j - 1) * 0.27, rec, 0.27, color=col, label=lab)
        ax.bar(xs[-1] + (j - 1) * 0.27, skips / 809, 0.27, color=col)
    ax.set_xticks(xs); ax.set_xticklabels([CLASS_NAMES[k] for k in phases] + ["skipped\nsegments / 809"]); ax.set_ylabel("decoded recall (fold 0, seed 0)"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig_crossfocal.png", dpi=200); plt.close(fig); print("fig_crossfocal.png")


# ------------------------------------------------------------------------------------------------------ Fig. 4
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--skip_gpu", action="store_true"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    import torch
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if a.only in ("", "crossfocal"): fig_crossfocal(device, a.skip_gpu)
    if a.only in ("", "defects"): fig_defects(device, a.skip_gpu)


if __name__ == "__main__":
    main()
