#!/usr/bin/env python
"""Post-hoc diagnostics for an already-trained kinetic run (TEMPO). No retraining: loads ``best.pt``, runs inference on
val and test with the run's *own* eval window (``training.eval_len/eval_stride``, default 150/150 -- the window the run
was scored with), and writes everything ``src/stseg/kinetic/diagnostics.py`` now logs during training plus the analyses
that need the whole set of predictions at once. Purpose: decide, from evidence rather than taste, which bottleneck a new
module/loss should attack:

  representation vs temporal   -> linear probe on backbone vs head features (val -> test), adjacent-phase separability,
                                  argmax-vs-Viterbi confusion, error rate as a function of distance to the nearest
                                  ground-truth transition (errors piled at boundaries = timing; scattered = features)
  bias vs variance of timings  -> per-event signed error: median, MAE, bias, MAD, sign test
  calibration                  -> entropy near vs far from transitions, ECE, AUROC of entropy for frame errors

Outputs to ``results/diagnostics/<run>/``: ``frame_probs_{val,test}.npz``, ``features_{val,test}.npz`` (backbone and
head features, float16, separable models only), ``summary.json``, PNGs, ``REPORT.md``. With several ``--runs`` a
``results/diagnostics/COMPARE.md`` table is written as well.

Usage:
  PYTHONPATH=src python scripts/diagnose_run.py --runs runs/h7/resnet50_lstm_L8_split0_seed1 \
      runs/h7/resnet18_transformer_L16_multifocal3_evalfix_split0_seed0 --device cuda:1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stseg.data.nantes_kinetic import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from stseg.eval.kinetic_metrics import THETA_BY_CLASS, evaluate_videos, viterbi  # noqa: E402
from stseg.kinetic import diagnostics as diag  # noqa: E402
from stseg.kinetic.datasets import _frames  # noqa: E402
from stseg.kinetic.models import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIST_BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]


# ----------------------------------------------------------------------------------------------------- inference
@torch.no_grad()
def infer_video(model, frames: torch.Tensor, eval_len: int, eval_stride: int, device, amp: bool):
    """Returns (log_probs (T,K), backbone feats (T,D) | None, head feats (T,H) | None). Same windowing rule as
    ``process.sequence_log_probs``; backbone runs once, head/cls per window, features averaged over overlaps."""
    T = len(frames)
    separable = hasattr(model, "backbone") and hasattr(model, "head") and hasattr(model, "cls")
    if not model.is_sequence:  # per-frame model
        lps, fs = [], []
        for i in range(0, T, 256):
            x = frames[i:i + 256].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                if separable:
                    f = model.backbone(x).float(); fs.append(f.cpu())
                    out = model.cls(model.head(f[:, None])).float()[:, 0]
                else:
                    out = model(x)["logits"].float().reshape(-1, NUM_CLASSES)
            lps.append(torch.log_softmax(out, 1).cpu())
        return torch.cat(lps).numpy(), (torch.cat(fs).numpy() if fs else None), None
    starts = list(range(0, max(T - eval_len, 0) + 1, eval_stride))
    if starts[-1] + eval_len < T:
        starts.append(T - eval_len)
    acc = torch.zeros(T, NUM_CLASSES); cnt = torch.zeros(T, 1)
    feats = hf = None
    if separable:
        fs = []
        for i in range(0, T, 256):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                fs.append(model.backbone(frames[i:i + 256].to(device, non_blocking=True)).float())
        feats = torch.cat(fs)
        hf = torch.zeros(T, model.head.d_out)
    for s in starts:
        e = min(s + eval_len, T)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            if separable:
                h = model.head(feats[s:e][None]).float()[0]
                out = model.cls(h).float()
                hf[s:e] += h.cpu()
            else:
                out = model(frames[s:e][None].to(device, non_blocking=True))["logits"].float()[0]
        acc[s:e] += torch.log_softmax(out, 1).cpu(); cnt[s:e] += 1
    lp = (acc / cnt).numpy()
    return lp, (feats.cpu().numpy() if feats is not None else None), (hf / cnt).numpy() if hf is not None else None


def collect(run: Path, partition: str, device, num_workers: int, want_feats: bool):
    cfg = json.loads((run / "config.resolved.json").read_text())
    t = cfg.get("training", {})
    eval_len = int(t.get("eval_len", 150)); eval_stride = int(t.get("eval_stride", eval_len))
    ds = _frames(cfg["data"], partition, "eval", 0, None, None)
    model = build_model(dict(cfg["model"]), ds.in_channels)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    model.to(device).eval()
    amp = device.type == "cuda"
    seqs, feats = [], {}
    for vid, g in ds.rows.groupby("video", sort=False):
        loader = DataLoader(Subset(ds, g.index.to_numpy()), batch_size=150, shuffle=False, num_workers=num_workers)
        frames = torch.cat([b["image"] for b in loader])
        lp, bf, hf = infer_video(model, frames, eval_len, eval_stride, device, amp)
        seqs.append({"video": str(vid), "labels": g.label.to_numpy(), "log_probs": lp, "times_h": g.time_h.to_numpy(dtype=float)})
        if want_feats and bf is not None:
            feats[f"{vid}__bb"] = bf.astype(np.float16)
            if hf is not None:
                feats[f"{vid}__hd"] = hf.astype(np.float16)
    return cfg, seqs, feats, (eval_len, eval_stride)


# ----------------------------------------------------------------------------------------------------- analyses
def dist_to_boundary(y: np.ndarray) -> np.ndarray:
    """Per frame: distance (frames) to the nearest ground-truth transition (frame where the label changes)."""
    change = np.where(y[1:] != y[:-1])[0] + 1  # index of first frame of the new phase
    if len(change) == 0:
        return np.full(len(y), 10**9)
    idx = np.arange(len(y))
    # distance to the transition point measured on both sides: frame t belongs to boundary at change c if |t - c| or |t - (c-1)|
    d = np.min(np.abs(idx[:, None] - change[None, :]), axis=1)
    d_prev = np.min(np.abs(idx[:, None] - (change[None, :] - 1)), axis=1)
    return np.minimum(d, d_prev)


def error_vs_distance(seqs, log_trans) -> dict:
    tot = {b: [0, 0, 0] for b in DIST_BINS}  # n, err_argmax, err_viterbi
    for s in seqs:
        y = np.asarray(s["labels"]); lp = np.asarray(s["log_probs"])
        a = lp.argmax(1); v = viterbi(lp, log_trans); d = dist_to_boundary(y)
        for lo, hi in DIST_BINS:
            m = (d >= lo) & (d <= hi)
            tot[(lo, hi)][0] += int(m.sum()); tot[(lo, hi)][1] += int((a[m] != y[m]).sum()); tot[(lo, hi)][2] += int((v[m] != y[m]).sum())
    out = {}
    for (lo, hi), (n, ea, ev) in tot.items():
        key = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}")
        out[key] = {"n_frames": n, "err_argmax": ea / n if n else None, "err_viterbi": ev / n if n else None,
                    "share_of_all_errors_viterbi": None}
    tot_err = sum(v[2] for v in tot.values())
    for (lo, hi), (n, ea, ev) in tot.items():
        key = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}")
        out[key]["share_of_all_errors_viterbi"] = ev / tot_err if tot_err else None
    return out


def per_phase_prf(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=float)
    out = {}
    for k in range(NUM_CLASSES):
        tp = cm[k, k]; fn = cm[k].sum() - tp; fp = cm[:, k].sum() - tp
        p = tp / (tp + fp) if tp + fp else None; r = tp / (tp + fn) if tp + fn else None
        out[CLASS_NAMES[k]] = {"support": int(cm[k].sum()), "precision": p, "recall": r,
                               "f1": (2 * p * r / (p + r)) if p and r else (0.0 if cm[k].sum() else None)}
    off = cm.copy(); np.fill_diagonal(off, 0)
    adj = sum(off[i, j] for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if abs(i - j) == 1)
    out["_adjacent_share_of_errors"] = float(adj / off.sum()) if off.sum() else None
    out["_under_prediction_share"] = float(np.tril(off, -1).sum() / off.sum()) if off.sum() else None  # pred earlier phase than truth
    return out


def timing_stats(per_video: list[dict]) -> dict:
    errs: dict[str, list[float]] = {}
    for pv in per_video:
        for ph, gt in pv["gt_times"].items():
            if ph in pv["pred_times"]:
                errs.setdefault(ph, []).append(pv["pred_times"][ph] - gt)
    out = {}
    for ph in CLASS_NAMES:
        if ph not in errs:
            continue
        e = np.asarray(errs[ph]); c = CLASS_NAMES.index(ph); th = THETA_BY_CLASS.get(c)
        out[ph] = {"n": int(len(e)), "bias_h": float(e.mean()), "median_signed_h": float(np.median(e)),
                   "mae_h": float(np.abs(e).mean()), "median_abs_h": float(np.median(np.abs(e))),
                   "mad_h": float(np.median(np.abs(e - np.median(e)))), "std_h": float(e.std()),
                   "share_late": float((e > 0).mean()), "theta_h": th,
                   "within_theta": float((np.abs(e) <= th).mean()) if th else None,
                   "within_theta_if_debiased": float((np.abs(e - np.median(e)) <= th).mean()) if th else None,
                   "p95_abs_h": float(np.percentile(np.abs(e), 95))}
    return out, errs


def calibration(seqs) -> dict:
    conf, correct, ent = [], [], []
    for s in seqs:
        lp = np.asarray(s["log_probs"], dtype=np.float64); y = np.asarray(s["labels"])
        p = np.exp(lp); conf.append(p.max(1)); correct.append(lp.argmax(1) == y); ent.append(-(p * lp).sum(1))
    conf = np.concatenate(conf); correct = np.concatenate(correct); ent = np.concatenate(ent)
    bins = np.linspace(0, 1, 11); ece = 0.0; curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean()); curve.append((float(hi), float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    # AUROC of entropy as an error detector (rank-based)
    err = ~correct
    if err.any() and (~err).any():
        r = ent.argsort().argsort().astype(float)
        auroc = float((r[err].mean() - (err.sum() - 1) / 2) / (~err).sum())
    else:
        auroc = None
    return {"ece": float(ece), "mean_conf": float(conf.mean()), "acc": float(correct.mean()), "auroc_entropy_error": auroc, "curve": curve}


def segment_stats(seqs, log_trans) -> dict:
    n_gt = n_a = n_v = skips = 0
    for s in seqs:
        y = np.asarray(s["labels"]); lp = np.asarray(s["log_probs"]); v = viterbi(lp, log_trans); a = lp.argmax(1)
        n_gt += 1 + int((y[1:] != y[:-1]).sum()); n_a += 1 + int((a[1:] != a[:-1]).sum()); n_v += 1 + int((v[1:] != v[:-1]).sum())
        skips += len(set(y.tolist()) - set(v.tolist()))
    return {"gt_segments": n_gt, "argmax_segments": n_a, "viterbi_segments": n_v, "phases_skipped_by_viterbi": skips, "n_videos": len(seqs)}


def sample_frames(feats: dict, seqs, key: str, per_video: int, rng) -> tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for s in seqs:
        f = feats.get(f"{s['video']}__{key}")
        if f is None:
            continue
        idx = rng.choice(len(f), size=min(per_video, len(f)), replace=False)
        X.append(f[idx].astype(np.float32)); Y.append(np.asarray(s["labels"])[idx])
    return (np.concatenate(X), np.concatenate(Y)) if X else (np.zeros((0, 1)), np.zeros(0, int))


def linear_probe(feats_val, seqs_val, feats_test, seqs_test, key: str, rng) -> dict | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr = sample_frames(feats_val, seqs_val, key, 200, rng)
    Xte, yte = sample_frames(feats_test, seqs_test, key, 200, rng)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=0.5, n_jobs=8).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = float((pred == yte).mean())
    # adjacent-pair separability: binary probe restricted to frames of phases (c, c+1)
    pairs = {}
    for c in range(NUM_CLASSES - 1):
        mtr = np.isin(ytr, [c, c + 1]); mte = np.isin(yte, [c, c + 1])
        if mtr.sum() < 40 or mte.sum() < 20 or len(np.unique(ytr[mtr])) < 2 or len(np.unique(yte[mte])) < 2:
            continue
        b = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(Xtr[mtr]), ytr[mtr])
        pairs[f"{CLASS_NAMES[c]}|{CLASS_NAMES[c + 1]}"] = {"acc": float((b.predict(sc.transform(Xte[mte])) == yte[mte]).mean()),
                                                            "n_test": int(mte.sum()),
                                                            "chance": float(max(np.bincount(yte[mte]).max() / mte.sum(), 0))}
    return {"test_acc": acc, "n_train": int(len(ytr)), "n_test": int(len(yte)), "dim": int(Xtr.shape[1]), "adjacent_pairs": pairs}


def embedding_plot(feats, seqs, key: str, path: Path, rng, title: str) -> dict | None:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    X, y = sample_frames(feats, seqs, key, 60, rng)
    if len(X) < 100:
        return None
    Z50 = PCA(n_components=min(50, X.shape[1] - 1)).fit_transform(X)
    sil = float(silhouette_score(Z50, y, sample_size=min(5000, len(y)), random_state=0))
    Z = Z50[:, :2]
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=3, alpha=0.6, vmin=0, vmax=NUM_CLASSES - 1)
    cb = fig.colorbar(sc, ticks=range(NUM_CLASSES)); cb.ax.set_yticklabels(CLASS_NAMES)
    ax.set_title(f"{title}\nPCA-2 of {key} features, silhouette(PCA-50)={sil:.3f}"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return {"silhouette_pca50": sil, "n": int(len(y))}


def plot_confusion(cm_a, cm_v, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, cm, name in zip(axes, (cm_a, cm_v), ("argmax", "Viterbi")):
        cm = np.asarray(cm, float); row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7); ax.set_yticklabels(CLASS_NAMES, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("truth"); ax.set_title(f"{name} (row-normalised)")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if row[i, j] >= 0.05 and i != j:
                    ax.text(j, i, f"{row[i, j]:.2f}", ha="center", va="center", fontsize=5, color="red")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_timing(errs: dict, path: Path, title: str) -> None:
    phases = [p for p in CLASS_NAMES if p in errs]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.boxplot([np.clip(errs[p], -12, 12) for p in phases], showfliers=False, tick_labels=phases)
    for i, p in enumerate(phases, 1):
        e = np.asarray(errs[p]); ax.scatter(np.full(len(e), i) + (np.random.rand(len(e)) - 0.5) * 0.3, np.clip(e, -12, 12), s=3, alpha=0.3)
        th = THETA_BY_CLASS.get(CLASS_NAMES.index(p))
        if th:
            ax.plot([i - 0.4, i + 0.4], [th, th], "r--", lw=0.7); ax.plot([i - 0.4, i + 0.4], [-th, -th], "r--", lw=0.7)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("pred - gt (h), clipped ±12"); ax.set_title(f"{title}\nsigned onset error per event (red dashed = ±θ)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_error_vs_distance(evd: dict, path: Path, title: str) -> None:
    keys = list(evd); fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [evd[k]["err_argmax"] or 0 for k in keys], 0.4, label="argmax")
    ax.bar(x + 0.2, [evd[k]["err_viterbi"] or 0 for k in keys], 0.4, label="Viterbi")
    ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_xlabel("frames to nearest GT transition"); ax.set_ylabel("frame error rate")
    ax.legend(); ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------------------------------- report
def fmt(x, spec=".3f"):
    return "--" if x is None else format(x, spec)


def write_report(run_name: str, S: dict, out: Path) -> None:
    m = S["metrics_test"]; L = [f"# Diagnostics: `{run_name}`", "",
         f"eval window {S['eval_window'][0]}/{S['eval_window'][1]}; test videos {m['n_videos']}; "
         f"p={m['p']:.3f} p_v={m['p_v']:.3f} p_t={m['p_t']:.3f} edit={m['edit']:.1f} F1@50={m['f1']['50']:.1f} MAE={m['mae_h_all']:.2f}h", ""]
    L += ["## 1. Where are the frame errors? (test, Viterbi)", "", "| frames to nearest GT transition | n | err argmax | err Viterbi | share of all Viterbi errors |", "|---|---|---|---|---|"]
    for k, v in S["error_vs_distance"].items():
        L.append(f"| {k} | {v['n_frames']} | {fmt(v['err_argmax'])} | {fmt(v['err_viterbi'])} | {fmt(v['share_of_all_errors_viterbi'])} |")
    seg = S["segments"]
    L += ["", f"segments: GT {seg['gt_segments']}, argmax {seg['argmax_segments']}, Viterbi {seg['viterbi_segments']}; phases skipped by Viterbi: {seg['phases_skipped_by_viterbi']}", ""]
    L += ["## 2. Confusion (test)", "", f"adjacent-phase share of Viterbi errors: {fmt(S['prf_viterbi']['_adjacent_share_of_errors'])}; "
          f"share predicting an *earlier* phase than truth: {fmt(S['prf_viterbi']['_under_prediction_share'])}", "",
          "| phase | support | P argmax | R argmax | P Viterbi | R Viterbi | F1 Viterbi |", "|---|---|---|---|---|---|---|"]
    for ph in CLASS_NAMES:
        a, v = S["prf_argmax"][ph], S["prf_viterbi"][ph]
        L.append(f"| {ph} | {a['support']} | {fmt(a['precision'])} | {fmt(a['recall'])} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['f1'])} |")
    L += ["", "![confusion](confusion_test.png)", ""]
    L += ["## 3. Timing errors per event (test, Viterbi + skip-fill)", "", "| event | n | bias (h) | median signed | MAE | median abs | MAD | share late | θ | within θ | within θ if debiased | p95 abs |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ph, t in S["timing"].items():
        L.append(f"| {ph} | {t['n']} | {t['bias_h']:+.2f} | {t['median_signed_h']:+.2f} | {t['mae_h']:.2f} | {t['median_abs_h']:.2f} | {t['mad_h']:.2f} | {t['share_late']:.2f} | {fmt(t['theta_h'], '.2f')} | {fmt(t['within_theta'], '.2f')} | {fmt(t['within_theta_if_debiased'], '.2f')} | {t['p95_abs_h']:.1f} |")
    L += ["", "![timing](timing_test.png)", ""]
    c = S["calibration"]; e = S["entropy"]
    L += ["## 4. Uncertainty / calibration (test, argmax)", "", f"ECE={c['ece']:.3f} (mean conf {c['mean_conf']:.3f} vs acc {c['acc']:.3f}); AUROC(entropy → frame error)={fmt(c['auroc_entropy_error'])}; "
          f"entropy boundary(±2)={fmt(e['entropy_boundary'])} vs interior={fmt(e['entropy_interior'])} nats", ""]
    L += ["## 5. Representation (val → test linear probe on frozen features)", ""]
    for key, name in (("bb", "backbone"), ("hd", "head")):
        pr = S.get(f"probe_{key}")
        if pr:
            L += [f"**{name}** (dim {pr['dim']}): probe test acc = {pr['test_acc']:.3f} (model argmax p = {m['p']:.3f}); silhouette(PCA-50) = {fmt((S.get(f'emb_{key}') or {}).get('silhouette_pca50'))}", "",
                  "| adjacent pair | probe acc | chance | n test |", "|---|---|---|---|"]
            for pair, v in pr["adjacent_pairs"].items():
                L.append(f"| {pair} | {v['acc']:.3f} | {v['chance']:.3f} | {v['n_test']} |")
            L += ["", f"![emb_{key}](embeddings_{key}_test.png)", ""]
    L += ["## 6. Worst test videos (Viterbi accuracy)", "", "| video | acc_v | acc | frames | far/transitions |", "|---|---|---|---|---|"]
    for w in S["worst_test"]:
        L.append(f"| {w['video']} | {w['acc_viterbi']:.3f} | {w['acc']:.3f} | {w['n_frames']} | {w['n_far']}/{w['n_transitions']} |")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out_root", default="results/diagnostics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--no_feats", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="skip inference if frame_probs/features npz already exist")
    a = ap.parse_args()
    device = torch.device(a.device)
    rng = np.random.default_rng(0)
    compare = []
    for run in a.runs:
        run = ROOT / run; name = run.name; out = ROOT / a.out_root / name; out.mkdir(parents=True, exist_ok=True)
        log_trans = np.load(run / "transition_log_matrix.npy")
        data = {}
        for part in ("val", "test"):
            fp, ff = out / f"frame_probs_{part}.npz", out / f"features_{part}.npz"
            if a.reuse and fp.exists():
                z = np.load(fp); vids = sorted({k.rsplit("__", 1)[0] for k in z.files})
                seqs = [{"video": v, "log_probs": z[f"{v}__lp"].astype(np.float32), "labels": z[f"{v}__y"].astype(int), "times_h": z[f"{v}__t"]} for v in vids]
                feats = dict(np.load(ff)) if ff.exists() else {}
                cfg = json.loads((run / "config.resolved.json").read_text()); t = cfg.get("training", {})
                win = (int(t.get("eval_len", 150)), int(t.get("eval_stride", t.get("eval_len", 150))))
            else:
                cfg, seqs, feats, win = collect(run, part, device, a.num_workers, not a.no_feats)
                diag.dump_frame_probs(seqs, fp)
                if feats:
                    np.savez_compressed(ff, **feats)
            data[part] = (seqs, feats)
            print(f"{name} {part}: {len(seqs)} videos, feats={'yes' if feats else 'no'}")
        seqs_v, feats_v = data["val"]; seqs_t, feats_t = data["test"]
        S: dict = {"run": name, "eval_window": win}
        mt = evaluate_videos(seqs_t, log_trans); S["metrics_test"] = {k: v for k, v in mt.items() if k != "per_video"}
        mv = evaluate_videos(seqs_v, log_trans); S["metrics_val"] = {k: v for k, v in mv.items() if k != "per_video"}
        cms = diag.confusion_matrices(seqs_t, log_trans); S["confusion_test"] = cms
        S["prf_argmax"] = per_phase_prf(cms["argmax"]); S["prf_viterbi"] = per_phase_prf(cms["viterbi"])
        S["error_vs_distance"] = error_vs_distance(seqs_t, log_trans)
        S["segments"] = segment_stats(seqs_t, log_trans)
        S["timing"], errs = timing_stats(mt["per_video"])
        S["calibration"] = calibration(seqs_t); S["entropy"] = diag.boundary_entropy(seqs_t)
        S["worst_test"] = diag.worst_videos(mt["per_video"], 8)
        title = name
        plot_confusion(cms["argmax"], cms["viterbi"], out / "confusion_test.png", title)
        plot_timing(errs, out / "timing_test.png", title)
        plot_error_vs_distance(S["error_vs_distance"], out / "error_vs_distance_test.png", title)
        for key in ("bb", "hd"):
            if any(k.endswith(f"__{key}") for k in feats_t):
                S[f"probe_{key}"] = linear_probe(feats_v, seqs_v, feats_t, seqs_t, key, rng)
                S[f"emb_{key}"] = embedding_plot(feats_t, seqs_t, key, out / f"embeddings_{key}_test.png", rng, title)
                print(f"  probe {key}: {S[f'probe_{key}']['test_acc']:.3f}")
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
        write_report(name, S, out)
        compare.append(S)
        print(f"  -> {out / 'REPORT.md'}")

    if len(compare) > 1:
        L = ["# TEMPO diagnostics: cross-run comparison", "", "| run | p | p_v | p_t | Edit | F1@50 | MAE h | err@boundary(0-2) | err@interior(>20) | adj. share | under-pred share | probe bb | probe hd | sil bb | sil hd | ECE | H_bnd/H_int |", "|" + "---|" * 17]
        for S in compare:
            m = S["metrics_test"]; ev = S["error_vs_distance"]
            eb = np.mean([ev[k]["err_viterbi"] for k in ("0", "1", "2") if ev[k]["err_viterbi"] is not None])
            ei = ev[">20"]["err_viterbi"]; e = S["entropy"]
            L.append(f"| {S['run']} | {m['p']:.3f} | {m['p_v']:.3f} | {m['p_t']:.3f} | {m['edit']:.1f} | {m['f1']['50']:.1f} | {m['mae_h_all']:.2f} | {eb:.3f} | {fmt(ei)} | "
                     f"{fmt(S['prf_viterbi']['_adjacent_share_of_errors'], '.2f')} | {fmt(S['prf_viterbi']['_under_prediction_share'], '.2f')} | "
                     f"{fmt((S.get('probe_bb') or {}).get('test_acc'))} | {fmt((S.get('probe_hd') or {}).get('test_acc'))} | "
                     f"{fmt((S.get('emb_bb') or {}).get('silhouette_pca50'))} | {fmt((S.get('emb_hd') or {}).get('silhouette_pca50'))} | {S['calibration']['ece']:.3f} | "
                     f"{fmt(e['entropy_boundary'], '.2f')}/{fmt(e['entropy_interior'], '.2f')} |")
        L += ["", "## Timing bias per event (h, pred - gt, median signed)", "", "| run | " + " | ".join(p for p in CLASS_NAMES if p in compare[0]["timing"]) + " |"]
        L.append("|" + "---|" * (1 + len(compare[0]["timing"])))
        for S in compare:
            L.append(f"| {S['run']} | " + " | ".join(f"{S['timing'][p]['median_signed_h']:+.2f}" for p in CLASS_NAMES if p in S["timing"]) + " |")
        L += ["", "## within-θ as-is → if per-event median bias were removed", "", "| run | " + " | ".join(p for p in CLASS_NAMES if p in compare[0]["timing"]) + " |", "|" + "---|" * (1 + len(compare[0]["timing"]))]
        for S in compare:
            L.append(f"| {S['run']} | " + " | ".join(f"{fmt(S['timing'][p]['within_theta'], '.2f')}→{fmt(S['timing'][p]['within_theta_if_debiased'], '.2f')}" for p in CLASS_NAMES if p in S["timing"]) + " |")
        (ROOT / a.out_root / "COMPARE.md").write_text("\n".join(L) + "\n")
        print(f"-> {ROOT / a.out_root / 'COMPARE.md'}")


if __name__ == "__main__":
    main()
