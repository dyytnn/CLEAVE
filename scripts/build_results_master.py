#!/usr/bin/env python
"""Single source of truth for every number in the CLEAVE paper.

Reads every ``runs/h7/*_seed*/{results.json, config.resolved.json, per_video_test.json}`` once and writes:

  results/runs_master.csv                    one row per run (Supplementary Data 1)
  outputs/tables/ed_table1.tex    grouped configuration summary: one row per configuration x split,
                                             n seeds, mean +- sample s.d. (ddof=1) per metric, human-readable columns
  outputs/tables/table_folds.tex  main Table 2: five-fold paired comparison, 3 seeds per cell, fold 0 marked
                                             as selection fold, folds 1-4 confirmatory statistic, pooled per-video bootstrap
  outputs/tables/ed_table_negative.tex  negative-results compendium (one row per idea, Delta vs its own reference,
                                             seed-matched paired bootstrap over videos where the split is shared)
  outputs/tables/ed_table_hsmm.tex      semi-Markov decoder summary over all checkpoints (from results/hsmm_leaderboard.csv)
  outputs/numbers.tex             \\newcommand macros for every in-text number, so text and tables cannot diverge

Conventions (stated once here and in every caption): mean +- sample s.d. over seeds (ddof=1); paired comparisons pair
seed i with seed i on the same split and bootstrap over videos (and, for the five-fold statistic, over seeds jointly);
two-tailed tests, alpha = 0.05.
"""
from __future__ import annotations

import hashlib
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from stseg.data.patient import patient_of

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs/tables"
TAB.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
NBOOT = 10_000

BACKBONE_NAMES = {"resnet18": "ResNet-18", "resnet34": "ResNet-34", "resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-T", "efficientnet_b0": "EfficientNet-B0",
                  "efficientnet_v2_s": "EfficientNet-V2-S", "efficientnet_v2_l": "EfficientNet-V2-L", "swin_t": "Swin-T", "vit_b_16": "ViT-B/16",
                  "kinetic_cnn": "small CNN (scratch)", "crossfocal": "cross-focal", "focalattn": "focus-adaptive", "dinov2_vitb14": "DINOv2-B/14"}
HEAD_NAMES = {"none": "per-frame", "lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "transformer", "transformer_relpos": "transformer (rotary)",
              "mamba": "selective SSM", "node": "ODE-RNN", "diffact_lite": "diffusion refinement", "asformer_lite": "ASFormer-lite", "mstcn": "MS-TCN"}
LOSS_NAMES = {"ce": "CE", "ce_weighted": "weighted CE", "ce_ordinal": "ordinal CE", "ce_boundary_weighted": "boundary-weighted CE", "ce_cellcount": "CE + cell-count"}


def split_family(cfg: dict) -> tuple[str, str]:
    d = cfg["data"]; sp = Path(d["split"]).stem
    if d.get("split_mode") == "image":
        return "image-level (frames of every video split 70/10/20)", "image"
    if sp.startswith("split"):
        return f"official fold {sp[-1]}", f"fold{sp[-1]}"
    if "grouped" in sp:
        # v1 keeps the bare key it has always had; a later version gets its own so the two are never pooled
        ver = sp.rsplit("grouped_", 1)[-1] if "grouped_" in sp else "v1"
        if ver != "v1":
            return f"patient-grouped {ver} (CLEAVE)", f"grouped_{ver}"
        return "patient-grouped (CLEAVE)", "grouped"
    if "random" in sp:
        return "random 70/10/20 video split (EmbryoDiff-style)", "random73"
    return sp, sp


def describe(cfg: dict) -> dict:
    m, d, t = cfg["model"], cfg["data"], cfg.get("training", {})
    bb = m.get("backbone", {}); hd = m.get("head", {"name": "none"})
    if m.get("type") == "r2plus1d":
        backbone, head = "R(2+1)D-18", "3D CNN"
    else:
        backbone = BACKBONE_NAMES.get(bb.get("name"), bb.get("name"))
        if bb.get("name") in ("crossfocal", "focalattn"):
            backbone = f"{BACKBONE_NAMES.get(bb.get('cnn', 'resnet18'))} + {backbone} attention"
            if bb.get("name") == "focalattn":
                backbone += " (" + "+".join(k for k in ("spatial", "sharpness") if bb.get(k, True)) + ")"
        head = HEAD_NAMES.get(hd.get("name"), hd.get("name"))
    planes = len(d["planes"]) if d.get("planes") else 1
    L = int(d.get("clip_len", 4)) if hd.get("name") != "none" else 1
    ev = int(t.get("eval_len", 150)); st = int(t.get("eval_stride", ev))
    window = f"{ev}/{st}"
    fam, fam_key = split_family(cfg)
    extras = []
    if d.get("resize", 250) != 250: extras.append(f"{d.get('crop')}px")
    if "recipe_e20" in cfg.get("experiment", {}).get("id", ""): extras.append("tuned recipe (Methods)")
    elif int(t.get("epochs", 10)) != 10: extras.append(f"{int(t.get('epochs', 10))} epochs")
    if d.get("photometric"): extras.append("photometric aug")
    if d.get("instance_norm"): extras.append("instance norm")
    if m.get("aux"): extras.append("cell-count aux")
    if bb.get("init_from"): extras.append("MAE init")
    if d.get("merge_last_class"): extras.append("15 classes")
    if cfg.get("optimizer", {}).get("backbone_lr_mult"): extras.append("fair recipe")
    return {"backbone": backbone, "head": head, "clip_len": L, "planes": planes, "window": window,
            "loss": LOSS_NAMES.get(cfg.get("loss", {}).get("name", "ce"), cfg.get("loss", {}).get("name")),
            "optimizer": cfg.get("optimizer", {}).get("name", "sgd"), "epochs": int(t.get("epochs", 10)),
            "split_family": fam, "split_key": fam_key, "extras": "; ".join(extras)}


# Paper 1 (the measurement/protocol paper) reports a frozen sweep. Rungs run after the two-paper split are excluded
# here and reported separately: the division-signal line (dualbranch / divchannel / divconsist) depends on
# external artefacts Paper 1 never introduces (N3 detector weights, embryo-crop and division-score caches) and would
# add a tenth backbone family and a fourth loss, contradicting the enumeration in main.tex ("nine backbone families,
# seven temporal-head families, three losses"); Sampling/position variants (planeaugbias, rareboost, L32) were already
# in the reviewed sweep and stay. Removing an entry from this list silently changes a published count.
PAPER2_ONLY = ("dualbranch", "divchannel", "divconsist")


def load_runs() -> pd.DataFrame:
    rows = []
    for rd in sorted(glob.glob(str(ROOT / "runs/h7/*_seed*"))):
        rd = Path(rd)
        if not (rd / "results.json").exists():
            continue
        if any(tag in rd.name for tag in PAPER2_ONLY):
            continue
        # not reported: v37 (stopped, split from an in-sample audit) and the five out-of-fold audit classifiers
        if "_trainclean2_" in rd.name or "_oof_audit_" in rd.name:
            continue
        cfg = json.loads((rd / "config.resolved.json").read_text()); res = json.loads((rd / "results.json").read_text()); t = res["test"]
        exp, _, seed = rd.name.rpartition("_seed")
        rows.append({"run": rd.name, "experiment": exp, "seed": int(seed), **describe(cfg), "best_epoch": res.get("best_epoch"),
                     "p": t["p"], "p_v": t["p_v"], "p_t": t["p_t"], "edit": t.get("edit"), "f1_50": (t.get("f1") or {}).get("50"),
                     "mae_h": t.get("mae_h_all"), "n_videos": t["n_videos"], "val_p_t": res.get("val", {}).get("p_t"), "val_p_v": res.get("val", {}).get("p_v")})
    return pd.DataFrame(rows)


def per_video(run: str) -> dict[str, tuple[float, float]]:
    """video -> (p_t contribution = 1 - far/transitions, acc_viterbi)."""
    out = {}
    for v in json.loads((ROOT / "runs/h7" / run / "per_video_test.json").read_text()):
        out[v["video"]] = ((v["n_transitions"] - v["n_far"]) / v["n_transitions"] if v["n_transitions"] else np.nan, v["acc_viterbi"])
    return out


def paired_videos(exp_a: str, exp_b: str, df: pd.DataFrame, metric: int = 0) -> np.ndarray | None:
    """Seed-matched paired per-video differences a-b (videos x matched seeds, flattened after averaging over seeds)."""
    ra = df[df.experiment == exp_a]; rb = df[df.experiment == exp_b]
    seeds = sorted(set(ra.seed) & set(rb.seed))
    if not seeds:
        return None
    diffs = {}
    for s in seeds:
        pa = per_video(ra[ra.seed == s].run.iloc[0]); pb = per_video(rb[rb.seed == s].run.iloc[0])
        for v in pa:
            if v in pb and not (np.isnan(pa[v][metric]) or np.isnan(pb[v][metric])):
                diffs.setdefault(v, []).append(pa[v][metric] - pb[v][metric])
    return np.array([np.mean(x) for x in diffs.values()])


def paired_videos_labelled(exp_a: str, exp_b: str, df: pd.DataFrame, metric: int = 0):
    """As ``paired_videos`` but returns (video_ids, differences) so the patient cluster is recoverable."""
    ra = df[df.experiment == exp_a]; rb = df[df.experiment == exp_b]
    seeds = sorted(set(ra.seed) & set(rb.seed))
    if not seeds:
        return None, None
    diffs: dict[str, list[float]] = {}
    for s in seeds:
        pa = per_video(ra[ra.seed == s].run.iloc[0]); pb = per_video(rb[rb.seed == s].run.iloc[0])
        for v in pa:
            if v in pb and not (np.isnan(pa[v][metric]) or np.isnan(pb[v][metric])):
                diffs.setdefault(v, []).append(pa[v][metric] - pb[v][metric])
    ids = list(diffs)
    return ids, np.array([np.mean(diffs[v]) for v in ids])


def boot_ci(d: np.ndarray, n: int = NBOOT) -> tuple[float, float, float]:
    bs = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cluster_boot_ci(d: np.ndarray, videos: list[str], n: int = NBOOT) -> tuple[float, float, float, int]:
    """Patient-clustered bootstrap: resample PATIENTS with replacement, not videos.

    This paper's own argument is that the patient, not the video, is the independent unit; a
    video-level resample would contradict it wherever a patient contributes several test videos
    (13 of 78 patients on the grouped split, up to 4 videos each). Returns the extra cluster count
    so the interval can be reported with its true n."""
    pat = np.array([patient_of(v) for v in videos])
    groups = {p: d[pat == p] for p in np.unique(pat)}
    keys = list(groups)
    bs = np.empty(n)
    for i in range(n):
        drawn = [groups[keys[j]] for j in RNG.integers(0, len(keys), len(keys))]
        bs[i] = np.concatenate(drawn).mean()
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(keys)


def mdd(sd_diff: float, n: int, power: float = 0.80, alpha: float = 0.05) -> tuple[float, float, float]:
    """Minimum detectable paired difference for a two-sample-per-video paired comparison: the smallest true mean
    difference that a test of this size has ``power`` chance of detecting at level ``alpha`` (two-tailed), given the
    observed standard deviation of the per-video paired differences. Standard normal-approximation power-analysis
    formula (e.g. Cohen 1988): MDD = (z_{alpha/2} + z_beta) * sd_diff / sqrt(n). Returns (z_alpha/2, z_beta, mdd)."""
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    return z_a, z_b, (z_a + z_b) * sd_diff / np.sqrt(n)


def seed_arrays(exp: str, df: pd.DataFrame) -> dict[str, np.ndarray] | None:
    """video -> per-video p_t of every seed of ``exp`` (seed order), for videos scored in every seed; None if < 3 seeds."""
    runs = df[df.experiment == exp].sort_values("seed")
    if len(runs) < 3:
        return None
    acc: dict[str, list[float]] = {}
    for r in runs.run:
        for v, (pt, _) in per_video(r).items():
            if not np.isnan(pt):
                acc.setdefault(v, []).append(pt)
    return {v: np.array(x) for v, x in acc.items() if len(x) == len(runs)}


def pooled_step(pairs: list[tuple[str, str]], df: pd.DataFrame, n: int = NBOOT, seed: int = 39) -> dict | None:
    """Paired difference A - B pooled over one or more test partitions (one (A, B) experiment pair per partition).

    Point: mean over the pooled test videos of the seed-mean difference. Interval: two-level bootstrap -- patients are
    resampled with replacement within each partition, and within every replicate the seeds of each arm are resampled
    independently (the seed-aware design of the cleaned-training comparison), so the interval carries seed as well as
    patient variance. Also returns the per-partition difference of the run-level means and how many are positive."""
    parts = []
    for a_exp, b_exp in pairs:
        A, B = seed_arrays(a_exp, df), seed_arrays(b_exp, df)
        if A is None or B is None:
            return None
        vids = sorted(set(A) & set(B))
        pat = np.array([patient_of(v) for v in vids])
        groups = [np.where(pat == p)[0] for p in np.unique(pat)]
        parts.append((np.stack([A[v] for v in vids]), np.stack([B[v] for v in vids]), groups, len(groups)))
    point = float(np.concatenate([a.mean(1) - b.mean(1) for a, b, _, _ in parts]).mean())
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for i in range(n):
        tot, cnt = 0.0, 0
        for a, b, groups, _ in parts:
            idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            sa, sb = rng.integers(0, a.shape[1], a.shape[1]), rng.integers(0, b.shape[1], b.shape[1])
            tot += float((a[idx][:, sa].mean(1) - b[idx][:, sb].mean(1)).sum())
            cnt += len(idx)
        bs[i] = tot / cnt
    fd = np.array([df[df.experiment == a].p_t.mean() - df[df.experiment == b].p_t.mean() for a, b in pairs])
    return {"d": point, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "npat": sum(p[3] for p in parts), "nvid": sum(len(p[0]) for p in parts),
            "part_d": fd, "wins": int((fd > 0).sum()), "nparts": len(pairs)}


def fmt_pm(x: pd.Series, nd: int = 3) -> str:
    x = x.dropna()
    if len(x) == 0: return "--"
    if len(x) == 1: return f"{x.iloc[0]:.{nd}f}"
    return f"${x.mean():.{nd}f}\\pm{x.std(ddof=1):.{nd}f}$"


def is_common_recipe(df: pd.DataFrame) -> pd.Series:
    """The common training recipe of the Methods: 10 epochs and not one of the tuned-recipe runs."""
    return (df.epochs == 10) & ~df.experiment.str.contains("recipe", regex=False)


def footer_to_notes(lines: list[str]) -> tuple[list[str], list[str]]:
    """Move full-width ``\\multicolumn{..}{l}{\\footnotesize ...}`` footer rows out of a tabular. Inside it they cannot
    wrap, so they set the table's width and \\resizebox then shrinks every cell to ~5 pt (round-4/5 review)."""
    keep, notes = [], []
    for ln in lines:
        m = re.match(r"\\multicolumn\{\d+\}\{l\}\{\\footnotesize (.*)\}\\\\$", ln)
        if m:
            prev = next((k for k in reversed(keep) if " & " in k), "")
            label = prev.split(" & ")[0].strip() if prev.split(" & ")[0].strip().startswith("$\\Delta$") else ""
            notes.append((label + ": " if label else "") + m.group(1))
        else:
            keep.append(ln)
    while keep and keep[-1] == "\\midrule":
        keep.pop()
    return keep, notes


def notes_block(notes: list[str]) -> list[str]:
    return ["\\par\\smallskip\\begin{minipage}{\\textwidth}\\footnotesize " + " ".join(n.rstrip(".") + "." for n in notes)
            + "\\end{minipage}"] if notes else []


def esc(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


# ------------------------------------------------------------------------------------------------- ED Table 1 (grouped)
def write_ed_table1(df: pd.DataFrame) -> None:
    g = df.groupby(["experiment"], sort=False)
    order = {"fold0": 0, "fold1": 1, "fold2": 1, "fold3": 1, "fold4": 1, "grouped": 2, "grouped_v2": 3,
             "grouped_v2_trainclean_v1": 3.5, "grouped_v2_trainclean_v3": 3.6, "random73": 4, "image": 5}
    blocks = {0: "Official fold 0 (selection fold for every ablation)", 1: "Official folds 1--4",
              2: "Patient-grouped split, cycle level (\\texttt{nantes\\_grouped\\_v1})",
              3: "Patient-grouped split, couple level (\\texttt{nantes\\_grouped\\_v2}, recommended)",
              3.5: "The same split without every video of the first, untiered defect log in training (\\texttt{nantes\\_grouped\\_v2\\_trainclean\\_v1})",
              3.6: "The same split without tier A of the defect log in training, tier-B labels cut (\\texttt{nantes\\_grouped\\_v2\\_trainclean\\_v3})",
              4: "Random 70/10/20 video split, 15 classes (EmbryoDiff-style)",
              5: "Image-level split (frames of every video in train and test; leakage demonstration, not comparable)"}
    # the optimiser follows the head family in the common recipe (Methods); v39 showed that it, not the head, carries the
    # reference -> transformer step, so every row whose optimiser differs from its head family's default is marked
    fam_opt = df[is_common_recipe(df)].groupby("head").optimizer.agg(lambda x: x.value_counts().index[0]).to_dict()
    rows = []
    for exp, r in g:
        r0 = r.iloc[0]
        tag = hashlib.sha1(str(exp).encode()).hexdigest()[:4]
        extras = str(r0.extras).strip() if isinstance(r0.extras, str) else ""
        if fam_opt.get(r0["head"]) and r0.optimizer != fam_opt[r0["head"]]:
            extras = "; ".join(x for x in (extras, {"adamw": "AdamW recipe", "sgd": "SGD recipe"}[r0.optimizer]) if x)
        extras_cell = f"{extras} [{tag}]" if extras else f"[{tag}]"
        rows.append((order[r0.split_key], r0.split_key, r0.backbone, r0["head"], r0.clip_len, r0.planes, r0.window, r0.loss, extras_cell, len(r), int(r0.n_videos),
                     fmt_pm(r.p), fmt_pm(r.p_v), fmt_pm(r.p_t), -r.p_t.mean()))
    rows.sort(key=lambda x: (x[0], x[1], x[-1]))
    cap = ("\\textbf{Supplementary Table 1 | Every trained configuration, grouped.} One row per configuration $\\times$ split; values are mean $\\pm$ sample "
           "s.d.\\ over $n$ seeds (single value when $n=1$). Window: evaluation window length/stride in frames (150/150 is the released protocol's "
           "non-overlapping chunking; $L$/$L{/}2$ the corrected rule). Planes: number of focal planes in the input. The optimiser follows the head family (Methods): "
           + ", ".join(f"{h} {o.replace('adamw', 'AdamW').replace('sgd', 'SGD')}" for h, o in sorted(fam_opt.items())) + "; a row that departs from its family's optimiser says so under Extras. "
           "Extras: deviations from the reference recipe, followed in brackets by a four-character configuration hash that identifies the run directory uniquely and matches the per-run table released with the code. "
           "Every row of this table is exploratory with respect to fold 0 (Statistical analysis, Methods of the article); inference is reserved for Table~3 of the article. "
           "Edit score, F1@50, MAE, per-seed rows, validation metrics and configuration hashes are in the per-run table released with the code.")
    # 12 columns (Edit/F1@50/MAE dropped to Supplementary Data 1 -- see caption), narrow L-columns, tiny font, tight tabcolsep:
    # this table has to fit \textwidth on both venues (Nature ~16.8cm at margin 2.2cm, MedIA ~16.2cm at margin 2.4cm), and a
    # longtable can't be wrapped in \resizebox across page breaks, so column budget is set by hand, not scaled automatically.
    L = ["\\centering\\tiny\\setlength{\\tabcolsep}{2pt}",
         "\\begin{longtable}{L{1.7cm}L{1.2cm}ccL{0.9cm}L{1.0cm}L{1.3cm}ccccc}",
         "\\multicolumn{12}{p{\\textwidth}}{" + cap + "}\\\\[4pt]", "\\toprule",
         "Backbone & Head & $L$ & Planes & Window & Loss & Extras & $n$ & videos & $p$ & $p_v$ & $p_t$ \\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", "Backbone & Head & $L$ & Planes & Window & Loss & Extras & $n$ & videos & $p$ & $p_v$ & $p_t$ \\\\", "\\midrule", "\\endhead"]
    last = None
    for r in rows:
        if r[0] != last:
            L.append(f"\\multicolumn{{12}}{{l}}{{\\textit{{{blocks[r[0]]}}}}}\\\\"); last = r[0]
        fold = "" if r[1] in ("fold0", "grouped", "grouped_v2", "grouped_v2_trainclean_v1", "grouped_v2_trainclean_v3", "random73", "image") else f" ({r[1].replace('fold', 'fold ')})"
        L.append(f"{esc(r[2])}{fold} & {esc(r[3])} & {r[4]} & {r[5]} & {r[6]} & {esc(r[7])} & {esc(r[8])} & {r[9]} & {r[10]} & {r[11]} & {r[12]} & {r[13]} \\\\")
    L += ["\\bottomrule", "\\end{longtable}"]
    (TAB / "ed_table1.tex").write_text("\n".join(L) + "\n")
    print(f"ED Table 1: {len(rows)} grouped rows from {len(df)} runs")


# ------------------------------------------------------------------------------------------------- Label-quality outliers
# Three videos where every model, every seed, decodes (almost) nothing right -- not a model failure, an annotation
# defect confirmed by direct visual inspection: DRL1048-1 and LV683-2-8 never visibly divide in
# any of the seven focal planes at any timepoint although annotated as reaching t9+/tEB; CC938-4 divides normally but
# its annotated event times run 1-2 division stages ahead of what is visible (its "t4" frame shows 2 cells, its "t9+"
# frame shows 4). Excluded only from the CLEAVE sensitivity numbers below, never silently from the released-
# protocol headline (kept there for comparability with prior literature, exactly as released-protocol numbers always
# include every official test video).
LABEL_QUALITY_OUTLIERS = {"DRL1048-1", "LV683-2-8", "CC938-4"}


def _pt_excluding(run: str, exclude: set[str]) -> float:
    pv = json.loads((ROOT / "runs/h7" / run / "per_video_test.json").read_text())
    vals = [(v["n_transitions"] - v["n_far"]) / v["n_transitions"] for v in pv if v["n_transitions"] and v["video"] not in exclude]
    return float(np.mean(vals)) if vals else float("nan")


def outlier_sensitivity(df: pd.DataFrame, numbers: dict) -> None:
    """With vs without the 3 label-quality outliers, pooled the same way as the main 5-fold table (mean over seeds
    per fold, then mean over the 5 fold means) -- a sensitivity check, not a replacement of the headline number."""
    for pat, tag in [("resnet18_lstm_L4_split{k}", "ref"), ("resnet18_transformer_L16_crossfocal7_evalfix_split{k}", "cf")]:
        fold_with, fold_without = [], []
        for k in range(5):
            runs_k = df[df.experiment == pat.format(k=k)].run.tolist()
            if not runs_k:
                continue
            fold_with.append(np.mean([_pt_excluding(r, set()) for r in runs_k]))
            fold_without.append(np.mean([_pt_excluding(r, LABEL_QUALITY_OUTLIERS) for r in runs_k]))
        if len(fold_with) < 5:
            continue
        numbers[f"{tag}_pt_withoutliers"] = f"{np.mean(fold_with):.3f}"
        numbers[f"{tag}_pt_nooutliers"] = f"{np.mean(fold_without):.3f}"
        numbers[f"{tag}_pt_outlierdelta"] = f"{np.mean(fold_without) - np.mean(fold_with):+.3f}"
    numbers["nOutliers"] = str(len(LABEL_QUALITY_OUTLIERS))


# ------------------------------------------------------------------------------------------------- Table 2 (five folds)
FOLD_MODELS = [("resnet18_lstm_L4_split{k}", "ResNet-18-LSTM, 4 fr., 1 plane (reference)", "ref"),
               ("resnet18_lstm_L16_win_split{k}", "ResNet-18-LSTM, 16 fr., 1 plane, corrected window", "lstmwin"),
               ("resnet18_lstm_L16_win_adamw_split{k}", "ResNet-18-LSTM, 16 fr., 1 plane, corrected window, AdamW recipe", "lstmwinadamw"),
               ("r2plus1d_L8_split{k}", "R(2+1)D-18 (released 3D baseline; 8 fr.\\ on fold 0, 4 fr.\\ on folds 1--4)", "r2p1d"),
               ("resnet18_lstm_L4_multifocal3_split{k}", "ResNet-18-LSTM, 4 fr., 3 planes stacked", "mf3lstm"),
               ("resnet18_transformer_L16_evalfix_split{k}", "Transformer, 16 fr., 1 plane (control E3)", "tr1"),
               ("resnet18_transformer_L16_multifocal3_evalfix_split{k}", "Transformer, 16 fr., 3 planes stacked (control E4)", "tr3"),
               ("resnet18_transformer_L16_multifocal7_evalfix_split{k}", "Transformer, 16 fr., 7 planes stacked (plane-matched control)", "mfSeven"),
               ("resnet18_transformer_L16_crossfocal7_evalfix_split{k}", "Transformer, 16 fr., 7 planes, cross-focal", "cf")]
#: rows that carry a paired-vs-reference line in Table 2; the other rows enter through the attribution block
FOLD_PAIRED = ("cf",)


def fold_table(df: pd.DataFrame, numbers: dict) -> None:
    def cell(exp, key):
        r = df[df.experiment == exp]
        return r[key].dropna()
    lines = []
    stats_rows = []
    for pat, label, tag in FOLD_MODELS:
        cells, n_seeds, means = [], [], {"p_t": [], "p_v": [], "f1_50": []}
        for k in range(5):
            pt = cell(pat.format(k=k), "p_t")
            if len(pt) == 0:
                cells.append("--"); n_seeds.append(0); [means[m].append(np.nan) for m in means]; continue
            cells.append(f"{pt.mean():.3f}" + (f"\\,{{\\scriptsize$\\pm${pt.std(ddof=1):.3f}}}" if len(pt) > 1 else ""))
            n_seeds.append(len(pt))
            for m in means: means[m].append(cell(pat.format(k=k), m).mean())
        if all(n == 0 for n in n_seeds):
            continue
        arr = {m: np.array(v) for m, v in means.items()}
        ok = ~np.isnan(arr["p_t"])
        ns = "/".join(str(n) for n in n_seeds) if len(set(n_seeds)) > 1 else str(n_seeds[0])
        summ = " & ".join(f"${np.nanmean(arr[m]):.{3 if m != 'f1_50' else 1}f}\\pm{np.nanstd(arr[m], ddof=1):.{3 if m != 'f1_50' else 1}f}$" if ok.sum() > 1 else "--" for m in ("p_t", "p_v", "f1_50"))
        lines.append({"label": label, "ns": ns, "cells": list(cells), "summ": summ,
                      "fold_vals": [float(x) if not np.isnan(x) else None for x in arr["p_t"]],
                      "sum_val": float(np.nanmean(arr["p_t"])) if ok.sum() > 1 else None})
        stats_rows.append((tag, arr, ok))
    # bold the best model in each fold column and in the summary p_t column
    for col in range(5):
        vals = {i: r["fold_vals"][col] for i, r in enumerate(lines) if r["fold_vals"][col] is not None}
        if vals:
            b = max(vals, key=lambda i: vals[i])
            lines[b]["cells"][col] = bold_cell(lines[b]["cells"][col])
    sv = {i: r["sum_val"] for i, r in enumerate(lines) if r["sum_val"] is not None}
    if sv:
        b = max(sv, key=lambda i: sv[i])
        parts = lines[b]["summ"].split(" & ")
        parts[0] = bold_cell(parts[0])
        lines[b]["summ"] = " & ".join(parts)
    lines = [f"{r['label']} & {r['ns']} & " + " & ".join(r["cells"]) + f" & {r['summ']} \\\\" for r in lines]
    # paired statistics vs reference, per model with all five folds
    ref = next(a for t, a, ok in stats_rows if t == "ref")
    L = ["\\midrule", "\\multicolumn{10}{l}{\\textit{Paired difference to the reference on the same fold (fold means)}}\\\\"]
    full = {tag for pat, _, tag in FOLD_MODELS if all(len(df[df.experiment == pat.format(k=k)]) >= 3 for k in range(5))}
    for tag, arr, ok in stats_rows:
        if ok.sum() == 5:
            numbers[f"{tag}_fold_pt"] = f"{arr['p_t'].mean():.3f}"; numbers[f"{tag}_fold_pt_sd"] = f"{arr['p_t'].std(ddof=1):.3f}"
            numbers[f"{tag}_fold_pv"] = f"{arr['p_v'].mean():.3f}"; numbers[f"{tag}_fold_f1"] = f"{arr['f1_50'].mean():.1f}"
            numbers[f"{tag}_fold_dptref"] = f"{(arr['p_t'] - ref['p_t']).mean():+.3f}"
        if tag == "ref" or ok.sum() < 5 or tag not in full:
            continue
        d = {m: arr[m] - ref[m] for m in arr}
        show = tag in FOLD_PAIRED
        t5 = stats.ttest_rel(arr["p_t"], ref["p_t"]); t4 = stats.ttest_rel(arr["p_t"][1:], ref["p_t"][1:])
        w5 = stats.wilcoxon(arr["p_t"], ref["p_t"]).pvalue
        cells = " & ".join(f"${d['p_t'][k]:+.3f}$" for k in range(5))
        summ = f"${d['p_t'].mean():+.3f}\\pm{d['p_t'].std(ddof=1):.3f}$ & ${d['p_v'].mean():+.3f}\\pm{d['p_v'].std(ddof=1):.3f}$ & ${d['f1_50'].mean():+.1f}\\pm{d['f1_50'].std(ddof=1):.1f}$"
        label = next(l for p, l, tg in FOLD_MODELS if tg == tag)
        if show:
            L.append(f"$\\Delta$ {esc(label)} $-$ reference & & {cells} & {summ} \\\\")
        if show:
            L.append(f"\\multicolumn{{10}}{{l}}{{\\footnotesize five folds: paired $t(4)$ two-tailed $p={t5.pvalue:.4f}$, Wilcoxon $p={w5:.3f}$, $d_z={d['p_t'].mean()/d['p_t'].std(ddof=1):.1f}$; "
                 f"confirmatory folds 1--4 only: $\\Delta p_t={d['p_t'][1:].mean():+.3f}$, $t(3)$ $p={t4.pvalue:.4f}$, wins {int((d['p_t'][1:] > 0).sum())}/4}}\\\\")
        numbers[f"{tag}_fold_dpt"] = f"{d['p_t'].mean():+.3f}"; numbers[f"{tag}_fold_dpt_sd"] = f"{d['p_t'].std(ddof=1):.3f}"
        numbers[f"{tag}_fold_p5"] = f"{t5.pvalue:.4f}"; numbers[f"{tag}_fold_p14"] = f"{t4.pvalue:.3f}"; numbers[f"{tag}_fold_dpv"] = f"{d['p_v'].mean():+.3f}"
        numbers[f"{tag}_fold_df1"] = f"{d['f1_50'].mean():+.1f}"; numbers[f"{tag}_fold_wins14"] = str(int((d["p_t"][1:] > 0).sum()))
        numbers[f"{tag}_fold_pt"] = f"{arr['p_t'].mean():.3f}"; numbers[f"{tag}_fold_pt_sd"] = f"{arr['p_t'].std(ddof=1):.3f}"
        numbers[f"{tag}_fold_pv"] = f"{arr['p_v'].mean():.3f}"; numbers[f"{tag}_fold_pv_sd"] = f"{arr['p_v'].std(ddof=1):.3f}"
        numbers[f"{tag}_fold_f1"] = f"{arr['f1_50'].mean():.1f}"
    numbers["ref_fold_pt"] = f"{ref['p_t'].mean():.3f}"; numbers["ref_fold_pt_sd"] = f"{ref['p_t'].std(ddof=1):.3f}"
    numbers["ref_fold_pv"] = f"{ref['p_v'].mean():.3f}"; numbers["ref_fold_pv_sd"] = f"{ref['p_v'].std(ddof=1):.3f}"; numbers["ref_fold_f1"] = f"{ref['f1_50'].mean():.1f}"
    # pooled per-video paired difference across the five test partitions (RM-15): patients resampled within each fold and
    # seeds within each arm -- the same interval as the attribution steps below, so one contrast carries one interval
    cf_pat = "resnet18_transformer_L16_crossfocal7_evalfix_split{k}"; ref_pat = "resnet18_lstm_L4_split{k}"
    attr = attribution_numbers(df, numbers)
    tot = attr.get("total") or pooled_step([(cf_pat.format(k=k), ref_pat.format(k=k)) for k in range(5)], df)
    labelled = [paired_videos_labelled(cf_pat.format(k=k), ref_pat.format(k=k), df) for k in range(5)]
    if tot is not None and all(d is not None for _, d in labelled):
        allv = np.concatenate([d for _, d in labelled])
        numbers["cf_pooled_dpt"] = f"{tot['d']:+.3f}"; numbers["cf_pooled_lo"] = f"{tot['lo']:+.3f}"; numbers["cf_pooled_hi"] = f"{tot['hi']:+.3f}"
        numbers["cf_pooled_n"] = str(tot["nvid"]); numbers["cf_pooled_share_pos"] = f"{(allv > 0).mean() * 100:.0f}"
        numbers["cf_pooled_npat"] = str(tot["npat"])
        L.append(f"\\multicolumn{{10}}{{l}}{{\\footnotesize pooled per-video paired difference, cross-focal $-$ reference, seed-averaged, over the five test partitions ($n={tot['npat']}$ patients, {tot['nvid']} videos): "
                 f"$\\Delta p_t={tot['d']:+.3f}$ (95\\,\\% CI ${tot['lo']:+.3f}$ to ${tot['hi']:+.3f}$; patients resampled within each fold, seeds within each arm)}}\\\\")
    # attribution: one change per step along CHAIN, pooled over the five test partitions, seed-aware joint interval
    if all(t in attr for t in STEP_TAGS):
        L.append("\\midrule")
        L.append("\\multicolumn{10}{l}{\\textit{Attribution, reference to cross-focal: one change per step; cells are differences of fold means, the last column the pooled paired estimate}}\\\\")
        for tag, (_, _, label) in zip(STEP_TAGS, CHAIN[1:]):
            r = attr[tag]
            cells = " & ".join(f"${x:+.3f}$" for x in r["part_d"])
            ph = r["p_holm"]
            est = f"${r['d']:+.3f}$ [${r['lo']:+.3f}$, ${r['hi']:+.3f}$]; {r['wins']}/5; " + ("$p_{\\mathrm{Holm}}<0.001$" if ph < 0.001 else f"$p_{{\\mathrm{{Holm}}}}={ph:.2f}$")
            L.append(f"{esc(label)} & & {cells} & \\multicolumn{{3}}{{l}}{{{est}}} \\\\")
        t = attr["total"]
        L.append(f"\\multicolumn{{10}}{{l}}{{\\footnotesize attribution: per-video $p_t$ differences, seed-averaged, pooled over the five test partitions ({t['nvid']} videos of {t['ncouples']} couples, {t['npat']} couple-fold clusters); "
                 f"95\\,\\% interval from one joint bootstrap that resamples couples within each fold and the seeds of each configuration; $k/5$ = folds on which the step is positive; "
                 f"$p_{{\\mathrm{{Holm}}}}$ = bootstrap $p$ adjusted over the five steps. The steps telescope, so they sum to the total by construction and follow one order of the changes; "
                 f"every seven-plane configuration trains at batch 2, the one- and three-plane transformers at batch 4.}}\\\\")
    cap = ("\\textbf{Five official folds.} Test partition of each fold; cell = mean over seeds (\\scriptsize$\\pm$ sample s.d.). Seed counts vary by row and are given in the seeds column; the best model in each fold column and in the summary $p_t$ column is in bold. "
           "Fold~0 is the fold on which every configuration in this study was selected and is marked $^{\\dagger}$; folds 1--4 were run after selection and carry the confirmatory statistic. "
           "Summary columns: mean $\\pm$ s.d.\\ over the five fold means. Paired tests are two-tailed at $\\alpha=0.05$; the fold-level $t$-test treats folds as independent although they share training patients (Methods).")
    body_l, notes_l = footer_to_notes(lines + L)
    out = ["\\begin{table*}[t]", "\\centering\\small\\setlength{\\tabcolsep}{4pt}", "\\caption{" + cap + "}\\label{tab:folds}",
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{L{4.6cm}cccccccccc}", "\\toprule",
           "Model & seeds & fold 0$^{\\dagger}$ & fold 1 & fold 2 & fold 3 & fold 4 & $p_t$ & $p_v$ & F1@50 \\\\", "\\midrule"] + body_l + ["\\bottomrule", "\\end{tabular}}"] + notes_block(notes_l) + ["\\end{table*}"]
    (TAB / "table_folds.tex").write_text("\n".join(out) + "\n")
    print(f"Table folds: {len(lines)} model rows")


# ------------------------------------------------------------------------------------------------- ED Table 2 (negative results)
NEGATIVE = [  # (label, experiment, reference experiment, condition note)
    ("Rotary (relative-position) attention, released chunking", "resnet18_transformer_relpos_L16_split0", "resnet18_transformer_L16_evalfix_split0", "evaluated with 150-frame chunking (no window fix) -- confounded, see Methods"),
    ("Rotary attention, corrected window (E7)", "resnet18_transformer_relpos_L16_evalfix_split0", "resnet18_transformer_L16_evalfix_split0", ""),
    ("Selective state-space head (Mamba-style)", "resnet18_mamba_L16_split0", "resnet18_transformer_L16_evalfix_split0", ""),
    ("ODE-RNN head (uniform time steps; preliminary)", "resnet18_node_L16_split0", "resnet18_transformer_L16_evalfix_split0", ""),
    ("Diffusion-refinement head", "resnet18_diffactlite_L16_split0", "resnet18_transformer_L16_evalfix_split0", ""),
    ("ASFormer-lite head (re-implementation)", "resnet18_asformer_L16_evalfix_split0", "resnet18_transformer_L16_evalfix_split0", "last-stage supervision only"),
    ("MS-TCN head (re-implementation)", "resnet18_mstcn_L64_evalfix_split0", "resnet18_transformer_L16_evalfix_split0", "did not converge under this recipe"),
    ("Class-weighted cross-entropy", "resnet18_lstm_L4_ceweighted_split0", "resnet18_lstm_L4_split0", ""),
    ("Ordinal cross-entropy", "resnet18_lstm_L4_ordinal_split0", "resnet18_lstm_L4_split0", ""),
    ("Boundary-weighted cross-entropy", "resnet18_lstm_L4_boundaryloss_split0", "resnet18_lstm_L4_split0", ""),
    ("Masked-frame video pretraining (init)", "resnet18_lstm_L4_maeinit_split0", "resnet18_lstm_L4_split0", ""),
    ("Photometric augmentation + instance norm", "resnet50_lstm_L8_photo_split0", "resnet50_lstm_L8_split0", ""),
    ("Ordinal cell-count auxiliary head", "resnet50_lstm_L8_cellcount_split0", "resnet50_lstm_L8_split0", ""),
    ("Photometric + cell-count (3 seeds)", "resnet50_lstm_L8_photo_cellcount_split0", "resnet50_lstm_L8_split0", ""),
    ("Native resolution (448 px)", "resnet50_lstm_L8_hires448_split0", "resnet50_lstm_L8_split0", ""),
    ("Cross-focal, ResNet-50 per-plane encoder", "resnet50_transformer_L16_crossfocal7_evalfix_split0", "resnet18_transformer_L16_crossfocal7_evalfix_split0", "batch 1; did not converge"),
    ("Focus-adaptive: sharpness prior only", "resnet18_transformer_L16_focalattn_sharp_evalfix_split0", "resnet18_transformer_L16_crossfocal7_evalfix_split0", ""),
    ("Focus-adaptive: spatial fusion only", "resnet18_transformer_L16_focalattn_spatial_evalfix_split0", "resnet18_transformer_L16_crossfocal7_evalfix_split0", ""),
    ("Focus-adaptive: spatial + sharpness", "resnet18_transformer_L16_focalattn_spatial_sharp_evalfix_split0", "resnet18_transformer_L16_crossfocal7_evalfix_split0", ""),
]


def negative_table(df: pd.DataFrame, numbers: dict) -> None:
    rows = []
    for label, exp, ref, note in NEGATIVE:
        r = df[df.experiment == exp]; rr = df[df.experiment == ref]
        if len(r) == 0:
            continue
        d = paired_videos(exp, ref, df)
        if d is not None and len(d) > 5:
            m, lo, hi = boot_ci(d); delta = f"${m:+.3f}$ [{lo:+.3f}, {hi:+.3f}]"
        else:
            delta = f"${r.p_t.mean() - rr.p_t.mean():+.3f}$ (unpaired)" if len(rr) else "--"
        if len(rr) and r.optimizer.iloc[0] != rr.optimizer.iloc[0]:   # v39: the optimiser alone moves p_t by ~0.04
            opt = {"adamw": "AdamW", "sgd": "SGD"}
            note = "; ".join(x for x in (note, f"optimiser differs ({opt[r.optimizer.iloc[0]]} vs {opt[rr.optimizer.iloc[0]]} reference)") if x)
        rows.append(f"{esc(label)} & {len(r)} & {fmt_pm(r.p_t)} & {fmt_pm(rr.p_t)} & {delta} & {fmt_pm(r.f1_50, 1)} & {esc(note)} \\\\")
    cap = ("\\textbf{Supplementary Table 2 | Ideas that did not improve the headline metric.} Official fold 0, one row per idea; $\\Delta p_t$ is the seed-matched, "
           "per-video paired difference to the row's own reference with a 95\\,\\% bootstrap CI over videos (10\\,000 resamples). Values mean $\\pm$ sample s.d.\\ over $n$ seeds. "
           "All rows are exploratory on the selection fold; none is corrected for multiplicity. A row whose optimiser differs from its reference's is marked: "
           f"for an LSTM, the AdamW recipe (optimiser, learning rate, weight decay and schedule) is worth ${numbers.get('stepopt_d', '?')}$ in $p_t$ over the five folds (Table~3 of the article), so such a difference is not attributable to the idea alone.")
    out = ["\\centering\\tiny\\setlength{\\tabcolsep}{2.5pt}",
           "\\begin{longtable}{L{3.4cm}cccL{2.2cm}cL{2.3cm}}", "\\multicolumn{7}{p{\\textwidth}}{" + cap + "}\\\\[4pt]", "\\toprule",
           "Idea & $n$ & $p_t$ & reference $p_t$ & $\\Delta p_t$ [95\\,\\% CI] & F1@50 & note \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{longtable}"]
    (TAB / "ed_table_negative.tex").write_text("\n".join(out) + "\n")
    print(f"ED negative table: {len(rows)} rows")


# ------------------------------------------------------------------------------------------------- ED Table 4 (HSMM)
def hsmm_table(numbers: dict) -> None:
    f = ROOT / "results/hsmm_leaderboard.csv"
    if not f.exists():
        return
    h = pd.read_csv(f)
    d = h.p_t_hsmm - h.p_t_viterbi; de = h.edit_hsmm - h.edit_viterbi; df1 = h.f1_50_hsmm - h.f1_50_viterbi; dm = h.mae_hsmm - h.mae_viterbi
    flat_ge = int((h.p_t_flat_seg_markov >= h.p_t_hsmm - 1e-9).sum())
    sel = h.groupby(["kind", "lam"]).size().sort_values(ascending=False)
    numbers.update({"hsmm_n": str(len(h)), "hsmm_dpt": f"{d.mean():+.3f}", "hsmm_dpt_med": f"{d.median():+.3f}", "hsmm_improved": str(int((d > 0).sum())),
                    "hsmm_dedit": f"{de.mean():+.1f}", "hsmm_df1": f"{df1.mean():+.1f}", "hsmm_dmae": f"{dm.mean():+.2f}", "hsmm_flat_ge": str(flat_ge),
                    "hsmm_skip_ratio": f"{(h.skipped_hsmm / h.skipped_viterbi.clip(lower=1)).median():.2f}"})
    rows = [f"{k}, $\\lambda={l:g}$ & {n} \\\\" for (k, l), n in sel.items()]
    cap = (f"\\textbf{{Semi-Markov decoding on all {len(h)} checkpoints.}} Duration model and $\\lambda$ selected per checkpoint on its validation partition; "
           "test-partition differences to the released frame-level Viterbi decoder. The $\\lambda=0$ flat control is a segment-level Markov model with no duration information.")
    out = ["\\begin{table}[t]", "\\centering\\small", "\\caption{" + cap + "}\\label{tab:edhsmm}", "\\begin{tabular}{lc}", "\\toprule", "Quantity & value \\\\", "\\midrule",
           f"$\\Delta p_t$ mean (median) & ${d.mean():+.3f}$ (${d.median():+.3f}$) \\\\", f"checkpoints with $\\Delta p_t>0$ & {int((d > 0).sum())}/{len(h)} \\\\",
           f"$\\Delta$ edit / $\\Delta$ F1@50 / $\\Delta$ MAE (h) & ${de.mean():+.1f}$ / ${df1.mean():+.1f}$ / ${dm.mean():+.2f}$ \\\\",
           f"skipped ground-truth segments, ratio semi-Markov/Viterbi (median) & {(h.skipped_hsmm / h.skipped_viterbi.clip(lower=1)).median():.2f} \\\\",
           f"checkpoints where the duration-free control $\\geq$ the selected model on test & {flat_ge}/{len(h)} \\\\", "\\midrule",
           "\\multicolumn{2}{l}{\\textit{Validation-selected setting (kind, $\\lambda$): count}} \\\\"] + [f"{r}" for r in rows] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "ed_table_hsmm.tex").write_text("\n".join(out) + "\n")


# ------------------------------------------------------------------------------------------------- numbers
#: the five-fold attribution chain: each step changes one thing
CHAIN = [("ref", "resnet18_lstm_L4_split{k}", "released reference: LSTM, 4-frame clips, 150-frame chunks, SGD"),
         ("lstmwin", "resnet18_lstm_L16_win_split{k}", "16-fr.\\ clips, batch 4, window fix (LSTM, SGD)"),
         ("lstmwinadamw", "resnet18_lstm_L16_win_adamw_split{k}", "SGD $\\to$ AdamW recipe (LSTM)"),
         ("tr1", "resnet18_transformer_L16_evalfix_split{k}", "LSTM head $\\to$ transformer head"),
         ("mfSeven", "resnet18_transformer_L16_multifocal7_evalfix_split{k}", "+6 planes, stacked; batch 4 $\\to$ 2"),
         ("cf", "resnet18_transformer_L16_crossfocal7_evalfix_split{k}", "stacking $\\to$ cross-focal (7 planes)")]
STEP_TAGS = ["stepclip", "stepopt", "stephead", "stepseven", "stepfuseseven"]


def _step_macros(numbers: dict, tag: str, r: dict, total: float | None = None) -> None:
    numbers[f"{tag}_d"] = f"{r['d']:+.3f}"; numbers[f"{tag}_lo"] = f"{r['lo']:+.3f}"; numbers[f"{tag}_hi"] = f"{r['hi']:+.3f}"
    numbers[f"{tag}_npat"] = str(r["npat"]); numbers[f"{tag}_nvid"] = str(r["nvid"])
    numbers[f"{tag}_wins"] = str(r["wins"]); numbers[f"{tag}_nparts"] = str(r["nparts"])
    numbers[f"{tag}_fold"] = f"{r['part_d'].mean():+.3f}"
    numbers[f"{tag}_excl"] = "1" if (r["lo"] > 0 or r["hi"] < 0) else "0"
    if total:
        numbers[f"{tag}_share"] = f"{100 * r['d'] / total:.0f}"


#: every configuration that enters a five-fold contrast; the per-frame pair (v40) is optional until it has run
ATTR_CONFIGS = {t: p for t, p, _ in CHAIN} | {"tr3": "resnet18_transformer_L16_multifocal3_evalfix_split{k}",
                                            "none": "resnet18_none_split{k}", "noneadamw": "resnet18_none_adamw_split{k}"}
#: secondary contrasts (tag, a, b): reported with unadjusted intervals and labelled exploratory in the text
SECONDARY = [("stepwin", "tr1", "ref"), ("stepplanes", "tr3", "tr1"), ("stepfusion", "cf", "tr3"),
             ("stepsevenvsthree", "mfSeven", "tr3"), ("stepcfadamw", "cf", "lstmwinadamw"), ("stepcfone", "cf", "tr1"),
             ("stepnoneadamw", "noneadamw", "none"), ("steplstmvsnone", "lstmwinadamw", "noneadamw"),
             ("steptrvsnone", "tr1", "noneadamw"), ("steprefvsnone", "ref", "none")]


def attribution_numbers(df: pd.DataFrame, numbers: dict, n: int = NBOOT, seed: int = 39) -> dict:
    """Joint five-fold bootstrap of the attribution chain.

    Every configuration's per-video seed-mean p_t enters on the same pooled videos (328 test videos of the five released
    folds). One replicate resamples patients within each fold once and the seeds of each configuration once, and every
    contrast, the total and the shares are read off that same replicate, so shares get intervals and the chain steps
    get a joint family for the Holm adjustment. Steps telescope: they sum to the total by construction."""
    have = {t: p for t, p in ATTR_CONFIGS.items() if all(len(df[df.experiment == p.format(k=k)]) >= 3 for k in range(5))}
    if not all(t in have for t, _, _ in CHAIN):
        return {}
    folds = []
    for k in range(5):
        arr = {t: seed_arrays(p.format(k=k), df) for t, p in have.items()}
        vids = sorted(set.intersection(*(set(a) for a in arr.values())))
        pat = np.array([patient_of(v) for v in vids])
        groups = [np.where(pat == q)[0] for q in np.unique(pat)]
        folds.append(({t: np.stack([a[v] for v in vids]) for t, a in arr.items()}, groups, vids))
    tags = list(have)
    point = {t: float(np.concatenate([f[0][t].mean(1) for f in folds]).mean()) for t in tags}
    rng = np.random.default_rng(seed)
    bs = {t: np.empty(n) for t in tags}
    for i in range(n):
        tot = dict.fromkeys(tags, 0.0); cnt = 0
        for arrs, groups, _ in folds:
            idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            cnt += len(idx)
            for t in tags:
                a = arrs[t]
                tot[t] += float(a[idx][:, rng.integers(0, a.shape[1], a.shape[1])].mean(1).sum())
        for t in tags:
            bs[t][i] = tot[t] / cnt
    fold_means = {t: np.array([df[df.experiment == have[t].format(k=k)].p_t.mean() for k in range(5)]) for t in tags}
    vids_all = [v for _, _, vids in folds for v in vids]
    n_couples = len({patient_of(v) for v in vids_all})
    n_clusters = sum(len(g) for _, g, _ in folds)
    total_bs = bs["cf"] - bs["ref"]
    total_d = point["cf"] - point["ref"]

    def contrast(a: str, b: str) -> dict:
        d = bs[a] - bs[b]
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        share = d / total_bs
        pd_ = fold_means[a] - fold_means[b]
        return {"d": point[a] - point[b], "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                "lo90": float(np.percentile(d, 5)), "hi90": float(np.percentile(d, 95)), "p": p,
                "share": 100 * (point[a] - point[b]) / total_d, "share_lo": float(np.percentile(share, 2.5) * 100),
                "share_hi": float(np.percentile(share, 97.5) * 100), "part_d": pd_, "wins": int((pd_ > 0).sum()),
                "npat": n_clusters, "ncouples": n_couples, "nvid": len(vids_all), "nparts": 5}
    out = {"total": contrast("cf", "ref")}
    chain_tags = [c[0] for c in CHAIN]
    for tag, (a, b) in zip(STEP_TAGS, zip(chain_tags[1:], chain_tags[:-1])):
        out[tag] = contrast(a, b)
    # Holm over the five chain steps (the inferential family of Table 2)
    order = sorted(STEP_TAGS, key=lambda t: out[t]["p"])
    running = 0.0
    for r, t in enumerate(order):
        running = max(running, min(1.0, (len(order) - r) * out[t]["p"]))
        out[t]["p_holm"] = running
    for tag, a, b in SECONDARY:
        if a in have and b in have:
            out[tag] = contrast(a, b)
    for key, r in out.items():
        tag = "steptotal" if key == "total" else key
        _step_macros(numbers, tag, r)
        numbers[f"{tag}_share"] = f"{r['share']:.0f}"
        numbers[f"{tag}_sharelo"] = f"{r['share_lo']:.0f}"; numbers[f"{tag}_sharehi"] = f"{r['share_hi']:.0f}"
        numbers[f"{tag}_ncouples"] = str(r["ncouples"])
        numbers[f"{tag}_p"] = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        numbers[f"{tag}_lonine"] = f"{r['lo90']:+.3f}"; numbers[f"{tag}_hinine"] = f"{r['hi90']:+.3f}"
        if "p_holm" in r:
            numbers[f"{tag}_pholm"] = "<0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}"
    numbers["attr_nholmsig"] = str(sum(out[t]["p_holm"] < 0.05 for t in STEP_TAGS))
    numbers["attr_pernine"] = "1" if "noneadamw" in have else "0"      # v40 present
    (ROOT / "results/attribution_v39.json").write_text(json.dumps(
        {k: {**{kk: vv for kk, vv in v.items() if kk != "part_d"}, "part_d": [float(x) for x in v["part_d"]]} for k, v in out.items()}, indent=1))
    return out


#: single-plane configurations re-run off the selection fold on the couple-level split (v39 family C), plus the two
#: released baselines and the per-frame probe that were already there; optimiser follows the head family (Methods)
V2_CEILING = [("none", "resnet18_none_grouped_v2"), ("lstm", "resnet18_lstm_L4_grouped_v2"), ("rFiftylstm", "resnet50_lstm_L8_grouped_v2"),
              ("gru", "resnet18_gru_L8_grouped_v2"), ("tcn", "resnet18_tcn_L16_grouped_v2"), ("mamba", "resnet18_mamba_L16_grouped_v2"),
              ("tr", "resnet18_transformer_L16_evalfix_grouped_v2"), ("cf", "resnet18_transformer_L16_crossfocal7_evalfix_grouped_v2")]


def v2_ceiling_numbers(df: pd.DataFrame, numbers: dict) -> None:
    """The ceiling off the selection fold: single-plane heads on nantes_grouped_v2, three seeds each (v39 family C)."""
    have = {t: e for t, e in V2_CEILING if len(df[df.experiment == e]) >= 3}
    if len(have) < len(V2_CEILING):
        return
    for t, e in have.items():
        r = df[df.experiment == e].p_t
        numbers[f"vtwoceil_{t}_pt"] = f"{r.mean():.3f}"; numbers[f"vtwoceil_{t}_sd"] = f"{r.std(ddof=1):.3f}"
    sgd = [t for t in ("lstm", "rFiftylstm", "gru", "tcn", "mamba")]
    means = {t: df[df.experiment == have[t]].p_t.mean() for t in sgd}
    numbers["vtwoceil_sgd_lo"] = f"{min(means.values()):.3f}"; numbers["vtwoceil_sgd_hi"] = f"{max(means.values()):.3f}"
    nomamba = {t: m for t, m in means.items() if t != "mamba"}
    numbers["vtwoceil_sgdnm_lo"] = f"{min(nomamba.values()):.3f}"; numbers["vtwoceil_sgdnm_hi"] = f"{max(nomamba.values()):.3f}"
    best = max(nomamba, key=nomamba.get)
    for tag, a, b in (("vtwobestsgd", have[best], have["none"]), ("vtwotr", have["tr"], have["none"]),
                      ("vtwotrlstm", have["tr"], have["lstm"]), ("vtwocftr", have["cf"], have["tr"])):
        r = pooled_step([(a, b)], df)
        if r is not None:
            _step_macros(numbers, tag, r)
    numbers["vtwoceil_nconfigs"] = str(sum(t in have for t in ("tr", "rFiftylstm", "gru", "tcn", "mamba")))  # v39 family C


def core_numbers(df: pd.DataFrame, numbers: dict) -> None:
    def stat(exp, key="p_t", nd=3):
        r = df[df.experiment == exp][key].dropna()
        return (f"{r.mean():.{nd}f}", f"{r.std(ddof=1):.{nd}f}" if len(r) > 1 else "", len(r), f"{r.min():.{nd}f}", f"{r.max():.{nd}f}")
    for tag, exp in [("ref", "resnet18_lstm_L4_split0"), ("refgrouped", "resnet18_lstm_L4_grouped_v1"), ("r50lstm", "resnet50_lstm_L8_split0"),
                     ("tr1", "resnet18_transformer_L16_evalfix_split0"), ("tr3", "resnet18_transformer_L16_multifocal3_evalfix_split0"),
                     ("cf", "resnet18_transformer_L16_crossfocal7_evalfix_split0"), ("cfgrouped", "resnet18_transformer_L16_crossfocal7_evalfix_grouped_v1"),
                     ("trchunk", "resnet18_transformer_L16_split0"), ("mamba", "resnet18_mamba_L16_split0"), ("node", "resnet18_node_L16_split0"),
                     ("diffact", "resnet18_diffactlite_L16_split0"), ("relpos", "resnet18_transformer_relpos_L16_split0"), ("relposfix", "resnet18_transformer_relpos_L16_evalfix_split0"),
                     ("imgr18", "resnet18_none_imagesplit"), ("imgeffl", "efficientnet_v2_l_none_imagesplit"), ("r18none", "resnet18_none_split0"), ("efflnone", "efficientnet_v2_l_none_split0"),
                     ("asformer16", "resnet18_asformer_L16_evalfix_split0"), ("r2p1d", "r2plus1d_L8_split0"), ("bound", "resnet18_lstm_L4_boundaryloss_split0"),
                     ("mf3lstm", "resnet18_lstm_L4_multifocal3_split0"), ("hires50", "resnet50_lstm_L8_hires448_split0"), ("photocc", "resnet50_lstm_L8_photo_cellcount_split0")]:
        for key in ("p_t", "p_v", "p", "f1_50", "edit", "mae_h"):
            nd = 1 if key in ("f1_50", "edit") else (2 if key == "mae_h" else 3)
            m, s, n, lo, hi = stat(exp, key, nd)
            if n == 0: continue
            numbers[f"{tag}_{key}"] = m; numbers[f"{tag}_{key}_sd"] = s; numbers[f"{tag}_n"] = str(n); numbers[f"{tag}_{key}_min"] = lo; numbers[f"{tag}_{key}_max"] = hi
    # frame-level leakage inflation, computed rather than typed: an earlier draft quoted +0.16 for a
    # difference that is +0.13 in these very macros (R3-M1/R1-M2).
    for tag, img, vid in (("imgr18", "resnet18_none_imagesplit", "resnet18_none_split0"),
                          ("imgeffl", "efficientnet_v2_l_none_imagesplit", "efficientnet_v2_l_none_split0")):
        a_ = df[df.experiment == img].p_t.dropna(); b_ = df[df.experiment == vid].p_t.dropna()
        if len(a_) and len(b_):
            numbers[f"{tag}_infl"] = f"{a_.mean() - b_.mean():+.3f}"
            numbers[f"{tag}_infl_nimg"] = str(len(a_)); numbers[f"{tag}_infl_nvid"] = str(len(b_))
    # plane-matched control for the fusion mechanism: seven planes stacked as channels (R3-M3)
    mf7 = df[df.experiment == "resnet18_transformer_L16_multifocal7_evalfix_split0"].p_t.dropna()
    cf0 = df[df.experiment == "resnet18_transformer_L16_crossfocal7_evalfix_split0"].p_t.dropna()
    if len(mf7) and len(cf0):
        numbers["mfSevenfoldZeropt"] = f"{mf7.mean():.3f}"; numbers["mfSevenfoldZeroptsd"] = f"{mf7.std(ddof=1):.3f}"
        numbers["mfSevenfoldZeron"] = str(len(mf7))
        numbers["cfminusmfSeven"] = f"{cf0.mean() - mf7.mean():+.3f}"
    # the tuned-recipe family: reported in Methods, excluded from every headline comparison
    rec = [f"resnet18_transformer_L16_crossfocal7_recipe_e20_evalfix_split{k}" for k in range(5)]
    fm = [df[df.experiment == e].p_t.mean() for e in rec]
    if not any(np.isnan(x) for x in fm):
        numbers["recipecf_fold_pt"] = f"{np.mean(fm):.3f}"; numbers["recipecf_fold_pt_sd"] = f"{np.std(fm, ddof=1):.3f}"
    # the attribution decomposition along the chain reference -> cross-focal (v39 completed it on all five folds), one
    # macro set per step: pooled per-video paired difference with a seed-aware, patient-clustered interval
    # (attribution_numbers runs inside fold_table, which renders the attribution block of Table 2)
    numbers["n_runs"] = str(len(df)); numbers["n_configs"] = str(df.experiment.nunique())
    # distinct model configurations: the same recipe evaluated on another fold or split is one
    # configuration, not several, so the sweep count does not grow when a recipe is re-run elsewhere.
    recipe = df.experiment.str.replace(r"_(split\d|grouped_v\d|grouped|imagesplit)$", "", regex=True)
    numbers["n_recipes"] = str(recipe.nunique())
    f0 = df[df.split_key == "fold0"]
    common = f0[is_common_recipe(f0)]
    # the sweep is the factor-at-a-time set at the common recipe (Methods); tuned-recipe runs are reported separately
    numbers["n_configs_fold0"] = str(common.experiment.nunique())
    numbers["n_configs_fold0_all"] = str(f0.experiment.nunique())
    numbers["n_configs_fold0_tuned"] = str(f0.experiment.nunique() - common.experiment.nunique())
    numbers["n_runs_fold0"] = str(len(f0))
    enc = common.backbone.str.replace(r" \+ .*", "", regex=True).str.replace(r" \(.*", "", regex=True)
    numbers["n_encoders"] = str(enc.nunique())
    # temporal heads proper: the per-frame baseline and R(2+1)D's built-in 3D convolution are not separate heads
    numbers["n_heads"] = str(common["head"][~common["head"].isin(["per-frame", "3D CNN"])].nunique())
    numbers["n_losses"] = str(common.loss.nunique())
    # Minimum detectable paired difference (Statistical analysis section): computed, not hand-typed, from the same
    # fold-0 control-vs-cross-focal per-video paired differences quoted elsewhere in the text (sd = 0.100, n = 67).
    d = paired_videos("resnet18_transformer_L16_crossfocal7_evalfix_split0", "resnet18_transformer_L16_evalfix_split0", df)
    if d is not None and len(d) > 5:
        sd_diff = float(d.std(ddof=1))
        z_a, z_b, m = mdd(sd_diff, len(d))
        numbers["mddsd"] = f"{sd_diff:.3f}"; numbers["mddn"] = str(len(d))
        numbers["mddptSixSeven"] = f"{m:.3f}"  # name kept for the fold-0 (n=67) figure already cited in-text
        numbers["mddzalpha"] = f"{z_a:.3f}"; numbers["mddzbeta"] = f"{z_b:.3f}"


def add_leakage_numbers(numbers: dict) -> None:
    """Patient-leakage macros, from results/protocol_audit.json (scripts/audit_protocol_splits.py).

    The manuscript quotes these in Results 'defects'/'structural'; they must not be hand-typed, because
    the count depends on the grouping level and three irregular name forms (stseg.data.patient)."""
    src = ROOT / "results/protocol_audit.json"
    if not src.exists():
        print("protocol_audit.json missing -- run scripts/audit_protocol_splits.py; leakage macros skipped")
        return
    a = json.loads(src.read_text())
    # The headline leakage numbers are quoted on the 652-video list the paper actually evaluates on,
    # not on the 704 released names (which include the 52 JPEG-truncated videos the paper excludes).
    # The pre-exclusion figures stay available under the "pre" tags for the Methods.
    for level, tag in (("couple", "leakcouple"), ("cycle", "leakcycle")):
        pool = a["official_pooled"][f"clean652.{level}"]
        f0 = a["official_folds"]["split0"]["clean652"][level]
        prepool = a["official_pooled"][f"all704.{level}"]
        numbers[f"{tag}_pre_pooled_exposed"] = str(prepool["pooled_exposed"])
        numbers[f"{tag}_pre_pooled_test"] = str(prepool["pooled_test"])
        numbers[f"{tag}_pre_pooled_pct"] = f"{100 * prepool['pooled_frac_exposed']:.0f}"
        numbers[f"{tag}_pre_fold0_patients"] = str(a["official_folds"]["split0"]["all704"][level]["n_leaking_patients"])
        numbers[f"{tag}_pre_codes"] = str(a["official_folds"]["split0"]["all704"][level]["n_patients"])
        numbers[f"{tag}_fold0_patients"] = str(f0["n_leaking_patients"])
        numbers[f"{tag}_fold0_of"] = str(f0["n_patients"])
        numbers[f"{tag}_fold0_pct"] = f"{100 * f0['frac_leaking_patients']:.1f}"
        numbers[f"{tag}_fold0_exposed"] = str(f0["n_exposed_test_videos"])
        numbers[f"{tag}_fold0_test"] = str(f0["n_test_videos"])
        numbers[f"{tag}_exposed_min"] = str(min(pool["exposed_test_videos_per_fold"]))
        numbers[f"{tag}_exposed_max"] = str(max(pool["exposed_test_videos_per_fold"]))
        numbers[f"{tag}_pooled_exposed"] = str(pool["pooled_exposed"])
        numbers[f"{tag}_pooled_test"] = str(pool["pooled_test"])
        numbers[f"{tag}_pooled_pct"] = f"{100 * pool['pooled_frac_exposed']:.0f}"
        sim = a["simulation"][f"clean652.{level}.released"]
        numbers[f"{tag}_sim_mean"] = f"{sim['leaked_patients_mean']:.1f}"
        numbers[f"{tag}_sim_min"] = str(sim["leaked_patients_min"])
        numbers[f"{tag}_sim_max"] = str(sim["leaked_patients_max"])
        numbers[f"{tag}_sim_seeds"] = str(sim["seeds"])
        numbers[f"{tag}_sim_exposed_pct"] = f"{100 * sim['frac_exposed_test_mean']:.0f}"
    for name, tag in (("nantes_grouped_v1", "gvOne"), ("nantes_grouped_v2", "gvTwo")):
        own = a["own_splits"].get(name)
        if not own:
            continue
        numbers[f"{tag}_leaking_couples"] = str(own["couple"]["n_leaking_patients"])
        numbers[f"{tag}_exposed"] = str(own["couple"]["n_exposed_test_videos"])
        numbers[f"{tag}_test"] = str(own["couple"]["n_test_videos"])
        numbers[f"{tag}_exposed_pct"] = f"{100 * own['couple']['frac_exposed_test_videos']:.1f}"
    # legacy macro names used in Results 'structural'; recomputed here at couple level so that the
    # hand-maintained results/numbers_extra.json can no longer disagree with the released split files.
    sim = a["simulation"]["clean652.couple.released"]
    sim7030 = a["simulation"]["clean652.couple.7030"]
    f0 = a["official_folds"]["split0"]["clean652"]["couple"]
    numbers["leak7030_mean"] = f"{sim7030['leaked_patients_mean']:.1f}"
    numbers["leak7030_exposed_pct"] = f"{100 * sim7030['frac_exposed_test_mean']:.0f}"
    numbers["leak_mean"] = f"{sim['leaked_patients_mean']:.1f}"
    numbers["leak_frac"] = f"{100 * sim['leaked_patients_mean'] / f0['n_patients']:.1f}"
    numbers["leak_min"] = str(sim["leaked_patients_min"])
    numbers["leak_max"] = str(sim["leaked_patients_max"])
    pct = int(sim["released_fold0_percentile"])
    suffix = "th" if 11 <= pct % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th")
    numbers["leak_fold0_pctile"] = str(pct)
    numbers["leak_fold0_pctile_ordinal"] = f"{pct}{suffix}"
    numbers["leak_videos_frac"] = f"{100 * sim['frac_exposed_test_mean']:.1f}"
    # distinct couples in the clean fold-0 test set: DC307 contributes two videos, so it is not 67
    f0c = a["official_folds"]["split0"]["clean652"]["couple"]
    numbers["fold0_test_couples"] = str(f0c.get("n_test_patients", "?"))
    numbers["clean_videos"] = str(a["official_folds"]["split0"]["clean652"]["couple"]["n_videos"])
    numbers["overview_onsets"] = "15"   # onsets annotated for the Fig. 1 example embryo (BA958-2)
    hv = ROOT / "results/hard_videos.json"
    if hv.exists():
        h = json.loads(hv.read_text())
        numbers["hard_scanned"] = str(h["n_videos_scanned"])
        numbers["hard_thresh"] = f"{h['threshold']:.2f}"
        numbers["hard_flagged"] = str(h["n_flagged"])
    bv = ROOT / "results/blank_videos.json"
    if bv.exists():
        b = json.loads(bv.read_text())
        numbers["blank_scanned"] = str(b["n_videos"])
        numbers["blank_flagged"] = str(b["n_flagged"])
    # focus-adaptive plane-fusion variants: the nearest competitors of the proposed mechanism, which
    # the text has to name rather than leave in Extended Data only (round-3 R1-M4 / R3-m7)
    import pandas as _pd  # noqa: PLC0415
    _df = _pd.read_csv(ROOT / "results/runs_master.csv")
    for tag, exp in (("faspatial", "resnet18_transformer_L16_focalattn_spatial_evalfix_split0"),
                     ("fasharp", "resnet18_transformer_L16_focalattn_sharp_evalfix_split0"),
                     ("faboth", "resnet18_transformer_L16_focalattn_spatial_sharp_evalfix_split0")):
        r_ = _df[_df.experiment == exp].p_t.dropna()
        if len(r_):
            numbers[f"{tag}_pt"] = f"{r_.mean():.3f}"
            numbers[f"{tag}_n"] = str(len(r_))
            numbers[f"{tag}_sd"] = f"{r_.std(ddof=1):.3f}" if len(r_) > 1 else ""
    _f0 = _df[_df.split_key == "fold0"]
    _g = _f0[is_common_recipe(_f0)].groupby("experiment").p_t.agg(["mean", "size"])
    numbers["fold0_best_pt"] = f"{_g['mean'].max():.3f}"                      # best common-recipe configuration
    numbers["fold0_best_pt_n"] = str(int(_g.loc[_g['mean'].idxmax(), 'size']))
    numbers["fold0_best3_pt"] = f"{_g[_g['size'] >= 3]['mean'].max():.3f}"      # best with three seeds
    numbers["fold0_best_tuned_pt"] = f"{_f0.groupby('experiment').p_t.mean().max():.3f}"
    _s0 = _df[(_df.experiment == "resnet18_transformer_L16_evalfix_split0") & (_df.seed == 0)].p_t
    if len(_s0):
        numbers["tr1_seed0_pt"] = f"{float(_s0.iloc[0]):.3f}"
    # derived clinical intervals cc2 and s2 (scripts/derived_intervals.py)
    dpath = ROOT / "results/derived_intervals.json"
    if dpath.exists():
        di = json.loads(dpath.read_text())
        for model, mtag in (("reference", "ref"), ("crossfocal", "cf")):
            if model not in di:
                continue
            for iv in ("cc2", "s2"):
                x = di[model].get(iv)
                if not x:
                    continue
                t = f"{iv}{mtag}"
                numbers[f"{t}_abs"] = f"{x['abs_mean']:.2f}"
                numbers[f"{t}_abs_lo"] = f"{x['abs_lo']:.2f}"; numbers[f"{t}_abs_hi"] = f"{x['abs_hi']:.2f}"
                numbers[f"{t}_med"] = f"{x['median_abs']:.2f}"
                numbers[f"{t}_signed"] = f"{x['signed_mean']:+.2f}"
                numbers[f"{t}_within"] = f"{100 * x['within_1h']:.0f}"
                numbers[f"{t}_n"] = str(x["n_scored"]); numbers[f"{t}_spurious"] = str(x["n_spurious"])
                numbers[f"{iv}_gt_med"] = f"{x['gt_median']:.1f}"
                numbers[f"{iv}_gt_lo"] = f"{x['gt_iqr_lo']:.1f}"; numbers[f"{iv}_gt_hi"] = f"{x['gt_iqr_hi']:.1f}"
                numbers[f"{iv}_gt_iqr"] = f"{x['gt_iqr']:.1f}"
                numbers[f"{iv}_defined"] = str(x["n_gt_defined"])
            for ck, cv in di[model].get("comparisons", {}).items():
                t = ck.replace("cc2", "cctwo").replace("s2", "stwo").replace("_skip3_as_zero", "skip")
                for stat in ("model_mae", "const_mae", "single_seed_mae", "model_median", "const_median"):
                    numbers[f"cmp{t}{mtag}{stat}"] = f"{cv[stat]:.2f}"
                for stat in ("model_within_1h", "const_within_1h"):
                    numbers[f"cmp{t}{mtag}{stat}"] = f"{100 * cv[stat]:.0f}"
                numbers[f"cmp{t}{mtag}n"] = str(cv["n"])
                if "within_1h_diff" in cv:
                    numbers[f"cmp{t}{mtag}diff"] = f"{100 * cv['within_1h_diff']:+.0f}"
                    numbers[f"cmp{t}{mtag}difflo"] = f"{100 * cv['within_1h_diff_ci'][0]:+.0f}"
                    numbers[f"cmp{t}{mtag}diffhi"] = f"{100 * cv['within_1h_diff_ci'][1]:+.0f}"
                    numbers[f"cmp{t}{mtag}kappa"] = f"{cv['category_kappa']:.2f}"
                    numbers[f"cmp{t}{mtag}favmodel"] = f"{100 * cv['favourable_share_model']:.0f}"
                    numbers[f"cmp{t}{mtag}favgt"] = f"{100 * cv['favourable_share_gt']:.0f}"
                    numbers[f"cmp{t}cut"] = f"{cv['category_cutoff_h']:g}"
                numbers[f"cmp{t}const"] = f"{cv['const_value_h']:.1f}"
            for ev in ("t2", "t3", "t4"):
                mm = di[model].get("marginal_abs", {}).get(ev)
                if mm:
                    numbers[f"onset{ev}{mtag}"] = f"{mm['abs_mean']:.2f}"
        numbers["di_nvideos"] = str(di.get("crossfocal", {}).get("n_videos", "?"))
        numbers["di_skipped"] = str(int(numbers.get("di_nvideos", 0)) - int(numbers.get("cc2_defined", 0))) \
            if numbers.get("di_nvideos", "?").isdigit() and numbers.get("cc2_defined", "?").isdigit() else "?"
    did = ROOT / "results/siblingleak_did.json"
    if did.exists():
        d = json.loads(did.read_text())
        numbers["sibdid"] = f"{d['DiD']:+.3f}"
        numbers["sibdidlo"] = f"{d['ci95'][0]:+.3f}"
        numbers["sibdidhi"] = f"{d['ci95'][1]:+.3f}"
        numbers["sibdidp"] = f"{d['permutation_p']:.3f}"
        numbers["sibdeltal"] = f"{d['Delta_L']:+.3f}"
        numbers["sibdeltac"] = f"{d['Delta_C']:+.3f}"
        numbers["sibinflfolds"] = f"{d['implied_inflation']['released_folds_pooled']:+.3f}"
        exp = d["exposure"]["released_folds_pooled"]
        numbers["sibinfllo"] = f"{d['ci95'][0] * exp:+.3f}"
        numbers["sibinflhi"] = f"{d['ci95'][1] * exp:+.3f}"
        numbers["sibexpfolds"] = f"{100 * exp:.0f}"
        numbers["sibnl"] = str(d["n_L"]); numbers["sibnc"] = str(d["n_C"])
        if "nantes_grouped_v1" in d["implied_inflation"]:
            numbers["sibinflgvOne"] = f"{d['implied_inflation']['nantes_grouped_v1']:+.4f}"
        # how much smaller the patient-level mechanism is than the frame-level one, on the same backbone
        try:
            frame = float(numbers["imgr18_p_t"]) - float(numbers["r18none_p_t"])
            infl = abs(d["implied_inflation"]["released_folds_pooled"])
            numbers["sibratio"] = f"{frame / infl:.0f}" if infl > 0 else "n/a"
        except (KeyError, ValueError, ZeroDivisionError):
            pass


def protocol_extra_numbers(numbers: dict) -> None:
    """The model-free timeline mechanism (scripts/timeline_mechanism.py) and the temporal-accuracy definitions
    (scripts/pt_definitions.py)."""
    m = ROOT / "results/frame_timing/mechanism.json"
    if m.exists():
        d = json.loads(m.read_text())
        numbers["mechn"] = str(d["mismatch"]["n_interval_0_23_0_27"])
        numbers["mechintervalmis"] = f"{d['mismatch']['image_interval_median']:.3f}"
        numbers["mechintervalother"] = f"{d['other']['image_interval_median']:.3f}"
        numbers["mechstepmis"] = f"{d['mismatch']['file_step_median']:.3f}"
        numbers["mechstepother"] = f"{d['other']['file_step_median']:.3f}"
        numbers["mechnother"] = str(d["other"]["n"])
        numbers["mechnboth"] = str(d["n_mismatch_step_0_2_interval_0_23_0_27"])
        numbers["mechnstepmis"] = str(d["n_mismatch_step_0_2"])
        numbers["mechntwentymin"] = str(d["n_other_step_above_0_3"])
        numbers["mechotheriqrlo"], numbers["mechotheriqrhi"] = (f"{x:.2f}" for x in d["other"]["image_interval_iqr"])
    nt = ROOT / "results/native_tas_heads.json"
    if nt.exists():
        d = json.loads(nt.read_text())
        for kind, t in (("mstcn", "mstcn"), ("asformer", "asformer")):
            x = [r["test"]["p_t"] for r in d if r["kind"] == kind]
            e = [r["test"]["edit"] for r in d if r["kind"] == kind]
            if len(x) == 3:
                numbers[f"native{t}pt"] = f"{np.mean(x):.3f}"
                numbers[f"native{t}sd"] = f"{np.std(x, ddof=1):.3f}"
                numbers[f"native{t}editlo"], numbers[f"native{t}edithi"] = f"{min(e):.0f}", f"{max(e):.0f}"
        if "nativemstcnpt" in numbers and "nativeasformerpt" in numbers and "tr1_p_t" in numbers:
            g = (float(numbers["nativemstcnpt"]) + float(numbers["nativeasformerpt"])) / 2 - float(numbers["tr1_p_t"])
            numbers["nativegain"] = f"{g:.2f}"
    q = ROOT / "results/pt_definitions.json"
    if q.exists():
        d = json.loads(q.read_text())
        if "named" in d:
            for k, t in (("impute_next_video", "ours"), ("skip_as_miss_video", "missvideo"), ("skip_as_miss_pooled", "misspooled")):
                numbers[f"ptdef{t}"] = f"{d['named'][k]:.3f}"
            numbers["ptdefn"] = str(d["n_definitions"])
            numbers["ptdeflo"], numbers["ptdefhi"] = f"{d['range_theta_as_published'][0]:.3f}", f"{d['range_theta_as_published'][1]:.3f}"
            numbers["ptdeffolds"] = str(d["n_folds"])
            numbers["ptdefruns"] = str(len(d["runs"]))


PRIMARY_RETRAIN = "v38"


def figure_numbers(numbers: dict) -> None:
    """Numbers quoted from the four figures about the task (scripts/make_fig_problem.py -> results/figdata)."""
    f = ROOT / "results/figdata/figure_numbers.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    t = d["task"]
    numbers["timeonlyacc"] = f"{t['time_only_acc']:.2f}"
    rec = {k: v for k, v in t["time_only_recall"].items() if v == v}
    zero = [k for k, v in rec.items() if v == 0.0]
    numbers["timeonlyzero"] = str(len(zero))
    numbers["ttwotfourshare"] = f"{100 * t['t2_to_t4_share']:.0f}"
    adj = d["latent"]["adjacent_head"]
    pick = lambda pair: [a[pair] for a in adj.values() if pair in a]  # noqa: E731
    for pair, tag in (("t6|t7", "sixseven"), ("t7|t8", "seveneight"), ("tPNf|t2", "pnftwo"), ("t2|t3", "twothree"),
                      ("t3|t4", "threefour"), ("t4|t5", "fourfive"), ("t5|t6", "fivesix")):
        numbers[f"probe{tag}lo"], numbers[f"probe{tag}hi"] = f"{min(pick(pair)):.2f}", f"{max(pick(pair)):.2f}"
    v = d["latent"]["velocity_peak_ratio"]
    numbers["velsync"] = f"{v['synchronous division']:.1f}"
    numbers["veltrans"] = f"{v['transient division']:.1f}"
    numbers["velmorph"] = f"{v['morphological']:.1f}"
    fl = d["dynamics"]["final_loss_median"]
    trans, stable = ("t3", "t5", "t6", "t7", "tB"), ("tPNa", "t2", "t4", "t8", "t9+")
    tv = [x[p] for x in fl.values() for p in trans]
    sv = [x[p] for x in fl.values() for p in stable]
    numbers["losstranslo"], numbers["losstranshi"] = f"{min(tv):.1f}", f"{max(tv):.1f}"
    numbers["lossstablelo"], numbers["lossstablehi"] = f"{min(sv):.2f}", f"{max(sv):.2f}"
    dr = d["dynamics"]["drift"]
    numbers["driftshare"] = f"{100 * dr['cross-focal']['offset_share_of_sq_error']:.0f}"
    numbers["driftshareref"] = f"{100 * dr['reference']['offset_share_of_sq_error']:.0f}"
    numbers["driftslopecf"] = f"{dr['cross-focal']['offset_on_tempo_slope']:+.2f}"
    numbers["driftsloperef"] = f"{dr['reference']['offset_on_tempo_slope']:+.2f}"
    numbers["ncurves"] = str(d["dynamics"]["n_curves"])
    lat = json.loads((ROOT / "results/figdata/latent.json").read_text())
    for pair, tag in (("t7|t8", "seveneight"), ("t6|t8", "sixeight"), ("t5|t8", "fiveeight"), ("t4|t8", "foureight")):
        vals = [lat[m]["pairs_ci"][f"head|{pair}"] for m in ("single-plane transformer", "cross-focal") if "pairs_ci" in lat[m]]
        if vals:
            numbers[f"pair{tag}lo"] = f"{min(v[0] for v in vals):.2f}"
            numbers[f"pair{tag}hi"] = f"{max(v[0] for v in vals):.2f}"
            numbers[f"pair{tag}cilo"] = f"{min(v[1] for v in vals):.2f}"
            numbers[f"pair{tag}cihi"] = f"{max(v[2] for v in vals):.2f}"
    numbers["attnrho"] = f"{d['where']['rho_median']:.2f}"
    numbers["attnrhofixed"] = f"{d['where']['rho_fixed_median']:.2f}"
    numbers["attndevr"] = f"{d['where']['r_deviation']:.2f}"
    numbers["attnvarphase"] = f"{100 * d['where']['var_share_phase']:.0f}"
    numbers["attnvarvideo"] = f"{100 * d['where']['var_share_video_phase']:.0f}"
    z = np.load(ROOT / "results/figdata/plane_attention_test.npz", allow_pickle=True)   # written by this repo
    numbers["attnframes"] = f"{len(z['label']):,}".replace(",", "\\,")


def defect_numbers(numbers: dict) -> None:
    """Per-video defect audit (data/qc/nantes_video_defects_v4.json: three tiers, inspection plus the image--annotation
    timeline audit) and its sensitivity analysis (results/noise_impact.json, results/derived_intervals_nodefect.json,
    results/v36_trainclean.json, results/v37_trainclean.json). Also writes Supplementary Tables 4 and 5."""
    qc_path, ni_path = ROOT / "data/qc/nantes_video_defects_v4.json", ROOT / "results/noise_impact.json"
    if not (qc_path.exists() and ni_path.exists()):
        return
    qc, ni = json.loads(qc_path.read_text()), json.loads(ni_path.read_text())
    v1 = json.loads((ROOT / "data/qc/nantes_video_defects_v1.json").read_text())["videos"]
    hard = json.loads((ROOT / "results/hard_videos.json").read_text())
    vids = qc["videos"]
    tiers = {t: [v for v, e in vids.items() if e["tier"] == t] for t in "ABC"}
    numbers["ndefect"] = str(len(vids))
    numbers["ndefectvone"] = str(len(v1))
    for t in "ABC":
        numbers[f"ndefecttier{t}"] = str(len(tiers[t]))
    numbers["ndefectimg"] = str(ni["n_image_scan_only"])
    numbers["ndefectmodel"] = str(len(v1) - ni["n_image_scan_only"])
    cat = ni["categories"]
    for k, tag in (("timeline_mismatch", "timeline"), ("embryo_removed", "removed"), ("dark_frames", "dark"),
                   ("no_division", "nodiv"), ("heavy_fragmentation", "frag"), ("blank_throughout", "blank"),
                   ("annotation_ahead", "ahead"), ("abrupt_cleavage", "abrupt"), ("out_of_focus", "oof")):
        numbers[f"cat{tag}"] = str(cat.get(k, 0))
    tl = {v: e for v, e in vids.items() if "timeline_mismatch" in e["defects"]}
    numbers["ntimelinenew"] = str(sum(1 for v in tl if v not in v1))
    numbers["ntimelinebehind"] = str(sum(1 for v in tl if v in v1 and "annotation_behind" in v1[v]["defects"]))
    numbers["ntimelinemiscat"] = str(sum(1 for v in tl if v in v1 and "annotation_behind" not in v1[v]["defects"]))
    numbers["nbehindvone"] = str(sum(1 for e in v1.values() if "annotation_behind" in e["defects"]))
    sl = np.array([e["timeline"]["slope"] for e in tl.values()])
    red = np.array([e["timeline"]["phase_error_identity"] / e["timeline"]["phase_error_fit"] for e in tl.values()])
    numbers["tlslopemed"], numbers["tlslopelo"], numbers["tlslopehi"] = f"{np.median(sl):.2f}", f"{np.percentile(sl, 10):.2f}", f"{np.percentile(sl, 90):.2f}"
    numbers["tlredlo"], numbers["tlredhi"] = f"{np.min(red):.1f}", f"{np.max(red):.1f}"
    numbers["tlredmed"] = f"{np.median(red):.1f}"
    numbers["ncutlabels"] = str(sum(1 for e in vids.values() if "cutoff_h" in e and e["tier"] != "A"))
    al = pd.read_csv(ROOT / "results/frame_timing/alignment_oof.csv")
    numbers["tlnvideos"] = str(len(al))
    avs = qc.get("audit_vs_signature", {})
    numbers["tlsignature"] = str(int(al.file_signature.sum()))
    numbers["tlagree"] = str(sum(1 for v, e in tl.items() if e["timeline"].get("file_signature") and not e["timeline"].get("decided_by_inspection")))
    numbers["tldisagree"] = str(len(avs.get("disagree", [])))
    numbers["tlbyeye"] = str(sum(1 for e in tl.values() if e["timeline"].get("decided_by_inspection")))
    numbers["tlnotinspected"] = str(sum(1 for e in tl.values() if e.get("basis", "").startswith("out-of-fold alignment and time-file")))
    numbers["tlinspected"] = str(len(tl) - int(numbers["tlnotinspected"]))
    numbers["tlrejectedrsix"] = str(sum(1 for v, e in vids.items() if "timeline_fit_v3" in e))
    numbers["ncleared"] = str(len(qc.get("inspected_and_cleared", {})))
    numbers["ncatbehind"] = str(sum(1 for e in vids.values() if "annotation_behind" in e["defects"]))
    numbers["ncattrunc"] = str(sum(1 for e in vids.values() if "annotation_truncated" in e["defects"]))
    numbers["ncatarrest"] = str(sum(1 for e in vids.values() if "developmental_arrest" in e["defects"]))
    tr_v1 = set(json.loads((ROOT / "data/splits/nantes_grouped_v2_trainclean_v1.json").read_text())["videos"]["train"])
    numbers["tlinsamplemissed"] = str(sum(1 for v in tl if v in tr_v1))
    numbers["tlidmed"] = f"{al.cost_identity.median():.2f}"
    # hard-video statistics quoted for the individual outliers
    by = {r["video"]: r for r in hard["flagged"]}
    two = [by[v] for v in ("LV683-2-8", "DRL1048-1") if v in by]
    if two:
        numbers["nodivmedlo"] = f"{min(r['median_acc'] for r in two):.2f}"
        numbers["nodivmedhi"] = f"{max(r['median_acc'] for r in two):.2f}"
        numbers["nodivbest"] = f"{max(r['best_acc'] for r in two):.2f}"
        numbers["nodivnconfigs"] = f"{min(r['n_configs'] for r in two)}--{max(r['n_configs'] for r in two)}"
    if "DC307-1" in by:
        r = by["DC307-1"]
        numbers["dcmed"], numbers["dcbest"] = f"{r['median_acc']:.2f}", f"{r['best_acc']:.2f}"
    # training contamination by tier-A videos
    tc = ni["training_contamination"]
    own = {k: x for k, x in tc.items() if not k.startswith("nantes_siblingleak")}
    pct = [100 * x["tier_A"] / x["videos"] for x in own.values()]
    numbers["trainbadpctlo"], numbers["trainbadpcthi"] = f"{min(pct):.1f}", f"{max(pct):.1f}"
    numbers["trainbadframehi"] = f"{100 * max(x['tier_A_frame_share'] for x in own.values()):.1f}"
    numbers["trainbadvtwo"] = str(tc["nantes_grouped_v2"]["tier_A"])
    numbers["trainbadsibclean"] = str(tc["nantes_siblingleak_clean_v1"]["tier_A"])
    numbers["trainbadsibleaky"] = str(tc["nantes_siblingleak_leaky_v1"]["tier_A"])
    # evaluation-side sensitivity; columns: none, image-scan set, tier A, every logged video
    cols = (("none", ""), ("image_scan_only", "img"), ("tier_A", "tA"), ("all_inspected", "clean"))
    rk = ni["fold0_ranking"]
    numbers["dfnconfigs"] = str(rk["n_configs"])
    numbers["dfbest"] = f"{rk['original_best']:.3f}"
    for dn, dt in cols[1:]:
        numbers[f"dfrho{dt}"] = f"{rk[dn]['spearman']:.3f}"
        numbers[f"dftopfive{dt}"] = str(rk[dn]["top5_kept"])
        numbers[f"dfbest{dt}"] = f"{rk[dn]['best']:.3f}"
    for split, tag in (("grouped_v1", "gvOne"), ("grouped_v2", "gvTwo")):
        h = ni["headline_comparison"][split]
        numbers[f"df{tag}ntest"] = str(h["n_defective_test"])
        for dn, dt in cols:
            numbers[f"df{tag}gap{dt}"] = f"{h[dn]['gap']:+.3f}"
            numbers[f"df{tag}cf{dt}"] = f"{h[dn]['crossfocal']:.3f}"
            numbers[f"df{tag}ref{dt}"] = f"{h[dn]['reference']:.3f}"
    ff = ni["released_fivefold"]
    numbers["dfslots"] = str(ff["n_defective_test_slots"])
    for dn, dt in cols:
        numbers[f"dfff{dt}ref"] = f"{ff[dn]['reference']:.3f}"
        numbers[f"dfff{dt}cf"] = f"{ff[dn]['crossfocal']:.3f}"
        numbers[f"dfff{dt}gap"] = f"{ff[dn]['gap']:+.3f}"
    dg = ni["diagnosis"]
    for key, t in (("share_of_errors_on_tier_A", "tA"), ("share_of_errors_on_defective", "")):
        shares = [100 * dg[m][key] for m in ("reference", "crossfocal")]
        numbers[f"dferrshare{t}lo"], numbers[f"dferrshare{t}hi"] = f"{min(shares):.0f}", f"{max(shares):.0f}"
    numbers["dffzerotA"], numbers["dffzerotest"] = str(dg["crossfocal"]["n_tier_A_test"]), str(dg["crossfocal"]["n_test"])
    numbers["dffzerotApct"] = f"{100 * dg['crossfocal']['n_tier_A_test'] / dg['crossfocal']['n_test']:.1f}"
    # computed from the three-decimal values the table prints, so the text and Supplementary Table 5 agree
    img_moves = [abs(round(ff["image_scan_only"]["gap"], 3) - round(ff["none"]["gap"], 3))] + [
        abs(round(ni["headline_comparison"][s_]["image_scan_only"]["gap"], 3) - round(ni["headline_comparison"][s_]["none"]["gap"], 3))
        for s_ in ("grouped_v1", "grouped_v2")]
    numbers["dfimggapmax"] = f"{max(img_moves):.3f}"
    for m, t in (("reference", "ref"), ("crossfocal", "cf")):
        for dn, dt in (("none", ""), ("tier_A", "tA"), ("all_inspected", "clean")):
            numbers[f"dffar{t}{dt}"] = f"{100 * dg[m][dn]['far_share']:.0f}"
            numbers[f"dfmid{t}{dt}"] = f"{100 * dg[m][dn]['mid_cleavage_share']:.0f}"
    nd = ROOT / "results/derived_intervals_nodefect.json"
    if nd.exists():
        d = json.loads(nd.read_text())
        for model, mtag in (("reference", "ref"), ("crossfocal", "cf")):
            for iv in ("cc2", "s2"):
                x = d.get(model, {}).get(iv)
                if x:
                    numbers[f"{iv}{mtag}_nd_abs"] = f"{x['abs_mean']:.2f}"
                    numbers[f"{iv}{mtag}_nd_med"] = f"{x['median_abs']:.2f}"
                    numbers[f"{iv}{mtag}_nd_within"] = f"{100 * x['within_1h']:.0f}"
            for ck, cv in d.get(model, {}).get("comparisons", {}).items():
                if ck in ("cc2", "s2"):
                    t = ck.replace("cc2", "cctwo").replace("s2", "stwo")
                    numbers[f"nd{t}{mtag}modelwithin"] = f"{100 * cv['model_within_1h']:.0f}"
                    numbers[f"nd{t}{mtag}constwithin"] = f"{100 * cv['const_within_1h']:.0f}"
    # retraining without the defective training videos: v36 (log v1) and v37 (log v2, tier A, labels cut)
    done = {}
    for tag, pre in (("v36", "tcone"), (PRIMARY_RETRAIN, "tc")):
        f = ROOT / f"results/{tag}_trainclean.json"
        done[tag] = False
        if not f.exists():
            continue
        doc = json.loads(f.read_text())
        rec = doc["recipes"]
        done[tag] = all(r.get("status") == "done" for r in rec.values()) and "contrasts" in doc
        for name, t in (("per_frame", "frame"), ("reference_lstm", "ref"), ("crossfocal", "cf")):
            r = rec.get(name, {})
            if r.get("status") != "done":
                continue
            x = r.get("seed_aware", r["all_test"])
            numbers[f"{pre}{t}delta"] = f"{x['delta']:+.3f}"
            numbers[f"{pre}{t}lo"], numbers[f"{pre}{t}hi"] = f"{x['ci95'][0]:+.3f}", f"{x['ci95'][1]:+.3f}"
            if "seed_ttest_p" in x:
                numbers[f"{pre}{t}p"] = f"{x['seed_ttest_p']:.3f}"
            ct = r["clean_test"]
            numbers[f"{pre}{t}ctdelta"] = f"{ct['delta']:+.3f}"
        for arm, t in (("noisy", "noisy"), ("trainclean", "clean")):
            g = doc.get("headline_gap_seed_aware", doc.get("headline_gap", {})).get(arm)
            if g:
                numbers[f"{pre}gap{t}"] = f"{g['gap']:+.3f}"
                numbers[f"{pre}gap{t}lo"], numbers[f"{pre}gap{t}hi"] = f"{g['ci95'][0]:+.3f}", f"{g['ci95'][1]:+.3f}"
        for k, c in doc.get("contrasts", {}).items():
            t = "ref" if k.endswith("lstm") else "cf"
            numbers[f"{pre}contrast{t}"] = f"{c['delta']:+.3f}"
            numbers[f"{pre}contrast{t}lo"], numbers[f"{pre}contrast{t}hi"] = f"{c['ci95'][0]:+.3f}", f"{c['ci95'][1]:+.3f}"
    numbers["tcdone"] = "1" if done.get(PRIMARY_RETRAIN) else "0"
    for tag, pre in ((PRIMARY_RETRAIN, "tc"), ("v36", "tcone")):
        lo = numbers.get(f"{pre}gapcleanlo")
        numbers[f"{pre}gapholds"] = "1" if lo is not None and float(lo) > 0 else "0"
    numbers["tconedone"] = "1" if done.get("v36") else "0"

    # Supplementary Table 5: sensitivity summary
    S = ["\\begin{table}[h]", "\\centering\\footnotesize",
         "\\caption{\\textbf{Sensitivity of each result to the logged videos.} Existing predictions re-scored without them; no "
         "model is retrained. \\emph{Image-scan set}: the " + numbers["ndefectimg"] + " videos found by the model-free image scan, "
         "which no model selected. \\emph{Tier A}: the " + numbers["ndefecttierA"] + " videos whose labels are not about their "
         "images. \\emph{All logged}: every video of Supplementary Table~4. Temporal accuracy \\pt{}; gap = cross-focal minus "
         "reference; best = best configuration of the common-recipe sweep.}\\label{tab:defect_sens}",
         "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         "Quantity & All videos & w/o image-scan set & w/o tier A & w/o all logged \\\\", "\\midrule"]

    def row(label, *vals):
        S.append(f"{label} & " + " & ".join(vals) + " \\\\")
    row("Fold 0, best of " + numbers["dfnconfigs"], numbers["dfbest"], numbers["dfbestimg"], numbers["dfbesttA"], numbers["dfbestclean"])
    row("Fold 0, Spearman $\\rho$ with all-video ranking", "1", numbers["dfrhoimg"], numbers["dfrhotA"], numbers["dfrhoclean"])
    for lab, k in (("Released 5 folds, reference", "ref"), ("Released 5 folds, cross-focal", "cf"), ("Released 5 folds, gap", "gap")):
        row(lab, *[numbers[f"dfff{dt}{k}"] for _, dt in cols])
    for tag, lab in (("gvOne", "Grouped v1, gap"), ("gvTwo", "Grouped v2, gap")):
        row(lab, *[numbers[f"df{tag}gap{dt}"] for _, dt in cols])
    S += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "table_defect_sensitivity.tex").write_text("\n".join(S) + "\n")

    # Supplementary Table 4: per-video defect log
    esc_ = lambda t: t.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_").replace("~", "$\\sim$")
    L = ["\\begin{footnotesize}",
         "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{2.1cm}>{\\centering\\arraybackslash}p{0.6cm}>{\\raggedright\\arraybackslash}p{3.0cm}>{\\raggedright\\arraybackslash}p{9.0cm}@{}}",
         "\\caption{\\textbf{Per-video defect log.} Tier A: the label is not about the image (removed from training in the "
         "cleaned split); B: image artefact with a correct label (kept; for an embryo removed from the well the label is cut "
         "where the well empties); C: atypical biology (kept). Descriptions of inspected videos are what is visible, not a "
         "guess at the cause. Timeline mismatches were found by the out-of-fold image--annotation alignment audit and the release's time files (Methods): $a$ is the "
         "fitted number of annotation frames per image and the error is the mean phase-order difference between a per-frame "
         "classifier and the annotation, pairing image $k$ with annotation frame $k$ versus with the fitted frame. Released "
         "as \\texttt{nantes\\_video\\_defects\\_v4.json}. Basis: what a defect rests on -- AI-assisted inspection of the frames "
         "(a vision-capable language model under the authors' direction), or, for timeline mismatches not inspected, agreement of the "
         "out-of-fold alignment and the time-file signature.}\\label{tab:defect_log}\\\\",
         "\\toprule", "Video & Tier & Defect & What is visible \\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", "Video & Tier & Defect & What is visible \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]
    for v, rec in sorted(vids.items(), key=lambda kv: (kv[1]["tier"], kv[1]["defects"][0], kv[0])):
        note = rec["note"]
        if "timeline" in rec:
            t = rec["timeline"]
            note = (note + " " if note else "") + (f"Timeline: $a={t['slope']:.2f}$; phase error "
                                                   f"{t['phase_error_identity']:.2f} $\\to$ {t['phase_error_fit']:.2f}.")
        if "cutoff_h" in rec and rec["tier"] != "A":
            note += f" Label cut at {rec['cutoff_h']:.1f} h."
        note += " Basis: " + ("alignment and time-file signature." if rec.get("basis", "").startswith("out-of-fold alignment and time-file") else "inspection.")
        L.append(f"{esc_(v)} & {rec['tier']} & {esc_(', '.join(c.replace('_', ' ') for c in rec['defects']))} & "
                 f"{esc_(note) if 'Timeline' not in note else note} \\\\")
    L += ["\\end{longtable}", "\\end{footnotesize}"]
    (TAB / "table_defect_log.tex").write_text("\n".join(L) + "\n")
    print(f"defect tables: {len(vids)} videos")


def write_numbers(numbers: dict) -> None:
    def key(k):  # LaTeX macro names: letters only
        return "num" + re.sub(r"[^A-Za-z]", lambda m: {"_": "", "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}.get(m.group(0), ""), k)
    L = ["% generated by scripts/build_results_master.py -- do not edit; every in-text number comes from here"]
    for k, v in sorted(numbers.items()):
        L.append(f"\\newcommand{{\\{key(k)}}}{{{v}}}")
    (ROOT / "outputs/numbers.tex").write_text("\n".join(L) + "\n")
    (ROOT / "results/numbers.json").write_text(json.dumps(numbers, indent=1))
    print(f"numbers.tex: {len(numbers)} macros")


PROTO_MODELS = [("resnet18_lstm_L4_split0", "resnet18_lstm_L4_split{k}", "resnet18_lstm_L4_grouped_v1", "ResNet-18-LSTM, 4 frames, 1 plane (released reference)"),
                ("r2plus1d_L8_split0", "r2plus1d_L8_split{k}", "r2plus1d_L8_grouped_v1", "R(2+1)D-18, 8 frames (released 3D baseline)"),
                ("resnet50_lstm_L8_split0", None, "resnet50_lstm_L8_grouped_v1", "ResNet-50-LSTM, 8 frames, 1 plane"),
                ("resnet18_transformer_L16_evalfix_split0", "resnet18_transformer_L16_evalfix_split{k}", "resnet18_transformer_L16_evalfix_grouped_v1", "ResNet-18 transformer, 16 frames, 1 plane, corrected window"),
                ("resnet18_transformer_L16_crossfocal7_evalfix_split0", "resnet18_transformer_L16_crossfocal7_evalfix_split{k}", "resnet18_transformer_L16_crossfocal7_evalfix_grouped_v1", "ResNet-18 transformer, 16 frames, 7 planes, cross-focal attention")]


def _paired_footnote(df: pd.DataFrame, numbers: dict, foot: list, ver: str, tag: str) -> None:
    """Per-video paired test of cross-focal minus reference ON a patient-grouped split.

    The corrected protocol has to be able to carry a comparison, not only a pair of point estimates
    (reviewer objection C2). Silently does nothing while the runs for ``ver`` do not exist."""
    cf = f"resnet18_transformer_L16_crossfocal7_evalfix_grouped_{ver}"
    ref = f"resnet18_lstm_L4_grouped_{ver}"
    ids, d = paired_videos_labelled(cf, ref, df)
    if d is None or len(d) <= 5:
        return
    ps = pooled_step([(cf, ref)], df)                     # seed-aware, the same interval design as every paired comparison
    if ps is None:
        return
    m, lo, hi, n_pat = ps["d"], ps["lo"], ps["hi"], ps["npat"]
    # the test is at the patient level too: videos of one patient are averaged first, so the t-test
    # and the win count use the same independent unit as the interval and as the split itself.
    by_pat: dict[str, list[float]] = {}
    for v, x in zip(ids, d):
        by_pat.setdefault(patient_of(v), []).append(float(x))
    per_pat = np.array([np.mean(x) for x in by_pat.values()])
    tt = stats.ttest_1samp(per_pat, 0.0)
    wins = int((per_pat > 0).sum())
    mant, exp = f"{tt.pvalue:.1e}".split("e")
    ptex = f"{mant}\\times 10^{{{int(exp)}}}"
    numbers[f"{tag}_delta"] = f"{m:+.3f}"; numbers[f"{tag}_lo"] = f"{lo:+.3f}"; numbers[f"{tag}_hi"] = f"{hi:+.3f}"
    numbers[f"{tag}_n"] = str(len(d)); numbers[f"{tag}_wins"] = str(wins); numbers[f"{tag}_df"] = str(len(per_pat) - 1)
    numbers[f"{tag}_npat"] = str(n_pat)
    numbers[f"{tag}_p"] = ptex
    name = "patient-grouped split" if ver == "v1" else f"couple-level split ({ver})"
    foot.append(f"\\multicolumn{{10}}{{l}}{{\\footnotesize per-video paired bootstrap on the {name}, "
                f"cross-focal $-$ reference, seed-averaged, resampling patients and seeds ($n={n_pat}$ patients, "
                f"{len(d)} videos): $\\Delta p_t={m:+.3f}$ "
                f"(95\\,\\% CI ${lo:+.3f}$ to ${hi:+.3f}$), paired $t({len(per_pat) - 1})$ $p={ptex}$, wins {wins}/{n_pat} patients}}\\\\")


def bold_cell(c: str) -> str:
    """Bold a table cell, handling the `$a\\pm b$` form that most cells take."""
    if c.startswith("$") and c.endswith("$"):
        inner = c[1:-1]
        if "\\pm" in inner:
            a, b = inner.split("\\pm", 1)
            return "$\\mathbf{" + a + "}\\pm\\mathbf{" + b + "}$"
        return "$\\mathbf{" + inner + "}$"
    return "\\textbf{" + c + "}"


def protocol_table(df: pd.DataFrame, numbers: dict) -> None:
    """Main Table 1: reference models under the released protocol (fold 0; five-fold mean) and the corrected patient-grouped split."""
    def cell(exp, key, nd=3):
        if exp is None: return "--", 0
        r = df[df.experiment == exp][key].dropna()
        if len(r) == 0: return "--", 0
        return (f"{r.iloc[0]:.{nd}f}" if len(r) == 1 else f"${r.mean():.{nd}f}\\pm{r.std(ddof=1):.{nd}f}$"), len(r)
    rows = []
    for f0, fk, gr, label in PROTO_MODELS:
        pt0, n0 = cell(f0, "p_t"); f10, _ = cell(f0, "f1_50", 1); mae0, _ = cell(f0, "mae_h", 2)
        if fk:
            fm = [df[df.experiment == fk.format(k=k)].p_t.mean() for k in range(5)]
            fm = [x for x in fm if not np.isnan(x)]
            pt5 = f"${np.mean(fm):.3f}\\pm{np.std(fm, ddof=1):.3f}$" if len(fm) == 5 else "--"
        else:
            pt5 = "--"
        ptg, ng = cell(gr, "p_t"); f1g, _ = cell(gr, "f1_50", 1); maeg, _ = cell(gr, "mae_h", 2)
        def _v(exp, key):
            r = df[df.experiment == exp][key].dropna() if exp else []
            return float(r.mean()) if len(r) else None
        fm5 = float(np.mean(fm)) if fk and len(fm) == 5 else None
        rows.append({"cells": [label, str(n0), pt0, f10, mae0, pt5, str(ng) if ng else "--", ptg, f1g, maeg],
                     "vals": {2: _v(f0, "p_t"), 3: _v(f0, "f1_50"), 4: _v(f0, "mae_h"), 5: fm5,
                              7: _v(gr, "p_t"), 8: _v(gr, "f1_50"), 9: _v(gr, "mae_h")}})
        tag = {"resnet18_lstm_L4_split0": "ref", "resnet18_transformer_L16_crossfocal7_evalfix_split0": "cf"}.get(f0)
        if tag and ng:
            rg = df[df.experiment == gr].p_t.dropna()
            numbers[f"grouped{tag}_pt"] = f"{rg.mean():.3f}"; numbers[f"grouped{tag}_pt_sd"] = f"{rg.std(ddof=1):.3f}" if ng > 1 else ""; numbers[f"grouped{tag}_n"] = str(ng)
    cap = ("\\textbf{Reference models under the released and the corrected protocol.} Released protocol: official video-level fold 0 (67 test videos; the fold on which "
           "every configuration in this study was selected) and the mean $\\pm$ s.d.\\ over the five official fold means (Table~\\ref{tab:folds}). Corrected protocol "
           "(CLEAVE): patient-grouped split (\\texttt{nantes\\_grouped\\_v1}, \\numgroupedTest{} test videos, grouped at treatment-cycle level; four couples straddle a partition at couple level and \\numgvOneexposed{} test video is exposed, Methods). "
           "$n$: seeds. Cells are mean $\\pm$ sample s.d.\\ over seeds, single value when $n=1$. Test sets differ between protocols, so the two blocks are not paired; the best value in each metric column is in bold. "
           "The grouped-split cells are not an estimate of leakage inflation, because the two protocols also differ in which videos are tested and in how many are trained on; that estimate comes from the paired design of Results~\\S\\ref{sec:structural}.")
    # bold the best value in each metric column: highest p_t and F1@50, lowest timing error
    best = {}
    for col in (2, 3, 4, 5, 7, 8, 9):
        vals = {i: r["vals"].get(col) for i, r in enumerate(rows) if r["vals"].get(col) is not None}
        if vals:
            best[col] = (min if col in (4, 9) else max)(vals, key=lambda i: vals[i])
    body = []
    for i, r in enumerate(rows):
        c = list(r["cells"])
        for col, bi in best.items():
            if bi == i and c[col] != "--":
                c[col] = bold_cell(c[col])
        body.append(" & ".join(c) + " \\\\")
    rows = body
    # paired per-video test ON the grouped split itself: the corrected protocol has to be able to carry a
    # comparison, not only a pair of point estimates (reviewer objection C2).
    foot = []
    for ver, tag in (("v1", "groupedpaired"), ("v2", "groupedpairedTwo")):
        _paired_footnote(df, numbers, foot, ver, tag)
    if foot:
        rows = rows + ["\\midrule"] + foot
    body_p, notes_p = footer_to_notes(rows)
    out = ["\\begin{table*}[t]", "\\centering\\small\\setlength{\\tabcolsep}{4pt}", "\\caption{" + cap + "}\\label{tab:protocols}",
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{L{5.6cm}cccccccccc}", "\\toprule",
           " & \\multicolumn{4}{c}{released protocol, fold 0} & five folds & \\multicolumn{4}{c}{corrected protocol (patient-grouped)} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-10}",
           "Model & $n$ & $p_t$ & F1@50 & MAE (h) & $p_t$ & $n$ & $p_t$ & F1@50 & MAE (h) \\\\", "\\midrule"] + body_p + ["\\bottomrule", "\\end{tabular}}"] + notes_block(notes_p) + ["\\end{table*}"]
    (TAB / "table_protocols.tex").write_text("\n".join(out) + "\n")
    g = df[df.split_key == "grouped"]
    numbers["groupedTest"] = str(int(g.n_videos.iloc[0])) if len(g) else "?"


def main() -> None:
    df = load_runs()
    df.to_csv(ROOT / "results/runs_master.csv", index=False)
    print(f"runs_master.csv: {len(df)} runs, {df.experiment.nunique()} configurations")
    numbers: dict[str, str] = {}
    write_ed_table1(df)
    fold_table(df, numbers)
    negative_table(df, numbers)
    hsmm_table(numbers)
    core_numbers(df, numbers)
    v2_ceiling_numbers(df, numbers)
    outlier_sensitivity(df, numbers)
    protocol_table(df, numbers)
    extra = ROOT / "results/numbers_extra.json"
    if extra.exists():
        numbers.update(json.loads(extra.read_text()))
    add_leakage_numbers(numbers)
    defect_numbers(numbers)
    figure_numbers(numbers)
    protocol_extra_numbers(numbers)
    write_numbers(numbers)


if __name__ == "__main__":
    main()
