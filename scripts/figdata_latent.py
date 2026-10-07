#!/usr/bin/env python3
"""Latent-space statistics for the "what the representation encodes" figure. CPU only.

For three fold-0 checkpoints with stored per-frame features (reference LSTM, single-plane transformer, cross-focal
transformer), on the test partition:

* ``adjacent``: for every pair of consecutive phases, balanced accuracy of a logistic-regression probe separating
  the two, 5-fold cross-validated with folds grouped by video, on backbone (per-frame) and head (temporal) features;
* ``velocity``: latent speed ||z_{t+1} - z_t|| (head features, z-scored per video) averaged in a +/- 20-frame window
  around each annotated onset, grouped by the kind of event;
* ``pca``: a 2-D PCA projection of every test frame (head features) for the trajectory panels.

    PYTHONPATH=src python scripts/figdata_latent.py      # -> results/figdata/latent.json, latent_pca.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stseg.data.nantes_kinetic import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "results/diagnostics"
MODELS = {"reference": "resnet18_lstm_L4_split0_seed1",
          "single-plane transformer": "resnet18_transformer_L16_evalfix_split0_seed0",
          "cross-focal": "resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0"}
EVENT_KIND = {"synchronous division": ["t2", "t4", "t8"], "transient division": ["t3", "t5", "t6", "t7"],
              "morphological": ["tM", "tSB", "tB", "tEB"]}
W = 20
PAIRS = [("t6", "t7"), ("t7", "t8"), ("t6", "t8"), ("t5", "t8"), ("t4", "t8"), ("t2", "t4")]
NBOOT_PROBE = 300


def load(run: str):
    f = np.load(DIAG / run / "features_test.npz", allow_pickle=True)
    p = np.load(DIAG / run / "frame_probs_test.npz", allow_pickle=True)
    vids = sorted({k.split("__")[0] for k in f.files})
    return {v: (f[f"{v}__bb"].astype(np.float32), f[f"{v}__hd"].astype(np.float32), p[f"{v}__y"].astype(int)) for v in vids}


def main() -> None:
    out, pca_store = {}, {}
    for name, run in MODELS.items():
        data = load(run)
        vids = list(data)
        y = np.concatenate([data[v][2] for v in vids])
        g = np.concatenate([[i] * len(data[v][2]) for i, v in enumerate(vids)])
        res = {"run": run, "n_videos": len(vids), "n_frames": int(len(y)), "adjacent": {}}
        for kind, idx in (("backbone", 0), ("head", 1)):
            X = np.concatenate([data[v][idx] for v in vids])
            row = {}
            present = [c for c in range(len(CLASS_NAMES)) if (y == c).sum() >= 50]
            for a, b in zip(present, present[1:]):
                s = (y == a) | (y == b)
                if len(np.unique(g[s])) < 5:
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"))
                pr = cross_val_predict(clf, X[s], y[s], groups=g[s], cv=GroupKFold(5))
                row[f"{CLASS_NAMES[a]}|{CLASS_NAMES[b]}"] = float(balanced_accuracy_score(y[s], pr))
            res["adjacent"][kind] = row
            print(name, kind, {k: round(v, 2) for k, v in row.items()}, flush=True)
        # latent velocity around annotated onsets (head features)
        prof = {k: [] for k in EVENT_KIND}
        for v in vids:
            z = data[v][1]
            z = (z - z.mean(0)) / (z.std(0) + 1e-6)
            sp = np.r_[np.nan, np.linalg.norm(np.diff(z, axis=0), axis=1)]
            sp = sp / np.nanmedian(sp)
            yy = data[v][2]
            for t in np.where(np.diff(yy) != 0)[0] + 1:
                ev = CLASS_NAMES[yy[t]]
                for kind, evs in EVENT_KIND.items():
                    if ev in evs and W <= t < len(yy) - W:
                        prof[kind].append(sp[t - W:t + W + 1])
        res["velocity"] = {k: {"mean": np.nanmean(p, 0).tolist(), "sem": (np.nanstd(p, 0) / np.sqrt(len(p))).tolist(),
                               "n_events": len(p)} for k, p in prof.items() if p}
        # PCA of head features for trajectories
        X = np.concatenate([data[v][1] for v in vids])
        pc = PCA(2, random_state=0).fit(X)
        pca_store[name] = (pc.transform(X).astype(np.float32), y, g, np.array(vids), pc.explained_variance_ratio_)
        res["pca_var"] = pc.explained_variance_ratio_.tolist()
        # non-adjacent pairs, tier-A videos left out, video-bootstrap intervals: does separability fall with cell count
        # (a representation limit) or only at t7, the least reliable label (inter-observer s.d. above its duration)?
        tier_a = {v for v, e in json.loads((ROOT / "data/qc/nantes_video_defects_v3.json").read_text())["videos"].items()
                  if e["tier"] == "A"}
        keep = np.isin(np.concatenate([[v] * len(data[v][2]) for v in vids]), sorted(tier_a), invert=True)
        rng = np.random.default_rng(0)
        res["pairs_ci"] = {}
        for kind, idx in (("backbone", 0), ("head", 1)):
            X = np.concatenate([data[v][idx] for v in vids])[keep]
            yk, gk = y[keep], g[keep]
            for a, b in PAIRS:
                s = (yk == CLASS_NAMES.index(a)) | (yk == CLASS_NAMES.index(b))
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"))
                pr = cross_val_predict(clf, X[s], yk[s], groups=gk[s], cv=GroupKFold(5))
                ys, gs = yk[s], gk[s]
                u = np.unique(gs)
                bs = []
                for _ in range(NBOOT_PROBE):
                    m = np.concatenate([np.where(gs == q)[0] for q in rng.choice(u, len(u))])
                    bs.append(balanced_accuracy_score(ys[m], pr[m]))
                res["pairs_ci"][f"{kind}|{a}|{b}"] = [float(balanced_accuracy_score(ys, pr)),
                                                      float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[name] = res
    (ROOT / "results/figdata/latent.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / "results/figdata/latent_pca.npz",
                        **{f"{n}|{k}": v for n, (xy, y, g, vids, var) in pca_store.items()
                           for k, v in (("xy", xy), ("y", y), ("g", g), ("videos", vids), ("var", var))})
    print("wrote results/figdata/latent.json, latent_pca.npz")


if __name__ == "__main__":
    main()
