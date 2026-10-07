#!/usr/bin/env python3
"""Do the released images and the released annotation share one time axis?

The released loader, and ours, pair the k-th image of a video (run-number order) with annotation frame k. That is only
right when both come from the same acquisition sequence. For 36 clean videos the time and annotation files list 115-160
more frames than there are images, and a round-4 reviewer found that their "annotated too late" labels cluster there.

This script lets the images decide. A per-frame classifier predicts a phase for every image, labelled or not; for each
video a linear map from image index k to annotation frame j = a*k + b is fitted by minimising the mean absolute
difference, in phase order, between prediction and annotation; the fitted slope is then compared with the identity
(a = 1, b = 0) and with the slope the counts imply (a = time rows / images).

    PYTHONPATH=src python scripts/audit_frame_timing.py --infer     # CPU inference -> results/frame_timing/preds.npz
    PYTHONPATH=src python scripts/audit_frame_timing.py             # fit -> results/frame_timing/alignment.csv

Out-of-fold:

    for k in 0 1 2 3 4; do PYTHONPATH=src python scripts/audit_frame_timing.py --infer --fold $k --device cuda:0; done
    PYTHONPATH=src python scripts/audit_frame_timing.py --oof       # -> results/frame_timing/alignment_oof.csv

The fit also records a model-free signature from the release's own files: a time file on an exactly uniform 0.2-h
grid, or annotation frames beyond the last image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stseg.data.nantes_kinetic import CLASS_NAMES, NantesKineticFrames
from stseg.kinetic.models import build_model

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("data/raw/nantes_embryo_dataset")
RUN = ROOT / "runs/h7/resnet18_none_grouped_v2_trainclean_seed0"
OUT = ROOT / "results/frame_timing"
SLOPES = np.round(np.arange(0.60, 1.601, 0.005), 3)
OFFSETS = np.arange(-80, 81, 1)
STRIDE = 3


def infer(run: Path = RUN, split: str | None = None, out: Path | None = None, device: str = "cpu") -> None:
    torch.set_num_threads(32)
    cfg = json.loads((run / "config.resolved.json").read_text())
    d = cfg["data"]
    ds = NantesKineticFrames(d["manifest"], d["split"], "test", mode="eval", plane=d.get("plane", "embryo_dataset"),
                             resize=d.get("resize", 250), crop=d.get("crop", 224), cache_dir=d.get("cache_dir"))
    m = pd.read_csv(ROOT / d["manifest"])
    m = m[m.plane == "embryo_dataset"].sort_values(["video", "frame_index"]).reset_index(drop=True)
    m = m[m.frame_index % STRIDE == 1].reset_index(drop=True)   # every STRIDE-th image is plenty to fit a slope
    if split:                                                   # out-of-fold: only the videos this model never saw
        held = set(json.loads((ROOT / split).read_text())["videos"]["test"])
        m = m[m.video.isin(held)].reset_index(drop=True)
    ds.rows = m.assign(label=0)                      # every image, labelled or not; the label is never used here
    model = build_model(dict(cfg["model"]), ds.in_channels)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
    model.to(device).eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=12)
    lp = []
    with torch.no_grad():
        for i, b in enumerate(loader):
            lp.append(torch.log_softmax(model(b["image"].to(device))["logits"][:, 0].float(), 1).cpu().numpy().astype(np.float16))
            if i % 100 == 0:
                print(f"batch {i}/{len(loader)}", flush=True)
    np.savez_compressed(out or OUT / "preds.npz", log_probs=np.concatenate(lp), video=m.video.to_numpy().astype(str),
                        frame_index=m.frame_index.to_numpy())
    print(f"wrote {len(m)} frame predictions")


def annotation(video: str) -> np.ndarray:
    """Phase index for annotation frames 1..last (-1 where unannotated), from the released *_phases.csv."""
    a = pd.read_csv(RAW / "ann/embryo_dataset_annotations" / f"{video}_phases.csv", header=None, names=["phase", "s", "e"])
    lab = np.full(int(a.e.max()) + 1, -1)
    for p, s, e in a.itertuples(index=False):
        lab[int(s):int(e) + 1] = CLASS_NAMES.index(p)
    return lab


def fit(pred: np.ndarray, k: np.ndarray, lab: np.ndarray) -> tuple[float, float, float, float]:
    """Best (a, b) of j = a*k + b over the grid; returns a, b, cost at best, cost at identity. Cost = mean |phase
    difference| over images that land on an annotated frame, required to cover at least half the images."""
    def cost(a: float, b: float) -> float:
        j = np.rint(a * k + b).astype(int)
        ok = (j >= 0) & (j < len(lab))
        ok[ok] &= lab[j[ok]] >= 0
        if ok.sum() < min(len(pred), int((lab >= 0).sum()) / STRIDE) / 2:   # cover half of what both sides share
            return np.inf
        return float(np.abs(pred[ok] - lab[j[ok]]).mean())

    best = min(((cost(a, b), a, b) for a in SLOPES for b in OFFSETS), key=lambda t: t[0])
    return best[1], float(best[2]), best[0], cost(1.0, 0.0)


def signature(video: str, n_images: int) -> bool:
    """Model-free: time file on an exactly uniform 0.2-h grid, or annotation frames beyond the last image."""
    t = pd.read_csv(RAW / "time_elapsed/embryo_dataset_time_elapsed" / f"{video}_timeElapsed.csv").time.to_numpy()
    d = np.round(np.diff(t), 2)
    d = d[d > 0]
    return bool(np.mean(d == 0.2) > 0.95 or len(annotation(video)) - 1 > n_images + 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--fold", type=int, default=None, help="out-of-fold inference with the audit model of this fold")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--oof", action="store_true", help="fit on the five out-of-fold prediction files")
    args = ap.parse_args()
    if args.infer:
        if args.fold is None:
            infer(device=args.device)
        else:
            infer(ROOT / f"runs/h7/resnet18_none_oof_audit_k{args.fold}_seed0", f"data/splits/nantes_oof_audit_v1_k{args.fold}.json",
                  OUT / f"preds_oof_k{args.fold}.npz", args.device)
        return
    files = [OUT / f"preds_oof_k{k}.npz" for k in range(5)] if args.oof else [OUT / "preds.npz"]
    zs = [np.load(f) for f in files]
    pred_all = np.concatenate([z["log_probs"].astype(np.float32).argmax(1) for z in zs])
    video = np.concatenate([z["video"] for z in zs])
    fidx = np.concatenate([z["frame_index"] for z in zs])
    rows = []
    for v in np.unique(video):
        sel = video == v
        order = np.argsort(fidx[sel])
        pred, kk = pred_all[sel][order], fidx[sel][order]
        lab = annotation(v)
        n_time = sum(1 for _ in open(RAW / "time_elapsed/embryo_dataset_time_elapsed" / f"{v}_timeElapsed.csv")) - 1
        a, b, c_best, c_id = fit(pred, kk, lab)
        rows.append({"video": v, "images": int(kk.max()), "time_rows": n_time, "ann_last": len(lab) - 1,
                     "file_signature": signature(v, int(kk.max())),
                     "slope_implied": n_time / int(kk.max()), "slope_fit": a, "offset_fit": b,
                     "cost_identity": c_id, "cost_fit": c_best})
        print(f"{v:14} img {int(kk.max()):4d} rows {n_time:4d}  fit a={a:.3f} b={b:+4.0f}  cost {c_id:.2f} -> {c_best:.2f}", flush=True)
    df = pd.DataFrame(rows)
    name = "alignment_oof.csv" if args.oof else "alignment.csv"
    df.to_csv(OUT / name, index=False)
    print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
