#!/usr/bin/env python3
"""MS-TCN and ASFormer in the regime they were designed for.

Action-segmentation heads are trained on frozen per-frame features of whole videos, with a stage-wise loss. The
sweep trained them on 16-/64-frame clips end to end, where MS-TCN did not converge. Here, on released fold 0:

1. per-frame features are extracted with the frozen backbone of the single-plane window-corrected transformer of the
   same seed (trained on fold-0 training videos only), for every fold-0 video;
2. a head is trained on whole training videos: MS-TCN (4 stages x 10 dilated layers, 64 channels; Farha and Gall
   2019) or ASFormer (the repository's asformer_lite, encoder + 2 decoders), cross-entropy on every stage plus the
   truncated-MSE smoothing loss of MS-TCN (lambda 0.15, tau 4), Adam 5e-4, 50 epochs, one video per step;
3. the checkpoint with the best validation p_t is decoded on test with the run's own train transition matrix and
   scored with the paper's metric code.

    PYTHONPATH=src python scripts/native_tas_heads.py --device cuda:0      # -> results/native_tas_heads.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stseg.eval.kinetic_metrics import evaluate_videos
from stseg.kinetic.datasets import _frames
from stseg.kinetic.heads import ASFormerLiteHead
from stseg.kinetic.models import FullVideoMSTCN, build_model

ROOT = Path(__file__).resolve().parents[1]
SRC = "runs/h7/resnet18_transformer_L16_evalfix_split0_seed{s}"
CACHE = ROOT / "runs/h7/native_tas"
SEEDS = (0, 1, 2)
EPOCHS, LR, LAMBDA, TAU = 50, 5e-4, 0.15, 4.0


def extract(seed: int, device: str) -> dict:
    out_f = CACHE / f"features_seed{seed}.npz"
    if out_f.exists():
        return dict(np.load(out_f, allow_pickle=True))      # written by this script
    rd = ROOT / SRC.format(s=seed)
    cfg = json.loads((rd / "config.resolved.json").read_text())
    model = build_model(dict(cfg["model"]), 3)
    model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu", weights_only=True)["model"])
    bb = model.backbone.to(device).eval()
    store = {}
    for part in ("train", "val", "test"):
        ds = _frames(cfg["data"], part, "eval", 0)
        for vid, g in ds.rows.groupby("video", sort=False):
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, g.index.to_numpy()), batch_size=256, num_workers=8)
            with torch.no_grad():
                f = torch.cat([bb(b["image"].to(device)).float().cpu() for b in loader]).numpy().astype(np.float16)
            store[f"{part}|{vid}|f"] = f
            store[f"{part}|{vid}|y"] = g.label.to_numpy()
            store[f"{part}|{vid}|t"] = g.time_h.to_numpy(dtype=float)
        print(f"seed {seed} {part}: {sum(1 for k in store if k.startswith(part) and k.endswith('|f'))} videos", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_f, **store)
    return store


class ASFormerFull(nn.Module):
    """asformer_lite on whole videos with a classifier on every stage (encoder and each decoder)."""

    def __init__(self, d_in: int, num_classes: int = 16) -> None:
        super().__init__()
        self.body = ASFormerLiteHead(d_in, hidden=64, layers=9, n_decoders=3, num_classes=num_classes, dropout=0.3)
        self.final = nn.Linear(64, num_classes)

    def forward(self, x):
        b = self.body
        h = b.inp(x)
        for blk in b.enc:
            h = blk(h)
        enc, stages = h, [b.enc_cls(h)]
        logits = stages[0]
        for s, blocks in enumerate(b.dec):
            z = b.dec_in[s](logits.softmax(-1))
            for blk in blocks:
                z = blk(z, enc)
            logits = b.dec_cls[s](z) if s < len(b.dec_cls) else self.final(z)
            stages.append(logits)
        return {"logits": stages[-1], "stage_logits": torch.stack(stages)}


def loss_fn(stage_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for lg in stage_logits:                                   # (B, T, K)
        loss = loss + F.cross_entropy(lg[0], y)
        lp = F.log_softmax(lg[0], -1)
        loss = loss + LAMBDA * torch.clamp((lp[1:] - lp[:-1].detach()) ** 2, max=TAU ** 2).mean()
    return loss


def videos(store: dict, part: str):
    vids = sorted({k.split("|")[1] for k in store if k.startswith(part + "|")})
    return [(v, store[f"{part}|{v}|f"].astype(np.float32), store[f"{part}|{v}|y"].astype(np.int64), store[f"{part}|{v}|t"]) for v in vids]


def evaluate(model, vids, lt, device) -> dict:
    model.eval()
    seqs = []
    with torch.no_grad():
        for v, f, y, t in vids:
            lg = model(torch.from_numpy(f)[None].to(device))["logits"][0]
            seqs.append({"video": v, "labels": y, "times_h": t, "log_probs": F.log_softmax(lg, -1).cpu().numpy()})
    return evaluate_videos(seqs, lt)


def train_one(kind: str, seed: int, store: dict, device: str, feat_seed: int | None = None) -> dict:
    """``seed`` seeds the head; ``feat_seed`` (default ``seed``) names the backbone run whose features and train
    transition matrix are used, so more head seeds than backbone seeds can share one feature cache."""
    feat_seed = seed if feat_seed is None else feat_seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va, te = videos(store, "train"), videos(store, "val"), videos(store, "test")
    d = tr[0][1].shape[1]
    model = (FullVideoMSTCN(d, hidden=64, layers=10, stages=4) if kind == "mstcn" else ASFormerFull(d)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lt = np.load(ROOT / SRC.format(s=feat_seed) / "transition_log_matrix.npy")
    best, best_state, hist = -1.0, None, []
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for i in rng.permutation(len(tr)):
            _, f, y, _ = tr[i]
            out = model(torch.from_numpy(f)[None].to(device))
            loss = loss_fn(out["stage_logits"], torch.from_numpy(y).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        val = evaluate(model, va, lt, device)["p_t"]
        hist.append({"epoch": ep + 1, "loss": tot / len(tr), "val_p_t": val})
        if val > best:
            best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"{kind} seed {seed} epoch {ep + 1}: loss {tot / len(tr):.3f} val p_t {val:.4f}", flush=True)
    model.load_state_dict(best_state)
    r = evaluate(model, te, lt, device)
    return {"kind": kind, "seed": seed, "best_val_p_t": best, "test": {k: r[k] for k in ("p", "p_v", "p_t", "edit")},
            "history": hist}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    dev = ap.parse_args().device
    out = []
    for s in SEEDS:
        store = extract(s, dev)
        for kind in ("mstcn", "asformer"):
            out.append(train_one(kind, s, store, dev))
            (ROOT / "results/native_tas_heads.json").write_text(json.dumps(out, indent=1))
    for kind in ("mstcn", "asformer"):
        x = [r["test"]["p_t"] for r in out if r["kind"] == kind]
        print(f"{kind}: test p_t {np.mean(x):.3f} +- {np.std(x, ddof=1):.3f} ({len(x)} seeds)")


if __name__ == "__main__":
    main()
