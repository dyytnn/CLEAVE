#!/usr/bin/env python3
"""Grad-CAM on single frames of one fold-0 test embryo, single-plane vs cross-focal transformer (CPU only).

For each target phase the frame in the middle of that phase is placed at the centre of a 16-frame clip (the models'
training clip length); the gradient of the annotated class's logit at the centre frame, taken with respect to the last
ResNet stage of that frame, gives the Grad-CAM map. For the cross-focal model one map is computed per focal plane
and the plane maps are combined with the model's own attention weights..

    PYTHONPATH=src python scripts/figdata_gradcam.py --video BM016-5   # -> results/figdata/gradcam_<video>.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stseg.data.nantes_kinetic import CLASS_NAMES
from stseg.kinetic.datasets import _frames
from stseg.kinetic.models import build_model

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"single": "resnet18_transformer_L16_evalfix_split0_seed0",
        "crossfocal": "resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0"}
PHASES = ["t2", "t3", "t4", "t5", "t7", "t8", "tM", "tB"]
L = 16


def load(run: str):
    rd = ROOT / "runs/h7" / run
    cfg = json.loads((rd / "config.resolved.json").read_text())
    ds = _frames(cfg["data"], "test", "eval", 0, None, None)
    model = build_model(dict(cfg["model"]), ds.in_channels)
    model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu", weights_only=True)["model"])
    return model.eval(), ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="BM016-5")
    video = ap.parse_args().video
    torch.set_num_threads(16)
    out = {}
    for tag, run in RUNS.items():
        model, ds = load(run)
        rows = ds.rows[ds.rows.video == video].reset_index()
        cnn = model.backbone.cnn.net if tag == "crossfocal" else model.backbone.net
        store = {}
        cnn.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("A", o))
        for ph in PHASES:
            idx = np.where(rows.phase == ph)[0]
            if not len(idx):
                continue
            c = int(idx[len(idx) // 2])
            s = min(max(c - L // 2, 0), len(rows) - L)
            clip = torch.stack([ds[int(rows.loc[i, "index"])]["image"] for i in range(s, s + L)])[None]
            store.clear()
            logits = model(clip)["logits"][0, c - s]
            A = store["A"]
            grad = torch.autograd.grad(logits[CLASS_NAMES.index(ph)], A)[0]
            P = clip.shape[2] if tag == "crossfocal" else 1
            Ag = A.view(L, P, *A.shape[1:])[c - s]                                                 # (P, C, h, w)
            Gg = grad.view(L, P, *grad.shape[1:])[c - s]
            cam = F.relu((Gg.mean((2, 3), keepdim=True) * Ag).sum(1))                              # (P, h, w)
            cam = F.interpolate(cam[:, None], size=clip.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            img = clip[0, c - s].detach().numpy()
            rec = {"cam": cam.detach().numpy(), "img": img, "prob": float(torch.softmax(logits, 0)[CLASS_NAMES.index(ph)])}
            if tag == "crossfocal":
                bb = model.backbone
                with torch.no_grad():
                    x = clip[0, c - s][None]
                    f = bb.cnn(x.reshape(P, 1, *x.shape[-2:]).expand(-1, 3, -1, -1)).view(1, P, -1)
                    tok = bb.norm(f + bb.plane_emb[None])
                    _, w = bb.attn(bb.query.expand(1, -1, -1), tok, tok, need_weights=True)
                rec["attn"] = w[0, 0].numpy()
            out[f"{tag}|{ph}"] = rec
            print(tag, ph, "frame", int(rows.loc[c, "frame_index"]), f"p={rec['prob']:.2f}", flush=True)
    np.savez_compressed(ROOT / f"results/figdata/gradcam_{video}.npz",
                        **{f"{k}|{f}": v for k, r in out.items() for f, v in r.items()})
    print("wrote", f"results/figdata/gradcam_{video}.npz")


if __name__ == "__main__":
    main()
