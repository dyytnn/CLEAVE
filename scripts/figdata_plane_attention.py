#!/usr/bin/env python3
"""Cross-focal plane attention and per-plane focus on every fold-0 test frame (CPU inference only).

Figure "where the model looks". For each test frame of the cross-focal transformer (fold 0,
seed 0) this stores the attention weight the fusion query puts on each of the seven focal planes, and a focus measure
per plane (variance of the Laplacian of the normalised 224-px input), so the figure can test whether the attention
follows focus and how it depends on the developmental phase.

    PYTHONPATH=src python scripts/figdata_plane_attention.py      # -> results/figdata/plane_attention_test.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stseg.kinetic.datasets import _frames
from stseg.kinetic.models import build_model

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/h7/resnet18_transformer_L16_crossfocal7_evalfix_split0_seed0"
OUT = ROOT / "results/figdata/plane_attention_test.npz"
LAP = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]


def main() -> None:
    torch.set_num_threads(24)
    cfg = json.loads((RUN / "config.resolved.json").read_text())
    ds = _frames(cfg["data"], "test", "eval", 0, None, None)
    model = build_model(dict(cfg["model"]), ds.in_channels)
    model.load_state_dict(torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)["model"])
    model.eval()
    bb = model.backbone
    loader = torch.utils.data.DataLoader(ds, batch_size=48, shuffle=False, num_workers=8)
    attn, sharp = [], []
    with torch.no_grad():
        for i, b in enumerate(loader):
            x = b["image"]                                           # (N, P, H, W)
            N, P, H, W = x.shape
            f = bb.cnn(x.reshape(N * P, 1, H, W).expand(-1, 3, -1, -1)).view(N, P, -1)
            tok = bb.norm(f + bb.plane_emb[None])
            _, w = bb.attn(bb.query.expand(N, -1, -1), tok, tok, need_weights=True)   # heads averaged
            attn.append(w[:, 0].numpy())
            lap = F.conv2d(x.reshape(N * P, 1, H, W), LAP, padding=1)
            sharp.append(lap.flatten(1).var(1).view(N, P).numpy())
            if i % 50 == 0:
                print(f"batch {i}/{len(loader)}", flush=True)
    rows = ds.rows
    np.savez_compressed(OUT, attn=np.concatenate(attn), sharp=np.concatenate(sharp), label=rows.label.to_numpy(),
                        video=rows.video.to_numpy().astype(str), frame_index=rows.frame_index.to_numpy(),
                        time_h=rows.time_h.to_numpy(dtype=float), planes=np.array(cfg["data"]["planes"]))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} frames")


if __name__ == "__main__":
    main()
