#!/usr/bin/env python
"""TEMPO rung T7: clip-level masked-frame reconstruction pretraining on unlabelled Nantes frames (1.9M+ frames
available, `` §12-I). Standalone script, same pattern as `scripts/pretrain_mae.py` (reverse-lever P1) --
outside the registry/YAML kinetic pipeline because it is unsupervised and has no classes/loss to register.

**Scope, stated plainly**: this is a *simplified* video MAE, not the full spatiotemporal-tube-masking ViT of Tong et
al. 2022. A fraction of *whole frames* in a clip are masked (replaced with the training-mean grey value, so the
backbone still runs but sees no content) instead of masking spatial patches within every frame; the backbone stays a
2D CNN (unchanged architecture, so the resulting checkpoint loads straight into any existing `configs/h7/*.yaml` via
`model.backbone.init_from`) instead of a video transformer. The reconstruction target is a heavily downsampled
(default 16x16) version of the true frame, predicted from the temporal head's context at that position -- so the
pretext task is genuinely "infer the masked frame's coarse appearance from its neighbours in time", the same idea as
VideoMAE, at a fraction of the engineering cost.

Usage:
  PYTHONPATH=src python scripts/pretrain_video_mae.py --config configs/mae/video_mae_resnet18.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stseg.kinetic.backbones import BACKBONE_REGISTRY  # noqa: E402
from stseg.kinetic.heads import HEAD_REGISTRY  # noqa: E402
import stseg.kinetic.heads  # noqa: E402,F401  (populate the registry)

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"run_id", "manifest", "backbone", "head", "pretrain", "seed"}
PRE_KEYS = {"epochs", "batch_size", "lr", "weight_decay", "clip_len", "mask_ratio", "target_size", "resize", "crop",
            "num_workers", "clips_per_epoch"}


class ClipFrames(Dataset):
    """Random clips of `clip_len` consecutive frames per video, any phase (including unlabelled/out-of-window)."""

    def __init__(self, manifest: str, clip_len: int, resize: int, crop: int, target_size: int, clips_per_epoch: int, seed: int) -> None:
        df = pd.read_csv(manifest, usecols=["video", "frame_index", "path"])
        self.groups = [g.sort_values("frame_index").path.tolist() for _, g in df.groupby("video", sort=False) if len(g) >= clip_len]
        self.L, self.resize, self.crop, self.target = clip_len, resize, crop, target_size
        self.n = clips_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _i: int):
        paths = self.groups[self.rng.integers(0, len(self.groups))]
        start = int(self.rng.integers(0, len(paths) - self.L + 1))
        m = (self.resize - self.crop) // 2
        imgs, targets = [], []
        for p in paths[start:start + self.L]:
            im = Image.open(p).convert("L").resize((self.resize, self.resize), Image.Resampling.BILINEAR)
            a = np.asarray(im, dtype=np.float32)[m:m + self.crop, m:m + self.crop] / 255.0
            imgs.append(torch.from_numpy(a))
            tgt = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((self.target, self.target), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            targets.append(torch.from_numpy(tgt).flatten())
        return torch.stack(imgs)[:, None].expand(-1, 3, -1, -1).clone(), torch.stack(targets)


class VideoMAELite(nn.Module):
    def __init__(self, backbone_cfg: dict, head_cfg: dict, target_size: int, mask_value: float = 0.5) -> None:
        super().__init__()
        self.backbone = BACKBONE_REGISTRY.build(backbone_cfg, in_channels=3)
        self.head = HEAD_REGISTRY.build(head_cfg, d_in=self.backbone.feat_dim)
        self.decoder = nn.Linear(self.head.d_out, target_size * target_size)
        self.mask_value = mask_value

    def forward(self, clip: torch.Tensor, mask_ratio: float):
        B, L = clip.shape[:2]
        mask = torch.rand(B, L, device=clip.device) < mask_ratio
        masked = clip.clone()
        masked[mask] = self.mask_value
        f = self.backbone(masked.flatten(0, 1)).view(B, L, -1)
        h = self.head(f)
        recon = self.decoder(h)
        return recon, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    extra = set(cfg) - KEYS
    if extra:
        raise ValueError(f"unknown top-level keys {extra}; allowed {KEYS}")
    pextra = set(cfg["pretrain"]) - PRE_KEYS
    if pextra:
        raise ValueError(f"unknown pretrain keys {pextra}; allowed {PRE_KEYS}")

    pt = cfg["pretrain"]
    torch.manual_seed(int(cfg.get("seed", 0)))
    ds = ClipFrames(cfg["manifest"], pt["clip_len"], pt.get("resize", 250), pt.get("crop", 224),
                    pt.get("target_size", 16), pt["clips_per_epoch"], int(cfg.get("seed", 0)))
    loader = DataLoader(ds, batch_size=pt["batch_size"], num_workers=pt.get("num_workers", 8), pin_memory=True)
    device = torch.device(a.device)
    model = VideoMAELite(cfg["backbone"], cfg["head"], pt.get("target_size", 16)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=pt.get("lr", 1e-4), weight_decay=pt.get("weight_decay", 0.05))
    out_dir = ROOT / "runs/mae" / cfg["run_id"]; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))

    history = []
    for epoch in range(1, pt["epochs"] + 1):
        model.train(); t0 = time.time(); tot, n = 0.0, 0
        for clip, target in loader:
            clip, target = clip.to(device, non_blocking=True), target.to(device, non_blocking=True)
            recon, mask = model(clip, pt.get("mask_ratio", 0.5))
            if mask.sum() == 0:
                continue
            loss = nn.functional.mse_loss(recon[mask], target[mask])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss.item()) * mask.sum().item(); n += int(mask.sum().item())
        avg = tot / max(n, 1)
        history.append({"epoch": epoch, "loss": avg, "min": (time.time() - t0) / 60})
        print(f"epoch {epoch}/{pt['epochs']} masked-mse={avg:.4f} ({history[-1]['min']:.1f} min)")
        torch.save(model.backbone.net.state_dict(), out_dir / "backbone.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    print(f"done -> {out_dir / 'backbone.pt'} (use as model.backbone.init_from in a configs/h7/*.yaml)")


if __name__ == "__main__":
    main()
