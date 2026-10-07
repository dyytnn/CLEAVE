"""TEMPO v33: end-to-end whole-video training -- trainable cross-focal backbone + SCE head.

Every frozen-cache probe saturated (12.ggg, 12.hhh) while whole-video context gave the largest development gain
(12.xx), yet the backbone behind the cache was only ever trained on 16-frame clips. This rung gives the backbone a
whole-video gradient. The model consumes raw seven-plane frames of one complete video, encodes them with the v25
cross-focal backbone in gradient-checkpointed chunks (one chunk's activations live at a time), standardises the
features exactly as the v26 cache producer did (train-cache mean/std, so the v27 SCE head can be warm-started
unchanged), and applies the SCE encoder + framewise head. Output/decoding/metrics are the v27 SCE ones. Validation
only: the producer never constructs the test partition. BatchNorm statistics stay frozen (batch = one video).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

from stseg.data.nantes_kinetic import NantesKineticFrames

from .datasets import _frames
from .embryodiff import _ConditionEncoder
from .interfaces import AbsKineticModel
from .registry import BACKBONE_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


class WholeVideoDataset(Dataset):
    """One complete labelled video per item, one augmentation draw applied identically to every frame."""

    def __init__(self, frames: NantesKineticFrames) -> None:
        self.f = frames
        self.groups = [g.index.to_numpy() for _, g in frames.rows.groupby("video", sort=False)]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> dict[str, Any]:
        sel = self.f.rows.iloc[self.groups[i]]
        y, x, fh, fv = self.f.sample_aug()
        ex = self.f.sample_extra()
        imgs = [self.f.load_frame(p, y, x, fh, fv, ex["photo"], ex["angle"], ex["plane"], v, int(fi))
                for p, v, fi in zip(sel.path, sel.video, sel.frame_index)]
        return {"image": torch.stack(imgs), "label": torch.tensor(sel.label.to_numpy(), dtype=torch.long),
                "video": sel.video.iloc[0], "frame_index": torch.tensor(sel.frame_index.to_numpy()),
                "time_h": torch.tensor(sel.time_h.to_numpy(dtype=np.float32))}


@DATASET_REGISTRY.register("nantes_video_trainval")
class NantesVideoTrainValProducer:
    """Whole-video train/validation producer on raw frames; never constructs the test partition."""

    def __init__(self, d: dict, seed: int, smoke: bool = False) -> None:
        lim = 1 if smoke else None
        max_fpv = 20 if smoke else d.get("max_frames_per_video")
        self.train_frames = _frames(d, "train", "train", seed, max_fpv, lim)
        self.train_full = _frames(d, "train", "eval", seed, None, lim)
        self.val = _frames(d, "val", "eval", seed, 20 if smoke else None, lim)
        self.in_channels = self.train_frames.in_channels
        self.default_batch = 1
        self.train_set = WholeVideoDataset(self.train_frames)
        self.sequence_mode = "full"


def cache_feature_stats(cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Train-frame mean/std of a v26 feature cache, computed exactly as ``cached_video.CachedSequenceTrainValProducer``."""
    cache_dir = _resolve(cache_dir)
    meta = json.loads((cache_dir / "meta.json").read_text())
    if meta.get("partitions") != ["train", "val"] or meta.get("test_cached"):
        raise ValueError(f"{cache_dir}: expected a train/val-only v26 cache")
    raw = np.load(cache_dir / "train.npz", allow_pickle=False)
    sum_x = sum_x2 = None
    count = 0
    for key in raw.files:
        if not key.endswith("__emb"):
            continue
        emb = np.asarray(raw[key], dtype=np.float64)
        ex, ex2 = emb.sum(0), np.square(emb).sum(0)
        sum_x = ex if sum_x is None else sum_x + ex
        sum_x2 = ex2 if sum_x2 is None else sum_x2 + ex2
        count += len(emb)
    raw.close()
    if sum_x is None or count < 1:
        raise ValueError(f"{cache_dir}: empty train cache")
    mean = sum_x / count
    std = np.sqrt(np.maximum(sum_x2 / count - np.square(mean), 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


@MODEL_REGISTRY.register("e2e_sce")
class EndToEndSce(AbsKineticModel):
    """Cross-focal backbone (chunked, checkpointed) -> cache-style standardisation -> SCE encoder + framewise head.

    Input ``(B, L, P, H, W)`` raw plane frames. Output ``logits`` = ``semantic_logits`` ``(B, L, K)``, so
    ``embryodiff_objective`` (semantic + smoothing terms) and the unchanged decoder/metrics apply. Attribute
    ``backbone`` keeps ``optimizer.backbone_lr_mult`` working; there is deliberately no ``head``/``cls`` attribute, so
    evaluation goes through ``forward`` (chunked, no_grad) rather than the separable fast path.
    """

    is_sequence = True
    requires_labels = False

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        hidden: int,
        layers: int,
        intermediate_layers: tuple[int, ...],
        attention_reduction: int,
        dropout: float,
        chunk: int,
        freeze_backbone: bool,
        feat_mean: np.ndarray | None = None,
        feat_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        self.backbone = backbone
        d = int(backbone.feat_dim)
        self.encoder = _ConditionEncoder(d, hidden, layers, intermediate_layers, attention_reduction, dropout)
        self.frame_classifier = nn.Linear(hidden, num_classes)
        self.chunk = int(chunk)
        self.freeze_backbone = bool(freeze_backbone)
        mean = torch.zeros(d) if feat_mean is None else torch.as_tensor(feat_mean, dtype=torch.float32)
        std = torch.ones(d) if feat_std is None else torch.as_tensor(feat_std, dtype=torch.float32)
        if mean.shape != (d,) or std.shape != (d,):
            raise ValueError(f"feature stats must have shape ({d},)")
        self.register_buffer("feat_mean", mean)
        self.register_buffer("feat_std", std)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "EndToEndSce":
        if cfg.get("head", {"name": "none"}).get("name") != "none":
            raise ValueError("e2e_sce requires model.head.name=none")
        e = cfg.get("e2e")
        if not isinstance(e, dict):
            raise ValueError("e2e_sce requires model.e2e mapping")
        backbone = BACKBONE_REGISTRY.build(cfg["backbone"], in_channels=in_channels)
        mean = std = None
        if e.get("feature_stats_cache"):
            mean, std = cache_feature_stats(e["feature_stats_cache"])
        model = cls(
            backbone=backbone,
            num_classes=int(cfg.get("num_classes", 16)),
            hidden=int(e.get("hidden", 96)),
            layers=int(e.get("layers", 6)),
            intermediate_layers=tuple(e.get("intermediate_layers", [2, 4, 6])),
            attention_reduction=int(e.get("attention_reduction", 2)),
            dropout=float(e.get("dropout", cfg.get("dropout", 0.5))),
            chunk=int(e.get("chunk", 64)),
            freeze_backbone=bool(e.get("freeze_backbone", False)),
            feat_mean=mean,
            feat_std=std,
        )
        if e.get("init_backbone_from"):
            model.load_backbone(e["init_backbone_from"])
        if e.get("init_head_from"):
            model.load_head(e["init_head_from"])
        return model

    # ------------------------------------------------------------------ warm starts
    def load_backbone(self, path: str | Path) -> None:
        """Strict load of the ``backbone.*`` tensors of a ``SeqKinetic`` checkpoint (e.g. v25 ``swa.pt``)."""
        sd = torch.load(_resolve(path), map_location="cpu", weights_only=True)["model"]
        sub = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        if not sub:
            raise ValueError(f"{path}: no backbone.* tensors")
        self.backbone.load_state_dict(sub, strict=True)

    def load_head(self, path: str | Path) -> None:
        """Strict load of a v27 ``sce`` checkpoint (``semantic_encoder.*`` / ``semantic_classifier.*``)."""
        sd = torch.load(_resolve(path), map_location="cpu", weights_only=True)["model"]
        enc = {k[len("semantic_encoder."):]: v for k, v in sd.items() if k.startswith("semantic_encoder.")}
        cls = {k[len("semantic_classifier."):]: v for k, v in sd.items() if k.startswith("semantic_classifier.")}
        if not enc or not cls:
            raise ValueError(f"{path}: not an SCE checkpoint")
        self.encoder.load_state_dict(enc, strict=True)
        self.frame_classifier.load_state_dict(cls, strict=True)

    # ------------------------------------------------------------------ forward
    def train(self, mode: bool = True) -> "EndToEndSce":
        super().train(mode)
        for m in self.backbone.modules():  # frozen statistics: batch = one video, and the v25 stats were SWA-re-estimated
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """``(N, P, H, W)`` -> ``(N, D)`` in chunks; checkpointed when the backbone is being trained."""
        n = frames.shape[0]
        outs = []
        train_backbone = self.training and torch.is_grad_enabled() and not self.freeze_backbone
        for i in range(0, n, self.chunk):
            c = frames[i:i + self.chunk]
            if train_backbone:
                outs.append(checkpoint(self.backbone, c, use_reentrant=False))
            else:
                with torch.no_grad():
                    outs.append(self.backbone(c))
        return torch.cat(outs, dim=0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 4:  # one video (L, P, H, W); a length-1 clip would break the SCE instance norms
            x = x[None]
        if x.ndim != 5:
            raise ValueError(f"e2e_sce input must be (B,L,P,H,W), got {tuple(x.shape)}")
        b, l_len = x.shape[:2]
        feats = self.encode_frames(x.flatten(0, 1)).float().view(b, l_len, -1)
        feats = (feats - self.feat_mean) / self.feat_std
        _, final = self.encoder(feats)
        logits = self.frame_classifier(final)
        return {"logits": logits, "semantic_logits": logits}


__all__ = ["EndToEndSce", "NantesVideoTrainValProducer", "WholeVideoDataset", "cache_feature_stats"]
