"""TEMPO v30: SCE plus one training-time auxiliary branch.

The decoded output is exactly the v27 SCE framewise head; every auxiliary head reads the encoder's final hidden
sequence and is dropped at inference. All auxiliary targets are derived from the phase labels alone (no new
annotation, no clock):

* ``change_contrastive`` -- frames ``k`` apart inside one phase are pulled together (cosine distance), frames
  ``k`` apart across a transition are pushed beyond a margin; pairs touching the +-1-frame annotation uncertainty
  band are ignored. No head, acts on the hidden sequence directly.
* ``peak_count`` -- ordinal head: number of transitions in the last ``window`` frames, thresholds ">=1" and ">=2".
* ``time_to_event`` -- regression head: log1p(frames until the next transition), capped; censored after the last one.
* ``boundary`` -- binary head: first frame of a new phase (the v27 boundary target, without the diffusion decoder
  and with a shared rather than separate encoder).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .embryodiff import _ConditionEncoder
from .interfaces import AbsKineticModel, AbsLoss
from .registry import LOSS_REGISTRY, MODEL_REGISTRY

AUX_VARIANTS = ("change_contrastive", "peak_count", "time_to_event", "boundary")
PEAK_LEVELS = 2  # ordinal thresholds: >=1, >=2 transitions in the window


def transitions(y: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` bool: frame ``t`` is the first frame of a new phase (never true at ``t = 0``)."""
    out = torch.zeros_like(y, dtype=torch.bool)
    out[:, 1:] = y[:, 1:] != y[:, :-1]
    return out


def peak_count_targets(y: torch.Tensor, window: int) -> torch.Tensor:
    """``(B, T)`` long: transitions in ``(t - window, t]``, clipped to ``PEAK_LEVELS``."""
    tr = transitions(y).float()
    cum = torch.cat([tr.new_zeros(tr.shape[0], 1), tr.cumsum(1)], dim=1)
    idx = torch.arange(y.shape[1], device=y.device)
    lo = (idx - window + 1).clamp_min(0)
    count = cum[:, idx + 1] - cum[:, lo]
    return count.long().clamp(max=PEAK_LEVELS)


def time_to_event_targets(y: torch.Tensor, cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(target, observed)``: ``log1p(min(frames to next transition, cap))`` and its validity mask.

    Frames at or after the last transition have no observed next event and are masked (right-censored).
    """
    tr = transitions(y)
    b, t_len = y.shape
    idx = torch.arange(t_len, device=y.device)
    next_idx = torch.full((b, t_len), t_len, dtype=torch.long, device=y.device)
    for i in range(b):
        pos = torch.nonzero(tr[i], as_tuple=False).flatten()
        if pos.numel():
            k = torch.searchsorted(pos, idx, right=True)
            hit = k < pos.numel()
            next_idx[i, hit] = pos[k[hit]]
    delta = next_idx - idx
    observed = next_idx < t_len
    return torch.log1p(delta.clamp(max=cap).float()), observed


def contrastive_pairs(y: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Masks over pair index ``t`` (pair = frames ``t-k`` and ``t``): ``(same, diff)``.

    ``diff``: the pair straddles a transition. ``same``: both frames share a phase and neither endpoint sits
    within one frame of any transition (annotation uncertainty band), so the pair is interior to its phase.
    """
    tr = transitions(y)
    near = tr.clone()
    near[:, :-1] |= tr[:, 1:]  # frame before a transition
    diff = y[:, k:] != y[:, :-k]
    same = (~diff) & ~near[:, k:] & ~near[:, :-k]
    return same, diff


@MODEL_REGISTRY.register("sce_aux_adapter")
class SceAuxAdapter(AbsKineticModel):
    """v27 SCE with one auxiliary branch on the encoder's final hidden sequence; ``logits`` = SCE logits."""

    is_sequence = True
    requires_labels = False

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        aux: str,
        hidden: int,
        layers: int,
        intermediate_layers: tuple[int, ...],
        attention_reduction: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if aux not in AUX_VARIANTS:
            raise ValueError(f"unknown sce_aux variant {aux!r}; expected {AUX_VARIANTS}")
        self.aux = aux
        self.num_classes = int(num_classes)
        self.encoder = _ConditionEncoder(input_dim, hidden, layers, intermediate_layers, attention_reduction, dropout)
        self.frame_classifier = nn.Linear(hidden, num_classes)
        self.aux_head: nn.Module | None = None
        if aux == "peak_count":
            self.aux_head = nn.Linear(hidden, PEAK_LEVELS)
        elif aux in {"time_to_event", "boundary"}:
            self.aux_head = nn.Linear(hidden, 1)

    @classmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "SceAuxAdapter":
        if cfg.get("backbone", {}).get("name") != "cached_features":
            raise ValueError("sce_aux_adapter requires cached_features backbone")
        if cfg.get("head", {}).get("name") != "none":
            raise ValueError("sce_aux_adapter requires model.head.name=none")
        a = cfg.get("sce_aux")
        if not isinstance(a, dict):
            raise ValueError("sce_aux_adapter requires model.sce_aux mapping")
        return cls(
            input_dim=int(in_channels),
            num_classes=int(cfg.get("num_classes", 16)),
            aux=str(a.get("aux", "change_contrastive")),
            hidden=int(a.get("hidden", 96)),
            layers=int(a.get("layers", 6)),
            intermediate_layers=tuple(a.get("intermediate_layers", [2, 4, 6])),
            attention_reduction=int(a.get("attention_reduction", 2)),
            dropout=float(a.get("dropout", cfg.get("dropout", 0.5))),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"sce_aux input must be (B,T,D), got {tuple(x.shape)}")
        _, final = self.encoder(x)
        out = {"logits": self.frame_classifier(final), "hidden": final}
        if self.aux_head is not None:
            out["aux_logits"] = self.aux_head(final)
        return out


@LOSS_REGISTRY.register("sce_aux_objective")
class SceAuxObjective(AbsLoss):
    """v27 SCE objective (CE + truncated smoothing) plus ``aux_weight`` times the selected auxiliary term."""

    needs_outputs = True

    def __init__(
        self,
        class_counts: torch.Tensor | None = None,
        num_classes: int = 16,
        aux: str = "change_contrastive",
        semantic_weight: float = 0.8,
        smooth_weight: float = 0.3,
        aux_weight: float = 0.5,
        truncation: float = 4.0,
        pair_offset: int = 2,
        margin: float = 0.5,
        window: int = 8,
        cap: int = 64,
    ) -> None:
        del class_counts, num_classes
        if aux not in AUX_VARIANTS:
            raise ValueError(f"unknown sce_aux variant {aux!r}; expected {AUX_VARIANTS}")
        if min(semantic_weight, smooth_weight, aux_weight) < 0 or truncation <= 0:
            raise ValueError("loss weights must be nonnegative and truncation positive")
        if min(pair_offset, window, cap) < 1 or margin <= 0:
            raise ValueError("pair_offset/window/cap must be >= 1 and margin > 0")
        self.aux = aux
        self.semantic_weight, self.smooth_weight = float(semantic_weight), float(smooth_weight)
        self.aux_weight, self.truncation = float(aux_weight), float(truncation)
        self.pair_offset, self.margin, self.window, self.cap = int(pair_offset), float(margin), int(window), int(cap)

    def aux_term(self, out: dict, y: torch.Tensor) -> torch.Tensor:
        if self.aux == "change_contrastive":
            h = F.normalize(out["hidden"].float(), dim=-1)
            k = self.pair_offset
            if y.shape[1] <= k:
                return h.sum() * 0.0
            d = 1.0 - (h[:, k:] * h[:, :-k]).sum(-1)
            same, diff = contrastive_pairs(y, k)
            pull = d[same].mean() if same.any() else d.sum() * 0.0
            push = F.relu(self.margin - d[diff]).mean() if diff.any() else d.sum() * 0.0
            return pull + push
        logits = out["aux_logits"].float()
        if self.aux == "peak_count":
            level = peak_count_targets(y, self.window)
            ks = torch.arange(PEAK_LEVELS, device=y.device)
            target = (level[..., None] > ks).float()
            return F.binary_cross_entropy_with_logits(logits, target)
        if self.aux == "time_to_event":
            target, observed = time_to_event_targets(y, self.cap)
            if not observed.any():
                return logits.sum() * 0.0
            return F.smooth_l1_loss(logits.squeeze(-1)[observed], target[observed])
        target = transitions(y).float()
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), target)

    def __call__(self, out, y):
        if not isinstance(out, dict) or "logits" not in out or "hidden" not in out:
            raise ValueError("sce_aux_objective requires logits and hidden")
        logits = out["logits"]
        if logits.shape[:2] != y.shape:
            raise ValueError("logits must align with (B,T) labels")
        loss = self.semantic_weight * F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
        if y.shape[1] > 1 and self.smooth_weight > 0:
            logp = F.log_softmax(logits.float(), dim=-1)
            delta = logp[:, 1:] - logp[:, :-1].detach()
            loss = loss + self.smooth_weight * delta.square().clamp(max=self.truncation**2).mean()
        if self.aux_weight > 0:
            loss = loss + self.aux_weight * self.aux_term(out, y)
        return loss


__all__ = [
    "AUX_VARIANTS",
    "PEAK_LEVELS",
    "SceAuxAdapter",
    "SceAuxObjective",
    "contrastive_pairs",
    "peak_count_targets",
    "time_to_event_targets",
    "transitions",
]
