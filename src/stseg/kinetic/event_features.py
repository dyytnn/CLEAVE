"""Feature-side scalar conditioning shared by training and cached-window inference."""

from __future__ import annotations

import math

import torch
from torch import nn

from .interfaces import AbsFrameBackbone


class EventFeatureBackbone(AbsFrameBackbone):
    """(N, visual_channels + 1, H, W) -> (N, D), excluding the score from the CNN.

    The final channel is scalar transport only, populated by the strict loader.
    Keeping the adapter inside ``model.backbone`` also covers the evaluator's
    separable fast path, which deliberately bypasses ``model.forward``.
    """

    def __init__(
        self,
        visual: AbsFrameBackbone,
        visual_channels: int,
        hidden: int = 32,
        enabled: bool = True,
        mode: str = "predicted",
        neutral_score: float = 0.5,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"predicted", "constant"}:
            raise ValueError("event mode must be predicted or constant")
        if hidden < 1 or visual_channels < 1:
            raise ValueError("event hidden and visual_channels must be positive")
        if not math.isfinite(neutral_score) or not 0 <= neutral_score <= 1:
            raise ValueError("neutral_score must be finite and in [0, 1]")
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("event scale must be finite and nonnegative")
        self.visual, self.visual_channels = visual, visual_channels
        self.feat_dim = visual.feat_dim
        self.enabled, self.mode = enabled, mode
        self.neutral_score, self.scale = float(neutral_score), float(scale)
        self.event = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, self.feat_dim)
        )
        # Exact identity at initialization; final layer learns on the first step.
        # Unlike a zero scalar gate AND zero branch, this does not trap gradients.
        nn.init.zeros_(self.event[-1].weight)
        nn.init.zeros_(self.event[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode visual channels only; add an optional centered-score residual."""
        if x.ndim != 4 or x.shape[1] != self.visual_channels + 1:
            raise ValueError("event backbone expects visual channels plus one score channel")
        visual = self.visual(x[:, : self.visual_channels])
        if not self.enabled or self.scale == 0:
            return visual
        score = x[:, -1].mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
        if self.mode == "constant":
            score = torch.full_like(score, self.neutral_score)
        residual = self.event((score - self.neutral_score).to(visual.dtype))
        return visual + self.scale * residual
