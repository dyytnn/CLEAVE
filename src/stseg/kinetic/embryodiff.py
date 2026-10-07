"""Transparent EmbryoDiff architecture adaptation for TEMPO v27.

The publication specifies the macro-architecture but does not release model
code.  This module keeps the stated dimensions and objectives while making the
remaining implementation choices explicit in ```` section 12.yy.
It consumes the validation-only frozen visual caches introduced in v26.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .interfaces import AbsKineticModel, AbsLoss
from .registry import LOSS_REGISTRY, MODEL_REGISTRY


class _LocalSlidingAttention(nn.Module):
    """ASFormer-style local attention over ``(B, C, T)`` features.

    The official ASFormer implementation groups queries into sliding blocks.
    Here the equivalent local support is expressed with ``unfold`` so it is
    device-independent, supports ``B > 1``, and masks boundary padding rather
    than treating padded zeros as observations.
    """

    def __init__(self, channels: int, radius: int, reduction: int) -> None:
        super().__init__()
        if channels < 1 or radius < 1 or reduction < 1:
            raise ValueError("attention channels/radius/reduction must be positive")
        projected = max(1, channels // reduction)
        self.radius = int(radius)
        self.query = nn.Conv1d(channels, projected, 1)
        self.key = nn.Conv1d(channels, projected, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.output = nn.Conv1d(channels, channels, 1)
        self.scale = projected**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local attention.

        Args:
            x: Features shaped ``(B, C, T)``.

        Returns:
            Locally attended features shaped ``(B, C, T)``.
        """
        if x.ndim != 3:
            raise ValueError(f"local attention expects (B,C,T), got {tuple(x.shape)}")
        radius = self.radius
        width = 2 * radius + 1
        query = self.query(x).unsqueeze(-1)
        key = F.pad(self.key(x), (radius, radius)).unfold(2, width, 1)
        value = F.pad(self.value(x), (radius, radius)).unfold(2, width, 1)
        score = (query * key).sum(dim=1) * self.scale

        valid = x.new_ones(1, 1, x.shape[-1])
        valid = F.pad(valid, (radius, radius)).unfold(2, width, 1).squeeze(1)
        score = score.masked_fill(~valid.bool(), torch.finfo(score.dtype).min)
        attention = torch.softmax(score.float(), dim=-1).to(value.dtype)
        attended = (value * attention.unsqueeze(1)).sum(dim=-1)
        return self.output(F.relu(attended))


class _ASFormerLayer(nn.Module):
    """One corrected ASFormer encoder layer over ``(B, C, T)`` features."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        attention_reduction: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feed_forward = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm = nn.InstanceNorm1d(channels, track_running_stats=False)
        self.attention = _LocalSlidingAttention(
            channels,
            radius=dilation,
            reduction=attention_reduction,
        )
        self.project = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.feed_forward(x))
        h = h + self.attention(self.norm(h))
        return x + self.dropout(self.project(h))


class _ConditionEncoder(nn.Module):
    """Six-layer ASFormer condition encoder with selected intermediate taps."""

    def __init__(
        self,
        input_dim: int,
        hidden: int,
        layers: int,
        intermediate_layers: tuple[int, ...],
        attention_reduction: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("condition encoder needs at least one layer")
        if not intermediate_layers or min(intermediate_layers) < 1:
            raise ValueError("intermediate layers use one-based positive indices")
        if max(intermediate_layers) > layers:
            raise ValueError("intermediate layer exceeds condition-encoder depth")
        self.intermediate_layers = tuple(intermediate_layers)
        self.input = nn.Conv1d(input_dim, hidden, 1)
        self.layers = nn.ModuleList(
            [
                _ASFormerLayer(
                    hidden,
                    dilation=2**index,
                    attention_reduction=attention_reduction,
                    dropout=dropout,
                )
                for index in range(layers)
            ]
        )
        self.output_dim = hidden * len(self.intermediate_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode cached features.

        Args:
            x: Frozen frame features shaped ``(B, T, D)``.

        Returns:
            ``(condition, final)`` shaped ``(B, T, C*n_taps)`` and
            ``(B, T, C)`` respectively.
        """
        h = self.input(x.transpose(1, 2))
        taps = []
        for one_based, layer in enumerate(self.layers, start=1):
            h = layer(h)
            if one_based in self.intermediate_layers:
                taps.append(h)
        return torch.cat(taps, dim=1).transpose(1, 2), h.transpose(1, 2)


class _TimestepEmbedding(nn.Module):
    """Sinusoidal diffusion-time embedding followed by a small MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angle = timestep.float()[:, None] * frequency[None]
        embedding = torch.cat([angle.sin(), angle.cos()], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return self.mlp(embedding)


class _HybridConditionBlock(nn.Module):
    """Semantic Q/K attention plus optional local boundary conditioning."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        use_boundary: bool,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("diffusion dimension must be divisible by attention heads")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.use_boundary = bool(use_boundary)
        self.query = nn.Linear(dim * 2, dim)
        self.key = nn.Linear(dim * 2, dim)
        self.value = nn.Linear(dim, dim)
        self.attention_output = nn.Linear(dim, dim)
        if self.use_boundary:
            self.boundary = nn.Conv1d(dim * 2, dim, kernel_size=3, padding=1)
        else:
            self.boundary = None
        self.condition_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _semantic_attention(
        self,
        noisy: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, dim = noisy.shape
        conditioned = torch.cat([self.condition_norm(noisy), semantic], dim=-1)

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

        query = heads(self.query(conditioned))
        key = heads(self.key(conditioned))
        value = heads(self.value(noisy))
        score = torch.matmul(query.float(), key.float().transpose(-2, -1)) * self.head_dim**-0.5
        attention = torch.softmax(score, dim=-1).to(value.dtype)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return self.attention_output(attended)

    def forward(
        self,
        noisy: torch.Tensor,
        semantic: torch.Tensor,
        boundary: torch.Tensor | None,
    ) -> torch.Tensor:
        semantic_path = self._semantic_attention(noisy, semantic)
        conditioned = semantic_path
        if self.use_boundary:
            if boundary is None or self.boundary is None:
                raise ValueError("boundary-conditioned block requires boundary features")
            joined = torch.cat([noisy, boundary], dim=-1).transpose(1, 2)
            conditioned = conditioned + self.boundary(joined).transpose(1, 2)
        h = noisy + self.dropout(conditioned)
        return h + self.dropout(self.ffn(self.ffn_norm(h)))


class _DiffusionDecoder(nn.Module):
    """DiT-style label-space decoder used for training and DDIM inference."""

    def __init__(
        self,
        condition_dim: int,
        dim: int,
        blocks: int,
        heads: int,
        num_classes: int,
        dropout: float,
        use_boundary: bool,
    ) -> None:
        super().__init__()
        self.semantic_projection = nn.Linear(condition_dim, dim)
        self.boundary_projection = nn.Linear(condition_dim, dim) if use_boundary else None
        self.time = _TimestepEmbedding(dim)
        self.blocks = nn.ModuleList(
            [_HybridConditionBlock(dim, heads, dropout, use_boundary) for _ in range(blocks)]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.output_embedding = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        semantic: torch.Tensor,
        boundary: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = noisy + self.time(timestep)[:, None]
        semantic = self.semantic_projection(semantic)
        if self.boundary_projection is not None:
            if boundary is None:
                raise ValueError("diffusion decoder requires boundary condition")
            boundary = self.boundary_projection(boundary)
        for block in self.blocks:
            h = block(h, semantic, boundary)
        embedding = self.output_embedding(self.output_norm(h))
        return embedding, self.classifier(embedding)


def _cosine_alpha_bars(steps: int, offset: float = 0.008) -> torch.Tensor:
    """Return the standard cosine cumulative noise schedule."""
    if steps < 2:
        raise ValueError("diffusion schedule needs at least two steps")
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    cumulative = torch.cos(((points / steps + offset) / (1.0 + offset)) * math.pi / 2.0).square()
    cumulative = cumulative / cumulative[0]
    betas = (1.0 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()


@MODEL_REGISTRY.register("embryodiff_adapter")
class EmbryoDiffAdapter(AbsKineticModel):
    """EmbryoDiff SCE/diffusion/boundary ablation family for frozen features.

    Args:
        x: Complete-video cached features shaped ``(B, T, D)``.
        y: Training labels shaped ``(B, T)``; required by diffusion variants.

    Returns:
        A dictionary whose ``logits`` are shaped ``(B, T, K)``. Auxiliary
        semantic, boundary, and diffusion outputs are included when active.
    """

    is_sequence = True

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        variant: str,
        hidden: int,
        layers: int,
        intermediate_layers: tuple[int, ...],
        attention_reduction: int,
        diffusion_dim: int,
        diffusion_blocks: int,
        diffusion_heads: int,
        diffusion_train_steps: int,
        inference_steps: int,
        selection_steps: int,
        label_scale: float,
        dropout: float,
        eval_seed: int,
    ) -> None:
        super().__init__()
        allowed = {"sce", "sce_diffusion", "sce_boundary_diffusion"}
        if variant not in allowed:
            raise ValueError(f"unknown EmbryoDiff variant {variant!r}; expected {sorted(allowed)}")
        if label_scale <= 0:
            raise ValueError("label scale must be positive")
        if min(inference_steps, selection_steps) < 1:
            raise ValueError("DDIM inference/selection steps must be positive")
        self.variant = variant
        self.num_classes = int(num_classes)
        self.requires_labels = variant != "sce"
        self.semantic_encoder = _ConditionEncoder(
            input_dim,
            hidden,
            layers,
            intermediate_layers,
            attention_reduction,
            dropout,
        )
        self.semantic_classifier = nn.Linear(hidden, num_classes)
        self.use_boundary = variant == "sce_boundary_diffusion"
        if self.use_boundary:
            self.boundary_encoder = _ConditionEncoder(
                input_dim,
                hidden,
                layers,
                intermediate_layers,
                attention_reduction,
                dropout,
            )
            self.boundary_classifier = nn.Linear(hidden, 1)
        else:
            self.boundary_encoder = None
            self.boundary_classifier = None

        self.diffusion_train_steps = int(diffusion_train_steps)
        self.inference_steps = int(inference_steps)
        self.selection_steps = int(selection_steps)
        self.current_inference_steps = self.selection_steps
        self.label_scale = float(label_scale)
        self.eval_seed = int(eval_seed)
        if self.requires_labels:
            self.label_embedding = nn.Embedding(num_classes, diffusion_dim)
            self.diffusion = _DiffusionDecoder(
                condition_dim=self.semantic_encoder.output_dim,
                dim=diffusion_dim,
                blocks=diffusion_blocks,
                heads=diffusion_heads,
                num_classes=num_classes,
                dropout=dropout,
                use_boundary=self.use_boundary,
            )
            self.register_buffer(
                "alpha_bars",
                _cosine_alpha_bars(self.diffusion_train_steps),
                persistent=True,
            )
        else:
            self.label_embedding = None
            self.diffusion = None
            self.register_buffer("alpha_bars", torch.empty(0), persistent=False)

    @classmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "EmbryoDiffAdapter":
        if cfg.get("backbone", {}).get("name") != "cached_features":
            raise ValueError("embryodiff_adapter requires cached_features backbone")
        if cfg.get("head", {}).get("name") != "none":
            raise ValueError("embryodiff_adapter requires model.head.name=none")
        ecfg = cfg.get("embryodiff")
        if not isinstance(ecfg, dict):
            raise ValueError("embryodiff_adapter requires model.embryodiff mapping")
        return cls(
            input_dim=int(in_channels),
            num_classes=int(cfg.get("num_classes", 16)),
            variant=str(ecfg.get("variant", "sce")),
            hidden=int(ecfg.get("hidden", 96)),
            layers=int(ecfg.get("layers", 6)),
            intermediate_layers=tuple(ecfg.get("intermediate_layers", [2, 4, 6])),
            attention_reduction=int(ecfg.get("attention_reduction", 2)),
            diffusion_dim=int(ecfg.get("diffusion_dim", 128)),
            diffusion_blocks=int(ecfg.get("diffusion_blocks", 8)),
            diffusion_heads=int(ecfg.get("diffusion_heads", 4)),
            diffusion_train_steps=int(ecfg.get("diffusion_train_steps", 1000)),
            inference_steps=int(ecfg.get("inference_steps", 25)),
            selection_steps=int(ecfg.get("selection_steps", 1)),
            label_scale=float(ecfg.get("label_scale", 0.1)),
            dropout=float(ecfg.get("dropout", cfg.get("dropout", 0.5))),
            eval_seed=int(ecfg.get("eval_seed", 27000)),
        )

    def use_selection_inference(self) -> None:
        """Use the preregistered cheap sampler for checkpoint selection."""
        self.current_inference_steps = self.selection_steps

    def use_final_inference(self) -> None:
        """Use the preregistered final sampler after checkpoint selection."""
        self.current_inference_steps = self.inference_steps

    def set_inference_steps(self, steps: int) -> None:
        """Set DDIM steps for fixed-checkpoint diagnostic evaluation."""
        if steps < 1:
            raise ValueError("DDIM steps must be positive")
        self.current_inference_steps = int(steps)

    def _conditions(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        semantic, semantic_final = self.semantic_encoder(x)
        semantic_logits = self.semantic_classifier(semantic_final)
        boundary = boundary_logits = None
        if self.boundary_encoder is not None and self.boundary_classifier is not None:
            boundary, boundary_final = self.boundary_encoder(x)
            boundary_logits = self.boundary_classifier(boundary_final).squeeze(-1)
        return semantic, semantic_logits, boundary, boundary_logits

    def _predict_clean(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        semantic: torch.Tensor,
        boundary: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.diffusion is None:
            raise RuntimeError("semantic-only variant has no diffusion decoder")
        return self.diffusion(noisy, timestep, semantic, boundary)

    def _evaluation_noise(
        self,
        batch: int,
        length: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.eval_seed + 1009 * length + 9176 * batch)
        return torch.randn(batch, length, dim, generator=generator).to(device=device, dtype=dtype)

    def _ddim(
        self,
        semantic: torch.Tensor,
        boundary: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.label_embedding is None:
            raise RuntimeError("semantic-only variant has no label embedding")
        batch, length = semantic.shape[:2]
        noisy = self._evaluation_noise(
            batch,
            length,
            self.label_embedding.embedding_dim,
            semantic.device,
            torch.float32,
        )
        count = min(self.current_inference_steps, self.diffusion_train_steps)
        timesteps = (
            torch.linspace(
                self.diffusion_train_steps - 1,
                0,
                count,
                device=semantic.device,
            )
            .round()
            .long()
        )
        predicted = logits = None
        for index, scalar_t in enumerate(timesteps):
            timestep = scalar_t.expand(batch)
            predicted, logits = self._predict_clean(noisy, timestep, semantic, boundary)
            if index + 1 == len(timesteps):
                break
            alpha_t = self.alpha_bars[scalar_t].to(noisy.dtype)
            alpha_previous = self.alpha_bars[timesteps[index + 1]].to(noisy.dtype)
            epsilon = (noisy - alpha_t.sqrt() * predicted) / (1.0 - alpha_t).sqrt()
            noisy = alpha_previous.sqrt() * predicted + (1.0 - alpha_previous).sqrt() * epsilon
        if predicted is None or logits is None:
            raise RuntimeError("DDIM sampler produced no prediction")
        return predicted, logits

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"EmbryoDiff input must be (B,T,D), got {tuple(x.shape)}")
        semantic, semantic_logits, boundary, boundary_logits = self._conditions(x)
        out = {"semantic_logits": semantic_logits}
        if boundary_logits is not None:
            out["boundary_logits"] = boundary_logits
        if not self.requires_labels:
            out["logits"] = semantic_logits
            return out
        if self.label_embedding is None:
            raise RuntimeError("diffusion variant is missing its label embedding")
        if self.training:
            if y is None:
                raise ValueError("diffusion training requires frame labels")
            if y.shape != x.shape[:2]:
                raise ValueError("diffusion labels must align with (B,T) input")
            timestep = torch.randint(
                self.diffusion_train_steps,
                (x.shape[0],),
                device=x.device,
            )
            clean = self.label_embedding(y) * self.label_scale
            alpha = self.alpha_bars[timestep].to(clean.dtype)[:, None, None]
            noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * torch.randn_like(clean)
            predicted, logits = self._predict_clean(noisy, timestep, semantic, boundary)
        else:
            predicted, logits = self._ddim(semantic, boundary)
        out["diffusion_embedding"] = predicted
        out["diffusion_logits"] = logits
        out["logits"] = logits
        return out


@LOSS_REGISTRY.register("embryodiff_objective")
class EmbryoDiffObjective(AbsLoss):
    """Published semantic/smooth/boundary/diffusion weighted objective."""

    needs_outputs = True

    def __init__(
        self,
        class_counts: torch.Tensor | None = None,
        num_classes: int = 16,
        semantic_weight: float = 0.8,
        smooth_weight: float = 0.3,
        boundary_weight: float = 0.5,
        diffusion_weight: float = 1.0,
        truncation: float = 4.0,
    ) -> None:
        del class_counts, num_classes
        weights = (
            semantic_weight,
            smooth_weight,
            boundary_weight,
            diffusion_weight,
        )
        if any(value < 0 for value in weights) or truncation <= 0:
            raise ValueError("loss weights must be nonnegative and truncation positive")
        self.semantic_weight = float(semantic_weight)
        self.smooth_weight = float(smooth_weight)
        self.boundary_weight = float(boundary_weight)
        self.diffusion_weight = float(diffusion_weight)
        self.truncation = float(truncation)

    def __call__(self, out, y):
        if not isinstance(out, dict) or "semantic_logits" not in out:
            raise ValueError("embryodiff_objective requires semantic_logits")
        semantic = out["semantic_logits"]
        if semantic.shape[:2] != y.shape:
            raise ValueError("semantic logits must align with (B,T) labels")
        loss = self.semantic_weight * F.cross_entropy(
            semantic.reshape(-1, semantic.shape[-1]), y.reshape(-1)
        )
        if semantic.shape[1] > 1 and self.smooth_weight > 0:
            logp = F.log_softmax(semantic.float(), dim=-1)
            delta = logp[:, 1:] - logp[:, :-1].detach()
            smooth = delta.square().clamp(max=self.truncation**2).mean()
            loss = loss + self.smooth_weight * smooth
        if "boundary_logits" in out and self.boundary_weight > 0:
            target = torch.zeros_like(out["boundary_logits"], dtype=torch.float32)
            target[:, 1:] = (y[:, 1:] != y[:, :-1]).float()
            boundary = F.binary_cross_entropy_with_logits(out["boundary_logits"].float(), target)
            loss = loss + self.boundary_weight * boundary
        if "diffusion_logits" in out and self.diffusion_weight > 0:
            logits = out["diffusion_logits"]
            diffusion = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss = loss + self.diffusion_weight * diffusion
        return loss
