"""Kinetic models (factory products). ``seq_kinetic`` = backbone (registry) + temporal head (registry);
``r2plus1d`` = the paper's ResNet-3D with temporal strides set to 1 (one prediction per frame)."""

from __future__ import annotations

import torch
from torch import nn

from .interfaces import AbsKineticModel
from .registry import BACKBONE_REGISTRY, HEAD_REGISTRY, MODEL_REGISTRY


@MODEL_REGISTRY.register("seq_kinetic")
class SeqKinetic(AbsKineticModel):
    """Backbone (per frame) -> temporal head -> linear classifier. Optional auxiliary head (``model.aux``, TEMPO rung
    T8b): ``{"name": "cellcount_ordinal", "levels": 9}`` adds ``levels-1`` cumulative ("is the cell count > k?")
    logits on the same head features, trained by ``loss: ce_cellcount`` -- the phases t2..t8 differ only by the
    number of cells, which the diagnostics showed the backbone cannot separate (t7|t8 probe below chance); an
    explicit ordinal counting target is a direct signal for that. Ignored at inference (Viterbi uses ``logits``)."""

    def __init__(self, backbone: nn.Module, head: nn.Module, num_classes: int = 16, dropout: float = 0.5, aux: dict | None = None,
                 echo_div_score: bool = False) -> None:
        super().__init__()
        self.backbone, self.head = backbone, head
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(head.d_out, num_classes))
        self.is_sequence = not isinstance(head, HEAD_REGISTRY._registry["none"])
        self.aux = None
        if aux:
            if aux.get("name") != "cellcount_ordinal":
                raise ValueError(f"unknown model.aux {aux.get('name')!r}")
            self.aux = nn.Sequential(nn.Dropout(dropout), nn.Linear(head.d_out, int(aux.get("levels", 9)) - 1))
        # follow-up: when ``data.division_score_path`` appends a division-event score as the LAST
        # input channel (a constant (H, W) map, see nantes_kinetic.py), ``echo_div_score`` reads it straight back off
        # ``x`` (before the backbone consumes it) into ``out["div_score"]`` -- for ``loss: ce_divconsistency`` only,
        # which needs the raw score alongside the model's own predictions; the count-channel-only use (plain ``ce``
        # loss, backbone free to use or ignore the extra channel) needs no echo.
        self.echo_div_score = bool(echo_div_score)

    @classmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "SeqKinetic":
        backbone = BACKBONE_REGISTRY.build(cfg["backbone"], in_channels=in_channels)
        head = HEAD_REGISTRY.build(cfg.get("head", {"name": "none"}), d_in=backbone.feat_dim)
        return cls(backbone, head, int(cfg.get("num_classes", 16)), float(cfg.get("dropout", 0.5)), cfg.get("aux"),
                   bool(cfg.get("echo_div_score", False)))

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, None]
        B, L = x.shape[:2]
        out: dict = {}
        if self.echo_div_score:
            out["div_score"] = x[:, :, -1].mean(dim=(-1, -2))
        f = self.backbone(x.flatten(0, 1)).view(B, L, -1)
        h = self.head(f)
        out["logits"] = self.cls(h)
        if self.aux is not None:
            out["aux_logits"] = self.aux(h)
        return out


@MODEL_REGISTRY.register("event_conditioned")
class EventConditionedKinetic(SeqKinetic):
    """v23: unchanged visual/head/loss path plus an opt-in scalar feature residual."""

    @classmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "EventConditionedKinetic":
        from .event_features import EventFeatureBackbone

        if in_channels < 2:
            raise ValueError("event_conditioned requires visual input and a score channel")
        if cfg.get("echo_div_score") or cfg.get("aux"):
            raise ValueError("v23 isolates event conditioning; echo_div_score/aux are unsupported")
        # Construct parent components first, preserving their seed initialization.
        model = super().from_config(cfg, in_channels - 1)
        # Adapter initialization must not perturb the parent's dropout/sampler RNG.
        # All components are constructed on CPU before the builder moves the model.
        with torch.random.fork_rng(devices=[]):
            model.backbone = EventFeatureBackbone(
                model.backbone, in_channels - 1, **cfg.get("event_conditioning", {})
            )
        return model


class _MSTCNDilatedResidual(nn.Module):
    """One MS-TCN residual layer over ``(B, C, T)`` features."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilated = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.project = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.dilated(x))
        return x + self.dropout(self.project(h))


class _MSTCNSingleStage(nn.Module):
    """A full-resolution temporal-convolution stage.

    Input shape is ``(B, D, T)`` and output logits are ``(B, K, T)``.
    """

    def __init__(
        self,
        d_in: int,
        hidden: int,
        layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(d_in, hidden, 1)
        self.layers = nn.ModuleList(
            [
                _MSTCNDilatedResidual(hidden, 2**i, dropout)
                for i in range(layers)
            ]
        )
        self.classifier = nn.Conv1d(hidden, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        for layer in self.layers:
            h = layer(h)
        return self.classifier(h)


@MODEL_REGISTRY.register("full_video_mstcn")
class FullVideoMSTCN(AbsKineticModel):
    """Multi-stage TCN for cached frame features and complete videos.

    The architecture follows the discriminative full-video structure of MS-TCN:
    one prediction stage consumes frame embeddings and each later stage refines
    the previous stage probabilities. All stages are returned for stage-wise
    supervision instead of the repository's older last-stage-only approximation.

    Input:
        x: ``(B, T, D)`` cached visual and/or clock features.

    Returns:
        ``logits``: final-stage ``(B, T, K)`` logits.
        ``stage_logits``: all-stage ``(S, B, T, K)`` logits.
    """

    is_sequence = True

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        layers: int = 10,
        stages: int = 4,
        num_classes: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden, layers, stages, num_classes) < 1:
            raise ValueError("full_video_mstcn dimensions/stages must be positive")
        self.first = _MSTCNSingleStage(
            input_dim, hidden, layers, num_classes, dropout
        )
        self.refine = nn.ModuleList(
            [
                _MSTCNSingleStage(
                    num_classes, hidden, layers, num_classes, dropout
                )
                for _ in range(stages - 1)
            ]
        )

    @classmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "FullVideoMSTCN":
        backbone = cfg.get("backbone", {})
        if backbone.get("name") != "cached_features":
            raise ValueError(
                "full_video_mstcn requires model.backbone.name=cached_features"
            )
        head = cfg.get("head", {})
        if head.get("name") != "mstcn":
            raise ValueError("full_video_mstcn requires model.head.name=mstcn")
        return cls(
            input_dim=int(in_channels),
            hidden=int(head.get("hidden", 64)),
            layers=int(head.get("layers", 10)),
            stages=int(head.get("stages", 4)),
            num_classes=int(cfg.get("num_classes", 16)),
            dropout=float(head.get("dropout", cfg.get("dropout", 0.5))),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"full-video input must be (B,T,D), got {tuple(x.shape)}")
        stage = self.first(x.transpose(1, 2))
        outputs = [stage]
        for refine in self.refine:
            stage = refine(torch.softmax(stage, dim=1))
            outputs.append(stage)
        all_logits = torch.stack(
            [logits.transpose(1, 2) for logits in outputs], dim=0
        )
        return {"logits": all_logits[-1], "stage_logits": all_logits}


@MODEL_REGISTRY.register("r2plus1d")
class R2Plus1DKinetic(AbsKineticModel):
    """R(2+1)D-18 (Kinetics-400) with every temporal stride = 1 (arXiv 2203.00531 §2.5)."""

    is_sequence = True

    def __init__(self, num_classes: int = 16, pretrained: bool = True, dropout: float = 0.5) -> None:
        super().__init__()
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

        m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        for mod in m.modules():
            if isinstance(mod, nn.Conv3d) and mod.stride[0] != 1:
                mod.stride = (1, mod.stride[1], mod.stride[2])
        self.body = nn.Sequential(m.stem, m.layer1, m.layer2, m.layer3, m.layer4)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))

    @classmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "R2Plus1DKinetic":
        return cls(int(cfg.get("num_classes", 16)), bool(cfg.get("backbone", {}).get("pretrained", True)), float(cfg.get("dropout", 0.5)))

    def forward(self, x):
        if x.dim() == 4:
            x = x[:, None]
        if x.shape[2] != 3:
            x = x.mean(2, keepdim=True).expand(-1, -1, 3, -1, -1)
        f = self.body(x.transpose(1, 2)).mean((3, 4)).transpose(1, 2)  # (B, L, 512)
        return {"logits": self.cls(f)}


def build_model(cfg: dict, in_channels: int) -> AbsKineticModel:
    """Factory: ``cfg["type"]`` selects the registered product; the product parses the rest via ``from_config``."""
    return MODEL_REGISTRY._registry[cfg.get("type", "seq_kinetic")].from_config(cfg, in_channels)
