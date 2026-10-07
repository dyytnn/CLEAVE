"""Architecture-faithful EMFiT common-protocol reproduction.

The implementation follows the authors' public MIT-licensed source pinned in
```` section 12.aaa.  The only task-level change is a 17-output
head: 16 frozen TEMPO phases plus the published frame-index regression scalar.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .interfaces import AbsKineticModel
from .registry import BACKBONE_REGISTRY, MODEL_REGISTRY


class FrozenBatchNorm3d(nn.Module):
    """Frozen 3-D batch normalization used by the released EMFiT code."""

    def __init__(
        self,
        scale: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        eps: float,
    ) -> None:
        super().__init__()
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("bias", bias.detach().clone())
        self.register_buffer("running_mean", running_mean.detach().clone())
        self.register_buffer("running_var", running_var.detach().clone())
        self.eps = float(eps)

    @classmethod
    def from_batch_norm(cls, layer: nn.BatchNorm3d) -> "FrozenBatchNorm3d":
        return cls(
            layer.weight,
            layer.bias,
            layer.running_mean,
            layer.running_var,
            layer.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` with fixed population statistics."""
        return torch.batch_norm(
            x,
            self.scale,
            self.bias,
            self.running_mean,
            self.running_var,
            False,
            0.1,
            self.eps,
            torch.backends.cudnn.enabled,
        )


def _freeze_batch_norm_3d(module: nn.Module) -> None:
    # Preserve a subtle behavior of the released helper: BatchNorm attributes
    # (``bn1``/``bn2``/``bn3``) are replaced, while a BatchNorm stored at a
    # numeric index inside ``nn.Sequential`` (the residual downsample path) is
    # not discoverable through ``dir(module)`` and remains trainable.
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d) and not isinstance(module, nn.Sequential):
            setattr(module, name, FrozenBatchNorm3d.from_batch_norm(child))
        else:
            _freeze_batch_norm_3d(child)


class InflatedBottleneck(nn.Module):
    """Inflated ResNet bottleneck from the official EMFiT I3D encoder."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int,
        downsample: nn.Module | None,
        temporal_conv: int,
        temporal_stride: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=(1 + 2 * temporal_conv, 1, 1),
            stride=(temporal_stride, 1, 1),
            padding=(temporal_conv, 0, 0),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a focal-volume tensor ``(B,C,F,H,W)``."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


@BACKBONE_REGISTRY.register("emfit_i3d")
class EMFiTI3D(nn.Module):
    """Released truncated I3D-ResNet-50: ``(B,3,7,224,224) -> (B,196,768)``."""

    feat_dim = 768

    def __init__(
        self,
        in_channels: int = 3,
        checkpoint: str | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"EMFiT I3D requires 3 image channels, got {in_channels}")
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(5, 7, 7),
            stride=(2, 2, 2),
            padding=(2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d((2, 3, 3), stride=(2, 2, 2))
        self.maxpool2 = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.layer1 = self._make_layer(64, 3, 1, [1, 1, 1])
        self.layer2 = self._make_layer(128, 4, 2, [1, 0, 1, 0])
        self.layer3 = self._make_layer(256, 6, 2, [1, 0, 1, 0, 1, 0])
        self.conv2 = nn.Conv2d(1024, self.feat_dim, 1)
        self.bn2 = nn.BatchNorm2d(self.feat_dim)

        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out")
            elif isinstance(layer, nn.BatchNorm3d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.pretrained_keys = 0
        if checkpoint is not None:
            self.pretrained_keys = self._load_pretrained(checkpoint, expected_sha256)
            _freeze_batch_norm_3d(self)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int,
        temporal_convs: list[int],
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * InflatedBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * InflatedBottleneck.expansion,
                    1,
                    stride=(1, stride, stride),
                    bias=False,
                ),
                nn.BatchNorm3d(planes * InflatedBottleneck.expansion),
            )
        layers: list[nn.Module] = [
            InflatedBottleneck(
                self.inplanes,
                planes,
                stride,
                downsample,
                temporal_convs[0],
                1,
            )
        ]
        self.inplanes = planes * InflatedBottleneck.expansion
        layers.extend(
            InflatedBottleneck(
                self.inplanes,
                planes,
                1,
                None,
                temporal_convs[index],
                1,
            )
            for index in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _load_pretrained(self, checkpoint: str, expected_sha256: str | None) -> int:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"EMFiT I3D checkpoint not found: {path}")
        if expected_sha256 is not None:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_sha256:
                raise ValueError(
                    f"EMFiT I3D checksum mismatch: expected {expected_sha256}, got {digest}"
                )
        source = torch.load(path, map_location="cpu", weights_only=True)
        own = self.state_dict()
        matched = {
            key: value
            for key, value in source.items()
            if key in own and own[key].shape == value.shape
        }
        if len(matched) < 200:
            raise ValueError(f"only {len(matched)} compatible I3D tensors in {path}")
        own.update(matched)
        self.load_state_dict(own)
        return len(matched)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return focal-collapsed spatial tokens, shape ``(B,196,768)`` at 224 px."""
        if x.ndim != 5 or x.shape[1:3] != (3, 7):
            raise ValueError(f"EMFiT I3D expects (B,3,7,H,W), got {tuple(x.shape)}")
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if x.shape[2] != 1:
            raise ValueError(f"EMFiT focal axis did not collapse to one: {tuple(x.shape)}")
        x = self.relu(self.bn2(self.conv2(x.squeeze(2))))
        return x.flatten(2).transpose(1, 2)


@MODEL_REGISTRY.register("emfit")
class EMFiT(AbsKineticModel):
    """Multi-focus I3D + frame-index-token ViT with auxiliary clock regression."""

    is_sequence = False
    requires_frame_index = True

    def __init__(
        self,
        backbone: EMFiTI3D,
        num_classes: int = 16,
        vit_pretrained: bool = True,
    ) -> None:
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        self.backbone = backbone
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if vit_pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.vit.conv_proj = nn.Identity()
        dim = self.vit.heads.head.in_features
        if dim != backbone.feat_dim:
            raise ValueError(f"I3D/ViT width mismatch: {backbone.feat_dim} vs {dim}")
        self.num_classes = int(num_classes)
        self.frame_encoder = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.vit.heads.head = nn.Linear(dim, self.num_classes + 1)
        nn.init.normal_(self.frame_encoder[0].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.frame_encoder[0].bias)
        nn.init.trunc_normal_(self.vit.heads.head.weight[: self.num_classes], std=0.02)
        nn.init.zeros_(self.vit.heads.head.bias[: self.num_classes])
        nn.init.normal_(self.vit.heads.head.weight[-1], mean=0.0, std=0.01)
        nn.init.zeros_(self.vit.heads.head.bias[-1])

    @classmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "EMFiT":
        backbone = BACKBONE_REGISTRY.build(cfg["backbone"], in_channels=in_channels)
        if not isinstance(backbone, EMFiTI3D):
            raise ValueError("model.type=emfit requires backbone.name=emfit_i3d")
        emfit = cfg.get("emfit", {})
        return cls(
            backbone,
            int(cfg.get("num_classes", 16)),
            bool(emfit.get("vit_pretrained", True)),
        )

    def forward(
        self,
        x: torch.Tensor,
        frame_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict phases from ``x`` `(B,3,7,224,224)` and normalized frame input `(B,1)`."""
        if frame_input is None:
            raise ValueError("EMFiT requires normalized frame_input")
        if frame_input.ndim == 1:
            frame_input = frame_input[:, None]
        if frame_input.ndim != 2 or frame_input.shape != (x.shape[0], 1):
            raise ValueError(
                f"frame_input must be (B,1), got {tuple(frame_input.shape)}"
            )
        patch_tokens = self.backbone(x)
        frame_token = self.frame_encoder(frame_input.float()).unsqueeze(1)
        encoded = self.vit.encoder(torch.cat([frame_token, patch_tokens], dim=1))
        outputs = self.vit.heads(encoded[:, 0])
        return {
            "logits": outputs[:, : self.num_classes],
            "frame_regression": outputs[:, self.num_classes],
            "frame_target": frame_input[:, 0],
        }
