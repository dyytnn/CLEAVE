"""Frame backbones. torchvision ImageNet models are registered generically; ``kinetic_cnn`` is the from-scratch ablation."""

from __future__ import annotations

import torch
from torch import nn

from .interfaces import AbsFrameBackbone
from .registry import BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register("cached_features")
class CachedFeatureBackbone(AbsFrameBackbone):
    """Identity adapter for precomputed frame embeddings.

    Args:
        in_channels: Embedding dimension ``D`` supplied by the cached-feature
            dataset.

    Input/return shape: ``(N, D)``. The registered adapter makes cached-feature
    experiments obey the same registry/config contract as image backbones; the
    dedicated full-video model validates the adapter name and consumes the
    features directly.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.feat_dim = int(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.feat_dim:
            raise ValueError(
                f"cached features must have shape (N, {self.feat_dim}), got {tuple(x.shape)}"
            )
        return x


def _replace_first_conv(model: nn.Module, in_channels: int) -> None:
    """Swap the first Conv2d for an ``in_channels`` one, initialised from the mean of the pretrained RGB filters."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            new = nn.Conv2d(in_channels, mod.out_channels, mod.kernel_size, mod.stride, mod.padding, mod.dilation, mod.groups, mod.bias is not None)
            with torch.no_grad():
                new.weight.copy_(mod.weight.mean(1, keepdim=True).repeat(1, in_channels, 1, 1))
                if mod.bias is not None:
                    new.bias.copy_(mod.bias)
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new)
            return
    raise RuntimeError("no Conv2d found to replace")


class TorchvisionBackbone(AbsFrameBackbone):
    """Any torchvision classification model, classifier stripped -> (B, feat_dim)."""

    HEAD_ATTRS = ("fc", "classifier", "head", "heads")

    def __init__(self, tv_name: str, pretrained: bool = True, in_channels: int = 3, init_from: str | None = None) -> None:
        super().__init__()
        from torchvision.models import get_model

        self.net = get_model(tv_name, weights="DEFAULT" if (pretrained and not init_from) else None)
        for attr in self.HEAD_ATTRS:
            if hasattr(self.net, attr):
                setattr(self.net, attr, nn.Identity()); break
        else:
            raise ValueError(f"cannot strip the head of {tv_name}")
        if in_channels != 3:
            _replace_first_conv(self.net, in_channels)
        with torch.no_grad():
            self.net.eval()
            self.feat_dim = int(self.net(torch.zeros(1, in_channels, 224, 224)).flatten(1).shape[1])
            self.net.train()
        if init_from:  # TEMPO rung T7: load backbone weights from scripts/pretrain_video_mae.py instead of ImageNet
            sd = torch.load(init_from, map_location="cpu", weights_only=True)
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            print(f"[backbone] loaded {init_from}: {len(missing)} missing, {len(unexpected)} unexpected keys")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-pool spatial features (N, C, h, w) -- what the cross-plane fusion modules consume."""
        net = self.net
        if hasattr(net, "layer4"):  # ResNet family
            for name, mod in net.named_children():
                if name in ("avgpool", "fc"):
                    break
                x = mod(x)
            return x
        if hasattr(net, "features"):  # ConvNeXt (N, C, h, w) / Swin (N, h, w, C)
            f = net.features(x)
            return f.permute(0, 3, 1, 2).contiguous() if f.shape[-1] == self.feat_dim and f.shape[1] != self.feat_dim else f
        raise ValueError(f"feature_map not supported for {type(net).__name__}")


@BACKBONE_REGISTRY.register("dinov2_vitb14")
class DinoV2Backbone(AbsFrameBackbone):
    """DINOv2 ViT-B/14 (self-supervised, Oquab et al. 2023) via torch.hub; frame feature = CLS token (768-d),
    ``feature_map`` = the 16x16 patch tokens as (N, 768, 16, 16) for the cross-plane fusion modules. Self-supervised
    natural-image features transfer to microscopy better than ImageNet-supervised ones in most reports, and a ViT has
    no BatchNorm -- which matters at the batch sizes the 7-plane clips force."""

    def __init__(self, pretrained: bool = True, in_channels: int = 3, init_from: str | None = None, arch: str = "dinov2_vitb14") -> None:
        super().__init__()
        self.net = torch.hub.load("facebookresearch/dinov2", arch, pretrained=pretrained, trust_repo=True, verbose=False)
        self.feat_dim = int(self.net.embed_dim)
        self.in_channels = int(in_channels)
        if init_from:
            sd = torch.load(init_from, map_location="cpu", weights_only=True)
            print("[backbone] loaded", init_from, self.net.load_state_dict(sd, strict=False))

    def _rgb(self, x):  # planes stacked as channels -> average to one grey image, replicated (ViT patch embed is RGB)
        return x if x.shape[1] == 3 else x.mean(1, keepdim=True).expand(-1, 3, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward_features(self._rgb(x))["x_norm_clstoken"]

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        t = self.net.forward_features(self._rgb(x))["x_norm_patchtokens"]  # (N, hw, C)
        N, L, C = t.shape; h = int(L ** 0.5)
        return t.transpose(1, 2).reshape(N, C, h, L // h)


def _register_torchvision(name: str) -> None:
    """One registered factory per torchvision architecture (explicit, no dynamic class creation)."""
    def factory(pretrained: bool = True, in_channels: int = 3, init_from: str | None = None) -> TorchvisionBackbone:
        return TorchvisionBackbone(name, pretrained, in_channels, init_from)
    factory.__name__ = f"{name}_backbone"
    BACKBONE_REGISTRY.register(name)(factory)


for _name in ["resnet18", "resnet34", "resnet50", "convnext_tiny", "efficientnet_b0", "efficientnet_v2_s", "efficientnet_v2_l",
              "swin_t", "vit_b_16", "regnet_y_800mf", "mobilenet_v3_large"]:
    _register_torchvision(_name)


class CrossFocalBackbone(AbsFrameBackbone):
    """TEMPO rung T9: per-plane shared CNN + attention fusion across focal planes, instead of stacking planes as input
    channels. Input (N, P, H, W) -- one grey channel per focal plane, as produced by ``data.planes`` -- each plane is
    replicated to 3 channels and encoded by the *same* ImageNet-pretrained CNN (so pretrained filters are used as
    intended, unlike the channel-stacking variant whose first conv is re-initialised from the filter mean); the P plane
    features receive a learned plane embedding and are fused by one multi-head attention step from a learned query
    (plus the plane-mean as a residual), giving one ``feat_dim`` vector per frame. Motivation: the 3-plane stacked
    input gave the only consistent per-phase gain in the sweep (tPNf), and the cleavage phases the models miss are
    decided by membranes that are in focus in only some of the seven planes. Exposes ``.feat_dim`` and behaves as a
    drop-in ``backbone`` for ``SeqKinetic`` (so whole-video inference, diagnostics and caching work unchanged)."""

    def __init__(self, cnn: str = "resnet18", pretrained: bool = True, in_channels: int = 3, nhead: int = 4, init_from: str | None = None) -> None:
        super().__init__()  # in_channels = number of focal planes (injected from the dataset)
        self.cnn = BACKBONE_REGISTRY.get(cnn, pretrained=pretrained, in_channels=3, init_from=init_from)
        d = self.cnn.feat_dim
        self.P, self.feat_dim = int(in_channels), d
        self.plane_emb = nn.Parameter(torch.zeros(self.P, d))
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.plane_emb, std=0.02); nn.init.normal_(self.query, std=0.02)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, P, H, W = x.shape
        f = self.cnn(x.reshape(N * P, 1, H, W).expand(-1, 3, -1, -1)).view(N, P, -1)
        tok = self.norm(f + self.plane_emb[None])
        fused, _ = self.attn(self.query.expand(N, -1, -1), tok, tok)
        return fused[:, 0] + f.mean(1)


@BACKBONE_REGISTRY.register("crossfocal")
def crossfocal_backbone(cnn: str = "resnet18", pretrained: bool = True, in_channels: int = 3, nhead: int = 4, init_from: str | None = None) -> CrossFocalBackbone:
    return CrossFocalBackbone(cnn, pretrained, in_channels, nhead, init_from)


def _laplacian_sharpness(x: torch.Tensor, grid: int) -> torch.Tensor:
    """Per-plane, per-region focus measure: variance of the Laplacian response pooled to a ``grid x grid`` map.
    x (N, P, H, W) in normalised intensity -> (N, P, grid, grid), log-scaled and standardised over planes per region, so
    that it is a relative 'which plane is sharp here' signal rather than an absolute contrast level."""
    N, P, H, W = x.shape
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    lap = torch.nn.functional.conv2d(x.reshape(N * P, 1, H, W), k, padding=1)
    e = torch.nn.functional.adaptive_avg_pool2d(lap.pow(2), grid).view(N, P, grid, grid)
    e = torch.log(e + 1e-6)
    return (e - e.mean(1, keepdim=True)) / (e.std(1, keepdim=True) + 1e-3)


class FocalAttnBackbone(AbsFrameBackbone):
    """TEMPO rung T10: *Focus-Adaptive Cross-Plane Attention* -- two extensions of ``CrossFocalBackbone`` motivated by
    what the diagnostics say the cross-focal model is doing right (fusing planes with attention) and what it still
    cannot see (cell membranes that are in focus at different depths in different image regions):

    * ``spatial=True``: fuse at the *feature-map* level rather than after global pooling. For every spatial position
      the P plane tokens are attended by a content query (a projection of the plane-mean at that position), so each
      region selects the plane(s) in which it is sharp; the fused map is then globally pooled. Attention is over
      planes only (P per position), so cost is linear in positions -- no h*w x h*w attention.
    * ``sharpness=True``: the attention logits receive a bias ``beta * sharpness_p(region)`` where sharpness is the
      log variance of the Laplacian of the *input* plane in that region, standardised across planes (learned scalar
      ``beta``, init 1). The prior is optics, not labels: it tells the fusion which plane is physically in focus
      before any feature is computed, and makes the attention map interpretable against focus.

    Both off reproduces ``CrossFocalBackbone`` up to the content query. Works with any registered backbone exposing ``feature_map``
    (ResNet, ConvNeXt, Swin, DINOv2). ``last_attn`` keeps the most recent attention weights (N, P, h, w) or (N, P) for figures."""

    def __init__(self, cnn: str = "resnet18", pretrained: bool = True, in_channels: int = 3, spatial: bool = True,
                 sharpness: bool = True, grid: int = 7, init_from: str | None = None) -> None:
        super().__init__()
        self.cnn = BACKBONE_REGISTRY.get(cnn, pretrained=pretrained, in_channels=3, init_from=init_from)
        if not hasattr(self.cnn, "feature_map"):
            raise ValueError(f"focalattn needs a backbone with .feature_map (ResNet/ConvNeXt/Swin/DINOv2), got {cnn}")
        d = self.cnn.feat_dim
        self.P, self.feat_dim, self.spatial, self.sharp, self.grid = int(in_channels), d, bool(spatial), bool(sharpness), int(grid)
        self.plane_emb = nn.Parameter(torch.zeros(self.P, d)); nn.init.normal_(self.plane_emb, std=0.02)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, P, H, W = x.shape
        fmap = self.cnn.feature_map(x.reshape(N * P, 1, H, W).expand(-1, 3, -1, -1))  # (N*P, C, h, w)
        C, h, w = fmap.shape[1:]
        if not self.spatial:
            fmap = fmap.mean((2, 3), keepdim=True); h = w = 1
        tok = fmap.view(N, P, C, h * w).permute(0, 3, 1, 2)  # (N, hw, P, C)
        tok = self.norm(tok + self.plane_emb[None, None])
        q = self.q(tok.mean(2, keepdim=True))  # content query per region (N, hw, 1, C)
        att = (q * self.k(tok)).sum(-1) / C ** 0.5  # (N, hw, P)
        if self.sharp:
            sh = _laplacian_sharpness(x.float(), h if self.spatial else 1).to(att.dtype)  # (N, P, h, w)
            att = att + self.beta * sh.view(N, P, h * w).transpose(1, 2)
        att = att.softmax(-1)
        self.last_attn = att.detach().transpose(1, 2).reshape(N, P, h, w)
        fused = (att.unsqueeze(-1) * self.v(tok)).sum(2) + tok.mean(2)  # (N, hw, C), residual plane-mean
        return fused.mean(1)  # global average pool over regions -> (N, C)


@BACKBONE_REGISTRY.register("focalattn")
def focalattn_backbone(cnn: str = "resnet18", pretrained: bool = True, in_channels: int = 3, spatial: bool = True,
                       sharpness: bool = True, grid: int = 7, init_from: str | None = None) -> FocalAttnBackbone:
    return FocalAttnBackbone(cnn, pretrained, in_channels, spatial, sharpness, grid, init_from)


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


@BACKBONE_REGISTRY.register("kinetic_cnn")
class KineticCNNBackbone(AbsFrameBackbone):
    """Small from-scratch CNN (no-pretraining ablation)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, pretrained: bool = False) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(nn.Conv2d(in_channels, c, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.stages = nn.Sequential(ConvBlock(c, c * 2), ConvBlock(c * 2, c * 4), ConvBlock(c * 4, c * 8), ConvBlock(c * 8, c * 8))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = c * 8

    def forward(self, x):
        return self.pool(self.stages(self.stem(x))).flatten(1)
