"""Optimizers and schedulers as registered factory functions (return the torch objects directly)."""

from __future__ import annotations

import torch

from .registry import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY


_LAYER_PATTERNS = (r"\.blocks\.(\d+)\.", r"\.layer(\d)\.", r"\.features\.(\d+)\.", r"\.stages\.(\d+)\.")


def _layer_id(name: str) -> int | None:
    import re
    for pat in _LAYER_PATTERNS:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def param_groups(model, lr: float, backbone_lr_mult: float = 1.0, layer_decay: float = 1.0, weight_decay: float | None = None) -> list[dict]:
    """Backbone-fair fine-tuning (TEMPO T11): parameters under ``backbone.`` get ``lr * backbone_lr_mult``, further scaled
    by ``layer_decay ** (n_layers - layer_id)`` (layer-wise lr decay, Clark et al. 2020 / BEiT) when the parameter name
    carries a block/stage index; everything else (heads, classifier, fusion) trains at ``lr``. Embedding / patch-embed /
    stem parameters without an index get the deepest decay. Returns torch param groups (with ``name`` for logging)."""
    names = [n for n, _ in model.named_parameters()]
    bb = [n for n in names if n.startswith("backbone.")]
    ids = [i for i in (_layer_id(n) for n in bb) if i is not None]
    n_layers = (max(ids) + 1) if ids else 0
    groups: dict[float, list] = {}
    for n, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        if n.startswith("backbone."):
            lid = _layer_id(n)
            depth = (n_layers - lid) if lid is not None else (n_layers + 1)
            scale = backbone_lr_mult * (layer_decay ** depth if n_layers else 1.0)
        else:
            scale = 1.0
        groups.setdefault(round(scale, 8), []).append(prm)
    out = [{"params": ps, "lr": lr * sc, "lr_scale": sc} for sc, ps in sorted(groups.items())]
    if weight_decay is not None:
        for g in out:
            g["weight_decay"] = weight_decay
    return out


@OPTIMIZER_REGISTRY.register("sgd")
def sgd(params, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.0):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


@OPTIMIZER_REGISTRY.register("adamw")
def adamw(params, lr: float = 1e-4, weight_decay: float = 1e-4, betas=(0.9, 0.999)):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas))


@OPTIMIZER_REGISTRY.register("adam")
def adam(params, lr: float = 1.6e-4, weight_decay: float = 1e-5, betas=(0.9, 0.999)):
    """Adam recipe used by the public EMFiT training script."""
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=tuple(betas))


class _NoScheduler:
    def step(self) -> None: ...
    def state_dict(self) -> dict: return {}
    def load_state_dict(self, d: dict) -> None: ...


@SCHEDULER_REGISTRY.register("none")
def none(optimizer, epochs: int):
    return _NoScheduler()


@SCHEDULER_REGISTRY.register("cosine")
def cosine(optimizer, epochs: int, eta_min: float = 0.0):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


@SCHEDULER_REGISTRY.register("cosine_warm_restarts")
def cosine_warm_restarts(
    optimizer,
    epochs: int,
    eta_min: float = 5e-5,
    t0: int = 30,
    t_mult: int = 1,
):
    """Epoch-stepped cosine warm restarts from the released EMFiT recipe."""
    del epochs
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t0,
        T_mult=t_mult,
        eta_min=eta_min,
    )


@SCHEDULER_REGISTRY.register("warmup_cosine")
def warmup_cosine(optimizer, epochs: int, warmup_epochs: int = 1, eta_min_ratio: float = 0.01):
    """Linear warm-up for ``warmup_epochs`` then cosine decay to ``eta_min_ratio`` x lr (stepped once per epoch);
    the standard recipe for transformer heads / AdamW."""
    import math

    def lam(e: int) -> float:
        if e < warmup_epochs:
            return (e + 1) / warmup_epochs
        t = (e - warmup_epochs) / max(1, epochs - warmup_epochs)
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lam)
