"""Registries for the kinetic-stage (H7/H8) pipeline — same semantics as NST_segmentation/registry/_registry_impl.py.

``REGISTRY.register("name")`` decorates a class; ``REGISTRY.get("name", **kwargs)`` instantiates it;
``REGISTRY.build(cfg)`` instantiates from a ``{"name": ..., <params>}`` dict. Every pluggable piece of the pipeline
(backbone, temporal head, model, loss, dataset, decoder, optimizer, scheduler, pipeline builder) is looked up by the
string written in the experiment YAML, so a new variant is one registered class + one YAML file, no edits elsewhere.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        def decorator(cls):
            if name in self._registry:
                raise KeyError(f"{name!r} already registered in {self._name}")
            self._registry[name] = cls
            return cls
        return decorator

    def check(self, name: str) -> bool:
        return name in self._registry

    def get(self, name: str, *args, **kwargs):
        if name not in self._registry:
            raise KeyError(f"{name!r} not in {self._name}; known: {sorted(self._registry)}")
        return self._registry[name](*args, **kwargs)

    def build(self, cfg: dict | None, **extra):
        """Instantiate from ``{"name": ..., **params}``; ``extra`` are injected kwargs (e.g. feat_dim)."""
        cfg = dict(cfg or {})
        name = cfg.pop("name")
        return self.get(name, **cfg, **extra)

    def names(self) -> list[str]:
        return sorted(self._registry)

    def __repr__(self) -> str:
        return f"Registry({self._name}): {self.names()}"


BACKBONE_REGISTRY = Registry("backbone")
HEAD_REGISTRY = Registry("temporal_head")
MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
DATASET_REGISTRY = Registry("dataset")
DECODER_REGISTRY = Registry("decoder")
OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")
BUILDER_REGISTRY = Registry("pipeline_builder")
