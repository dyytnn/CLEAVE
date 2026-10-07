"""Abstract interfaces of the kinetic pipeline (mirrors NST_segmentation/abs_interface/)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import nn


class AbsFrameBackbone(nn.Module, ABC):
    """(B, C, H, W) -> (B, feat_dim)."""

    feat_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class AbsTemporalHead(nn.Module, ABC):
    """(B, L, D_in) -> (B, L, d_out)."""

    d_out: int

    @abstractmethod
    def forward(self, f: torch.Tensor) -> torch.Tensor: ...


class AbsKineticModel(nn.Module, ABC):
    """Clip (B, L, C, H, W) or frame (B, C, H, W) -> {"logits": (B, L, K)}."""

    is_sequence: bool = False

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "AbsKineticModel": ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]: ...


class AbsLoss(ABC):
    @abstractmethod
    def __call__(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


class AbsDecoder(ABC):
    """Turns per-frame log-probabilities of one video into a transition matrix used by the evaluator."""

    @abstractmethod
    def log_transition(self, train_label_sequences: list[np.ndarray], n_classes: int) -> np.ndarray: ...


class AbsProcess(ABC):
    @abstractmethod
    def run(self) -> dict[str, Any]: ...


class AbsPipelineBuilder(ABC):
    """Builder pattern: produce_* assemble the parts, ``product`` returns the runnable process."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg

    @property
    @abstractmethod
    def product(self) -> AbsProcess: ...

    @abstractmethod
    def produce_pre_processor(self) -> None: ...   # data

    @abstractmethod
    def produce_main_processor(self) -> None: ...  # model, loss, optimizer

    @abstractmethod
    def produce_post_processor(self) -> None: ...  # decoder / evaluation / logging
