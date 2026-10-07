"""Pipeline builder (registered): YAML cfg -> KineticTrainProcess. Mirrors NST_segmentation's AbsAlgorithmBuilder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stseg.data.nantes_kinetic import NUM_CLASSES

from .interfaces import AbsPipelineBuilder
from .models import build_model
from .process import KineticTrainProcess
from .registry import BUILDER_REGISTRY, DATASET_REGISTRY, DECODER_REGISTRY, LOSS_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

ROOT = Path(__file__).resolve().parents[3]


@BUILDER_REGISTRY.register("kinetic_training")
class KineticTrainingBuilder(AbsPipelineBuilder):
    def __init__(self, cfg: dict, seed: int | None = None, smoke: bool = False, out_dir: str | None = None, resume: bool = True) -> None:
        super().__init__(cfg)
        exp = cfg.get("experiment", {})
        self.seed = int(seed if seed is not None else exp.get("seed", 0))
        self.smoke, self.resume = smoke, resume
        # Smoke runs (3 videos, 8 frames) go to runs/_smoke/, never the real run directory: on 2026-09-12 a --smoke
        # invocation with the real reference config silently overwrote runs/h7/resnet18_lstm_L4_split0_seed0
        # (best.pt/results.json/history.json), costing a retrain of the benchmark's reference seed.
        out_root = "runs/_smoke" if smoke else exp.get("out_root", "runs/h7")
        self.out_dir = Path(out_dir) if out_dir else ROOT / out_root / f"{exp['id']}_seed{self.seed}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self._data = self._model = self._criterion = self._opt = self._sched = self._log_trans = None

    def produce_pre_processor(self) -> None:
        d = self._cfg["data"]
        self._data = DATASET_REGISTRY.get(d.get("dataset", "nantes_clips"), d, self.seed, self.smoke)
        test = (
            f" | test {len(self._data.test)} / {len(self._data.test.videos())}"
            if hasattr(self._data, "test")
            else " | test NOT CONSTRUCTED"
        )
        print(f"data: train {len(self._data.train_frames)} frames / {len(self._data.train_frames.videos())} videos | "
              f"val {len(self._data.val)} / {len(self._data.val.videos())}{test} | "
              f"in_channels={self._data.in_channels}" + (f" | clips/epoch={len(self._data.train_set)} (L={self._data.clip_len})" if hasattr(self._data, "clip_len") else ""))

    def produce_main_processor(self) -> None:
        mc = dict(self._cfg["model"])
        self._model = build_model(mc, self._data.in_channels).to(self.device)
        counts = torch.tensor([self._data.train_frames.rows.label.value_counts().get(k, 0) for k in range(NUM_CLASSES)], dtype=torch.float32)
        self._criterion = LOSS_REGISTRY.build(self._cfg.get("loss", {"name": "ce"}), class_counts=counts, num_classes=NUM_CLASSES)
        ocfg = dict(self._cfg.get("optimizer", {"name": "sgd"}))
        mult, decay = float(ocfg.pop("backbone_lr_mult", 1.0)), float(ocfg.pop("layer_decay", 1.0))
        if mult != 1.0 or decay != 1.0:
            from .optim import param_groups
            params = param_groups(self._model, float(ocfg.get("lr", 1e-3)), mult, decay)
            print(f"optimizer: {len(params)} param groups (backbone_lr_mult={mult}, layer_decay={decay}); lr scales " + ", ".join(f"{g['lr_scale']:.3g}" for g in params))
        else:
            params = self._model.parameters()
        self._opt = OPTIMIZER_REGISTRY.build(ocfg, params=params)
        self._sched = SCHEDULER_REGISTRY.build(self._cfg.get("scheduler", {"name": "none"}), optimizer=self._opt, epochs=int(self._cfg["training"].get("epochs", 10)))
        print(f"model: {mc.get('type', 'seq_kinetic')} {mc.get('backbone')} + {mc.get('head', {'name': 'none'})} | "
              f"{sum(p.numel() for p in self._model.parameters()) / 1e6:.1f}M params | sequence={self._model.is_sequence} | loss={self._cfg.get('loss', {'name': 'ce'})}")

    def produce_post_processor(self) -> None:
        dec = DECODER_REGISTRY.build(self._cfg.get("decoder", {"name": "viterbi"}))
        self._log_trans = dec.log_transition(self._data.train_full.label_sequences(), NUM_CLASSES)

    @property
    def product(self) -> KineticTrainProcess:
        if self._data is None:
            self.produce_pre_processor()
        if self._model is None:
            self.produce_main_processor()
        if self._log_trans is None:
            self.produce_post_processor()
        return KineticTrainProcess(self._cfg, self._data, self._model, self._criterion, self._opt, self._sched, self._log_trans,
                                   self.out_dir, self.seed, self.smoke, self.device, self.resume)


@BUILDER_REGISTRY.register("kinetic_validation_training")
class KineticValidationBuilder(KineticTrainingBuilder):
    """Matched kinetic trainer that never constructs or evaluates test data."""

    def produce_pre_processor(self) -> None:
        allowed = {"nantes_clips_trainval", "nantes_emfit_frames_trainval", "nantes_video_trainval"}
        if self._cfg["data"].get("dataset") not in allowed:
            raise ValueError(
                "kinetic_validation_training requires a train/validation-only dataset"
            )
        super().produce_pre_processor()

    @property
    def product(self) -> KineticTrainProcess:
        if self._data is None:
            self.produce_pre_processor()
        if self._model is None:
            self.produce_main_processor()
        if self._log_trans is None:
            self.produce_post_processor()
        return KineticTrainProcess(
            self._cfg,
            self._data,
            self._model,
            self._criterion,
            self._opt,
            self._sched,
            self._log_trans,
            self.out_dir,
            self.seed,
            self.smoke,
            self.device,
            self.resume,
            evaluate_test=False,
        )
