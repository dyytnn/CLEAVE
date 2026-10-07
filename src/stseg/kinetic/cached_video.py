"""Validation-only cached-feature full-video pipeline for TEMPO v26.

The cache contains train/validation frame embeddings only. This module provides
registered data and builder components while reusing the established kinetic
training/evaluation process and metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .builder import KineticTrainingBuilder
from .process import KineticTrainProcess
from .registry import BUILDER_REGISTRY, DATASET_REGISTRY

ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def clock_features(
    time_h: np.ndarray,
    mean_h: float,
    std_h: float,
    periods_h: tuple[float, ...],
) -> np.ndarray:
    """Encode absolute acquisition time without using video duration.

    Args:
        time_h: ``(T,)`` absolute hours since insemination.
        mean_h: Training-frame mean hour.
        std_h: Training-frame standard deviation.
        periods_h: Fixed Fourier periods in hours.

    Returns:
        ``(T, 2 + 2 * len(periods_h))`` float32 clock features. The first two
        columns are standardised time and its bounded signed square.
    """
    t = np.asarray(time_h, dtype=np.float32)
    z = (t - float(mean_h)) / max(float(std_h), 1e-6)
    values = [z, np.sign(z) * np.minimum(z * z, 16.0)]
    for period in periods_h:
        angle = 2.0 * np.pi * t / float(period)
        values.extend([np.sin(angle), np.cos(angle)])
    return np.stack(values, axis=1).astype(np.float32)


class _FeaturePartition(Dataset):
    """Frame-indexable view of one cached partition."""

    def __init__(
        self,
        cache_path: Path,
        expected_videos: set[str],
        input_mode: str,
        visual_mean: np.ndarray,
        visual_std: np.ndarray,
        clock_mean_h: float,
        clock_std_h: float,
        clock_periods_h: tuple[float, ...],
        clock_offset_h: float = 0.0,
        clock_rate: float = 1.0,
    ) -> None:
        if input_mode not in {"visual", "clock", "visual_clock"}:
            raise ValueError(f"unknown cached input mode {input_mode!r}")
        if clock_rate <= 0:
            raise ValueError("clock_rate must be positive")
        self.input_mode = input_mode
        self.visual_mean = visual_mean.astype(np.float32)
        self.visual_std = visual_std.astype(np.float32)
        self.clock_mean_h = float(clock_mean_h)
        self.clock_std_h = float(clock_std_h)
        self.clock_periods_h = tuple(float(v) for v in clock_periods_h)
        self.clock_offset_h = float(clock_offset_h)
        self.clock_rate = float(clock_rate)

        raw = np.load(cache_path, allow_pickle=False)
        videos = sorted(k[: -len("__emb")] for k in raw.files if k.endswith("__emb"))
        if set(videos) != expected_videos:
            missing = sorted(expected_videos - set(videos))[:5]
            extra = sorted(set(videos) - expected_videos)[:5]
            raise ValueError(
                f"{cache_path}: cached video IDs differ from split; missing={missing}, extra={extra}"
            )
        self.by_video: dict[str, dict[str, np.ndarray]] = {}
        rows: list[dict[str, Any]] = []
        for video in videos:
            emb = np.asarray(raw[f"{video}__emb"], dtype=np.float16)
            labels = np.asarray(raw[f"{video}__y"], dtype=np.int64)
            times = np.asarray(raw[f"{video}__t"], dtype=np.float32)
            frames = np.asarray(raw[f"{video}__frame"], dtype=np.int64)
            n = len(labels)
            if emb.ndim != 2 or not (len(emb) == len(times) == len(frames) == n):
                raise ValueError(f"{cache_path}: inconsistent arrays for {video}")
            if n < 1 or not np.isfinite(emb).all() or not np.isfinite(times).all():
                raise ValueError(f"{cache_path}: empty/non-finite arrays for {video}")
            self.by_video[video] = {
                "emb": emb,
                "label": labels,
                "time_h": times,
                "frame_index": frames,
            }
            rows.extend(
                {
                    "video": video,
                    "position": i,
                    "label": int(labels[i]),
                    "time_h": float(times[i]),
                    "frame_index": int(frames[i]),
                }
                for i in range(n)
            )
        raw.close()
        self.rows = pd.DataFrame(rows)
        visual_dim = next(iter(self.by_video.values()))["emb"].shape[1]
        clock_dim = 2 + 2 * len(self.clock_periods_h)
        self.in_channels = {
            "visual": visual_dim,
            "clock": clock_dim,
            "visual_clock": visual_dim + clock_dim,
        }[input_mode]

    def __len__(self) -> int:
        return len(self.rows)

    def videos(self) -> list[str]:
        return list(self.by_video)

    def label_sequences(self) -> list[np.ndarray]:
        return [v["label"] for v in self.by_video.values()]

    def inputs(self, video: str, positions: np.ndarray | slice | None = None) -> np.ndarray:
        item = self.by_video[video]
        select = slice(None) if positions is None else positions
        parts = []
        if self.input_mode in {"visual", "visual_clock"}:
            emb = item["emb"][select].astype(np.float32)
            parts.append((emb - self.visual_mean) / self.visual_std)
        if self.input_mode in {"clock", "visual_clock"}:
            time_h = item["time_h"][select] * self.clock_rate + self.clock_offset_h
            parts.append(
                clock_features(
                    time_h,
                    self.clock_mean_h,
                    self.clock_std_h,
                    self.clock_periods_h,
                )
            )
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        pos = int(row.position)
        x = self.inputs(str(row.video), np.asarray([pos]))[0]
        return {
            "image": torch.from_numpy(x),
            "label": int(row.label),
            "video": str(row.video),
            "frame_index": int(row.frame_index),
            "time_h": float(row.time_h),
        }


class _CachedSequences(Dataset):
    """Full videos or random fixed-length clips from one cached partition."""

    def __init__(
        self,
        frames: _FeaturePartition,
        mode: str,
        clip_len: int,
        clips_per_video: int,
        seed: int,
    ) -> None:
        if mode not in {"full", "clip"}:
            raise ValueError(f"sequence_mode must be full|clip, got {mode!r}")
        self.frames = frames
        self.mode = mode
        self.clip_len = int(clip_len)
        self.clips_per_video = int(clips_per_video)
        self.videos_kept = [
            v
            for v, item in frames.by_video.items()
            if mode == "full" or len(item["label"]) >= self.clip_len
        ]
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        multiplier = 1 if self.mode == "full" else self.clips_per_video
        return len(self.videos_kept) * multiplier

    def __getitem__(self, index: int) -> dict[str, Any]:
        multiplier = 1 if self.mode == "full" else self.clips_per_video
        video = self.videos_kept[index // multiplier]
        item = self.frames.by_video[video]
        if self.mode == "full":
            pos = np.arange(len(item["label"]))
        else:
            start = int(self.rng.integers(0, len(item["label"]) - self.clip_len + 1))
            pos = np.arange(start, start + self.clip_len)
        return {
            "image": torch.from_numpy(self.frames.inputs(video, pos)),
            "label": torch.from_numpy(item["label"][pos].copy()),
            "video": video,
            "frame_index": torch.from_numpy(item["frame_index"][pos].copy()),
            "time_h": torch.from_numpy(item["time_h"][pos].copy()),
        }


@DATASET_REGISTRY.register("nantes_cached_sequences_trainval")
class CachedSequenceTrainValProducer:
    """Patient-grouped cached features with no test cache or test dataset."""

    def __init__(self, d: dict, seed: int, smoke: bool = False) -> None:
        cache_dir = _resolve(d["feature_cache"])
        split_path = _resolve(d["split"])
        split = json.loads(split_path.read_text())
        train_ids = set(split["videos"]["train"])
        val_ids = set(split["videos"]["val"])
        patient_sets = {name: set(split["patients"][name]) for name in ("train", "val", "test")}
        overlaps = {
            f"{left}/{right}": sorted(patient_sets[left] & patient_sets[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        overlaps = {key: value for key, value in overlaps.items() if value}
        if overlaps:
            raise ValueError(f"patient leakage across frozen partitions: {overlaps}")
        if (cache_dir / "test.npz").exists():
            raise ValueError("v26 cache directory must not contain test.npz")
        for required in ("train.npz", "val.npz", "meta.json"):
            if not (cache_dir / required).exists():
                raise FileNotFoundError(cache_dir / required)
        meta = json.loads((cache_dir / "meta.json").read_text())
        if meta.get("split_sha256") != _sha256(split_path):
            raise ValueError("feature cache split hash does not match configured split")
        if meta.get("partitions") != ["train", "val"]:
            raise ValueError("feature cache must contain exactly train and val partitions")

        # Compute all normalisation statistics from training frames only.
        train_raw = np.load(cache_dir / "train.npz", allow_pickle=False)
        train_videos = sorted(k[: -len("__emb")] for k in train_raw.files if k.endswith("__emb"))
        if set(train_videos) != train_ids:
            raise ValueError("training cache membership differs from frozen split")
        sum_x = sum_x2 = None
        sum_t = sum_t2 = 0.0
        count = 0
        for video in train_videos:
            emb = np.asarray(train_raw[f"{video}__emb"], dtype=np.float64)
            times = np.asarray(train_raw[f"{video}__t"], dtype=np.float64)
            ex = emb.sum(0)
            ex2 = np.square(emb).sum(0)
            sum_x = ex if sum_x is None else sum_x + ex
            sum_x2 = ex2 if sum_x2 is None else sum_x2 + ex2
            sum_t += float(times.sum())
            sum_t2 += float(np.square(times).sum())
            count += len(times)
        train_raw.close()
        if sum_x is None or sum_x2 is None or count < 1:
            raise ValueError("training feature cache is empty")
        visual_mean = sum_x / count
        visual_var = np.maximum(sum_x2 / count - np.square(visual_mean), 1e-6)
        visual_std = np.sqrt(visual_var)
        clock_mean = sum_t / count
        clock_std = max((sum_t2 / count - clock_mean**2) ** 0.5, 1e-6)
        periods = tuple(float(v) for v in d.get("clock_periods_h", [6, 12, 24, 48, 96]))
        mode = d.get("input_mode", "visual")

        self._partition_args = dict(
            input_mode=mode,
            visual_mean=visual_mean,
            visual_std=visual_std,
            clock_mean_h=clock_mean,
            clock_std_h=clock_std,
            clock_periods_h=periods,
        )
        self.train_frames = _FeaturePartition(
            cache_dir / "train.npz", train_ids, **self._partition_args
        )
        self.train_full = self.train_frames
        self.val = _FeaturePartition(cache_dir / "val.npz", val_ids, **self._partition_args)
        self.in_channels = self.train_frames.in_channels
        self.default_batch = 1
        sequence_mode = d.get("sequence_mode", "full")
        clip_len = int(d.get("clip_len", 16))
        clips = d.get("clips_per_video", "auto")
        if clips == "auto":
            clips = max(
                1,
                round(
                    len(self.train_frames) / (clip_len * max(1, len(self.train_frames.videos())))
                ),
            )
        if smoke:
            # Keep the true model/data path but bound the smoke to <=20 frames.
            first = self.train_frames.videos()[0]
            self.train_frames.by_video = {
                first: {
                    key: value[: min(20, len(value))]
                    for key, value in self.train_frames.by_video[first].items()
                }
            }
            self.train_frames.rows = (
                self.train_frames.rows[self.train_frames.rows.video == first]
                .iloc[:20]
                .reset_index(drop=True)
            )
            self.val = self._smoke_partition(self.val)
            clips = 2
        self.train_set = _CachedSequences(
            self.train_frames,
            sequence_mode,
            clip_len,
            int(clips),
            seed,
        )
        self.clip_len = clip_len
        self.clips_per_video = int(clips)
        self.sequence_mode = sequence_mode
        self.input_mode = mode
        self._cfg_cache = str(cache_dir)
        self._val_ids = val_ids

    @staticmethod
    def _smoke_partition(partition: _FeaturePartition) -> _FeaturePartition:
        first = partition.videos()[0]
        partition.by_video = {
            first: {
                key: value[: min(20, len(value))]
                for key, value in partition.by_video[first].items()
            }
        }
        partition.rows = (
            partition.rows[partition.rows.video == first].iloc[:20].reset_index(drop=True)
        )
        return partition

    def validation_with_clock(self, offset_h: float = 0.0, rate: float = 1.0) -> _FeaturePartition:
        """Return a fresh validation view under a prespecified clock perturbation."""
        cache_dir = _resolve(self._cfg_cache)
        return _FeaturePartition(
            cache_dir / "val.npz",
            self._val_ids,
            clock_offset_h=offset_h,
            clock_rate=rate,
            **self._partition_args,
        )


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CachedVideoTrainProcess(KineticTrainProcess):
    """Kinetic process plus prespecified validation clock stress diagnostics."""

    def run(self) -> dict[str, Any]:
        result = super().run()
        samples = len(self.data.train_set)
        batches = samples // self.bs
        accumulation = max(1, int(self.cfg.get("training", {}).get("grad_accum", 1)))
        steps = math.ceil(batches / accumulation)
        frames = (
            len(self.data.train_frames)
            if self.data.sequence_mode == "full"
            else steps * self.bs * self.data.clip_len
        )
        result["training_exposure"] = {
            "sequence_mode": self.data.sequence_mode,
            "samples_per_epoch": samples,
            "optimizer_steps_per_epoch": steps,
            "gradient_accumulation": accumulation,
            "frames_per_epoch": frames,
            "epochs": self.epochs,
        }
        if hasattr(self.model, "set_inference_steps") and not self.smoke:
            reports = {}
            final_steps = int(self.model.inference_steps)
            for steps in sorted({1, 15, final_steps}):
                self.model.set_inference_steps(steps)
                metric = self.evaluate(self.data.val)
                reports[str(steps)] = {
                    key: value for key, value in metric.items() if key != "per_video"
                }
                (self.out_dir / f"per_video_val_ddim{steps}.json").write_text(
                    json.dumps(metric["per_video"], indent=1)
                )
            self.model.set_inference_steps(final_steps)
            result["diffusion_steps_val"] = reports
        if self.data.input_mode not in {"clock", "visual_clock"} or self.smoke:
            (self.out_dir / "results.json").write_text(json.dumps(result, indent=2))
            return result
        stresses = {
            "plus6h": (6.0, 1.0),
            "minus6h": (-6.0, 1.0),
            "rate1p10": (0.0, 1.10),
        }
        reports = {}
        for name, (offset_h, rate) in stresses.items():
            ds = self.data.validation_with_clock(offset_h, rate)
            metric = self.evaluate(ds)
            reports[name] = {k: v for k, v in metric.items() if k != "per_video"}
            (self.out_dir / f"per_video_val_clock_{name}.json").write_text(
                json.dumps(metric["per_video"], indent=1)
            )
        result["clock_stress_val"] = reports
        (self.out_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result


@BUILDER_REGISTRY.register("cached_video_validation_training")
class CachedVideoValidationBuilder(KineticTrainingBuilder):
    """Build the cached full-video process without ever constructing test data."""

    def produce_pre_processor(self) -> None:
        if self._cfg["data"].get("dataset") != "nantes_cached_sequences_trainval":
            raise ValueError(
                "cached_video_validation_training requires "
                "data.dataset=nantes_cached_sequences_trainval"
            )
        super().produce_pre_processor()

    @property
    def product(self) -> CachedVideoTrainProcess:
        if self._data is None:
            self.produce_pre_processor()
        if self._model is None:
            self.produce_main_processor()
        if self._log_trans is None:
            self.produce_post_processor()
        process = CachedVideoTrainProcess(
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
        if self.smoke:
            process.epochs = 2
        return process
