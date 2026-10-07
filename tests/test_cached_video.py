"""TEMPO v26 cached full-video model, loss, controls and leakage tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import stseg.kinetic  # noqa: F401 - registry side effects
from stseg.kinetic.cached_video import (
    CachedSequenceTrainValProducer,
    clock_features,
)
from stseg.kinetic.config import ConfigError, load_pipeline, validate_pipeline
from stseg.kinetic.losses import MultiStageTemporalLoss
from stseg.kinetic.models import build_model

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cache(tmp_path: Path, leaked: bool = False) -> tuple[Path, Path]:
    split = {
        "patients": {
            "train": ["A"],
            "val": ["A" if leaked else "B"],
            "test": ["C"],
        },
        "videos": {
            "train": ["A-1", "A-2"],
            "val": ["B-1"],
            "test": ["C-1"],
        },
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    cache = tmp_path / "cache"
    cache.mkdir()

    def arrays(videos: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for number, video in enumerate(videos):
            length = 7 + number
            out[f"{video}__emb"] = np.arange(length * 4, dtype=np.float16).reshape(length, 4)
            out[f"{video}__y"] = np.repeat(number + 2, length).astype(np.int16)
            out[f"{video}__t"] = np.linspace(20, 60, length, dtype=np.float32)
            out[f"{video}__frame"] = np.arange(length, dtype=np.int32)
        return out

    np.savez_compressed(cache / "train.npz", **arrays(split["videos"]["train"]))
    np.savez_compressed(cache / "val.npz", **arrays(split["videos"]["val"]))
    (cache / "meta.json").write_text(
        json.dumps({"split_sha256": _sha(split_path), "partitions": ["train", "val"]})
    )
    return cache, split_path


def test_full_video_mstcn_shape_gradient_and_determinism() -> None:
    cfg = {
        "type": "full_video_mstcn",
        "backbone": {"name": "cached_features"},
        "head": {
            "name": "mstcn",
            "hidden": 8,
            "layers": 3,
            "stages": 3,
            "dropout": 0.0,
        },
        "num_classes": 16,
    }
    outputs = []
    for _ in range(2):
        torch.manual_seed(11)
        model = build_model(cfg, in_channels=6)
        x = torch.randn(2, 19, 6)
        out = model(x)
        assert out["logits"].shape == (2, 19, 16)
        assert out["stage_logits"].shape == (3, 2, 19, 16)
        out["stage_logits"].square().mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        outputs.append(out["logits"].detach())
    assert torch.equal(*outputs)


def test_multistage_loss_supervises_all_stages_and_penalises_flicker() -> None:
    y = torch.zeros(1, 6, dtype=torch.long)
    logits = torch.zeros(2, 1, 6, 3, requires_grad=True)
    with torch.no_grad():
        logits[:, :, ::2, 0] = 3.0
        logits[:, :, 1::2, 1] = 3.0
    plain = MultiStageTemporalLoss(lam=0.0)({"stage_logits": logits, "logits": logits[-1]}, y)
    smooth = MultiStageTemporalLoss(lam=0.15)({"stage_logits": logits, "logits": logits[-1]}, y)
    assert smooth > plain
    smooth.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert (logits.grad[0].abs().sum() > 0) and (logits.grad[1].abs().sum() > 0)


def test_clock_encoding_uses_absolute_time_not_video_duration() -> None:
    first = clock_features(np.array([24.0, 30.0]), 40.0, 10.0, (6.0, 24.0))
    extended = clock_features(np.array([24.0, 30.0, 100.0]), 40.0, 10.0, (6.0, 24.0))
    assert first.shape == (2, 6)
    assert np.allclose(first, extended[:2])
    shifted = clock_features(np.array([30.0, 36.0]), 40.0, 10.0, (6.0, 24.0))
    assert not np.allclose(first, shifted)


@pytest.mark.parametrize(
    "input_mode,expected_dim", [("visual", 4), ("clock", 12), ("visual_clock", 16)]
)
def test_cached_sequence_producer_modes_and_test_isolation(
    tmp_path: Path, input_mode: str, expected_dim: int
) -> None:
    cache, split = _write_cache(tmp_path)
    producer = CachedSequenceTrainValProducer(
        {
            "feature_cache": str(cache),
            "split": str(split),
            "input_mode": input_mode,
            "sequence_mode": "full",
            "clip_len": 4,
            "clips_per_video": "auto",
            "clock_periods_h": [6, 12, 24, 48, 96],
        },
        seed=3,
    )
    assert not hasattr(producer, "test")
    assert producer.in_channels == expected_dim
    assert set(producer.train_frames.videos()) == {"A-1", "A-2"}
    sample = producer.train_set[0]
    assert sample["image"].shape == (7, expected_dim)
    assert sample["label"].shape == (7,)


def test_cached_sequence_rejects_patient_leakage_and_test_cache(tmp_path: Path) -> None:
    cache, split = _write_cache(tmp_path, leaked=True)
    cfg = {
        "feature_cache": str(cache),
        "split": str(split),
        "input_mode": "visual",
        "sequence_mode": "full",
    }
    with pytest.raises(ValueError, match="patient leakage"):
        CachedSequenceTrainValProducer(cfg, seed=0)

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    cache, split = _write_cache(clean_root)
    np.savez_compressed(cache / "test.npz", forbidden=np.zeros(1))
    with pytest.raises(ValueError, match="test.npz"):
        CachedSequenceTrainValProducer(
            {**cfg, "feature_cache": str(cache), "split": str(split)}, seed=0
        )


def test_all_v26_configs_validate_and_full_batch_is_strict() -> None:
    sweep = yaml.safe_load((ROOT / "configs/h7/v26/sweep.yaml").read_text())["experiments"]
    assert len(sweep) == 12
    for entry in sweep:
        document = load_pipeline(ROOT / entry["config"])
        assert document["builder"] == "cached_video_validation_training"
        assert document["cfg"]["data"]["split"] == "data/splits/nantes_grouped_v1.json"
    bad = load_pipeline(ROOT / "configs/h7/v26/full_visual_src0.yaml")
    bad["cfg"]["training"]["batch_size"] = 2
    with pytest.raises(ConfigError, match="batch_size=1"):
        validate_pipeline(bad)
