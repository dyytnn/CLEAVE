"""Protocol and component tests for the v28 EMFiT reproduction."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import stseg.kinetic as K
from stseg.data.nantes_kinetic import EMFIT_PLANES, NantesEMFiTFrames
from stseg.kinetic.config import ConfigError, load_pipeline, validate_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_emfit_i3d_shape_gradient_and_determinism() -> None:
    torch.manual_seed(7)
    model = K.BACKBONE_REGISTRY.get(
        "emfit_i3d", in_channels=3, checkpoint=None
    ).eval()
    x = torch.randn(1, 3, 7, 64, 64, requires_grad=True)
    first = model(x)
    second = model(x)
    assert first.shape == (1, 16, 768)
    assert torch.equal(first, second)
    first.square().mean().backward()
    assert model.conv1.weight.grad is not None
    assert torch.isfinite(model.conv1.weight.grad).all()


def test_emfit_loss_matches_hand_computation() -> None:
    criterion = K.LOSS_REGISTRY.build(
        {"name": "emfit_multitask", "frame_weight": 5.0},
        class_counts=torch.ones(16),
        num_classes=16,
    )
    logits = torch.tensor([[2.0] + [0.0] * 15], requires_grad=True)
    prediction = torch.tensor([0.25], requires_grad=True)
    target = torch.tensor([0.5])
    out = {
        "logits": logits,
        "frame_regression": prediction,
        "frame_target": target,
    }
    observed = criterion(out, torch.tensor([0]))
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    expected = expected + 5.0 * torch.nn.functional.mse_loss(prediction, target)
    assert torch.allclose(observed, expected)
    observed.backward()
    assert logits.grad is not None and prediction.grad is not None


def test_emfit_config_is_official_recipe_and_validation_only() -> None:
    doc = load_pipeline(ROOT / "configs/h7/v28/emfit_src0.yaml")
    cfg = doc["cfg"]
    assert doc["builder"] == "kinetic_validation_training"
    assert cfg["data"]["dataset"] == "nantes_emfit_frames_trainval"
    assert cfg["data"]["planes"] == EMFIT_PLANES
    assert cfg["data"]["frame_index_divisor"] == 415.0
    assert cfg["model"]["type"] == "emfit"
    assert cfg["model"]["backbone"]["name"] == "emfit_i3d"
    assert cfg["loss"] == {"name": "emfit_multitask", "frame_weight": 5.0}
    assert cfg["optimizer"]["name"] == "adam"
    assert cfg["training"]["epochs"] == 30
    assert cfg["training"]["batch_size"] * cfg["training"]["grad_accum"] == 64

    bad = {"builder": doc["builder"], "cfg": {**cfg, "data": {**cfg["data"], "frame_index_divisor": 656.0}}}
    with pytest.raises(ConfigError, match="frame_index_divisor"):
        validate_pipeline(bad)


def test_emfit_dataset_shape_clock_and_patient_partition() -> None:
    manifest = ROOT / "data/derived/nantes_manifest_F0.csv"
    split = ROOT / "data/splits/nantes_grouped_v1.json"
    if not manifest.exists() or not split.exists():
        pytest.skip("Nantes data not available")
    train = NantesEMFiTFrames(
        manifest,
        split,
        "train",
        mode="eval",
        planes=EMFIT_PLANES,
        max_frames_per_video=1,
    )
    val = NantesEMFiTFrames(
        manifest,
        split,
        "val",
        mode="eval",
        planes=EMFIT_PLANES,
        max_frames_per_video=1,
    )
    item = train[0]
    assert item["image"].shape == (3, 7, 224, 224)
    assert item["frame_input"].shape == (1,)
    assert item["frame_input"].item() == pytest.approx(item["frame_index"] / 415.0)
    assert set(train.videos()).isdisjoint(val.videos())


def test_emfit_trainval_producer_never_constructs_test() -> None:
    manifest = ROOT / "data/derived/nantes_manifest_F0.csv"
    if not manifest.exists():
        pytest.skip("Nantes data not available")
    cfg = load_pipeline(ROOT / "configs/h7/v28/emfit_src0.yaml")["cfg"]
    producer = K.DATASET_REGISTRY.get(
        "nantes_emfit_frames_trainval", cfg["data"], seed=0, smoke=True
    )
    assert not hasattr(producer, "test")
    assert len(producer.train_set) <= 8
    assert len(producer.val) <= 8
