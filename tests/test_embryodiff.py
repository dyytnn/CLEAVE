"""TEMPO v27 EmbryoDiff adaptation shape, gradient and protocol tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from stseg.kinetic.config import ConfigError, load_pipeline, validate_pipeline
from stseg.kinetic.embryodiff import EmbryoDiffAdapter, EmbryoDiffObjective

ROOT = Path(__file__).resolve().parents[1]


def _model(variant: str, dropout: float = 0.0) -> EmbryoDiffAdapter:
    return EmbryoDiffAdapter(
        input_dim=12,
        num_classes=5,
        variant=variant,
        hidden=12,
        layers=3,
        intermediate_layers=(1, 2, 3),
        attention_reduction=2,
        diffusion_dim=16,
        diffusion_blocks=2,
        diffusion_heads=4,
        diffusion_train_steps=20,
        inference_steps=3,
        selection_steps=1,
        label_scale=0.1,
        dropout=dropout,
        eval_seed=27123,
    )


@pytest.mark.parametrize(
    "variant",
    ["sce", "sce_diffusion", "sce_boundary_diffusion"],
)
def test_embryodiff_variants_emit_aligned_logits(variant: str) -> None:
    torch.manual_seed(7)
    model = _model(variant)
    x = torch.randn(2, 11, 12)
    y = torch.randint(0, 5, (2, 11))
    out = model(x, y if model.requires_labels else None)
    assert out["logits"].shape == (2, 11, 5)
    assert out["semantic_logits"].shape == (2, 11, 5)
    assert ("diffusion_logits" in out) == (variant != "sce")
    assert ("boundary_logits" in out) == (variant == "sce_boundary_diffusion")


def test_full_embryodiff_objective_reaches_every_parameter() -> None:
    torch.manual_seed(9)
    model = _model("sce_boundary_diffusion")
    x = torch.randn(2, 13, 12)
    y = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4]] * 2,
        dtype=torch.long,
    )
    out = model(x, y)
    loss = EmbryoDiffObjective()(out, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_ddim_evaluation_is_deterministic_and_step_toggle_changes_output() -> None:
    torch.manual_seed(11)
    model = _model("sce_diffusion").eval()
    x = torch.randn(2, 10, 12)
    model.set_inference_steps(3)
    with torch.no_grad():
        first = model(x)["logits"]
        second = model(x)["logits"]
    assert torch.equal(first, second)
    model.set_inference_steps(1)
    with torch.no_grad():
        one_step = model(x)["logits"]
    assert not torch.allclose(first, one_step)


def test_diffusion_training_requires_labels() -> None:
    model = _model("sce_diffusion").train()
    with pytest.raises(ValueError, match="requires frame labels"):
        model(torch.randn(1, 8, 12))


def test_boundary_target_marks_first_frame_of_new_phase() -> None:
    objective = EmbryoDiffObjective(
        semantic_weight=0.0,
        smooth_weight=0.0,
        boundary_weight=1.0,
        diffusion_weight=0.0,
    )
    y = torch.tensor([[0, 0, 1, 1, 3]])
    semantic = torch.zeros(1, 5, 4)
    correct = torch.tensor([[-10.0, -10.0, 10.0, -10.0, 10.0]])
    wrong = -correct
    good = objective({"semantic_logits": semantic, "boundary_logits": correct}, y)
    bad = objective({"semantic_logits": semantic, "boundary_logits": wrong}, y)
    assert good < 0.001
    assert bad > 5.0


def test_all_v27_configs_validate_and_stay_trainval_only() -> None:
    entries = yaml.safe_load((ROOT / "configs/h7/v27/sweep.yaml").read_text())["experiments"]
    assert len(entries) == 9
    variants = set()
    for entry in entries:
        document = load_pipeline(ROOT / entry["config"])
        cfg = document["cfg"]
        assert document["builder"] == "cached_video_validation_training"
        assert cfg["data"]["split"] == "data/splits/nantes_grouped_v1.json"
        assert cfg["data"]["sequence_mode"] == "full"
        assert cfg["data"]["input_mode"] == "visual"
        variants.add(cfg["model"]["embryodiff"]["variant"])
    assert variants == {"sce", "sce_diffusion", "sce_boundary_diffusion"}

    bad = load_pipeline(ROOT / "configs/h7/v27/sce_diffusion_src0.yaml")
    bad["cfg"]["model"]["embryodiff"]["unknown_knob"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        validate_pipeline(bad)
