"""TEMPO v29 milestone-query tests: targets, censoring, valid frame distributions, gradients, configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from stseg.kinetic.config import load_pipeline
from stseg.kinetic.milestone import (
    VARIANTS,
    MilestoneObjective,
    MilestoneQueryAdapter,
    milestone_frame_log_probs,
    milestone_soft_targets,
    milestone_targets,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_onednn():
    """torch 2.0.1's oneDNN (mkldnn) CPU conv1d backward intermittently returns non-finite input gradients here
    (13-24 of 1500 identical-input trials; 0 of 1500 with mkldnn disabled; not reproducible on replay of the same
    input), which made gradient tests flaky. Training runs on CUDA/cuDNN, which does not use oneDNN."""
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def _model(variant: str, k: int = 5, dropout: float = 0.0) -> MilestoneQueryAdapter:
    return MilestoneQueryAdapter(
        input_dim=12, num_classes=k, variant=variant, hidden=12, layers=3, intermediate_layers=(1, 2, 3),
        attention_reduction=2, dropout=dropout, query_dim=16, query_layers=2, query_heads=4, query_dropout=0.0,
        local_kernel=5, refine_radius=2, combine_weight=1.0,
    )


def test_targets_handle_left_censoring_skips_and_right_censoring() -> None:
    # K=6 -> milestones for phases 1..5. Starts in phase 1 (onset of 1 at/before frame 0), skips phase 3,
    # never reaches phase 5 within the recording.
    y = torch.tensor([[1, 1, 2, 2, 4, 4, 4]])
    t = milestone_targets(y, 6)
    assert t.tolist() == [[0, 2, 4, 4, 7]]  # skipped phase 3 onset coincides with phase 4; phase 5 censored (=T)


def test_soft_targets_are_distributions_and_censored_slot_is_one_hot() -> None:
    t = torch.tensor([[0, 3, 6]])
    soft = milestone_soft_targets(t, 6, sigma=1.0)
    assert soft.shape == (1, 3, 7)
    assert torch.allclose(soft.sum(-1), torch.ones(1, 3))
    assert soft[0, 2, 6] == 1.0 and soft[0, 2, :6].sum() == 0
    assert soft[0, 1].argmax() == 3
    hard = milestone_soft_targets(t, 6, sigma=0.0)
    assert hard[0, 1, 3] == 1.0 and hard[0, 1].sum() == 1.0


def test_frame_probs_are_valid_and_recover_the_label_path_from_sharp_onsets() -> None:
    y = torch.tensor([[0, 0, 1, 1, 3, 3, 3]])  # skips phase 2
    t = milestone_targets(y, 4)
    logits = torch.full((1, 3, 8), -30.0)
    logits.scatter_(-1, t[..., None], 30.0)
    logp = milestone_frame_log_probs(logits)
    assert logp.shape == (1, 7, 4)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(1, 7), atol=1e-5)
    assert logp.argmax(-1).tolist() == y.tolist()


def test_frame_probs_stay_valid_when_raw_onsets_are_out_of_order() -> None:
    torch.manual_seed(0)
    logp = milestone_frame_log_probs(torch.randn(2, 6, 9) * 5)
    probs = logp.exp()
    assert torch.isfinite(logp).all() and (probs >= 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(2, 8), atol=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_emit_aligned_outputs_and_train_every_parameter(variant: str) -> None:
    torch.manual_seed(3)
    model = _model(variant)
    x = torch.randn(1, 13, 12)
    y = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4]])
    out = model(x)
    assert out["logits"].shape == (1, 13, 5) and out["frame_logits"].shape == (1, 13, 5)
    assert out["milestone_coarse_logits"].shape == (1, 4, 14)
    assert ("milestone_refined_logits" in out) == (variant == "query_refine")
    assert torch.allclose(out["logits"].exp().sum(-1), torch.ones(1, 13), atol=1e-4)
    loss = MilestoneObjective(num_classes=5)(out, y)
    loss.backward()
    assert torch.isfinite(loss)
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    if variant == "query_refine":  # refine_k gets no gradient while refine_q is still at its zero init
        missing = [n for n in missing if not n.startswith("refine_k.")]
    assert not missing, missing


def test_refinement_starts_as_identity_and_is_local() -> None:
    torch.manual_seed(5)
    model = _model("query_refine").eval()
    x = torch.randn(1, 20, 12)
    with torch.no_grad():
        out = model(x)
        assert torch.allclose(out["milestone_refined_logits"], out["milestone_coarse_logits"])
        torch.nn.init.normal_(model.refine_q.weight)
        out = model(x)
    diff = (out["milestone_refined_logits"] - out["milestone_coarse_logits"])[..., :20].abs() > 0
    centre = out["milestone_coarse_logits"][..., :20].argmax(-1)
    pos = torch.arange(20)
    outside = (pos[None, None] - centre[..., None]).abs() > model.refine_radius
    assert not (diff & outside).any()
    assert out["milestone_refined_logits"][..., 20].equal(out["milestone_coarse_logits"][..., 20])


def test_model_is_non_separable_label_free_and_deterministic_in_eval() -> None:
    model = _model("query_global").eval()
    assert not any(hasattr(model, a) for a in ("backbone", "head", "cls"))  # evaluated through forward()
    assert model.requires_labels is False and model.is_sequence
    x = torch.randn(1, 9, 12)
    with torch.no_grad():
        assert torch.equal(model(x)["logits"], model(x)["logits"])


def test_all_v29_configs_validate_stay_trainval_and_match_the_v27_sce_recipe() -> None:
    sce = load_pipeline(ROOT / "configs/h7/v27/sce_src0.yaml")["cfg"]
    paths = sorted((ROOT / "configs/h7/v29").glob("*_src*.yaml"))
    assert len(paths) == 9
    for path in paths:
        doc = load_pipeline(path)
        cfg = doc["cfg"]
        assert doc["builder"] == "cached_video_validation_training"
        assert cfg["data"]["dataset"] == "nantes_cached_sequences_trainval"
        assert cfg["data"]["input_mode"] == "visual" and cfg["data"]["sequence_mode"] == "full"
        seed = cfg["experiment"]["seed"]
        assert cfg["data"]["feature_cache"] == f"runs/h7/v26/features_seed{seed}"
        for section in ("optimizer", "scheduler", "training"):
            assert cfg[section] == sce[section], section
        enc = cfg["model"]["milestone"]
        ref = sce["model"]["embryodiff"]
        for key in ("hidden", "layers", "intermediate_layers", "attention_reduction", "dropout"):
            assert enc[key] == ref[key], key
    sweeps = [yaml.safe_load((ROOT / f"configs/h7/v29/sweep_gpu{i}.yaml").read_text())["experiments"] for i in (0, 1)]
    assert sorted(e["config"] for s in sweeps for e in s) == sorted(str(p.relative_to(ROOT)) for p in paths)


def test_refine_gradients_stay_finite_under_repeated_trials() -> None:
    torch.manual_seed(11)
    model = _model("query_refine")
    torch.nn.init.normal_(model.refine_q.weight, std=0.1)  # exercise the active refinement path
    y = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4]])
    for _ in range(60):
        model.zero_grad()
        MilestoneObjective(num_classes=5)(model(torch.randn(1, 13, 12)), y).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
