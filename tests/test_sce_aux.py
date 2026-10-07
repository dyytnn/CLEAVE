"""TEMPO v30 SCE + auxiliary-branch tests: hand-computed targets, gradients, unchanged decoding, configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from stseg.kinetic.config import load_pipeline
from stseg.kinetic.embryodiff import EmbryoDiffAdapter, EmbryoDiffObjective
from stseg.kinetic.sce_aux import (
    AUX_VARIANTS,
    SceAuxAdapter,
    SceAuxObjective,
    contrastive_pairs,
    peak_count_targets,
    time_to_event_targets,
    transitions,
)

ROOT = Path(__file__).resolve().parents[1]
Y = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2, 4, 4]])  # transitions at t=3, 5, 9


@pytest.fixture(autouse=True)
def _no_onednn():
    """See tests/test_milestone.py: torch 2.0.1 oneDNN CPU conv1d backward is intermittently non-finite."""
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def _model(aux: str, dropout: float = 0.0) -> SceAuxAdapter:
    return SceAuxAdapter(input_dim=12, num_classes=5, aux=aux, hidden=12, layers=3, intermediate_layers=(1, 2, 3),
                         attention_reduction=2, dropout=dropout)


def test_transition_and_boundary_target() -> None:
    assert transitions(Y).int().tolist() == [[0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]]


def test_peak_count_targets_count_transitions_in_trailing_window() -> None:
    # window 3 covers (t-3, t]: t=3 ->1, t=5 -> {3,5}=2, t=6 -> {5}=1 (3 fell out), t=9 -> 1
    assert peak_count_targets(Y, 3).tolist() == [[0, 0, 0, 1, 1, 2, 1, 1, 0, 1, 1]]
    assert peak_count_targets(Y, 100).max() == 2  # clipped at ">=2"


def test_time_to_event_targets_and_censoring() -> None:
    target, observed = time_to_event_targets(Y, cap=64)
    frames_to_next = [3, 2, 1, 2, 1, 4, 3, 2, 1]  # t=0..8; t=9,10 censored (no later transition)
    assert observed.tolist() == [[True] * 9 + [False, False]]
    assert torch.allclose(target[0, :9], torch.log1p(torch.tensor(frames_to_next, dtype=torch.float32)))
    capped, _ = time_to_event_targets(torch.tensor([[0] * 50 + [1]]), cap=4)
    assert capped[0, 0] == torch.log1p(torch.tensor(4.0))


def test_contrastive_pairs_exclude_uncertainty_band() -> None:
    same, diff = contrastive_pairs(Y, k=2)  # pair index i refers to frames (i, i+2), i = 0..8
    # diff: y[i+2] != y[i] -> i=1,2,3,4,7,8
    assert diff.int().tolist() == [[0, 1, 1, 1, 1, 0, 0, 1, 1]]
    # same candidates i=0,5,6; the uncertainty band (transition frame and the frame before: 2,3,4,5,8,9) excludes
    # all of them: (0,2) touches 2, (5,7) touches 5, (6,8) touches 8. Nothing interior survives here.
    assert same.int().tolist() == [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
    long = torch.tensor([[0] * 8 + [1] * 8])  # transition at 8, band = {7, 8}
    same_l, diff_l = contrastive_pairs(long, k=2)
    assert same_l[0, :5].all() and same_l[0, 9:].all() and not same_l[0, 5:9].any()
    assert diff_l.int().tolist() == [[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]]


@pytest.mark.parametrize("aux", AUX_VARIANTS)
def test_every_variant_trains_all_parameters_and_keeps_sce_logits(aux: str) -> None:
    torch.manual_seed(2)
    model = _model(aux)
    x = torch.randn(1, 11, 12)
    out = model(x)
    assert out["logits"].shape == (1, 11, 5) and out["hidden"].shape == (1, 11, 12)
    assert ("aux_logits" in out) == (aux != "change_contrastive")
    loss = SceAuxObjective(aux=aux, pair_offset=2, window=3)(out, Y)
    loss.backward()
    assert torch.isfinite(loss)
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing
    # decoding uses exactly the SCE framewise head: aux weight 0 recovers the v27 SCE objective value
    plain = SceAuxObjective(aux=aux, aux_weight=0.0)(out, Y)
    sce_ref = EmbryoDiffObjective()({"semantic_logits": out["logits"]}, Y)
    assert torch.allclose(plain, sce_ref)


def test_sce_aux_encoder_is_the_v27_sce_encoder() -> None:
    torch.manual_seed(4)
    ours = _model("boundary")
    ref = EmbryoDiffAdapter(input_dim=12, num_classes=5, variant="sce", hidden=12, layers=3, intermediate_layers=(1, 2, 3),
                            attention_reduction=2, diffusion_dim=16, diffusion_blocks=1, diffusion_heads=4,
                            diffusion_train_steps=2, inference_steps=1, selection_steps=1, label_scale=0.1,
                            dropout=0.0, eval_seed=1)
    ours.encoder.load_state_dict(ref.semantic_encoder.state_dict())
    ours.frame_classifier.load_state_dict(ref.semantic_classifier.state_dict())
    x = torch.randn(1, 9, 12)
    with torch.no_grad():
        assert torch.allclose(ours.eval()(x)["logits"], ref.eval()(x)["logits"])
    assert not any(hasattr(ours, a) for a in ("backbone", "head", "cls")) and ours.requires_labels is False


def test_aux_terms_are_finite_on_degenerate_sequences() -> None:
    flat = torch.zeros(1, 6, dtype=torch.long)  # no transitions: no diff pairs, all time-to-event censored
    for aux in AUX_VARIANTS:
        out = _model(aux)(torch.randn(1, 6, 12))
        assert torch.isfinite(SceAuxObjective(aux=aux)(out, flat))


def test_all_v30_configs_validate_and_match_the_v27_sce_recipe() -> None:
    sce = load_pipeline(ROOT / "configs/h7/v27/sce_src0.yaml")["cfg"]
    paths = sorted((ROOT / "configs/h7/v30").glob("*_src*.yaml"))
    assert len(paths) == 12
    seen = set()
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
        a = cfg["model"]["sce_aux"]
        assert a["aux"] == cfg["loss"]["aux"] and a["aux"] in AUX_VARIANTS
        seen.add((a["aux"], seed))
        for key in ("hidden", "layers", "intermediate_layers", "attention_reduction", "dropout"):
            assert a[key] == sce["model"]["embryodiff"][key], key
        assert (cfg["loss"]["semantic_weight"], cfg["loss"]["smooth_weight"], cfg["loss"]["truncation"]) == (
            sce["loss"]["semantic_weight"], sce["loss"]["smooth_weight"], sce["loss"]["truncation"])
    assert seen == {(a, s) for a in AUX_VARIANTS for s in range(3)}
    sweeps = [yaml.safe_load((ROOT / f"configs/h7/v30/sweep_gpu{i}.yaml").read_text())["experiments"] for i in (0, 1)]
    assert sorted(e["config"] for s in sweeps for e in s) == sorted(str(p.relative_to(ROOT)) for p in paths)
