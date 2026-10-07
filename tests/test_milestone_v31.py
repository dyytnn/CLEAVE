"""TEMPO v31 tests: hazard parameterisation, gated fusion, combined smoothing, v29 defaults unchanged, configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from stseg.kinetic.config import ConfigError, load_pipeline, validate_pipeline
from stseg.kinetic.milestone import (
    MilestoneObjective,
    MilestoneQueryAdapter,
    hazard_onset_log_probs,
    milestone_frame_log_probs,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_onednn():
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def _model(**kw) -> MilestoneQueryAdapter:
    args = dict(input_dim=12, num_classes=5, variant="onset_linear", hidden=12, layers=3, intermediate_layers=(1, 2, 3),
                attention_reduction=2, dropout=0.0, query_dim=16, query_layers=1, query_heads=4, query_dropout=0.0,
                local_kernel=5, refine_radius=2, combine_weight=1.0)
    args.update(kw)
    return MilestoneQueryAdapter(**args)


def test_hazard_rows_are_normalised_and_cdfs_ordered_without_projection() -> None:
    torch.manual_seed(0)
    scores = torch.randn(2, 4, 9) * 3
    logp = hazard_onset_log_probs(scores)
    assert logp.shape == (2, 4, 10)
    assert torch.allclose(logp.logsumexp(-1), torch.zeros(2, 4), atol=1e-5)
    cdf = logp[..., :9].exp().cumsum(-1)
    assert (cdf[:, 1:] <= cdf[:, :-1] + 1e-5).all()  # later milestones never precede earlier ones
    never = logp[..., 9].exp()
    assert (never[:, 1:] >= never[:, :-1] - 1e-5).all()  # never-reached mass propagates forward


def test_hazard_limits_skip_and_censoring() -> None:
    hot = torch.full((1, 3, 6), 30.0)  # every milestone fires immediately -> all onsets at frame 0 (skipped phases)
    p = hazard_onset_log_probs(hot).exp()
    assert torch.allclose(p[0, :, 0], torch.ones(3), atol=1e-4)
    cold = torch.full((1, 3, 6), -30.0)  # nothing ever fires -> all mass on the never slot
    p = hazard_onset_log_probs(cold).exp()
    assert torch.allclose(p[0, :, 6], torch.ones(3), atol=1e-4)
    # first milestone fires only at frame 2, second only at frame 4 -> deterministic ordered onsets
    s = torch.full((1, 2, 6), -30.0)
    s[0, 0, 2] = 30.0
    s[0, 1, 4] = 30.0
    p = hazard_onset_log_probs(s).exp()
    assert p[0, 0].argmax() == 2 and p[0, 1].argmax() == 4
    frame_logp = milestone_frame_log_probs(hazard_onset_log_probs(s))
    assert frame_logp.argmax(-1).tolist() == [[0, 0, 1, 1, 2, 2]]


def test_hazard_model_trains_and_has_no_never_head() -> None:
    torch.manual_seed(1)
    model = _model(parameterisation="hazard")
    assert not hasattr(model, "never_head")
    x = torch.randn(1, 13, 12)
    y = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4]])
    out = model(x)
    assert torch.allclose(out["milestone_coarse_logits"].logsumexp(-1), torch.zeros(1, 4), atol=1e-4)
    MilestoneObjective(num_classes=5)(out, y).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing


def test_gated_fusion_starts_near_the_fixed_product_and_is_learnable() -> None:
    torch.manual_seed(2)
    gated = _model(fusion="gated")
    plain = _model()
    plain.load_state_dict({k: v for k, v in gated.state_dict().items() if not k.startswith("gate.")})
    x = torch.randn(1, 9, 12)
    with torch.no_grad():
        og, op = gated.eval()(x), plain.eval()(x)
    assert "fusion_gate" in og and og["fusion_gate"].shape == (1, 9)
    assert torch.allclose(og["fusion_gate"], torch.full((1, 9), torch.sigmoid(torch.tensor(3.0)).item()), atol=1e-5)
    frame = torch.log_softmax(og["frame_logits"], -1)
    expect = torch.log_softmax(frame + og["fusion_gate"][..., None] * og["milestone_frame_log_probs"], -1)
    assert torch.allclose(og["logits"], expect, atol=1e-5)
    assert not torch.allclose(og["logits"], op["logits"])  # gate < 1 changes the fusion
    y = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 4, 4]])
    MilestoneObjective(num_classes=5)(gated.train()(x), y).backward()
    assert gated.gate.weight.grad is not None and torch.isfinite(gated.gate.weight.grad).all()


def test_combined_smoothing_is_off_by_default_and_penalises_flicker() -> None:
    torch.manual_seed(3)
    out = _model()(torch.randn(1, 11, 12))
    y = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2, 4, 4]])
    base = MilestoneObjective(num_classes=5)(out, y)
    same = MilestoneObjective(num_classes=5, combined_smooth_weight=0.0)(out, y)
    assert torch.equal(base, same)
    smooth = torch.log_softmax(torch.zeros(1, 11, 5), -1)
    flicker = torch.log_softmax(torch.stack([torch.full((5,), -8.0).scatter(0, torch.tensor(i % 2), 8.0) for i in range(11)])[None], -1)
    crit = MilestoneObjective(num_classes=5, semantic_weight=0, smooth_weight=0, milestone_weight=0, combined_weight=0,
                              combined_smooth_weight=1.0)
    fake = lambda lg: {"frame_logits": lg, "milestone_coarse_logits": torch.zeros(1, 4, 12), "logits": lg}  # noqa: E731
    assert crit(fake(smooth), y) == 0 and crit(fake(flicker), y) > 1.0


def test_v29_defaults_are_unchanged() -> None:
    model = _model()
    assert model.fusion == "product" and model.parameterisation == "onset" and hasattr(model, "never_head")
    assert not hasattr(model, "gate")
    out = model.eval()(torch.randn(1, 7, 12))
    assert "fusion_gate" not in out
    with pytest.raises(ValueError):
        _model(variant="query_global", parameterisation="hazard")
    with pytest.raises(ValueError):
        _model(fusion="mystery")


def test_all_v31_configs_validate_and_change_exactly_one_thing() -> None:
    base = load_pipeline(ROOT / "configs/h7/v29/onset_linear_src0.yaml")["cfg"]
    paths = sorted((ROOT / "configs/h7/v31").glob("*_src*.yaml"))
    assert len(paths) == 9
    expected = {"combined_smooth": ("loss", "combined_smooth_weight", 0.3),
                "gated_fusion": ("model.milestone", "fusion", "gated"),
                "duration_hazard": ("model.milestone", "parameterisation", "hazard")}
    seen = set()
    for path in paths:
        doc = load_pipeline(path)
        cfg = doc["cfg"]
        assert doc["builder"] == "cached_video_validation_training" and cfg["data"]["sequence_mode"] == "full"
        arm = next(a for a in expected if path.name.startswith(a))
        seed = cfg["experiment"]["seed"]
        seen.add((arm, seed))
        assert cfg["data"]["feature_cache"] == f"runs/h7/v26/features_seed{seed}"
        section, key, value = expected[arm]
        node = cfg["model"]["milestone"] if section == "model.milestone" else cfg["loss"]
        assert node[key] == value
        # everything else equals the v29 onset_linear resolved config (modulo identity fields)
        strip = lambda c: {k: v for k, v in c.items() if k not in ("experiment", "data")}  # noqa: E731
        ours, ref = strip(cfg), strip(base)
        ref_node = ref["model"]["milestone"] if section == "model.milestone" else ref["loss"]
        ref_node = dict(ref_node)
        ref_node[key] = value
        if section == "model.milestone":
            ref["model"] = {**ref["model"], "milestone": ref_node}
        else:
            ref["loss"] = ref_node
        assert ours == ref, path.name
    assert seen == {(a, s) for a in expected for s in range(3)}
    allruns = yaml.safe_load((ROOT / "configs/h7/v31/sweep_all.yaml").read_text())["experiments"]
    assert sorted(e["config"] for e in allruns) == sorted(str(p.relative_to(ROOT)) for p in paths)


def test_config_rejects_hazard_on_query_variants() -> None:
    doc = load_pipeline(ROOT / "configs/h7/v31/duration_hazard_src0.yaml")
    doc["cfg"]["model"]["milestone"]["variant"] = "query_global"
    with pytest.raises(ConfigError):
        validate_pipeline(doc, "test")
