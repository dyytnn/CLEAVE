"""TEMPO v33 end-to-end whole-video tests: shapes, chunk equivalence, frozen BN, freezing, warm starts, configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from stseg.kinetic.config import ConfigError, load_pipeline, validate_pipeline
from stseg.kinetic.e2e_video import EndToEndSce
from stseg.kinetic.registry import BACKBONE_REGISTRY

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_onednn():
    with torch.backends.mkldnn.flags(enabled=False):
        yield


def _model(chunk: int = 2, freeze: bool = False, mean=None, std=None) -> EndToEndSce:
    torch.manual_seed(0)
    bb = BACKBONE_REGISTRY.build({"name": "crossfocal", "cnn": "resnet18", "pretrained": False, "nhead": 4}, in_channels=7)
    return EndToEndSce(bb, num_classes=5, hidden=12, layers=2, intermediate_layers=(1, 2), attention_reduction=2,
                       dropout=0.0, chunk=chunk, freeze_backbone=freeze, feat_mean=mean, feat_std=std)


def test_forward_shapes_and_sce_output_identity() -> None:
    m = _model().eval()
    x = torch.randn(1, 5, 7, 64, 64)
    out = m(x)
    assert out["logits"].shape == (1, 5, 5) and torch.equal(out["logits"], out["semantic_logits"])
    assert m(x[0])["logits"].shape == (1, 5, 5)  # 4-D single-video input is promoted
    assert not any(hasattr(m, a) for a in ("head", "cls")) and hasattr(m, "backbone")  # non-separable, lr-mult-able


def test_chunking_does_not_change_the_result() -> None:
    a, b = _model(chunk=2).eval(), _model(chunk=100).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 5, 7, 64, 64)
    with torch.no_grad():
        assert torch.allclose(a(x)["logits"], b(x)["logits"], atol=1e-5)


def test_trainable_backbone_gets_gradients_through_checkpointed_chunks_and_bn_stays_frozen() -> None:
    m = _model(chunk=2).train()
    assert all(not mod.training for mod in m.backbone.modules() if isinstance(mod, torch.nn.BatchNorm2d))
    x = torch.randn(1, 5, 7, 64, 64)
    y = torch.tensor([[0, 0, 1, 1, 2]])
    loss = torch.nn.functional.cross_entropy(m(x)["logits"].reshape(-1, 5), y.reshape(-1))
    loss.backward()
    bb_grads = [p.grad for p in m.backbone.parameters() if p.requires_grad]
    assert bb_grads and any(g is not None and g.abs().sum() > 0 for g in bb_grads)
    assert all(p.grad is not None for p in m.encoder.parameters())


def test_frozen_backbone_has_no_gradients_and_no_grad_requirement() -> None:
    m = _model(chunk=2, freeze=True).train()
    assert all(not p.requires_grad for p in m.backbone.parameters())
    x = torch.randn(1, 4, 7, 64, 64)
    y = torch.tensor([[0, 1, 1, 2]])
    torch.nn.functional.cross_entropy(m(x)["logits"].reshape(-1, 5), y.reshape(-1)).backward()
    assert all(p.grad is None for p in m.backbone.parameters())
    assert all(p.grad is not None for p in m.frame_classifier.parameters())


def test_feature_standardisation_buffers_are_applied_and_saved() -> None:
    d = 512
    plain = _model().eval()
    scaled = _model(mean=torch.full((d,), 0.5).numpy(), std=torch.full((d,), 2.0).numpy()).eval()
    scaled.load_state_dict({**plain.state_dict(), "feat_mean": torch.full((d,), 0.5), "feat_std": torch.full((d,), 2.0)})
    x = torch.randn(1, 3, 7, 64, 64)
    with torch.no_grad():
        assert not torch.allclose(plain(x)["logits"], scaled(x)["logits"])
    assert "feat_mean" in scaled.state_dict() and "feat_std" in scaled.state_dict()
    with pytest.raises(ValueError):
        _model(mean=torch.zeros(3).numpy(), std=torch.ones(3).numpy())


def test_warm_start_mappings_are_strict(tmp_path: Path) -> None:
    src = _model()
    seq_like = {f"backbone.{k}": v for k, v in src.backbone.state_dict().items()}
    seq_like.update({"head.x": torch.zeros(1), "cls.1.weight": torch.zeros(1)})
    torch.save({"model": seq_like, "epochs_averaged": 5}, tmp_path / "swa.pt")
    sce_like = {f"semantic_encoder.{k}": v for k, v in src.encoder.state_dict().items()}
    sce_like.update({f"semantic_classifier.{k}": v for k, v in src.frame_classifier.state_dict().items()})
    torch.save({"model": sce_like, "epoch": 3, "val_p_t": 0.7}, tmp_path / "best.pt")
    dst = _model()
    dst.load_backbone(tmp_path / "swa.pt")
    dst.load_head(tmp_path / "best.pt")
    x = torch.randn(1, 3, 7, 64, 64)
    with torch.no_grad():
        assert torch.allclose(src.eval()(x)["logits"], dst.eval()(x)["logits"], atol=1e-6)
    with pytest.raises(ValueError):
        dst.load_head(tmp_path / "swa.pt")


def test_all_v33_configs_validate_and_differ_only_in_freeze_backbone() -> None:
    v25 = load_pipeline(ROOT / "configs/h7/v25_grouped_recipe_baseline.yaml")["cfg"]
    paths = sorted((ROOT / "configs/h7/v33").glob("*_src*.yaml"))
    assert len(paths) == 6
    by_seed: dict[int, dict[str, dict]] = {}
    for path in paths:
        doc = load_pipeline(path)
        cfg = doc["cfg"]
        assert doc["builder"] == "kinetic_validation_training" and cfg["data"]["dataset"] == "nantes_video_trainval"
        for key in ("manifest", "split", "planes", "resize", "crop", "cache_dir", "rotation"):
            assert cfg["data"][key] == v25["data"][key], key
        assert cfg["model"]["type"] == "e2e_sce" and cfg["training"]["batch_size"] == 1
        seed = cfg["experiment"]["seed"]
        e = cfg["model"]["e2e"]
        assert e["init_backbone_from"].endswith(f"seed{seed}/swa.pt") and e["init_head_from"].endswith(f"src{seed}_seed{seed}/best.pt")
        assert e["feature_stats_cache"].endswith(f"features_seed{seed}")
        arm = "frozen_control" if e["freeze_backbone"] else "e2e"
        by_seed.setdefault(seed, {})[arm] = {k: v for k, v in cfg.items() if k != "experiment"}
    for seed, arms in by_seed.items():
        a, b = arms["e2e"], arms["frozen_control"]
        diff = {k for k in a if a[k] != b[k]}
        assert diff == {"model"}, (seed, diff)
        ea, eb = dict(a["model"]["e2e"]), dict(b["model"]["e2e"])
        ea.pop("freeze_backbone"), eb.pop("freeze_backbone")
        assert ea == eb
    sweep = yaml.safe_load((ROOT / "configs/h7/v33/sweep_all.yaml").read_text())["experiments"]
    assert sorted(e["config"] for e in sweep) == sorted(str(p.relative_to(ROOT)) for p in paths)


def test_config_rejects_wrong_dataset_or_loss() -> None:
    doc = load_pipeline(ROOT / "configs/h7/v33/e2e_src0.yaml")
    bad = {**doc, "cfg": {**doc["cfg"], "data": {**doc["cfg"]["data"], "dataset": "nantes_clips_trainval"}}}
    with pytest.raises(ConfigError):
        validate_pipeline(bad, "test")
    bad = {**doc, "cfg": {**doc["cfg"], "loss": {"name": "ce"}}}
    with pytest.raises(ConfigError):
        validate_pipeline(bad, "test")
