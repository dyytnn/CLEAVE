"""v23 model, strict score transport, validation-only audit and inference parity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import stseg.kinetic as K
from stseg.data.event_scores import load_event_scores, require_event_coverage
from stseg.eval.kinetic_metrics import evaluate_videos, viterbi
from stseg.kinetic.config import ConfigError, load_pipeline, validate_pipeline
from stseg.kinetic.event_audit import (
    binary_summary,
    cleavage_windows,
    event_viterbi,
    patient_overlap,
    summarize_paths,
)
from stseg.kinetic.models import build_model
from stseg.kinetic.process import sequence_log_probs

ROOT = Path(__file__).resolve().parents[1]
PREDICTED = (
    ROOT / "configs/h7/v23/resnet18_transformer_L16_crossfocal7_recipe_event_predicted_split0.yaml"
)


def tiny_config() -> dict:
    return {
        "type": "event_conditioned",
        "backbone": {"name": "kinetic_cnn", "base_channels": 8},
        "head": {"name": "lstm", "hidden": 8, "layers": 1},
        "dropout": 0.0,
        "event_conditioning": {"hidden": 4},
    }


def inputs(length: int = 4) -> torch.Tensor:
    x = torch.randn(2, length, 4, 32, 32)
    x[:, :, -1] = torch.linspace(0.1, 0.9, length)[None, :, None, None]
    return x


def test_zero_init_and_disabled_are_exact_parent_and_visual_channels_unchanged():
    cfg = tiny_config()
    torch.manual_seed(4)
    event = build_model(cfg, 4).eval()
    event_rng = torch.get_rng_state()
    torch.manual_seed(4)
    parent = build_model({**cfg, "type": "seq_kinetic"}, 3).eval()
    assert torch.equal(event_rng, torch.get_rng_state())
    x = inputs()
    seen = []
    hook = event.backbone.visual.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].shape[1])
    )
    with torch.no_grad():
        expected = parent(x[:, :, :3])["logits"]
        assert torch.equal(event(x)["logits"], expected)
        event.backbone.event[-1].weight.fill_(0.1)
        assert not torch.equal(event(x)["logits"], expected)
        event.backbone.enabled = False
        assert torch.equal(event(x)["logits"], expected)
    hook.remove()
    assert seen == [3, 3, 3]


def test_branch_gradients_reach_first_layer_after_initial_identity_step():
    torch.manual_seed(5)
    model = build_model(tiny_config(), 4).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = inputs(), torch.arange(8).reshape(2, 4)
    for step in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x)["logits"].flatten(0, 1), y.flatten())
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.backbone.event[-1].weight.grad.abs().sum() > 0
        if step == 1:
            assert model.backbone.event[0].weight.grad.abs().sum() > 0
        opt.step()


def test_constant_control_ignores_scores_but_keeps_learnable_branch():
    cfg = tiny_config()
    cfg["event_conditioning"]["mode"] = "constant"
    model = build_model(cfg, 4).eval()
    with torch.no_grad():
        model.backbone.event[-1].weight.fill_(0.2)
    x = inputs()
    other = x.clone()
    other[:, :, -1] = 1 - other[:, :, -1]
    assert torch.equal(model(x)["logits"], model(other)["logits"])


def test_window_cache_fast_path_includes_nonzero_event_branch():
    torch.manual_seed(8)
    model = build_model(tiny_config(), 4).eval()
    with torch.no_grad():
        model.backbone.event[-1].weight.normal_()
    frames = inputs(9)[0]
    fast = sequence_log_probs(model, frames, 4, 2, torch.device("cpu"), False)
    acc, count = torch.zeros(9, 16), torch.zeros(9, 1)
    with torch.no_grad():
        for start in (0, 2, 4, 5):
            logits = model(frames[start : start + 4][None])["logits"][0]
            acc[start : start + 4] += logits.log_softmax(-1)
            count[start : start + 4] += 1
    np.testing.assert_allclose(fast, (acc / count).numpy(), atol=1e-6)


def test_crossfocal_receives_seven_real_planes_not_eighth_scalar():
    cfg = tiny_config()
    cfg["backbone"] = {"name": "crossfocal", "cnn": "resnet18", "pretrained": False}
    model = build_model(cfg, 8).eval()
    assert model.backbone.visual.P == 7
    with torch.no_grad():
        assert model(torch.zeros(1, 2, 8, 32, 32))["logits"].shape == (1, 2, 16)


def test_deterministic_initialization_and_outputs():
    x = inputs()
    outputs = []
    for _ in range(2):
        torch.manual_seed(19)
        model = build_model(tiny_config(), 4).eval()
        outputs.append(model(x)["logits"].detach())
    assert torch.equal(*outputs)


def test_event_model_optimizer_checkpoint_round_trip(tmp_path):
    torch.manual_seed(21)
    model = build_model(tiny_config(), 4).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = inputs()

    def step(m, opt):
        opt.zero_grad()
        m(x)["logits"].square().mean().backward()
        opt.step()

    step(model, optimizer)
    path = tmp_path / "last.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    restored = build_model(tiny_config(), 4).train()
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = torch.load(path, weights_only=True)
    restored.load_state_dict(state["model"])
    restored_opt.load_state_dict(state["optimizer"])
    step(model, optimizer)
    step(restored, restored_opt)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


@pytest.mark.parametrize(
    "rows",
    [
        "A-1,1,0.2\nA-1,1,0.3\n",
        "A-1,1,nan\n",
        "A-1,1,1.1\n",
        "A-1,1.5,0.3\n",
        "A-1,1,inf\n",
    ],
)
def test_strict_scores_reject_bad_cache(tmp_path, rows):
    path = tmp_path / "scores.csv"
    path.write_text("video,frame_index,div_score\n" + rows)
    with pytest.raises(ValueError):
        load_event_scores(path)


def test_cache_coverage_includes_supplied_unlabelled_rows(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("video,frame_index,div_score\nA-1,1,0.3\n")
    scores = load_event_scores(path)
    rows = pd.DataFrame({"video": ["A-1"], "frame_index": [1], "phase": [None]})
    require_event_coverage(scores, rows)
    with pytest.raises(ValueError, match="missing"):
        require_event_coverage(scores, rows.assign(frame_index=2))


@pytest.mark.parametrize(
    "change",
    [
        {"hidden": 0},
        {"hidden": None},
        {"hidden": True},
        {"mode": "shuffle_per_batch"},
        {"scale": float("nan")},
        {"neutral_score": 2.0},
        {"typo": 1},
        {"enabled": None},
    ],
)
def test_event_schema_rejects_invalid_settings(change):
    doc = load_pipeline(PREDICTED)
    doc["cfg"]["model"]["event_conditioning"].update(change)
    with pytest.raises(ConfigError):
        validate_pipeline(doc)


@pytest.mark.parametrize(
    "change",
    [
        {"division_score_path": None},
        {"division_score_strict": False},
        {"instance_norm": True},
        {"hires_crop_dir": "cache"},
    ],
)
def test_event_schema_rejects_unsafe_transport(change):
    doc = load_pipeline(PREDICTED)
    doc["cfg"]["data"].update(change)
    with pytest.raises(ConfigError):
        validate_pipeline(doc)


def test_v23_configs_validate_and_preserve_parent_recipe():
    parent = load_pipeline(
        ROOT / "configs/h7/v17/resnet18_transformer_L16_crossfocal7_recipe_e20_evalfix_split0.yaml"
    )["cfg"]
    for path in (ROOT / "configs/h7/v23").glob("*.yaml"):
        if path.name.startswith("sweep"):
            continue
        doc = load_pipeline(path)
        for section in ("training", "loss", "optimizer", "scheduler", "decoder"):
            assert doc["cfg"][section] == parent[section]
        if doc["builder"] == "event_signal_audit":
            builder = K.BUILDER_REGISTRY.get(doc["builder"], cfg=doc["cfg"])
            assert builder.product.cfg == doc["cfg"]


def test_generator_matches_checked_in_configs():
    spec = importlib.util.spec_from_file_location(
        "event_configs", ROOT / "scripts/make_h7_event_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for path, expected in module.render_configs().items():
        assert (ROOT / path).read_text() == expected


def test_centered_window_support_and_noncleavage_boundaries():
    y = np.array([2, 2, 2, 3, 3, 3, 11, 12])
    assert np.flatnonzero(cleavage_windows(y, 2)).tolist() == [1, 2, 3, 4]
    assert np.flatnonzero(cleavage_windows(y, 0)).tolist() == [3]


def test_event_decoder_baseline_identity_and_late_phase_invariance():
    rng = np.random.default_rng(2)
    lp = rng.normal(size=(10, 16))
    lt = np.full((16, 16), -np.inf)
    lt[np.triu_indices(16)] = 0
    expected = viterbi(lp, lt)
    assert np.array_equal(event_viterbi(lp, lt, np.full(10, 0.5), 4), expected)
    assert np.array_equal(event_viterbi(lp, lt, rng.random(10), 0), expected)
    lp[:, :11] = -np.inf
    assert np.array_equal(event_viterbi(lp, lt, rng.random(10), 4), viterbi(lp, lt))


def test_event_decoder_uses_observed_score_to_move_boundary():
    lp = np.full((4, 16), -np.inf)
    lp[:, 2:4] = 0
    lp[0, 3] = lp[-1, 2] = -np.inf
    lt = np.full((16, 16), -np.inf)
    np.fill_diagonal(lt, 0)
    lt[2, 3] = 0
    early = event_viterbi(lp, lt, np.array([0.5, 0.99, 0.01, 0.01]), 1)
    late = event_viterbi(lp, lt, np.array([0.5, 0.01, 0.01, 0.99]), 1)
    assert early.tolist() == [2, 3, 3, 3]
    assert late.tolist() == [2, 2, 2, 3]


def test_binary_metric_ties_and_missing_class_are_well_defined():
    result = binary_summary(np.array([0, 1, 0, 1]), np.full(4, 0.5))
    assert result["roc_auc"] == result["window_ap"] == 0.5
    assert result["brier"] == 0.25
    assert binary_summary(np.zeros(3), np.zeros(3))["roc_auc"] is None
    assert binary_summary(np.ones(3), np.ones(3))["window_ap"] == 1


def test_path_summary_matches_existing_metrics_without_filling_misses():
    y = np.array([2, 2, 3, 4, 4, 5, 5])
    pred = np.array([2, 2, 3, 3, 5, 5, 5])
    lp = np.full((len(y), 16), -10.0)
    lp[np.arange(len(y)), pred] = 0
    seq = {"video": "A-1", "labels": y, "log_probs": lp, "times_h": np.arange(len(y)) * 0.25}
    actual = summarize_paths([seq], [pred])
    expected = evaluate_videos([seq], np.zeros((16, 16)))
    assert actual["p_t"] == expected["p_t"]
    assert actual["p_v"] == expected["p_v"]
    assert actual["f1_50"] == expected["f1"]["50"]
    assert actual["short_missing"] == actual["short_eligible"] == 1


def test_patient_overlap_includes_unlabelled_sibling_videos(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"videos": {"train": ["A-1"], "val": ["B-1"], "test": ["A-2"]}}))
    assert patient_overlap(path) == {"train_val": 0, "train_test": 1, "val_test": 0}


def test_audit_schema_has_no_test_partition_option():
    doc = load_pipeline(ROOT / "configs/h7/v23/event_audit_parent_seed0.yaml")
    doc["cfg"]["event_audit"]["partition"] = "test"
    with pytest.raises(ConfigError):
        validate_pipeline(doc)


def test_audit_process_never_loads_test_predictions_and_refuses_overwrite(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from stseg.kinetic.event_audit import EventAuditProcess

    source = tmp_path / "source"
    (source / "diagnostics").mkdir(parents=True)
    split = tmp_path / "split.csv"
    split.write_text("A-1,train\nB-1,val\nC-1,test\n")
    (source / "config.resolved.json").write_text(json.dumps({"data": {"split": str(split)}}))
    (source / "meta.json").write_text(json.dumps({"seed": 2}))
    y = np.array([2, 2, 3, 3, 4, 4])
    times = np.arange(6) * 0.25
    lp = np.full((6, 16), -5.0)
    lp[np.arange(6), y] = 0.0
    np.save(source / "transition_log_matrix.npy", np.zeros((16, 16)))
    np.savez(
        source / "diagnostics/frame_probs_val.npz", **{"B-1__lp": lp, "B-1__y": y, "B-1__t": times}
    )
    (source / "diagnostics/frame_probs_test.npz").write_bytes(b"must never be opened")
    cache = tmp_path / "scores.csv"
    pd.DataFrame(
        {"video": ["B-1"] * 6, "frame_index": np.arange(6), "div_score": np.full(6, 0.5)}
    ).to_csv(cache, index=False)
    partitions = []

    def fake_frames(d, partition, mode, seed):
        partitions.append(partition)
        assert partition in {"train", "val"}
        video = "A-1" if partition == "train" else "B-1"
        rows = pd.DataFrame(
            {"video": [video] * 6, "label": y, "time_h": times, "frame_index": np.arange(6)}
        )
        return SimpleNamespace(rows=rows, videos=lambda: [video])

    monkeypatch.setattr("stseg.kinetic.event_audit._frames", fake_frames)
    cfg = {
        "event_audit": {
            "source_run": str(source),
            "score_path": str(cache),
            "weights": [0.5],
            "include_onset_oracle": True,
        }
    }
    process = EventAuditProcess(cfg, tmp_path / "output", 3)
    report = process.run()
    assert partitions == ["val", "train"]
    assert report["source_seed"] == 2 and report["control_seed"] == 3
    assert report["baseline"]["p_t"] == 1.0
    # Overlapping event windows cover this entire tiny fixture: no negative class.
    assert report["dense_windows"]["predicted"]["roc_auc"] is None
    assert any(t["signal"] == "oracle_onset_DIAGNOSTIC_ONLY" for t in report["trials"])
    with pytest.raises(FileExistsError):
        process.run()


def test_audit_rejects_test_cache_disguised_as_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from stseg.kinetic.event_audit import EventAuditProcess

    source = tmp_path / "source"
    (source / "diagnostics").mkdir(parents=True)
    split = tmp_path / "split.csv"
    split.write_text("A-1,train\nB-1,val\nC-1,test\n")
    (source / "config.resolved.json").write_text(json.dumps({"data": {"split": str(split)}}))
    cache = tmp_path / "scores.csv"
    cache.write_text("video,frame_index,div_score\nB-1,0,0.5\n")
    rows = pd.DataFrame({"video": ["B-1"], "frame_index": [0], "label": [2], "time_h": [0.0]})
    monkeypatch.setattr(
        "stseg.kinetic.event_audit._frames",
        lambda *args: SimpleNamespace(rows=rows, videos=lambda: ["B-1"]),
    )
    np.save(source / "transition_log_matrix.npy", np.zeros((16, 16)))
    np.savez(source / "diagnostics/frame_probs_val.npz", **{"C-1__lp": np.zeros((1, 16))})
    cfg = {"event_audit": {"source_run": str(source), "score_path": str(cache)}}
    with pytest.raises(ValueError, match="video IDs"):
        EventAuditProcess(cfg, tmp_path / "output", 0).run()


@pytest.mark.slow
def test_real_train_only_two_epoch_smoke():
    """16-frame clips from <=20 real training frames; no val/test evaluation or GPU."""
    from stseg.data.nantes_kinetic import NantesKineticClips
    from stseg.kinetic.datasets import _frames

    if not (ROOT / "data/derived/nantes_manifest_F0.csv").exists():
        pytest.skip("real Nantes training manifest unavailable")
    cfg = load_pipeline(PREDICTED)["cfg"]
    torch.manual_seed(0)
    frames = _frames(cfg["data"], "train", "train", 0)
    group = next(g for _, g in frames.rows.groupby("video", sort=True) if len(g) >= 20)
    frames.rows = group.iloc[:20].reset_index(drop=True)
    clips = NantesKineticClips(frames, clip_len=16, clips_per_video=1, seed=0)
    # Random init avoids downloads; this smoke tests plumbing, not performance.
    cfg["model"]["backbone"]["pretrained"] = False
    model = build_model(cfg["model"], frames.in_channels).cpu().train()
    loss_fn = K.LOSS_REGISTRY.build(cfg["loss"])
    opt = K.OPTIMIZER_REGISTRY.build(cfg["optimizer"], params=model.parameters())
    losses = []
    for _ in range(2):
        batch = clips[0]
        assert batch["image"].shape == (16, 8, 224, 224)
        opt.zero_grad()
        loss = loss_fn(model(batch["image"][None])["logits"], batch["label"][None])
        assert torch.isfinite(loss)
        loss.backward()
        assert model.backbone.event[-1].weight.grad.abs().sum() > 0
        opt.step()
        losses.append(float(loss.detach()))
    print(f"train-only CPU smoke, 2 epochs, 20 frames, losses={losses}")
