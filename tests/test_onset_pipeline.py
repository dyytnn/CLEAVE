"""v24/v25 dense onset targets, metrics, OOF provenance and train/val isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import stseg.kinetic  # noqa: F401 -- component registration
from stseg.kinetic.config import ConfigError, load_pipeline, validate_pipeline
from stseg.kinetic.onset_metrics import event_metrics, onset_targets, peak_indices
from stseg.kinetic.onset_decoder import event_viterbi_prior, patient_bootstrap_delta
from stseg.kinetic.onset_oof import calibrate, fit_platt, patient_folds, patient_of
from stseg.kinetic.onset_pipeline import ONSET_MODELS
from stseg.kinetic.registry import BUILDER_REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def test_onset_target_masks_gaps_and_ambiguous_stage_skips() -> None:
    labels = np.array([2, 3, 3, 5, 5, 6])
    indices = np.array([10, 11, 13, 14, 15, 16])
    target, valid = onset_targets(labels, indices)
    assert np.flatnonzero(target).tolist() == [1, 5]
    # start, missing-frame boundary, and t3->t5 skipped-label boundary are unknown.
    assert np.flatnonzero(~valid).tolist() == [0, 2, 3]


def test_window_target_changes_support_but_not_exact_reference() -> None:
    labels = np.array([2, 2, 3, 3, 4])
    indices = np.arange(5)
    exact, valid = onset_targets(labels, indices, radius=0)
    window, valid_window = onset_targets(labels, indices, radius=2)
    assert np.array_equal(valid, valid_window)
    assert np.flatnonzero(exact).tolist() == [2, 4]
    assert np.flatnonzero(window).tolist() == [0, 1, 2, 3, 4]


def test_peak_ties_and_nms_are_deterministic() -> None:
    scores = np.array([0.1, 0.8, 0.8, 0.7, 0.9])
    indices = np.arange(5)
    peaks = peak_indices(scores, indices, np.ones(5, dtype=bool), separation=2)
    assert peaks.tolist() == [4, 1]


def test_event_metric_matches_each_ground_truth_once() -> None:
    record = {
        "scores": np.array([0.1, 0.9, 0.8, 0.2, 0.95]),
        "indices": np.arange(5),
        "valid": np.ones(5, dtype=bool),
        "onset": np.array([0, 1, 0, 0, 1]),
    }
    metric = event_metrics([record], tolerance=0, separation=1)
    assert metric["n_gt"] == 2
    assert metric["best_f1"] == 1.0
    assert metric["event_ap"] == 1.0


def test_dense_tcn_shape_gradient_and_determinism() -> None:
    features = torch.randn(2, 11, 16)
    outputs = []
    for _ in range(2):
        torch.manual_seed(8)
        model = ONSET_MODELS.get(
            "dense_tcn", d_in=16, hidden=8, layers=2, kernel=3, dropout=0.0
        )
        logits = model(features)
        assert logits.shape == (2, 11)
        logits.square().mean().backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        outputs.append(logits.detach())
    assert torch.equal(*outputs)


def test_patient_folds_keep_all_videos_of_patient_together_and_balanced() -> None:
    records = [{"video": f"P{i}-1"} for i in range(9)] + [{"video": "P0-2"}]
    assignment = patient_folds(records, folds=3, seed=17)
    assert patient_of("P0-1") == patient_of("P0-2") == "P0"
    counts = np.bincount(list(assignment.values()))
    assert counts.max() - counts.min() <= 1
    assert assignment == patient_folds(records, folds=3, seed=17)


def test_platt_calibration_is_monotone_and_improves_simple_brier() -> None:
    scores = np.array([0.2, 0.3, 0.7, 0.8])
    targets = np.array([0, 0, 1, 1], dtype=np.float32)
    record = {
        "scores": scores,
        "valid": np.ones(4, dtype=bool),
        "onset": targets,
    }
    scale, offset = fit_platt([record], max_iter=30)
    calibrated = calibrate([record], scale, offset)[0]["scores"]
    assert scale > 0 and np.all(np.diff(calibrated) >= 0)
    assert np.mean((calibrated - targets) ** 2) < np.mean((scores - targets) ** 2)


@pytest.mark.parametrize(
    "path",
    [
        "configs/h7/v24_dense_onset.yaml",
        "configs/h7/v24_dense_window.yaml",
        "configs/h7/v25_onset_oof.yaml",
        "configs/h7/v25_grouped_recipe_baseline.yaml",
        "configs/h7/v25_onset_decoder.yaml",
    ],
)
def test_onset_and_grouped_configs_validate(path: str) -> None:
    load_pipeline(ROOT / path)


def test_oof_schema_rejects_non_exact_target() -> None:
    document = load_pipeline(ROOT / "configs/h7/v25_onset_oof.yaml")
    document["cfg"]["target"]["radius"] = 2
    with pytest.raises(ConfigError):
        validate_pipeline(document)


def test_trainval_builder_does_not_construct_test_and_smoke_has_at_most_20_train_frames() -> None:
    document = load_pipeline(ROOT / "configs/h7/v25_grouped_recipe_baseline.yaml")
    builder = BUILDER_REGISTRY.get(
        document["builder"], cfg=document["cfg"], seed=0, smoke=True
    )
    builder.produce_pre_processor()
    assert not hasattr(builder._data, "test")
    assert 16 <= len(builder._data.train_frames) <= 20


def test_prior_centered_event_decoder_constant_is_exact_baseline() -> None:
    rng = np.random.default_rng(9)
    log_probs = rng.normal(size=(12, 16))
    transition = np.full((16, 16), -np.inf)
    for phase in range(16):
        transition[phase, phase] = -0.2
        if phase + 1 < 16:
            transition[phase, phase + 1] = -1.0
    prior = 0.017
    baseline = event_viterbi_prior(
        log_probs, transition, np.linspace(0.01, 0.4, 12), prior, weight=0
    )
    constant = event_viterbi_prior(
        log_probs, transition, np.full(12, prior), prior, weight=1
    )
    assert np.array_equal(baseline, constant)


def test_patient_bootstrap_pairs_videos_and_groups_patients() -> None:
    baseline = {
        "per_video": [
            {"video": "A-1", "n_transitions": 2, "n_far": 1},
            {"video": "A-2", "n_transitions": 2, "n_far": 2},
            {"video": "B-1", "n_transitions": 1, "n_far": 1},
        ]
    }
    treatment = {
        "per_video": [
            {"video": "A-1", "n_transitions": 2, "n_far": 0},
            {"video": "A-2", "n_transitions": 2, "n_far": 1},
            {"video": "B-1", "n_transitions": 1, "n_far": 1},
        ]
    }
    summary = patient_bootstrap_delta(baseline, treatment, seed=2, draws=100)
    assert summary["n_patients"] == 2
    assert summary["mean_patient_delta"] == 0.25
