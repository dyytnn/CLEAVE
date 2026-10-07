"""Validation-only integration of calibrated onset evidence into phase decoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stseg.eval.kinetic_metrics import evaluate_videos

from .datasets import _frames
from .event_audit import CLEAVAGE_FROM, _sha256, patient_overlap, summarize_paths
from .interfaces import AbsPipelineBuilder, AbsProcess
from .onset_oof import patient_of
from .process import _commit
from .registry import BUILDER_REGISTRY

ROOT = Path(__file__).resolve().parents[3]


def event_viterbi_prior(
    log_probs: np.ndarray,
    log_trans: np.ndarray,
    scores: np.ndarray,
    prior: float,
    weight: float = 1.0,
    floor: float = 1e-6,
) -> np.ndarray:
    """Decode with onset likelihood ratios relative to the train-OOF prior.

    For adjacent cleavage transitions, add log(q/prior); for staying in a
    cleavage phase, add log((1-q)/(1-prior)). Thus q == prior is exactly the
    original decoder, unlike a hard-coded 0.5 centre for rare dense events.
    """
    lp, transition, q = (
        np.asarray(log_probs),
        np.asarray(log_trans),
        np.asarray(scores),
    )
    if lp.ndim != 2 or transition.shape != (lp.shape[1], lp.shape[1]):
        raise ValueError("invalid event-decoder phase shapes")
    if q.shape != (len(lp),) or not np.isfinite(q).all():
        raise ValueError("event decoder requires one finite score per frame")
    if not 0 < prior < 1 or weight < 0 or not 0 < floor < 0.5:
        raise ValueError("invalid event-decoder prior/weight/floor")
    if ((q < 0) | (q > 1)).any():
        raise ValueError("event scores must be in [0,1]")
    q = np.clip(q, floor, 1 - floor)
    prior = float(np.clip(prior, floor, 1 - floor))
    score = lp[0].copy()
    back = np.zeros(lp.shape, dtype=np.int64)
    for time in range(1, len(lp)):
        candidates = score[:, None] + transition
        for phase in CLEAVAGE_FROM:
            if phase + 1 < lp.shape[1]:
                candidates[phase, phase] += weight * np.log(
                    (1 - q[time]) / (1 - prior)
                )
                candidates[phase, phase + 1] += weight * np.log(q[time] / prior)
        back[time] = candidates.argmax(0)
        score = candidates.max(0) + lp[time]
    path = np.empty(len(lp), dtype=np.int64)
    path[-1] = int(score.argmax())
    for time in range(len(lp) - 1, 0, -1):
        path[time - 1] = back[time, path[time]]
    return path


def evaluate_fixed_paths(
    sequences: list[dict], log_trans: np.ndarray, paths: list[np.ndarray]
) -> dict:
    """Reuse the benchmark evaluator with already-decoded, aligned paths."""
    iterator = iter(paths)
    return evaluate_videos(sequences, log_trans, decode=lambda _: next(iterator))


def patient_bootstrap_delta(
    baseline: dict, treatment: dict, seed: int, draws: int = 10000
) -> dict:
    """Paired bootstrap validation p_t differences at the patient level."""
    base = {row["video"]: row for row in baseline["per_video"]}
    event = {row["video"]: row for row in treatment["per_video"]}
    if set(base) != set(event):
        raise ValueError("baseline/treatment video mismatch")
    grouped: dict[str, list[float]] = {}
    for video in sorted(base):
        b, e = base[video], event[video]
        if b["n_transitions"] != e["n_transitions"]:
            raise ValueError("paired transition count mismatch")
        count = b["n_transitions"]
        if count:
            delta = (b["n_far"] - e["n_far"]) / count
            grouped.setdefault(patient_of(video), []).append(delta)
    patient_delta = np.array([np.mean(values) for values in grouped.values()])
    rng = np.random.default_rng(seed)
    samples = patient_delta[
        rng.integers(0, len(patient_delta), (draws, len(patient_delta)))
    ].mean(1)
    return {
        "n_patients": len(patient_delta),
        "mean_patient_delta": float(patient_delta.mean()),
        "bootstrap_ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_probability_positive": float((samples > 0).mean()),
        "draws": draws,
        "validation_only": True,
    }


class OnsetDecoderProcess(AbsProcess):
    """Apply one preregistered fixed event potential to cached val predictions."""

    def __init__(self, cfg: dict, out_dir: Path, seed: int) -> None:
        self.cfg, self.out_dir, self.seed = cfg, out_dir, seed

    def run(self) -> dict[str, Any]:
        settings = self.cfg["integration"]
        source = ROOT / settings["source_run_template"].format(seed=self.seed)
        score_run = ROOT / settings["score_run"]
        source_cfg_path = source / "config.resolved.json"
        source_results_path = source / "results.json"
        predictions_path = source / "diagnostics/frame_probs_val.npz"
        transition_path = source / "transition_log_matrix.npy"
        score_path = score_run / "event_scores.csv"
        score_results_path = score_run / "results.json"
        for path in (
            source_cfg_path,
            source_results_path,
            predictions_path,
            transition_path,
            score_path,
            score_results_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        source_cfg = json.loads(source_cfg_path.read_text())
        source_results = json.loads(source_results_path.read_text())
        score_results = json.loads(score_results_path.read_text())
        data_cfg = source_cfg["data"]
        split = ROOT / data_cfg["split"]
        if source_results.get("test_evaluated") is not False:
            raise ValueError("v25 integration requires a train/val-only phase source")
        if data_cfg.get("dataset") != "nantes_clips_trainval":
            raise ValueError("v25 integration requires nantes_clips_trainval")
        if patient_overlap(split) != {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        }:
            raise ValueError("patient overlap in grouped split")
        if score_results.get("test_evaluated") is not False:
            raise ValueError("OOF cache unexpectedly evaluated test")
        prior = float(score_results["train_oof"]["dense"]["prevalence"])
        scores = pd.read_csv(score_path)
        scores = scores[scores.partition == "val_ensemble"]
        if scores.duplicated(["video", "frame_index"]).any():
            raise ValueError("duplicate validation event score keys")
        lookup = dict(
            zip(
                zip(scores.video.astype(str), scores.frame_index.astype(int)),
                scores.div_score.astype(float),
                strict=True,
            )
        )
        val = _frames(data_cfg, "val", "eval", self.seed)
        cache = np.load(predictions_path, allow_pickle=False)
        log_trans = np.load(transition_path, allow_pickle=False)
        sequences, signals = [], []
        for video, group in val.rows.groupby("video", sort=False):
            label_key, time_key, probability_key = (
                f"{video}__y",
                f"{video}__t",
                f"{video}__lp",
            )
            if any(key not in cache for key in (label_key, time_key, probability_key)):
                raise ValueError(f"missing cached phase predictions for {video}")
            labels = cache[label_key].astype(int)
            times = cache[time_key].astype(float)
            log_probs = cache[probability_key].astype(float)
            if not np.array_equal(labels, group.label.to_numpy()) or not np.allclose(
                times, group.time_h.to_numpy(dtype=float), atol=1e-4, rtol=0
            ):
                raise ValueError(f"phase cache alignment mismatch for {video}")
            signal = np.array(
                [lookup[(str(video), int(frame))] for frame in group.frame_index],
                dtype=float,
            )
            sequences.append(
                {
                    "video": str(video),
                    "labels": labels,
                    "times_h": times,
                    "log_probs": log_probs,
                }
            )
            signals.append(signal)
        baseline = evaluate_videos(sequences, log_trans)
        baseline_paths = [
            event_viterbi_prior(
                sequence["log_probs"], log_trans, signal, prior, weight=0
            )
            for sequence, signal in zip(sequences, signals, strict=True)
        ]
        predicted_paths = [
            event_viterbi_prior(
                sequence["log_probs"],
                log_trans,
                signal,
                prior,
                weight=settings["weight"],
                floor=settings["floor"],
            )
            for sequence, signal in zip(sequences, signals, strict=True)
        ]
        constant_paths = [
            event_viterbi_prior(
                sequence["log_probs"],
                log_trans,
                np.full_like(signal, prior),
                prior,
                weight=settings["weight"],
                floor=settings["floor"],
            )
            for sequence, signal in zip(sequences, signals, strict=True)
        ]
        rng = np.random.default_rng(settings["shuffle_seed"])
        shuffled_paths = [
            event_viterbi_prior(
                sequence["log_probs"],
                log_trans,
                rng.permutation(signal),
                prior,
                weight=settings["weight"],
                floor=settings["floor"],
            )
            for sequence, signal in zip(sequences, signals, strict=True)
        ]
        if not all(
            np.array_equal(a, b)
            for a, b in zip(baseline_paths, constant_paths, strict=True)
        ):
            raise AssertionError("constant-prior control must exactly recover baseline")
        predicted = evaluate_fixed_paths(sequences, log_trans, predicted_paths)
        constant = evaluate_fixed_paths(sequences, log_trans, constant_paths)
        shuffled = evaluate_fixed_paths(sequences, log_trans, shuffled_paths)
        report = {
            "kind": "validation_only_fixed_onset_likelihood_ratio",
            "source_run": str(source.relative_to(ROOT)),
            "source_seed": self.seed,
            "score_run": settings["score_run"],
            "split": data_cfg["split"],
            "test_evaluated": False,
            "weight": settings["weight"],
            "weight_selection": "fixed_theoretical_likelihood_ratio_not_val_tuned",
            "train_oof_prior": prior,
            "baseline": {key: value for key, value in baseline.items() if key != "per_video"},
            "constant_prior_control": {
                key: value for key, value in constant.items() if key != "per_video"
            },
            "shuffled_control": {
                key: value for key, value in shuffled.items() if key != "per_video"
            },
            "predicted": {
                key: value for key, value in predicted.items() if key != "per_video"
            },
            "delta": {
                "p_t": predicted["p_t"] - baseline["p_t"],
                "p_v": predicted["p_v"] - baseline["p_v"],
                "f1_50": predicted["f1"]["50"] - baseline["f1"]["50"],
                "mae_h_all": predicted["mae_h_all"] - baseline["mae_h_all"],
            },
            "short_phase": {
                "baseline": summarize_paths(sequences, baseline_paths),
                "predicted": summarize_paths(sequences, predicted_paths),
            },
            "paired_patient_bootstrap": patient_bootstrap_delta(
                baseline, predicted, self.seed
            ),
            "input_sha256": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (
                    source_cfg_path,
                    source_results_path,
                    predictions_path,
                    transition_path,
                    score_path,
                    score_results_path,
                )
            },
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "stseg_commit": _commit(ROOT),
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / "results.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        with (self.out_dir / "config.resolved.json").open("x") as stream:
            json.dump(self.cfg, stream, indent=2)
        return report


@BUILDER_REGISTRY.register("onset_decoder_validation")
class OnsetDecoderBuilder(AbsPipelineBuilder):
    """Registered no-training process for fixed v25 event integration."""

    def __init__(
        self,
        cfg: dict,
        seed: int | None = None,
        smoke: bool = False,
        resume: bool = True,
    ) -> None:
        super().__init__(cfg)
        self.seed = cfg["experiment"]["seed"] if seed is None else seed
        root = "runs/_smoke" if smoke else cfg["experiment"]["out_root"]
        self.out_dir = Path(root) / f"{cfg['experiment']['id']}_seed{self.seed}"

    def produce_pre_processor(self) -> None:
        """Input validation occurs in the read-only process."""

    def produce_main_processor(self) -> None:
        """No trainable model is constructed."""

    def produce_post_processor(self) -> None:
        """Fixed decoder is constructed lazily by the process."""

    @property
    def product(self) -> AbsProcess:
        return OnsetDecoderProcess(self._cfg, self.out_dir, self.seed)
