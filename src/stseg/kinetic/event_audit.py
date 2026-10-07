"""Validation-only event headroom diagnostics; no model fitting or test evaluation.

This is an exploratory transition-potential probe, not a learned event/state model.
The current score cache has legacy teacher-selection limitations recorded in PLAN.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from stseg.data.event_scores import load_event_scores, require_event_coverage
from stseg.data.nantes_kinetic import PHASE_TO_CLASS, load_split_videos
from stseg.eval.kinetic_metrics import THETA_BY_CLASS, f1_at_overlap, fill_skipped, first_times

from .datasets import _frames
from .interfaces import AbsPipelineBuilder, AbsProcess
from .process import _commit
from .registry import BUILDER_REGISTRY

ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger(__name__)
CLEAVAGE_FROM = tuple(PHASE_TO_CLASS[p] for p in ("tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8"))
SHORT_PHASES = tuple(PHASE_TO_CLASS[p] for p in ("t3", "t5", "t7"))


def cleavage_windows(labels: np.ndarray, radius: int) -> np.ndarray:
    """Binary windows whose (center-radius, center+radius) pair spans a cleavage.

    For a boundary b between b-1 and b, valid centers lie in [b-radius, b+radius).
    radius=0 is the separate exact-onset diagnostic, not a two-frame window.
    """
    if radius < 0:
        raise ValueError("event radius must be nonnegative")
    y = np.asarray(labels)
    target = np.zeros(len(y), dtype=float)
    for b in range(1, len(y)):
        if y[b - 1] in CLEAVAGE_FROM and y[b] == y[b - 1] + 1:
            if radius == 0:
                target[b] = 1
            else:
                target[max(0, b - radius) : min(len(y), b + radius)] = 1
    return target


def event_viterbi(
    log_probs: np.ndarray,
    log_trans: np.ndarray,
    scores: np.ndarray,
    weight: float,
    floor: float = 1e-4,
) -> np.ndarray:
    """Add a soft, image-score-dependent stay/adjacent-cleavage potential.

    Constant 0.5 and weight=0 exactly recover the base decoder. Unsupported
    transitions, including skips and late stages, retain the original potential.
    Centered scores are soft evidence, NOT calibrated adjacent-transition targets.
    """
    lp, lt, q = np.asarray(log_probs), np.asarray(log_trans), np.asarray(scores)
    if lp.ndim != 2 or not len(lp) or lt.shape != (lp.shape[1], lp.shape[1]):
        raise ValueError("invalid event decoder log-probability/transition shapes")
    if q.shape != (len(lp),) or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("event decoder requires one finite probability per frame")
    if not np.isfinite(weight) or weight < 0 or not 0 < floor < 0.5:
        raise ValueError("invalid event decoder weight/floor")
    if np.isnan(lp).any() or np.isposinf(lp).any() or np.isnan(lt).any() or np.isposinf(lt).any():
        raise ValueError("invalid log probabilities")
    score = lp[0].copy()
    back = np.zeros(lp.shape, dtype=np.int64)
    q = np.clip(q, floor, 1 - floor)
    for t in range(1, len(lp)):
        cand = score[:, None] + lt
        for c in CLEAVAGE_FROM:
            if c + 1 < lp.shape[1]:
                cand[c, c] += weight * np.log(2 * (1 - q[t]))
                cand[c, c + 1] += weight * np.log(2 * q[t])
        back[t] = cand.argmax(0)
        score = cand.max(0) + lp[t]
    if not np.isfinite(score).any():
        raise ValueError("event decoder has no finite path")
    path = np.empty(len(lp), dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(len(lp) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def binary_summary(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Dense-window statistics with tie-correct AUC/AP, not event-detection AP."""
    y, q = np.asarray(targets, dtype=int), np.asarray(scores, dtype=float)
    if y.shape != q.shape or y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("binary summary needs matching nonempty binary targets/scores")
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("binary summary scores must be probabilities")
    npos, nneg = int(y.sum()), int((1 - y).sum())
    auc = None
    if npos and nneg:
        auc = float((rankdata(q)[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    order = np.argsort(-q, kind="stable")
    end = np.r_[np.flatnonzero(np.diff(q[order]) != 0), len(q) - 1]
    tp = np.cumsum(y[order])[end]
    ap = float(np.sum(np.diff(np.r_[0, tp]) * tp / (end + 1)) / npos) if npos else None
    return {
        "n_frames": len(y),
        "n_positive_windows": npos,
        "prevalence": float(y.mean()),
        "roc_auc": auc,
        "window_ap": ap,
        "brier": float(np.mean((q - y) ** 2)),
        "mean_score": float(q.mean()),
        "fraction_score_ge_half": float((q >= 0.5).mean()),
    }


def summarize_paths(sequences: list[dict], paths: list[np.ndarray]) -> dict[str, Any]:
    """Existing p_t convention plus unfilled short-phase misses and segment F1@50."""
    pt, nframes, correct = [], 0, 0
    tp = fp = fn = missed = eligible = 0
    details = []
    for s, pred in zip(sequences, paths, strict=True):
        y, times = s["labels"], s["times_h"]
        gt, observed = first_times(y, times), first_times(pred, times)
        filled = fill_skipped(observed, sorted(gt))
        transitions = [c for c in sorted(gt)[1:] if c in THETA_BY_CLASS]
        hits = sum(c in filled and abs(filled[c] - gt[c]) <= THETA_BY_CLASS[c] for c in transitions)
        value = hits / len(transitions) if transitions else None
        if value is not None:
            pt.append(value)
        short = [c for c in SHORT_PHASES if c in gt and c != y[0]]
        misses = sum(c not in observed for c in short)
        missed += misses
        eligible += len(short)
        nframes += len(y)
        correct += int((pred == y).sum())
        a, b, c = f1_at_overlap(pred, y, 0.5)
        tp, fp, fn = tp + a, fp + b, fn + c
        details.append(
            {
                "video": s["video"],
                "p_t": value,
                "short_eligible": len(short),
                "short_missing": misses,
            }
        )
    return {
        "p_t": float(np.mean(pt)) if pt else None,
        "p_v": correct / max(nframes, 1),
        "f1_50": 200 * tp / max(2 * tp + fp + fn, 1),
        "short_eligible": eligible,
        "short_missing": missed,
        "short_missing_rate": missed / eligible if eligible else None,
        "per_video": details,
    }


def patient_overlap(split: Path) -> dict[str, int]:
    """Audit all split identities, including videos absent from labelled frames."""
    patients = {}
    for partition in ("train", "val", "test"):
        patients[partition] = set()
        for video in load_split_videos(split, partition):
            match = re.fullmatch(r"(.*)-_?(\d+[A-Za-z]?(?:-\d+)?)", video)
            if match is None:
                raise ValueError(f"unparseable public video ID: {video}")
            patients[partition].add(match[1])
    return {
        f"{a}_{b}": len(patients[a] & patients[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EventAuditProcess(AbsProcess):
    """Read cached validation predictions, report fixed controls, never open test predictions."""

    def __init__(self, cfg: dict, out_dir: Path, seed: int) -> None:
        self.cfg, self.out_dir, self.seed = cfg, out_dir, seed

    def run(self) -> dict[str, Any]:
        a = self.cfg["event_audit"]
        source = ROOT / a["source_run"]
        config_path = source / "config.resolved.json"
        source_cfg = json.loads(config_path.read_text())
        source_meta_path = source / "meta.json"
        source_meta = json.loads(source_meta_path.read_text()) if source_meta_path.exists() else {}
        d = dict(source_cfg["data"])
        if d.get("split_mode", "video") != "video" or d.get("division_score_path"):
            raise ValueError("audit parent must be a video-split appearance-only model")
        split = ROOT / d["split"]
        overlap = patient_overlap(split)
        if any(overlap.values()) and not a.get("allow_legacy_overlap", False):
            raise ValueError("patient overlap: audit requires explicit exploratory legacy opt-in")
        # Only metadata/labels are loaded. No __getitem__, image read or forward pass.
        frames = _frames(d, "val", "eval", self.seed)
        train = _frames(d, "train", "eval", self.seed)
        score_path = ROOT / a["score_path"]
        scores = load_event_scores(score_path)
        require_event_coverage(scores, frames.rows)
        cache = source / "diagnostics/frame_probs_val.npz"
        transition_path = source / "transition_log_matrix.npy"
        lt = np.load(transition_path, allow_pickle=False)
        sequences, signal, targets, times = [], [], [], []
        radius = a.get("radius", 2)
        with np.load(cache, allow_pickle=False) as z:
            video_ids = {k.rsplit("__", 1)[0] for k in z.files}
            if video_ids != set(frames.videos()):
                raise ValueError(
                    "validation cache video IDs differ from the declared validation split"
                )
            for video, g in frames.rows.groupby("video", sort=True):
                lp, y, th = (
                    z[f"{video}__lp"].astype(float),
                    z[f"{video}__y"],
                    z[f"{video}__t"].astype(float),
                )
                if not np.array_equal(y, g.label.to_numpy()) or not np.allclose(
                    th, g.time_h, atol=1e-4, rtol=0
                ):
                    raise ValueError("validation cache labels/times do not match manifest order")
                if lp.shape != (len(g), len(lt)):
                    raise ValueError("validation cache has invalid prediction dimensions")
                sequences.append({"video": video, "labels": y, "times_h": th, "log_probs": lp})
                signal.append(np.array([scores[(video, int(f))] for f in g.frame_index]))
                targets.append(cleavage_windows(y, radius))
                times.append(th)
        # Clock control: training-only empirical event-window rate per acquisition-time bin.
        width = a.get("clock_bin_hours", 2.0)
        train_times, train_y = [], []
        for _, g in train.rows.groupby("video", sort=False):
            train_times.extend(g.time_h.to_numpy())
            train_y.extend(cleavage_windows(g.label.to_numpy(), radius))
        bins = np.floor(np.asarray(train_times) / width).astype(int)
        train_y = np.asarray(train_y)
        prior = float(train_y.mean())
        strength = a.get("clock_prior_strength", 10.0)
        rates = {
            b: float((train_y[bins == b].sum() + strength * prior) / ((bins == b).sum() + strength))
            for b in np.unique(bins)
        }
        clock = [np.array([rates.get(int(np.floor(t / width)), prior) for t in th]) for th in times]
        rng = np.random.default_rng(self.seed)
        controls = {
            "constant": [np.full_like(q, 0.5) for q in signal],
            "shuffled": [rng.permutation(q) for q in signal],
            "clock_train_only": clock,
            "predicted": signal,
            "oracle_window_DIAGNOSTIC_ONLY": targets,
        }
        if a.get("include_onset_oracle", False):
            controls["oracle_onset_DIAGNOSTIC_ONLY"] = [
                cleavage_windows(s["labels"], 0) for s in sequences
            ]
        floor = a.get("floor", 1e-4)
        base_paths = [
            event_viterbi(s["log_probs"], lt, q, 0, floor) for s, q in zip(sequences, signal)
        ]
        baseline = summarize_paths(sequences, base_paths)
        trials = []
        for name, qs in controls.items():
            for weight in a.get("weights", [0.25, 0.5, 1.0]):
                paths = [
                    event_viterbi(s["log_probs"], lt, q, weight, floor)
                    for s, q in zip(sequences, qs)
                ]
                summary = summarize_paths(sequences, paths)
                trials.append(
                    {
                        "signal": name,
                        "weight": weight,
                        **summary,
                        "delta_p_t": summary["p_t"] - baseline["p_t"],
                    }
                )
        report = {
            "kind": "validation_only_exploratory_headroom",
            "split": str(split),
            "source_run": str(source),
            "source_seed": source_meta.get("seed"),
            "control_seed": self.seed,
            "n_videos": len(sequences),
            "patient_overlap": overlap,
            "limitations": [
                "Legacy teacher seed was test-selected; no confirmatory claim.",
                "Centered window AP/Brier are not dense onset detection AP/calibration.",
                "Oracle uses validation labels for diagnosis, never deployment.",
                "Float16 cached probabilities may introduce rounding differences.",
            ],
            "stseg_commit": _commit(ROOT),
            "implementation_sha256": {
                str(p): _sha256(p)
                for p in (
                    Path(__file__),
                    ROOT / "src/stseg/data/event_scores.py",
                    ROOT / "src/stseg/data/nantes_kinetic.py",
                    ROOT / "src/stseg/eval/kinetic_metrics.py",
                )
            },
            "input_sha256": {
                str(p): _sha256(p) for p in (config_path, cache, transition_path, score_path, split)
            },
            "dense_windows": {
                name: binary_summary(np.concatenate(targets), np.concatenate(qs))
                for name, qs in controls.items()
            },
            "baseline": baseline,
            "trials": trials,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # No source-run writes; exclusive creation prevents accidental result replacement.
        with (self.out_dir / "results.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        with (self.out_dir / "config.resolved.json").open("x") as stream:
            json.dump(self.cfg, stream, indent=2)
        LOGGER.info("event audit wrote %s", self.out_dir)
        return report


@BUILDER_REGISTRY.register("event_signal_audit")
class EventAuditBuilder(AbsPipelineBuilder):
    """YAML-selected no-training process; compatible with run_pipeline --validate-only."""

    def __init__(
        self, cfg: dict, seed: int | None = None, smoke: bool = False, resume: bool = True
    ) -> None:
        super().__init__(cfg)
        exp = cfg["experiment"]
        self.seed = int(exp.get("seed", 0) if seed is None else seed)
        out_root = "runs/_smoke" if smoke else exp.get("out_root", "runs/event_audit")
        self.out_dir = ROOT / out_root / f"{exp['id']}_seed{self.seed}"

    def produce_pre_processor(self) -> None:
        """Inputs are validated lazily by the read-only audit process."""

    def produce_main_processor(self) -> None:
        """No model, optimizer or training process is constructed."""

    def produce_post_processor(self) -> None:
        """Fixed-weight diagnostics are computed within the process."""

    @property
    def product(self) -> EventAuditProcess:
        return EventAuditProcess(self._cfg, self.out_dir, self.seed)
