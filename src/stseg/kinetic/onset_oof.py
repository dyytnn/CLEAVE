"""Patient-level OOF exact-onset teacher and calibrated train/val score cache."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .interfaces import AbsPipelineBuilder, AbsProcess
from .onset_metrics import dense_metrics, event_metrics
from .onset_pipeline import (
    ONSET_MODELS,
    atomic_torch_save,
    cached_features,
    contiguous_logits,
    json_write,
    seed_all,
)
from .process import _commit
from .registry import BUILDER_REGISTRY

ROOT = Path(__file__).resolve().parents[3]
PATIENT_RE = re.compile(r"(.*)-_?(\d+[A-Za-z]?(?:-\d+)?)$")


def patient_of(video: str) -> str:
    """Extract public patient identifier from a Nantes video identifier."""
    match = PATIENT_RE.fullmatch(video)
    if match is None:
        raise ValueError(f"unparseable public video ID: {video}")
    return match.group(1)


def patient_folds(records: list[dict], folds: int, seed: int) -> dict[str, int]:
    """Balanced deterministic fold assignment; every patient has one fold."""
    patients = sorted({patient_of(record["video"]) for record in records})
    order = np.random.default_rng(seed).permutation(len(patients))
    return {patients[index]: rank % folds for rank, index in enumerate(order)}


def predict_records(
    model: nn.Module,
    records: list[dict],
    device: torch.device,
    pos_weight: float,
) -> list[dict]:
    """Predict prior-corrected dense probabilities for aligned full videos."""
    model.eval()
    output = []
    with torch.inference_mode():
        for record in records:
            logits = contiguous_logits(
                model, record["features"].to(device), record["indices"]
            )
            scores = torch.sigmoid(logits - math.log(pos_weight)).cpu().numpy()
            output.append(
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("features", "target")
                }
                | {"scores": scores}
            )
    return output


def train_fold(
    cfg: dict,
    training_records: list[dict],
    heldout_records: list[dict],
    fold: int,
    seed: int,
    device: torch.device,
    out_dir: Path,
    fingerprint: str,
    resume: bool,
) -> tuple[dict, float, dict]:
    """Fit one OOF teacher, selecting its epoch only on its held-out train fold."""
    checkpoint = out_dir / f"fold{fold}_best.pt"
    report_path = out_dir / f"fold{fold}_results.json"
    if checkpoint.exists() and report_path.exists():
        if not resume:
            raise FileExistsError(
                f"OOF fold {fold} exists; omit --no-resume or use a new run id"
            )
        report = json.loads(report_path.read_text())
        if report.get("fingerprint") != fingerprint:
            raise ValueError(f"OOF fold {fold} checkpoint provenance mismatch")
        return torch.load(checkpoint, map_location="cpu", weights_only=True), float(
            report["pos_weight"]
        ), report
    seed_all(seed)
    model = ONSET_MODELS.build(cfg["model"], d_in=512).to(device)
    settings = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    positive = sum(
        float(record["target"][record["valid"]].sum())
        for record in training_records
    )
    total = sum(int(record["valid"].sum()) for record in training_records)
    pos_weight = min(settings["pos_weight_cap"], (total - positive) / positive)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device), reduction="none"
    )
    best, best_epoch, history, best_state = -1.0, -1, [], None
    for epoch in range(settings["epochs"]):
        model.train()
        losses = []
        order = np.random.default_rng(seed + epoch).permutation(len(training_records))
        for index in order:
            record = training_records[index]
            valid = torch.tensor(record["valid"], device=device)
            logits = contiguous_logits(
                model, record["features"].to(device), record["indices"]
            )
            target = torch.tensor(record["target"], device=device)
            loss = criterion(logits, target)[valid].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite OOF onset loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), settings["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        predictions = predict_records(model, heldout_records, device, pos_weight)
        metric = event_metrics(
            predictions,
            cfg["evaluation"]["tolerance"],
            cfg["evaluation"]["peak_separation"],
        )
        score = metric["event_ap"]
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "event": metric}
        )
        print(
            f"OOF fold={fold} epoch={epoch + 1}/{settings['epochs']} "
            f"loss={np.mean(losses):.6f} heldout_AP={score:.6f}",
            flush=True,
        )
        if score > best:
            best, best_epoch = score, epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("OOF fold did not produce a checkpoint")
    report = {
        "fold": fold,
        "seed": seed,
        "n_train_videos": len(training_records),
        "n_heldout_videos": len(heldout_records),
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_event_ap": best,
        "fingerprint": fingerprint,
        "history": history,
    }
    atomic_torch_save(best_state, checkpoint)
    json_write(report, report_path)
    return best_state, pos_weight, report


def fit_platt(records: list[dict], max_iter: int) -> tuple[float, float]:
    """Fit monotone Platt calibration using OOF-train predictions only."""
    scores = np.concatenate([record["scores"][record["valid"]] for record in records])
    targets = np.concatenate([record["onset"][record["valid"]] for record in records])
    epsilon = 1e-6
    x = torch.tensor(
        np.log(np.clip(scores, epsilon, 1 - epsilon) / np.clip(1 - scores, epsilon, 1)),
        dtype=torch.float64,
    )
    y = torch.tensor(targets, dtype=torch.float64)
    log_scale = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_scale, bias], max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            log_scale.exp() * x + bias, y
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    scale, offset = float(log_scale.detach().exp()), float(bias.detach())
    if not math.isfinite(scale + offset) or scale <= 0:
        raise FloatingPointError("invalid Platt calibration")
    return scale, offset


def calibrate(records: list[dict], scale: float, offset: float) -> list[dict]:
    """Apply a fixed monotone logit transform without changing alignment."""
    output = []
    epsilon = 1e-6
    for record in records:
        score = np.clip(record["scores"], epsilon, 1 - epsilon)
        logits = np.log(score / (1 - score))
        calibrated = 1 / (1 + np.exp(-(scale * logits + offset)))
        output.append(record | {"scores": calibrated.astype(np.float32)})
    return output


class OnsetOOFProcess(AbsProcess):
    """Create leakage-safe train OOF and untouched-validation onset scores."""

    def __init__(
        self, cfg: dict, out_dir: Path, seed: int, smoke: bool, resume: bool
    ) -> None:
        self.cfg, self.out_dir, self.seed, self.smoke, self.resume = (
            cfg,
            out_dir,
            seed,
            smoke,
            resume,
        )

    def run(self) -> dict:
        cfg = json.loads(json.dumps(self.cfg))
        if self.smoke:
            cfg["training"]["epochs"] = 2
            cfg["oof"]["folds"] = 2
        out = self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        device = torch.device(cfg["training"]["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA unavailable")
        data, provenance = cached_features(cfg, device, self.smoke)
        folds = cfg["oof"]["folds"]
        assignments = patient_folds(data["train"], folds, cfg["oof"]["seed"])
        json_write(assignments, out / "patient_folds.json")
        fingerprint = hashlib.sha256(
            json.dumps(
                {"cfg": cfg, "provenance": provenance, "assignments": assignments},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        fold_models = []
        oof_predictions = []
        reports = []
        for fold in range(folds):
            heldout = [
                record
                for record in data["train"]
                if assignments[patient_of(record["video"])] == fold
            ]
            training = [
                record
                for record in data["train"]
                if assignments[patient_of(record["video"])] != fold
            ]
            state, pos_weight, report = train_fold(
                cfg,
                training,
                heldout,
                fold,
                cfg["oof"]["seed"] + fold,
                device,
                out,
                fingerprint,
                self.resume,
            )
            model = ONSET_MODELS.build(cfg["model"], d_in=512).to(device)
            model.load_state_dict(state)
            oof_predictions.extend(predict_records(model, heldout, device, pos_weight))
            fold_models.append((state, pos_weight))
            reports.append(report)
        if {record["video"] for record in oof_predictions} != {
            record["video"] for record in data["train"]
        }:
            raise ValueError("OOF predictions do not cover each training video exactly once")
        scale, offset = fit_platt(
            oof_predictions, cfg["oof"]["calibration_max_iter"]
        )
        oof_calibrated = calibrate(oof_predictions, scale, offset)
        val_members: list[list[dict]] = []
        for state, pos_weight in fold_models:
            model = ONSET_MODELS.build(cfg["model"], d_in=512).to(device)
            model.load_state_dict(state)
            val_members.append(predict_records(model, data["val"], device, pos_weight))
        val_predictions = []
        for index, record in enumerate(data["val"]):
            scores = np.mean(
                [member[index]["scores"] for member in val_members], axis=0
            )
            val_predictions.append(
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("features", "target")
                }
                | {"scores": scores}
            )
        val_calibrated = calibrate(val_predictions, scale, offset)
        rows = []
        partitions = [("train_oof", oof_calibrated)]
        if not self.smoke:
            partitions.append(("val_ensemble", val_calibrated))
        for partition, records in partitions:
            for record in records:
                rows.extend(
                    {
                        "video": record["video"],
                        "frame_index": int(frame),
                        "div_score": float(score),
                        "partition": partition,
                    }
                    for frame, score in zip(
                        record["indices"], record["scores"], strict=True
                    )
                )
        frame = pd.DataFrame(rows)
        if frame.duplicated(["video", "frame_index"]).any():
            raise ValueError("duplicate OOF score keys")
        frame.to_csv(out / "event_scores.csv", index=False)
        settings = cfg["evaluation"]
        result = {
            "kind": "patient_oof_train_plus_fold_ensemble_val",
            "seed": self.seed,
            "folds": folds,
            "test_evaluated": False,
            "provenance": provenance,
            "patient_fold_sha256": hashlib.sha256(
                json.dumps(assignments, sort_keys=True).encode()
            ).hexdigest(),
            "fingerprint": fingerprint,
            "calibration": {
                "kind": "monotone_platt_fit_on_oof_train_only",
                "scale": scale,
                "offset": offset,
            },
            "fold_reports": reports,
            "train_oof": {
                "event": event_metrics(
                    oof_calibrated,
                    settings["tolerance"],
                    settings["peak_separation"],
                ),
                "dense": dense_metrics(oof_calibrated),
            },
            "val_ensemble": {
                "event": event_metrics(
                    val_calibrated,
                    settings["tolerance"],
                    settings["peak_separation"],
                ),
                "dense": dense_metrics(val_calibrated),
            },
            "stseg_commit": _commit(ROOT),
            "score_rows": len(frame),
        }
        json_write(cfg, out / "config.resolved.json")
        json_write(result, out / "results.json")
        return result


@BUILDER_REGISTRY.register("onset_oof_cache")
class OnsetOOFBuilder(AbsPipelineBuilder):
    """Registered builder for the patient-OOF score-generation process."""

    def __init__(
        self,
        cfg: dict,
        seed: int | None = None,
        smoke: bool = False,
        resume: bool = True,
    ) -> None:
        super().__init__(cfg)
        self.seed = cfg["experiment"]["seed"] if seed is None else seed
        self.smoke, self.resume = smoke, resume
        suffix = "_smoke" if smoke else ""
        self.out_dir = (
            Path(cfg["experiment"]["out_root"])
            / f"{cfg['experiment']['id']}_seed{self.seed}{suffix}"
        )

    def produce_pre_processor(self) -> None:
        from .onset_config import validate_onset_oof

        validate_onset_oof(self._cfg, "OnsetOOFBuilder")

    def produce_main_processor(self) -> None:
        self._product = OnsetOOFProcess(
            self._cfg, self.out_dir, self.seed, self.smoke, self.resume
        )

    def produce_post_processor(self) -> None:
        pass

    @property
    def product(self) -> AbsProcess:
        self.produce_pre_processor()
        self.produce_main_processor()
        self.produce_post_processor()
        return self._product
