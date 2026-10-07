"""Patient-safe dense event teacher; frozen public ImageNet features, train/val only.

This standalone family does not use the phase trainer, which evaluates test at
completion. No test images/predictions or legacy division-score caches are loaded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .datasets import _frames
from .event_audit import _sha256, patient_overlap
from .interfaces import AbsPipelineBuilder, AbsProcess
from .onset_metrics import dense_metrics, event_metrics, onset_targets
from .process import _commit
from .registry import BACKBONE_REGISTRY, BUILDER_REGISTRY, HEAD_REGISTRY, Registry

LOGGER = logging.getLogger(__name__)
ONSET_MODELS = Registry("feature_onset_model")


class AbsOnsetModel(nn.Module, ABC):
    """Explicit temporal feature contract (B,T,D) -> (B,T), unlike image models."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...


@ONSET_MODELS.register("dense_tcn")
class DenseOnsetTCN(AbsOnsetModel):
    """Whole contiguous sequence model; frozen features never mix time into batch."""

    def __init__(self, d_in: int, hidden: int, layers: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.head = HEAD_REGISTRY.get("tcn", d_in=d_in, hidden=hidden, layers=layers, kernel=kernel, dropout=dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map (B,T,D) cached visual features to (B,T) binary event logits."""
        if features.ndim != 3:
            raise ValueError("onset model requires (B,T,D)")
        return self.classifier(self.head(self.norm(features))).squeeze(-1)


def atomic_torch_save(value: dict, path: Path) -> None:
    """Replace only this run's own generated checkpoint, never another run."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def json_write(value: dict, path: Path) -> None:
    """Atomically persist JSON with no NaN/Inf extension."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed: int) -> None:
    """Seed every RNG; deterministic kernels, no cuDNN autotuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def contiguous_logits(model: AbsOnsetModel, features: torch.Tensor, indices: np.ndarray) -> torch.Tensor:
    """Run each contiguous (T,D) interval independently; no context across gaps."""
    cuts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1, len(indices)]
    return torch.cat([model(features[a:b][None])[0] for a, b in zip(cuts[:-1], cuts[1:], strict=True)])


def _code_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), Path(__file__).with_name("onset_metrics.py"), Path(__file__).with_name("onset_config.py"),
             Path(__file__).with_name("heads.py"), Path(__file__).with_name("backbones.py"), root / "data/nantes_kinetic.py"]
    return {str(p.relative_to(root)): _sha256(p) for p in paths}


def cached_features(cfg: dict, device: torch.device, smoke: bool) -> tuple[dict, dict]:
    """Extract train and val only, or <=20 TRAIN frames for smoke.

    Caches are keyed by preprocessing, source weights, manifest, split and source
    code hashes. Completed per-video artifacts permit interrupted extraction resume.
    """
    d, enc = cfg["data"], cfg["encoder"]
    overlaps = patient_overlap(Path(d["split"]))
    if any(overlaps.values()):
        raise ValueError(f"patient leakage in split (including unlabelled videos): {overlaps}")
    weights = Path(enc["weights"]).expanduser()
    # Fixed public upstream artifact: never silently accept embryo-finetuned weights.
    digest = _sha256(weights)
    if not digest.startswith("f37072fd"):
        raise ValueError("v24 requires torchvision ImageNet ResNet18 resnet18-f37072fd.pth")
    provenance = {"data": d, "weights_sha256": digest, "manifest_sha256": _sha256(Path(d["manifest"])),
                  "split_sha256": _sha256(Path(d["split"])), "code": _code_hashes(), "smoke": smoke,
                  "patient_overlap": overlaps, "partitions_loaded": ["train"] if smoke else ["train", "val"]}
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = Path(enc["feature_cache"]) / key
    cache.mkdir(parents=True, exist_ok=True)
    json_write(provenance, cache / "provenance.json")
    encoder = None
    outputs = {}
    for partition in provenance["partitions_loaded"]:
        ds = _frames(d, partition, "eval", 0)
        if smoke:
            selected_groups = []
            selected_patients = set()
            for _, group in ds.rows.groupby("video", sort=False):
                exact, valid = onset_targets(group.label.to_numpy(), group.frame_index.to_numpy())
                points = np.flatnonzero((exact > 0) & valid)
                patient = str(group.video.iloc[0]).rsplit("-", 1)[0]
                if len(points) and patient not in selected_patients:
                    start = max(0, int(points[0]) - 5)
                    selected_groups.append(group.iloc[start : start + 10])
                    selected_patients.add(patient)
                if len(selected_groups) == 2:
                    break
            if len(selected_groups) < 2:
                raise ValueError("smoke requires observed onsets from two patients")
            import pandas as pd

            ds.rows = pd.concat(selected_groups, ignore_index=True)
        records = []
        for v, group in ds.rows.groupby("video", sort=False):
            target = cache / f"{partition}_{v}.pt"
            indices = group.frame_index.to_numpy()
            labels = group.label.to_numpy()
            if target.exists():
                saved = torch.load(target, map_location="cpu", weights_only=True)
                if not torch.equal(saved["indices"], torch.tensor(indices)) or not torch.equal(saved["labels"], torch.tensor(labels)):
                    raise ValueError(f"cache alignment mismatch: {target}")
                features = saved["features"]
            else:
                if encoder is None:
                    # Isolate random classifier-stripping construction from training RNG.
                    with torch.random.fork_rng(devices=[]):
                        encoder = BACKBONE_REGISTRY.get(enc["name"], pretrained=False, init_from=str(weights), in_channels=3).eval().to(device)
                    encoder.requires_grad_(False)
                # One whole video per cache object; bounded image batch memory.
                subset = torch.utils.data.Subset(ds, group.index.tolist())
                loader = DataLoader(subset, batch_size=enc["batch_size"], num_workers=enc["num_workers"], shuffle=False)
                batches = []
                with torch.inference_mode():
                    for batch in loader:
                        batches.append(encoder(batch["image"].to(device)).cpu())
                features = torch.cat(batches)
                atomic_torch_save({"features": features, "indices": torch.tensor(indices), "labels": torch.tensor(labels)}, target)
            if features.shape != (len(group), 512) or not torch.isfinite(features).all():
                raise ValueError(f"invalid frozen feature cache: {target}")
            exact, valid = onset_targets(labels, indices)
            train_y, _ = onset_targets(labels, indices, cfg["target"]["radius"])
            records.append({"video": v, "features": features, "indices": indices, "onset": exact,
                            "valid": valid, "target": train_y})
            if len(records) % 20 == 0 or smoke:
                LOGGER.warning("frozen features %s videos=%d frames=%d", partition, len(records), sum(len(r["indices"]) for r in records))
        outputs[partition] = records
    if smoke:
        # NOT validation: same <=20 train frames exercise the entire pipeline only.
        outputs["val"] = outputs["train"]
    return outputs, {**provenance, "feature_cache_key": key}


class DenseOnsetProcess(AbsProcess):
    """Train one video per update with natural dense negatives; select on val AP."""

    def __init__(self, cfg: dict, out_dir: Path, seed: int, smoke: bool, resume: bool) -> None:
        self.cfg, self.out_dir, self.seed, self.smoke, self.resume = cfg, out_dir, seed, smoke, resume

    def run(self) -> dict:
        cfg, out = self.cfg, self.out_dir
        out.mkdir(parents=True, exist_ok=True)
        device = torch.device(cfg["training"]["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA unavailable; refusing silent CPU full training")
        seed_all(self.seed)
        data, provenance = cached_features(cfg, device, self.smoke)
        # Feature extraction loader RNG consumption cannot affect model init.
        seed_all(self.seed)
        model = ONSET_MODELS.build(cfg["model"], d_in=512).to(device)
        training = cfg["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
        count = sum(float(r["target"][r["valid"]].sum()) for r in data["train"])
        nvalid = sum(int(r["valid"].sum()) for r in data["train"])
        if count == 0 or count == nvalid:
            raise ValueError("training requires both positive and negative observed frames")
        pos_weight = min(training["pos_weight_cap"], (nvalid - count) / count)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device), reduction="none")
        identity = {"cfg": cfg, "seed": self.seed, "provenance": provenance}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        start, best, history = 0, -1.0, []
        last = out / "last.pt"
        if last.exists() and not self.resume:
            raise FileExistsError("refusing to overwrite an existing run with --no-resume")
        if last.exists():
            state = torch.load(last, map_location=device, weights_only=True)
            if state["fingerprint"] != fingerprint:
                raise ValueError("resume config/data/code mismatch; use a new run id")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start, best, history = state["epoch"] + 1, state["best"], state["history"]
            torch.set_rng_state(state["rng_cpu"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state(state["rng_cuda"].cpu(), device)
        json_write(cfg, out / "config.resolved.json")
        json_write({**identity, "fingerprint": fingerprint, "commit": _commit(Path.cwd()), "torch": torch.__version__,
                    "pos_weight": pos_weight, "prior_correction": "logits - log(pos_weight)",
                    "n_train_valid_frames": nvalid, "n_train_positive_frames": count,
                    "smoke_train_only": self.smoke, "test_evaluated": False}, out / "meta.json")
        # Frozen features fit in RAM; only each video is copied to the GPU.
        epochs = 2 if self.smoke else training["epochs"]
        for epoch in range(start, epochs):
            model.train()
            losses = []
            order = np.random.default_rng(self.seed + epoch).permutation(len(data["train"]))
            for i in order:
                r = data["train"][i]
                valid = torch.tensor(r["valid"], device=device)
                if not valid.any():
                    continue
                logits = contiguous_logits(model, r["features"].to(device), r["indices"])
                y = torch.tensor(r["target"], device=device)
                loss = criterion(logits, y)[valid].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite onset loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            predictions = []
            with torch.inference_mode():
                for r in data["val"]:
                    logits = contiguous_logits(model, r["features"].to(device), r["indices"])
                    q = torch.sigmoid(logits - math.log(pos_weight)).cpu().numpy()
                    predictions.append({k: v for k, v in r.items() if k not in ("features", "target") } | {"scores": q})
            settings = cfg["evaluation"]
            metrics = {f"event_at_{tol}": event_metrics(predictions, tol, settings["peak_separation"])
                       for tol in sorted({0, 1, settings["tolerance"]})}
            metrics["dense_onset"] = dense_metrics(predictions)
            metrics["epoch"] = epoch
            metrics["train_loss"] = float(np.mean(losses))
            score = metrics[f"event_at_{settings['tolerance']}"]["event_ap"]
            if score is None:
                raise ValueError("selection partition contains no observed onsets")
            history.append(metrics)
            improved = score > best
            best = max(best, score)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "best": best, "history": history, "fingerprint": fingerprint,
                     "rng_cpu": torch.get_rng_state(),
                     "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            if improved:
                atomic_torch_save(state, out / "best.pt")
                json_write(metrics, out / "best_val.json")
                # Lists only: safe torch serialization and exact alignment metadata.
                atomic_torch_save({"records": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in predictions]}, out / "val_predictions.pt")
            atomic_torch_save(state, last)
            json_write({"epochs": history}, out / "history.json")
            LOGGER.warning("%s seed=%d epoch=%d/%d loss=%.6f val_event_AP=%.6f best=%.6f", cfg["experiment"]["id"], self.seed, epoch + 1, epochs, metrics["train_loss"], score, best)
        result = {"split": cfg["data"]["split"], "partition": "smoke_train_subset" if self.smoke else "val",
                  "seed": self.seed, "n_seeds": 1, "test_evaluated": False,
                  "best": json.loads((out / "best_val.json").read_text()), "completed_epochs": epochs,
                  "target_radius": cfg["target"]["radius"], "fingerprint": fingerprint,
                  "note": "Standalone teacher screen; not downstream p_t or SOTA; scores are not OOF training features."}
        json_write(result, out / "results.json")
        return result


@BUILDER_REGISTRY.register("dense_onset_training")
class DenseOnsetBuilder(AbsPipelineBuilder):
    """Assemble the separate feature/teacher process with resume and smoke isolation."""

    def __init__(self, cfg: dict, seed: int | None = None, smoke: bool = False, resume: bool = True) -> None:
        super().__init__(cfg)
        self.seed = cfg["experiment"]["seed"] if seed is None else seed
        self.smoke, self.resume = smoke, resume
        tag = "_smoke" if smoke else ""
        self.out_dir = Path(cfg["experiment"]["out_root"]) / f"{cfg['experiment']['id']}_seed{self.seed}{tag}"

    def produce_pre_processor(self) -> None:
        from .onset_config import validate_onset
        validate_onset(self._cfg, "DenseOnsetBuilder")

    def produce_main_processor(self) -> None:
        self._product = DenseOnsetProcess(self._cfg, self.out_dir, self.seed, self.smoke, self.resume)

    def produce_post_processor(self) -> None:
        pass

    @property
    def product(self) -> AbsProcess:
        self.produce_pre_processor()
        self.produce_main_processor()
        self.produce_post_processor()
        return self._product
