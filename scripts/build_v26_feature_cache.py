#!/usr/bin/env python3
"""Cache train/validation backbone features from a matched v25 phase seed.

Only deterministic evaluation transforms are used. The script refuses to cache
test data and records hashes for the selected checkpoint, resolved source config,
frozen split, manifest and cache implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import stseg.kinetic  # noqa: E402,F401 - local src path + registry side effects
from stseg.kinetic.datasets import NantesClipsTrainValProducer  # noqa: E402
from stseg.kinetic.interfaces import AbsFrameBackbone  # noqa: E402
from stseg.kinetic.models import build_model  # noqa: E402


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    """Resolve repository-relative paths."""
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


@torch.no_grad()
def extract_partition(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, np.ndarray]:
    """Extract one array bundle per video from a frame dataset."""
    model.eval()
    backbone = cast(AbsFrameBackbone, model.backbone)
    amp = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=amp,
        persistent_workers=workers > 0,
    )
    features = np.empty((len(dataset), backbone.feat_dim), dtype=np.float16)
    cursor = 0
    for number, batch in enumerate(loader, start=1):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            output = backbone(batch["image"].to(device, non_blocking=True))
        values = output.float().cpu().numpy().astype(np.float16)
        features[cursor : cursor + len(values)] = values
        cursor += len(values)
        if number % 100 == 0 or cursor == len(dataset):
            print(f"  cached {cursor}/{len(dataset)} frames", flush=True)
    if cursor != len(dataset):
        raise RuntimeError(f"feature cache stopped at {cursor}/{len(dataset)} frames")

    arrays: dict[str, np.ndarray] = {}
    for video, group in dataset.rows.groupby("video", sort=False):
        indices = group.index.to_numpy()
        arrays[f"{video}__emb"] = features[indices]
        arrays[f"{video}__y"] = group.label.to_numpy(dtype=np.int16)
        arrays[f"{video}__t"] = group.time_h.to_numpy(dtype=np.float32)
        arrays[f"{video}__frame"] = group.frame_index.to_numpy(dtype=np.int32)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source_run = resolve(args.source_run)
    out_dir = resolve(args.out)
    config_path = source_run / "config.resolved.json"
    results_path = source_run / "results.json"
    if not config_path.exists() or not results_path.exists():
        raise FileNotFoundError("source run must contain config.resolved.json and results.json")
    cfg = json.loads(config_path.read_text())
    result = json.loads(results_path.read_text())
    if result.get("test_evaluated") is not False:
        raise ValueError("v26 feature source must be a train/validation-only run")
    data_cfg = dict(cfg["data"])
    if data_cfg.get("dataset") != "nantes_clips_trainval":
        raise ValueError("source run must use nantes_clips_trainval")
    split_path = resolve(data_cfg["split"])
    manifest_path = resolve(data_cfg["manifest"])
    split = json.loads(split_path.read_text())
    partitions = ("train", "val")
    patient_sets = [set(split["patients"][name]) for name in partitions]
    if patient_sets[0] & patient_sets[1]:
        raise ValueError("patient leakage between cached train and validation")

    weight_name = "swa.pt" if result.get("final_weights") == "swa" else "best.pt"
    checkpoint_path = source_run / weight_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    source_meta_path = source_run / "meta.json"
    source_meta = json.loads(source_meta_path.read_text()) if source_meta_path.exists() else {}
    seed = int(source_meta.get("seed", cfg.get("experiment", {}).get("seed", 0)))
    data = NantesClipsTrainValProducer(data_cfg, seed=seed, smoke=False)
    model = build_model(cfg["model"], data.in_channels)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    model.to(device).eval()
    backbone = cast(AbsFrameBackbone, model.backbone)
    print(
        f"source={source_run} weights={weight_name} device={device} "
        f"feature_dim={backbone.feat_dim}",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "test.npz").exists():
        raise ValueError("refusing output directory that contains test.npz")
    for name, dataset in (("train", data.train_full), ("val", data.val)):
        expected = set(split["videos"][name])
        if set(dataset.videos()) != expected:
            raise ValueError(f"{name} dataset membership differs from frozen split")
        print(f"extracting {name}: {len(dataset)} frames / {len(expected)} videos")
        arrays = extract_partition(model, dataset, device, args.batch_size, args.workers)
        pending = out_dir / f"{name}.pending.npz"
        np.savez_compressed(pending, **arrays)
        pending.replace(out_dir / f"{name}.npz")

    implementation_files = [
        Path(__file__),
        ROOT / "src/stseg/kinetic/cached_video.py",
        ROOT / "src/stseg/kinetic/models.py",
        ROOT / "src/stseg/kinetic/losses.py",
    ]
    implementation_digest = hashlib.sha256(
        "".join(sha256(path) for path in implementation_files).encode()
    ).hexdigest()
    meta = {
        "version": "v26_cached_features_v1",
        "source_run": str(source_run),
        "source_weights": weight_name,
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_config_sha256": sha256(config_path),
        "split": str(split_path),
        "split_sha256": sha256(split_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "implementation_sha256": implementation_digest,
        "partitions": ["train", "val"],
        "test_cached": False,
        "feature_dim": int(backbone.feat_dim),
        "seed": seed,
        "videos": {name: len(split["videos"][name]) for name in partitions},
        "patients": {name: len(split["patients"][name]) for name in partitions},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote train/val-only feature cache to {out_dir}")


if __name__ == "__main__":
    main()
