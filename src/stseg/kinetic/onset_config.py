"""Strict, separate schema for the validation-only dense-onset training family."""

from __future__ import annotations

import math


def validate_onset(cfg: dict, where: str) -> None:
    """Reject every unknown field and unsafe data/pretraining mode before building."""
    from .config import ConfigError, _check_keys

    schema = {
        "experiment": {"id": str, "seed": int, "out_root": str},
        "data": {"manifest": str, "split": str, "plane": str, "resize": int, "crop": int, "cache_dir": str},
        "encoder": {"name": str, "weights": str, "feature_cache": str, "batch_size": int, "num_workers": int},
        "model": {"name": str, "hidden": int, "layers": int, "kernel": int, "dropout": (int, float)},
        "target": {"radius": int},
        "training": {"epochs": int, "lr": (int, float), "weight_decay": (int, float), "pos_weight_cap": (int, float), "grad_clip": (int, float), "device": str},
        "evaluation": {"tolerance": int, "peak_separation": int},
    }
    if set(cfg) != set(schema):
        raise ConfigError(f"{where}: onset cfg requires exactly {sorted(schema)}")
    for section, fields in schema.items():
        if not isinstance(cfg[section], dict):
            raise ConfigError(f"{where}: {section} must be a mapping")
        _check_keys(section, cfg[section], {k: (v, True) for k, v in fields.items()}, where)
        for key, typ in fields.items():
            val = cfg[section][key]
            if val is None or isinstance(val, bool):
                raise ConfigError(f"{where}: {section}.{key} cannot be null/bool")
            if isinstance(val, (int, float)) and (not math.isfinite(val) or val < 0):
                raise ConfigError(f"{where}: {section}.{key} must be finite and nonnegative")
    if cfg["encoder"]["name"] != "resnet18" or cfg["model"]["name"] != "dense_tcn":
        raise ConfigError("v24 requires registered resnet18 / dense_tcn")
    for section, keys in {
        "data": ["resize", "crop"], "encoder": ["batch_size"],
        "model": ["hidden", "layers", "kernel"],
        "training": ["epochs", "lr", "pos_weight_cap", "grad_clip"],
    }.items():
        if any(cfg[section][key] <= 0 for key in keys):
            raise ConfigError(f"{where}: {section} positive values required for {keys}")
    if cfg["model"]["kernel"] % 2 != 1 or not 0 <= cfg["model"]["dropout"] < 1:
        raise ConfigError("onset kernel must be odd; dropout must be in [0,1)")
    if cfg["data"]["crop"] > cfg["data"]["resize"] or cfg["target"]["radius"] not in (0, 2):
        raise ConfigError("onset requires crop <= resize and radius 0 or 2")
    if cfg["training"]["pos_weight_cap"] < 1:
        raise ConfigError("pos_weight_cap must be >=1")
    if cfg["training"]["device"] not in ("cpu", "cuda"):
        raise ConfigError("select physical GPU using CUDA_VISIBLE_DEVICES, device cpu|cuda")


def validate_onset_oof(cfg: dict, where: str) -> None:
    """Validate the exact-onset OOF cache builder on top of the v24 schema."""
    from .config import ConfigError, _check_keys

    if "oof" not in cfg:
        raise ConfigError(f"{where}: cfg.oof is required")
    base = {key: value for key, value in cfg.items() if key != "oof"}
    validate_onset(base, where)
    allowed = {
        "folds": (int, True),
        "seed": (int, True),
        "calibration_max_iter": (int, True),
    }
    _check_keys("oof", cfg["oof"], allowed, where)
    if cfg["target"]["radius"] != 0:
        raise ConfigError("OOF cache must use exact-onset targets")
    if cfg["oof"]["folds"] < 2 or cfg["oof"]["calibration_max_iter"] < 1:
        raise ConfigError("OOF folds >=2 and positive calibration_max_iter required")


def validate_onset_decoder(cfg: dict, where: str) -> None:
    """Strict schema for the fixed validation-only event decoder."""
    from .config import ConfigError, _check_keys

    if set(cfg) != {"experiment", "integration"}:
        raise ConfigError(f"{where}: decoder cfg requires experiment and integration")
    _check_keys(
        "experiment",
        cfg["experiment"],
        {"id": (str, True), "seed": (int, True), "out_root": (str, True)},
        where,
    )
    _check_keys(
        "integration",
        cfg["integration"],
        {
            "source_run_template": (str, True),
            "score_run": (str, True),
            "weight": ((int, float), True),
            "floor": ((int, float), True),
            "shuffle_seed": (int, True),
        },
        where,
    )
    integration = cfg["integration"]
    if "{seed}" not in integration["source_run_template"]:
        raise ConfigError("source_run_template must contain {seed}")
    if integration["weight"] != 1.0:
        raise ConfigError("v25 preregisters fixed theoretical weight=1.0")
    if not 0 < integration["floor"] < 0.5:
        raise ConfigError("integration floor must be in (0,0.5)")
