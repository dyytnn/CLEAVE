"""Experiment-config loading: ``inherits`` chain → deep-merge → strict schema validation → registry-name checks.

Fails at load time, not at build time: unknown keys anywhere in ``cfg`` raise (a typo like ``optimzer:`` can no
longer fall back to defaults silently), every ``name`` is checked against its registry, and simple types/ranges are
enforced. The resolved dict is what the builder receives and what the run stores as ``config.resolved.json``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from . import registry as R

ROOT = Path(__file__).resolve().parents[3]

# section -> {key: (type or tuple of types, required)}
SCHEMA: dict[str, dict[str, tuple]] = {
    "event_audit": {"source_run": (str, True), "score_path": (str, True), "weights": (list, False),
                    "radius": (int, False), "floor": ((int, float), False), "clock_bin_hours": ((int, float), False),
                    "clock_prior_strength": ((int, float), False),
                    "include_onset_oracle": (bool, False),
                    "allow_legacy_overlap": (bool, False)},
    "experiment": {"id": (str, True), "seed": (int, False), "out_root": (str, False), "tags": (list, False), "notes": (str, False)},
    "data": {"dataset": (str, True), "manifest": (str, True), "split": (str, True), "plane": (str, False), "planes": ((list, type(None)), False),
             "resize": (int, False), "crop": (int, False), "clip_len": (int, False), "clips_per_video": ((int, str), False),
             "official_offset": (bool, False), "split_mode": (str, False), "merge_last_class": (bool, False), "max_frames_per_video": ((int, type(None)), False), "cache_dir": ((str, type(None)), False),
             "photometric": ((dict, type(None)), False), "instance_norm": (bool, False), "train_label_cutoff": ((str, type(None)), False),
             "plane_mode": ((str, type(None)), False), "eval_plane": ((str, type(None)), False), "rotation": ((dict, type(None)), False),
             "plane_center_weight": ((int, float, type(None)), False), "rare_phase_boost": ((dict, type(None)), False),
             "hires_crop_dir": ((str, type(None)), False), "division_score_path": ((str, type(None)), False),
             "division_score_strict": (bool, False),
             "feature_cache": (str, False), "input_mode": (str, False),
             "sequence_mode": (str, False), "clock_periods_h": (list, False),
             "frame_index_divisor": ((int, float), False),
             "emfit_augmentation": ((dict, type(None)), False),
             "emfit_cache_strict": (bool, False)},
    "model": {"type": (str, False), "backbone": (dict, True), "head": (dict, False), "aux": ((dict, type(None)), False), "num_classes": (int, False), "dropout": ((int, float), False),
              "echo_div_score": (bool, False), "event_conditioning": (dict, False), "embryodiff": (dict, False),
              "emfit": (dict, False), "milestone": (dict, False), "sce_aux": (dict, False), "e2e": (dict, False)},
    "loss": {"name": (str, True), "lam": ((int, float), False), "bonus": ((int, float), False), "sigma": ((int, float), False), "eps": ((int, float), False),
             "truncation": ((int, float), False), "semantic_weight": ((int, float), False),
             "smooth_weight": ((int, float), False), "boundary_weight": ((int, float), False),
             "diffusion_weight": ((int, float), False), "frame_weight": ((int, float), False),
             "milestone_weight": ((int, float), False), "combined_weight": ((int, float), False),
             "target_sigma": ((int, float), False), "combined_smooth_weight": ((int, float), False), "aux": (str, False), "aux_weight": ((int, float), False),
             "pair_offset": (int, False), "margin": ((int, float), False), "window": (int, False), "cap": (int, False)},
    "optimizer": {"name": (str, True), "lr": ((int, float), False), "momentum": ((int, float), False), "weight_decay": ((int, float), False), "betas": (list, False),
                  "backbone_lr_mult": ((int, float), False), "layer_decay": ((int, float), False)},
    "scheduler": {"name": (str, True), "eta_min": ((int, float), False), "warmup_epochs": (int, False), "eta_min_ratio": ((int, float), False),
                  "t0": (int, False), "t_mult": (int, False)},
    "training": {"epochs": (int, False), "batch_size": (int, False), "eval_len": (int, False), "eval_stride": (int, False), "eval_batch_size": (int, False),
                 "num_workers": (int, False), "amp": (bool, False), "monitor": (str, False), "grad_accum": (int, False), "swa_last": (int, False)},
    "decoder": {"name": (str, True)},
    "logging": {"frame_probs": (bool, False), "embeddings": (bool, False), "grad_norms": (bool, False), "confusion": (bool, False),
                "per_phase_loss": (bool, False), "worst_k": (int, False)},
}
COMPONENT_KEYS = {"backbone": {"name": (str, True), "pretrained": (bool, False), "base_channels": (int, False), "init_from": ((str, type(None)), False),
                               "checkpoint": ((str, type(None)), False), "expected_sha256": ((str, type(None)), False),
                               "cnn": (str, False), "arch": (str, False), "nhead": (int, False), "spatial": (bool, False), "sharpness": (bool, False), "grid": (int, False),
                               "low_res_cnn": (str, False), "hires_weights": (str, False)},
                  "head": {"name": (str, True), "hidden": (int, False), "layers": (int, False), "kernel": (int, False), "dropout": ((int, float), False), "nhead": (int, False), "n_substeps": (int, False), "steps": (int, False), "n_decoders": (int, False), "stages": (int, False), "num_classes": (int, False)},
                  "aux": {"name": (str, True), "levels": (int, False)},
                  "event_conditioning": {"hidden": (int, False), "enabled": (bool, False),
                                         "mode": (str, False), "neutral_score": ((int, float), False),
                                         "scale": ((int, float), False)},
                  "embryodiff": {"variant": (str, True), "hidden": (int, False),
                                  "layers": (int, False), "intermediate_layers": (list, False),
                                  "attention_reduction": (int, False), "diffusion_dim": (int, False),
                                  "diffusion_blocks": (int, False), "diffusion_heads": (int, False),
                                  "diffusion_train_steps": (int, False), "inference_steps": (int, False),
                                  "selection_steps": (int, False), "label_scale": ((int, float), False),
                                  "dropout": ((int, float), False), "eval_seed": (int, False)},
                  "emfit": {"vit_pretrained": (bool, False)}}
REGISTRY_OF = {("data", "dataset"): R.DATASET_REGISTRY, ("loss", "name"): R.LOSS_REGISTRY, ("optimizer", "name"): R.OPTIMIZER_REGISTRY,
               ("scheduler", "name"): R.SCHEDULER_REGISTRY, ("decoder", "name"): R.DECODER_REGISTRY, ("model", "type"): R.MODEL_REGISTRY,
               ("model.backbone", "name"): R.BACKBONE_REGISTRY, ("model.head", "name"): R.HEAD_REGISTRY}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge, except that a *component* dict (one carrying ``name``) whose name changes REPLACES the base
    dict instead of merging into it — otherwise ``optimizer: {name: adamw}`` would inherit SGD's ``momentum`` and
    ``head: {name: none}`` would inherit the LSTM's ``hidden`` (the 2026-09-08 sweep failures)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        cur = out.get(k)
        if isinstance(v, dict) and isinstance(cur, dict):
            if "name" in v and "name" in cur and v["name"] != cur["name"]:
                out[k] = copy.deepcopy(v)
            else:
                out[k] = deep_merge(cur, v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_pipeline(path: str | Path) -> dict:
    """Read a pipeline YAML, resolve ``inherits`` (path relative to the repo root or to the file), merge, validate."""
    path = Path(path)
    doc = yaml.safe_load(path.read_text()) or {}
    if "inherits" in doc:
        parent_ref = doc.pop("inherits")
        parent_path = (ROOT / parent_ref) if (ROOT / parent_ref).exists() else (path.parent / parent_ref)
        parent = load_pipeline(parent_path)
        doc = deep_merge(parent, doc)
    validate_pipeline(doc, str(path))
    return doc


def _check_keys(section: str, d: dict, allowed: dict[str, tuple], where: str) -> None:
    unknown = set(d) - set(allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) in '{section}': {sorted(unknown)} (allowed: {sorted(allowed)})")
    for k, (typ, required) in allowed.items():
        if required and k not in d:
            raise ConfigError(f"{where}: '{section}.{k}' is required")
        if k in d and d[k] is not None and not isinstance(d[k], typ):
            raise ConfigError(f"{where}: '{section}.{k}' must be {typ}, got {type(d[k]).__name__}")


def validate_pipeline(doc: dict, where: str = "<config>") -> None:
    if set(doc) - {"builder", "cfg"}:
        raise ConfigError(f"{where}: top level must be exactly {{builder, cfg}}, got {sorted(doc)}")
    if not R.BUILDER_REGISTRY.check(doc.get("builder", "")):
        raise ConfigError(f"{where}: builder {doc.get('builder')!r} not registered; known {R.BUILDER_REGISTRY.names()}")
    cfg = doc["cfg"]
    if doc["builder"] == "dense_onset_training":
        from .onset_config import validate_onset
        validate_onset(cfg, where)
        return
    if doc["builder"] == "onset_oof_cache":
        from .onset_config import validate_onset_oof
        validate_onset_oof(cfg, where)
        return
    if doc["builder"] == "sealed_grouped_test":
        from .sealed_test import validate_sealed_config
        validate_sealed_config(cfg, where)
        return
    if doc["builder"] == "onset_decoder_validation":
        from .onset_config import validate_onset_decoder
        validate_onset_decoder(cfg, where)
        return
    unknown = set(cfg) - set(SCHEMA)
    if unknown:
        raise ConfigError(f"{where}: unknown cfg section(s) {sorted(unknown)}; allowed {sorted(SCHEMA)}")
    for sec in ("experiment", "data", "model", "training"):
        if sec not in cfg:
            raise ConfigError(f"{where}: cfg.{sec} is required")
    for sec, allowed in SCHEMA.items():
        if sec in cfg:
            _check_keys(sec, cfg[sec], allowed, where)
    for comp, allowed in COMPONENT_KEYS.items():
        if cfg["model"].get(comp) is not None:
            _check_keys(f"model.{comp}", cfg["model"][comp], allowed, where)
    for (sec, key), reg in REGISTRY_OF.items():
        node: Any = cfg
        for part in sec.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and key in node and not reg.check(node[key]):
            raise ConfigError(f"{where}: '{sec}.{key}' = {node[key]!r} not registered; known {reg.names()}")
    # component kwargs must match the registered factory/class signature (catches typos and merge leftovers)
    import inspect
    def _check_sig(reg, node: dict, where_key: str, injected: set[str]) -> None:
        target = reg._registry[node["name"]]
        fn = target.__init__ if inspect.isclass(target) else target
        params = inspect.signature(fn).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return
        allowed = set(params) - {"self"} - injected
        extra = set(node) - {"name"} - allowed - injected  # injected keys may also legitimately appear in the YAML (consumed upstream)
        if extra:
            raise ConfigError(f"{where}: '{where_key}' ({node['name']}) got unknown parameter(s) {sorted(extra)}; accepts {sorted(allowed)}")
    _check_sig(R.OPTIMIZER_REGISTRY, cfg.get("optimizer", {"name": "sgd"}), "optimizer", {"params", "backbone_lr_mult", "layer_decay"})  # the two are consumed by the builder (param groups)
    _check_sig(R.SCHEDULER_REGISTRY, cfg.get("scheduler", {"name": "none"}), "scheduler", {"optimizer", "epochs"})
    _check_sig(R.LOSS_REGISTRY, cfg.get("loss", {"name": "ce"}), "loss", {"class_counts", "num_classes"})
    _check_sig(R.HEAD_REGISTRY, cfg["model"].get("head", {"name": "none"}), "model.head", {"d_in"})
    _check_sig(R.BACKBONE_REGISTRY, cfg["model"]["backbone"], "model.backbone", {"in_channels"})
    d = cfg["data"]
    if cfg["model"].get("type") == "emfit":
        from stseg.data.nantes_kinetic import EMFIT_PLANES

        if doc["builder"] != "kinetic_validation_training":
            raise ConfigError(f"{where}: EMFiT reproduction must use validation-only training")
        if d.get("dataset") != "nantes_emfit_frames_trainval":
            raise ConfigError(
                f"{where}: EMFiT requires data.dataset=nantes_emfit_frames_trainval"
            )
        if cfg["model"]["backbone"].get("name") != "emfit_i3d":
            raise ConfigError(f"{where}: EMFiT requires backbone.name=emfit_i3d")
        if cfg["model"].get("head", {"name": "none"}).get("name") != "none":
            raise ConfigError(f"{where}: EMFiT owns its head; model.head must be none")
        if cfg.get("loss", {}).get("name") != "emfit_multitask":
            raise ConfigError(f"{where}: EMFiT requires loss.name=emfit_multitask")
        if d.get("planes") != EMFIT_PLANES:
            raise ConfigError(f"{where}: EMFiT requires the ordered official seven planes")
        divisor = d.get("frame_index_divisor", 415.0)
        if isinstance(divisor, bool) or float(divisor) != 415.0:
            raise ConfigError(f"{where}: official EMFiT frame_index_divisor must be 415")
        if d.get("max_frames_per_video") is not None:
            raise ConfigError(f"{where}: full EMFiT training forbids frame subsampling")
        if not d.get("cache_dir") or not d.get("emfit_cache_strict", False):
            raise ConfigError(f"{where}: v28 requires the declared strict common-protocol cache")
    if cfg["model"].get("type") == "e2e_sce":
        if doc["builder"] != "kinetic_validation_training" or d.get("dataset") != "nantes_video_trainval":
            raise ConfigError(f"{where}: e2e_sce requires builder kinetic_validation_training + data.dataset=nantes_video_trainval")
        if cfg["model"]["backbone"].get("name") != "crossfocal" or cfg["model"].get("head", {"name": "none"}).get("name") != "none":
            raise ConfigError(f"{where}: e2e_sce requires the crossfocal backbone and head none")
        if cfg.get("loss", {}).get("name") != "embryodiff_objective":
            raise ConfigError(f"{where}: e2e_sce requires loss embryodiff_objective (SCE semantic + smoothing terms)")
        e = cfg["model"].get("e2e")
        if not isinstance(e, dict):
            raise ConfigError(f"{where}: e2e_sce requires model.e2e")
        allowed_e2e = {"hidden", "layers", "intermediate_layers", "attention_reduction", "dropout", "chunk",
                       "freeze_backbone", "feature_stats_cache", "init_backbone_from", "init_head_from"}
        if set(e) - allowed_e2e:
            raise ConfigError(f"{where}: unknown model.e2e key(s) {sorted(set(e) - allowed_e2e)}")
        if "chunk" in e and (isinstance(e["chunk"], bool) or int(e["chunk"]) < 1):
            raise ConfigError(f"{where}: model.e2e.chunk must be a positive integer")
        if not d.get("planes") or len(d["planes"]) != 7:
            raise ConfigError(f"{where}: e2e_sce expects the seven focal planes")
        if int(cfg["training"].get("batch_size", 1)) != 1:
            raise ConfigError(f"{where}: whole-video training requires batch_size=1")
    if doc["builder"] == "cached_video_validation_training":
        if d.get("dataset") != "nantes_cached_sequences_trainval":
            raise ConfigError(
                f"{where}: cached_video_validation_training requires "
                "data.dataset=nantes_cached_sequences_trainval"
            )
        if not d.get("feature_cache"):
            raise ConfigError(f"{where}: cached video training requires data.feature_cache")
        if d.get("input_mode", "visual") not in {"visual", "clock", "visual_clock"}:
            raise ConfigError(f"{where}: data.input_mode must be visual|clock|visual_clock")
        if d.get("sequence_mode", "full") not in {"full", "clip"}:
            raise ConfigError(f"{where}: data.sequence_mode must be full|clip")
        periods = d.get("clock_periods_h", [6, 12, 24, 48, 96])
        if not periods or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
            for v in periods
        ):
            raise ConfigError(f"{where}: clock_periods_h must be positive numbers")
        if cfg["model"]["backbone"].get("name") != "cached_features":
            raise ConfigError(f"{where}: cached video training requires cached_features backbone")
        model_type = cfg["model"].get("type")
        head_name = cfg["model"].get("head", {}).get("name")
        loss_name = cfg.get("loss", {}).get("name")
        if model_type == "full_video_mstcn":
            if head_name != "mstcn" or loss_name != "multistage_ce_tmse":
                raise ConfigError(
                    f"{where}: full_video_mstcn requires mstcn + multistage_ce_tmse"
                )
        elif model_type == "embryodiff_adapter":
            if head_name != "none" or loss_name != "embryodiff_objective":
                raise ConfigError(
                    f"{where}: embryodiff_adapter requires head none + embryodiff_objective"
                )
            if d.get("input_mode", "visual") != "visual" or d.get("sequence_mode", "full") != "full":
                raise ConfigError(
                    f"{where}: embryodiff_adapter requires visual complete-video inputs"
                )
            ecfg = cfg["model"].get("embryodiff")
            if not isinstance(ecfg, dict):
                raise ConfigError(f"{where}: embryodiff_adapter requires model.embryodiff")
            if ecfg.get("variant") not in {
                "sce",
                "sce_diffusion",
                "sce_boundary_diffusion",
            }:
                raise ConfigError(f"{where}: invalid EmbryoDiff variant")
            positive = (
                "hidden",
                "layers",
                "attention_reduction",
                "diffusion_dim",
                "diffusion_blocks",
                "diffusion_heads",
                "diffusion_train_steps",
                "inference_steps",
                "selection_steps",
            )
            if any(
                key in ecfg
                and (isinstance(ecfg[key], bool) or int(ecfg[key]) < 1)
                for key in positive
            ):
                raise ConfigError(f"{where}: EmbryoDiff dimensions/steps must be positive")
            taps = ecfg.get("intermediate_layers", [2, 4, 6])
            layers = int(ecfg.get("layers", 6))
            if not taps or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > layers
                for value in taps
            ):
                raise ConfigError(f"{where}: invalid EmbryoDiff intermediate layers")
        elif model_type == "milestone_query_adapter":
            if head_name != "none" or loss_name != "milestone_objective":
                raise ConfigError(
                    f"{where}: milestone_query_adapter requires head none + milestone_objective"
                )
            if d.get("input_mode", "visual") != "visual" or d.get("sequence_mode", "full") != "full":
                raise ConfigError(
                    f"{where}: milestone_query_adapter requires visual complete-video inputs (no clock)"
                )
            mcfg = cfg["model"].get("milestone")
            if not isinstance(mcfg, dict):
                raise ConfigError(f"{where}: milestone_query_adapter requires model.milestone")
            from .milestone import VARIANTS as _MILESTONE_VARIANTS
            if mcfg.get("variant") not in _MILESTONE_VARIANTS:
                raise ConfigError(f"{where}: invalid milestone variant; expected {_MILESTONE_VARIANTS}")
            for key in ("hidden", "layers", "attention_reduction", "query_dim", "query_layers",
                        "query_heads", "local_kernel", "refine_radius"):
                if key in mcfg and (isinstance(mcfg[key], bool) or int(mcfg[key]) < 1):
                    raise ConfigError(f"{where}: model.milestone.{key} must be a positive integer")
            from .milestone import FUSIONS as _FUSIONS, PARAMETERISATIONS as _PARAMS
            if mcfg.get("fusion", "product") not in _FUSIONS:
                raise ConfigError(f"{where}: model.milestone.fusion must be one of {_FUSIONS}")
            if mcfg.get("parameterisation", "onset") not in _PARAMS:
                raise ConfigError(f"{where}: model.milestone.parameterisation must be one of {_PARAMS}")
            if mcfg.get("parameterisation", "onset") == "hazard" and mcfg.get("variant") != "onset_linear":
                raise ConfigError(f"{where}: the hazard parameterisation requires variant onset_linear")
        elif model_type == "sce_aux_adapter":
            if head_name != "none" or loss_name != "sce_aux_objective":
                raise ConfigError(f"{where}: sce_aux_adapter requires head none + sce_aux_objective")
            if d.get("input_mode", "visual") != "visual" or d.get("sequence_mode", "full") != "full":
                raise ConfigError(f"{where}: sce_aux_adapter requires visual complete-video inputs (no clock)")
            acfg = cfg["model"].get("sce_aux")
            if not isinstance(acfg, dict):
                raise ConfigError(f"{where}: sce_aux_adapter requires model.sce_aux")
            from .sce_aux import AUX_VARIANTS as _AUX
            if acfg.get("aux") not in _AUX or cfg["loss"].get("aux") != acfg.get("aux"):
                raise ConfigError(f"{where}: model.sce_aux.aux and loss.aux must both be one of {_AUX} and equal")
        else:
            raise ConfigError(
                f"{where}: cached video training supports full_video_mstcn, embryodiff_adapter, "
                "milestone_query_adapter or sce_aux_adapter"
            )
        if d.get("sequence_mode", "full") == "full" and int(
            cfg["training"].get("batch_size", 1)
        ) != 1:
            raise ConfigError(f"{where}: variable-length full videos require batch_size=1")
    if doc["builder"] == "event_signal_audit":
        import math
        a = cfg.get("event_audit")
        if not isinstance(a, dict):
            raise ConfigError(f"{where}: event_signal_audit requires cfg.event_audit")
        for key in ("source_run", "score_path"):
            if not a.get(key):
                raise ConfigError(f"{where}: event_audit.{key} is required")
        weights = a.get("weights", [0.25, 0.5, 1.0])
        if not isinstance(weights, list) or not weights or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights):
            raise ConfigError(f"{where}: event audit weights must be nonnegative finite numbers")
        radius = a.get("radius", 2)
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
            raise ConfigError(f"{where}: event audit radius must be a nonnegative integer")
        floor, width = a.get("floor", 1e-4), a.get("clock_bin_hours", 2.0)
        if floor is None or width is None or isinstance(floor, bool) or isinstance(width, bool) or not 0 < floor < 0.5 or not math.isfinite(width) or width <= 0:
            raise ConfigError(f"{where}: invalid event audit floor/clock_bin_hours")
        strength = a.get("clock_prior_strength", 10.0)
        if strength is None or isinstance(strength, bool) or not math.isfinite(strength) or strength <= 0:
            raise ConfigError(f"{where}: clock_prior_strength must be positive and finite")
        if "allow_legacy_overlap" in a and not isinstance(a["allow_legacy_overlap"], bool):
            raise ConfigError(f"{where}: allow_legacy_overlap must be boolean")
        if "include_onset_oracle" in a and not isinstance(a["include_onset_oracle"], bool):
            raise ConfigError(f"{where}: include_onset_oracle must be boolean")
    elif "event_audit" in cfg:
        raise ConfigError(f"{where}: cfg.event_audit requires builder=event_signal_audit")
    ec = cfg["model"].get("event_conditioning")
    if "event_conditioning" in cfg["model"] and not isinstance(ec, dict):
        raise ConfigError(f"{where}: model.event_conditioning must be a mapping")
    if cfg["model"].get("type") == "event_conditioned":
        if not d.get("division_score_path") or not d.get("division_score_strict"):
            raise ConfigError(f"{where}: event_conditioned requires a strict division-score cache")
        if d.get("instance_norm") or d.get("hires_crop_dir") or cfg["model"]["backbone"]["name"] == "dualbranch":
            raise ConfigError(f"{where}: event_conditioned forbids instance_norm/hires/dualbranch")
        if cfg["model"].get("echo_div_score") or cfg["model"].get("aux"):
            raise ConfigError(f"{where}: event_conditioned isolates conditioning from echo/aux losses")
        ec = ec or {}
        import math
        if ec.get("mode", "predicted") not in {"predicted", "constant"}:
            raise ConfigError(f"{where}: event mode must be predicted or constant")
        if not isinstance(ec.get("hidden", 32), int) or isinstance(ec.get("hidden", 32), bool) or ec.get("hidden", 32) < 1:
            raise ConfigError(f"{where}: event hidden must be a positive integer")
        for key, default in (("scale", 1.0), ("neutral_score", 0.5)):
            value = ec.get(key, default)
            if value is None or isinstance(value, bool) or not math.isfinite(value) or value < 0 or (key == "neutral_score" and value > 1):
                raise ConfigError(f"{where}: invalid event {key}")
        for key in ("enabled", "mode"):
            if key in ec and ec[key] is None:
                raise ConfigError(f"{where}: event {key} cannot be null")
    elif ec is not None:
        raise ConfigError(f"{where}: event_conditioning requires model.type=event_conditioned")
    if d.get("dataset") == "nantes_clips" and int(d.get("clip_len", 4)) < 2:
        raise ConfigError(f"{where}: clip_len must be >= 2 for nantes_clips")
    if d.get("planes") and cfg["model"].get("type", "seq_kinetic") == "r2plus1d":
        raise ConfigError(f"{where}: r2plus1d takes 3 replicated channels; multi-plane input is not supported for it")
    if "crop" in d and "resize" in d and d["crop"] > d["resize"]:
        raise ConfigError(f"{where}: data.crop ({d['crop']}) > data.resize ({d['resize']})")
