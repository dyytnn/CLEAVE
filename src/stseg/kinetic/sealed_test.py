"""One-shot sealed evaluation on the patient-grouped test partition.

The grouped test partition of ``data/splits/nantes_grouped_v1.json`` is the confirmatory partition of CLEAVE.
This module is the only path that evaluates a checkpoint on it, and it may run exactly once: the first forward pass
on test data writes ``results/SEALED_GROUPED_TEST_USED``, and every later attempt is refused. A crash after the
marker is written still counts as a use -- the partition was seen -- which is the intended, deliberately unforgiving
behaviour.

Guards, all of which must pass before any test frame is loaded:

* ``results/SEALED_GROUPED_TEST_USED`` must not exist;
* ``sealed.confirm`` must equal :data:`CONFIRM_PHRASE` verbatim in the YAML;
* ``data.split`` must be the grouped split;
* every entry of ``sealed.models`` must name an existing config and existing checkpoints.

One YAML evaluates several models and seeds together, because a second invocation is impossible by construction.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stseg.data.nantes_kinetic import NUM_CLASSES

from .config import load_pipeline
from .interfaces import AbsPipelineBuilder
from .models import build_model
from .process import KineticTrainProcess, _commit
from .registry import BUILDER_REGISTRY, DATASET_REGISTRY, DECODER_REGISTRY

ROOT = Path(__file__).resolve().parents[3]
MARKER = ROOT / "results/SEALED_GROUPED_TEST_USED"
GROUPED_SPLIT = "data/splits/nantes_grouped_v1.json"
CONFIRM_PHRASE = "I confirm this is the single sealed-test run of nantes_grouped_v1"


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _rel(p: Path) -> str:
    """Repo-relative when possible; absolute otherwise (the marker path is patched in tests)."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def marker_state() -> dict[str, Any] | None:
    """Contents of the sealed-test marker, or ``None`` when the partition is still sealed."""
    if not MARKER.exists():
        return None
    return json.loads(MARKER.read_text())


SEALED_SECTIONS = {"experiment", "data", "sealed"}
SEALED_KEYS = {"confirm", "models", "reason"}


def validate_sealed_config(cfg: dict, where: str = "<config>") -> None:
    """Structural validation only: no marker check, no filesystem access, so tests can validate configs freely."""
    from .config import ConfigError

    unknown = set(cfg) - SEALED_SECTIONS
    if unknown:
        raise ConfigError(f"{where}: unknown cfg section(s) {sorted(unknown)}; allowed {sorted(SEALED_SECTIONS)}")
    for sec in ("data", "sealed"):
        if not isinstance(cfg.get(sec), dict):
            raise ConfigError(f"{where}: cfg.{sec} is required")
    s = cfg["sealed"]
    if set(s) - SEALED_KEYS:
        raise ConfigError(f"{where}: unknown sealed key(s) {sorted(set(s) - SEALED_KEYS)}")
    if s.get("confirm") != CONFIRM_PHRASE:
        raise ConfigError(f"{where}: sealed.confirm must be exactly {CONFIRM_PHRASE!r}")
    split = cfg["data"].get("split")
    if split is None or Path(split).name != Path(GROUPED_SPLIT).name:
        raise ConfigError(f"{where}: sealed test runs on {GROUPED_SPLIT}, got {split!r}")
    models = s.get("models")
    if not isinstance(models, list) or not models:
        raise ConfigError(f"{where}: sealed.models must be a non-empty list")
    for m in models:
        if not isinstance(m, dict) or set(m) - {"name", "config", "checkpoints"} or not {"name", "config", "checkpoints"} <= set(m):
            raise ConfigError(f"{where}: each sealed.models entry needs exactly name, config, checkpoints")
        if not isinstance(m["checkpoints"], list) or not m["checkpoints"]:
            raise ConfigError(f"{where}: {m.get('name')!r}: checkpoints must be a non-empty list")


def check_sealed(cfg: dict) -> list[dict]:
    """Full validation: structure, the marker, and that every named config and checkpoint exists on disk."""
    validate_sealed_config(cfg)
    if MARKER.exists():
        raise RuntimeError(
            f"the grouped test partition has already been used ({_rel(MARKER)}); refusing"
        )
    s = cfg.get("sealed")
    if not isinstance(s, dict):
        raise ValueError("sealed-test config requires a 'sealed' mapping")
    if s.get("confirm") != CONFIRM_PHRASE:
        raise ValueError(f"sealed.confirm must be exactly: {CONFIRM_PHRASE!r}")
    split = cfg.get("data", {}).get("split")
    if split is None or Path(split).name != Path(GROUPED_SPLIT).name:
        raise ValueError(f"sealed test runs on {GROUPED_SPLIT}, got {split!r}")
    models = s.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("sealed.models must be a non-empty list of {name, config, checkpoints}")
    out = []
    for m in models:
        for key in ("name", "config", "checkpoints"):
            if key not in m:
                raise ValueError(f"sealed.models entry missing {key!r}")
        cfg_path = _resolve(m["config"])
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        ckpts = [_resolve(c) for c in m["checkpoints"]]
        if not ckpts:
            raise ValueError(f"{m['name']}: no checkpoints")
        for c in ckpts:
            if not c.exists():
                raise FileNotFoundError(c)
        out.append({"name": str(m["name"]), "config": cfg_path, "checkpoints": ckpts})
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def burn_marker(entries: list[dict], reason: str) -> None:
    """Record the single use. Called immediately before the first test forward pass; never overwrites."""
    if MARKER.exists():
        raise RuntimeError("marker already exists")
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps({
        "version": "sealed_grouped_test_v1",
        "split": GROUPED_SPLIT,
        "split_sha256": _sha256(ROOT / GROUPED_SPLIT),
        "used_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _commit(ROOT),
        "reason": reason,
        "models": [{"name": e["name"], "config": _rel(e["config"]),
                    "checkpoints": [_rel(c) for c in e["checkpoints"]]} for e in entries],
    }, indent=2) + "\n")


class SealedTestProcess:
    """Evaluates the configured checkpoints on the grouped test partition exactly once."""

    def __init__(self, cfg: dict, entries: list[dict], out_path: Path, device: torch.device) -> None:
        self.cfg, self.entries, self.out_path, self.device = cfg, entries, out_path, device

    def run(self) -> dict[str, Any]:
        burn_marker(self.entries, self.cfg.get("sealed", {}).get("reason", "unspecified"))
        results: dict[str, Any] = {
            "version": "sealed_grouped_test_v1", "split": GROUPED_SPLIT,
            "commit": _commit(ROOT), "marker": _rel(MARKER), "models": {},
        }
        for e in self.entries:
            sub = load_pipeline(e["config"])["cfg"]
            seeds = []
            for k, ckpt in enumerate(e["checkpoints"]):
                seed = int(sub.get("experiment", {}).get("seed", k))
                torch.manual_seed(seed)
                np.random.seed(seed)
                data = DATASET_REGISTRY.get(sub["data"].get("dataset", "nantes_clips"), sub["data"], seed, False)
                if not hasattr(data, "test"):
                    raise ValueError(f"{e['name']}: dataset {sub['data'].get('dataset')!r} does not build a test partition")
                model = build_model(dict(sub["model"]), data.in_channels).to(self.device)
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                model.load_state_dict(state["model"], strict=True)
                dec = DECODER_REGISTRY.build(sub.get("decoder", {"name": "viterbi"}))
                log_trans = dec.log_transition(data.train_full.label_sequences(), NUM_CLASSES)
                proc = KineticTrainProcess(sub, data, model, None, None, None, log_trans,
                                           self.out_path.parent, seed, False, self.device, resume=False,
                                           evaluate_test=True)
                seeds.append({"checkpoint": _rel(ckpt), "seed": seed,
                              "metrics": proc.evaluate(data.test)})
                print(f"{e['name']} seed {seed}: p_t={seeds[-1]['metrics'].get('p_t'):.4f}")
            results["models"][e["name"]] = {"n_seeds": len(seeds), "seeds": seeds}
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {_rel(self.out_path)}; the grouped test partition is now spent")
        return results


@BUILDER_REGISTRY.register("sealed_grouped_test")
class SealedTestBuilder(AbsPipelineBuilder):
    """Builds :class:`SealedTestProcess`; every guard runs in ``__init__``, before any data is touched."""

    def __init__(self, cfg: dict, seed: int | None = None, smoke: bool = False, out_dir: str | None = None,
                 resume: bool = True) -> None:
        super().__init__(cfg)
        if smoke:
            raise ValueError("the sealed test has no smoke mode; validate with tests/test_sealed_test.py")
        self.entries = check_sealed(cfg)
        self.out_path = Path(out_dir) / "sealed_grouped_test.json" if out_dir else ROOT / "results/sealed_grouped_test.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def produce_pre_processor(self) -> None:  # data is built per model inside the process
        return None

    def produce_main_processor(self) -> None:
        return None

    def produce_post_processor(self) -> None:
        return None

    @property
    def product(self) -> SealedTestProcess:
        return SealedTestProcess(self._cfg, self.entries, self.out_path, self.device)


__all__ = ["CONFIRM_PHRASE", "MARKER", "SealedTestBuilder", "SealedTestProcess", "check_sealed", "marker_state", "validate_sealed_config"]
