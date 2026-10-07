#!/usr/bin/env python3
"""Run one experiment YAML (or a sweep list) through the registered pipeline builder — like NST_segmentation/main.py.
Configs may `inherits:` a base file; they are validated strictly (unknown keys / unregistered names fail before anything runs);
an interrupted run resumes from runs/.../last.pt unless --no-resume.

    PYTHONPATH=src python scripts/run_pipeline.py --pipeline_config configs/h7/resnet18_lstm_L4_split0.yaml [--seed 0] [--smoke]
    PYTHONPATH=src python scripts/run_pipeline.py --sweep configs/h7/sweep_v1.yaml          # sequential, skips finished runs
    PYTHONPATH=src python scripts/run_pipeline.py --list                                     # show every registered component
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import stseg.kinetic as K
from stseg.kinetic.config import ConfigError, load_pipeline

ROOT = Path(__file__).resolve().parents[1]


def run_one(path: Path, seed: int | None, smoke: bool, resume: bool = True, validate_only: bool = False) -> None:
    pipe = load_pipeline(path)  # resolves `inherits`, rejects unknown keys / unregistered names
    if validate_only:
        print(f"OK {path}"); return
    builder = K.BUILDER_REGISTRY.get(pipe["builder"], cfg=pipe["cfg"], seed=seed, smoke=smoke, resume=resume)
    if (builder.out_dir / "results.json").exists() and not smoke:
        print(f"skip {path.name}: {builder.out_dir}/results.json exists"); return
    print(f"=== {path.name} -> {builder.out_dir}")
    builder.product.run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_config")
    ap.add_argument("--sweep", help="YAML with a list under 'experiments': paths, or {config: path, seed: k} entries")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing last.pt and start from scratch")
    ap.add_argument("--validate-only", action="store_true", help="load + validate the config(s) without running")
    a = ap.parse_args()
    if a.list:
        for r in (K.BUILDER_REGISTRY, K.DATASET_REGISTRY, K.MODEL_REGISTRY, K.BACKBONE_REGISTRY, K.HEAD_REGISTRY, K.LOSS_REGISTRY, K.DECODER_REGISTRY, K.OPTIMIZER_REGISTRY, K.SCHEDULER_REGISTRY):
            print(r)
        return
    if a.sweep:
        for entry in yaml.safe_load(Path(a.sweep).read_text())["experiments"]:
            rel, seed = (entry, a.seed) if isinstance(entry, str) else (entry["config"], entry.get("seed", a.seed))
            try:
                run_one(ROOT / rel, seed, a.smoke, not a.no_resume, a.validate_only)
            except (ConfigError, Exception) as e:  # keep the sweep going, report at the end
                print(f"FAILED {rel}: {type(e).__name__}: {e}", flush=True)
        return
    run_one(Path(a.pipeline_config), a.seed, a.smoke, not a.no_resume, a.validate_only)


if __name__ == "__main__":
    main()
