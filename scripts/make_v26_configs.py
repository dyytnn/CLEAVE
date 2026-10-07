#!/usr/bin/env python3
"""Generate the four-arm, three-seed TEMPO v26 cached-video experiment grid."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/h7/v26"

ARMS = {
    "clip16_visual": {
        "input_mode": "visual",
        "sequence_mode": "clip",
        "batch_size": 28,
        "hypothesis": "controlled L16 frozen-feature parent",
    },
    "full_visual": {
        "input_mode": "visual",
        "sequence_mode": "full",
        "batch_size": 1,
        "hypothesis": "whole-video training improves over matched L16 windows",
    },
    "full_clock": {
        "input_mode": "clock",
        "sequence_mode": "full",
        "batch_size": 1,
        "hypothesis": "absolute acquisition time quantifies clock-only predictability",
    },
    "full_visual_clock": {
        "input_mode": "visual_clock",
        "sequence_mode": "full",
        "batch_size": 1,
        "hypothesis": "clock adds information beyond frozen visual full-video features",
    },
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sweep = []
    for seed in range(3):
        for arm, spec in ARMS.items():
            path = OUT / f"{arm}_src{seed}.yaml"
            doc = {
                "inherits": "configs/h7/v26/_base.yaml",
                "cfg": {
                    "experiment": {
                        "id": f"v26_{arm}_src{seed}",
                        "seed": seed,
                        "tags": [
                            "v26",
                            arm,
                            f"matched-source-seed-{seed}",
                            "trainval-only",
                        ],
                        "notes": spec["hypothesis"],
                    },
                    "data": {
                        "feature_cache": f"runs/h7/v26/features_seed{seed}",
                        "input_mode": spec["input_mode"],
                        "sequence_mode": spec["sequence_mode"],
                    },
                    "training": {"batch_size": spec["batch_size"]},
                },
            }
            header = (
                f"# v26 {arm}, matched source seed {seed}. Single scientific "
                f"contrast: {spec['hypothesis']}.\n"
            )
            path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
            sweep.append({"config": str(path.relative_to(ROOT)), "seed": seed})
    (OUT / "sweep.yaml").write_text(yaml.safe_dump({"experiments": sweep}, sort_keys=False))
    print(f"wrote {len(sweep)} configs and {OUT / 'sweep.yaml'}")


if __name__ == "__main__":
    main()
