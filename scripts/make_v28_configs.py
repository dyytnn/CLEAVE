#!/usr/bin/env python3
"""Regenerate the three v28 EMFiT seed configs and sweep from the locked base."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/h7/v28"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sweep = []
    for seed in range(3):
        name = f"emfit_src{seed}"
        doc = {
            "inherits": "configs/h7/v28/_base.yaml",
            "cfg": {
                "experiment": {
                    "id": f"v28_{name}",
                    "seed": seed,
                    "tags": ["v28", "emfit", f"seed-{seed}", "trainval-only"],
                }
            },
        }
        header = (
            f"# v28 EMFiT seed {seed}. Published architecture/recipe; only "
            "protocol/head adapt to frozen 16-class grouped data.\n"
        )
        (OUT / f"{name}.yaml").write_text(
            header + yaml.safe_dump(doc, sort_keys=False)
        )
        sweep.append({"config": f"configs/h7/v28/{name}.yaml", "seed": seed})
    (OUT / "sweep.yaml").write_text(
        yaml.safe_dump({"experiments": sweep}, sort_keys=False)
    )


if __name__ == "__main__":
    main()
