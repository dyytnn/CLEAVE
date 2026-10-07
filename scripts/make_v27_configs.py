#!/usr/bin/env python3
"""Generate the three-arm, three-seed TEMPO v27 EmbryoDiff grid."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs/h7/v27"

PARENTS = {
    "sce": "configs/h7/v27/_base.yaml",
    "sce_diffusion": "configs/h7/v27/_sce_diffusion.yaml",
    "sce_boundary_diffusion": "configs/h7/v27/_sce_boundary_diffusion.yaml",
}

HYPOTHESES = {
    "sce": "discriminative six-layer semantic condition encoder",
    "sce_diffusion": "label-space diffusion adds a sequence prior beyond SCE",
    "sce_boundary_diffusion": "boundary conditioning improves transition localization",
}


def main() -> None:
    """Write deterministic seed configs and the launch manifest."""
    OUT.mkdir(parents=True, exist_ok=True)
    sweep = []
    for seed in range(3):
        for arm, parent in PARENTS.items():
            path = OUT / f"{arm}_src{seed}.yaml"
            doc = {
                "inherits": parent,
                "cfg": {
                    "experiment": {
                        "id": f"v27_{arm}_src{seed}",
                        "seed": seed,
                        "tags": [
                            "v27",
                            arm,
                            f"matched-source-seed-{seed}",
                            "trainval-only",
                        ],
                        "notes": HYPOTHESES[arm],
                    },
                    "data": {
                        "feature_cache": f"runs/h7/v26/features_seed{seed}",
                    },
                    "model": {"embryodiff": {"eval_seed": 27000 + seed}},
                },
            }
            header = (
                f"# v27 {arm}, matched source seed {seed}. Single scientific "
                f"contrast: {HYPOTHESES[arm]}.\n"
            )
            path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
            sweep.append({"config": str(path.relative_to(ROOT)), "seed": seed})
    (OUT / "sweep.yaml").write_text(yaml.safe_dump({"experiments": sweep}, sort_keys=False))
    print(f"wrote {len(sweep)} configs and {OUT / 'sweep.yaml'}")


if __name__ == "__main__":
    main()
