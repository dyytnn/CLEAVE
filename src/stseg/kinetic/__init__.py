"""Kinetic-stage (H7) / temporal pipeline package — registry + factory + builder, configured by YAML.

Importing this package registers every component; ``scripts/run_pipeline.py`` is the entry point.
"""

from . import backbones, builder, datasets, decoders, heads, losses, models, optim  # noqa: F401  (registration side effects)
from . import event_audit  # noqa: F401  (registered validation-only process)
from . import onset_pipeline  # noqa: F401  (patient-safe standalone event training)
from . import onset_oof  # noqa: F401  (patient-level OOF event-score cache)
from . import onset_decoder  # noqa: F401  (fixed validation-only event integration)
from . import cached_video  # noqa: F401  (v26 cached-feature full-video controls)
from . import embryodiff  # noqa: F401  (v27 transparent EmbryoDiff adaptation)
from . import emfit  # noqa: F401  (v28 official-code EMFiT reproduction)
from . import milestone  # noqa: F401  (v29 milestone queries + local refinement)
from . import sce_aux  # noqa: F401  (v30 SCE + one auxiliary training branch)
from . import e2e_video  # noqa: F401  (v33 end-to-end whole-video training)
from . import sealed_test  # noqa: F401  (one-shot grouped sealed-test process)
from .registry import (BACKBONE_REGISTRY, BUILDER_REGISTRY, DATASET_REGISTRY, DECODER_REGISTRY, HEAD_REGISTRY,  # noqa: F401
                       LOSS_REGISTRY, MODEL_REGISTRY, OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY)
