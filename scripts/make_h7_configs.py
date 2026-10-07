#!/usr/bin/env python3
"""Generate configs/h7/: one `_base.yaml` (the official ResNet-LSTM protocol) + one small override YAML per experiment
(`inherits: configs/h7/_base.yaml`) + `sweep_v1.yaml`. Re-run after editing EXPERIMENTS; the files are committed so the
sweep is reproducible from git alone.

    python scripts/make_h7_configs.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parents[1] / "configs/h7"

BASE = {
    "builder": "kinetic_training",
    "cfg": {
        "experiment": {"id": "_base", "seed": 0, "out_root": "runs/h7", "tags": []},
        "data": {"dataset": "nantes_clips", "manifest": "data/derived/nantes_manifest_F0.csv", "split": "data/splits/nantes_official/split0.csv",
                 "plane": "embryo_dataset", "planes": None, "resize": 250, "crop": 224, "clip_len": 4, "clips_per_video": "auto", "official_offset": False,
                 "cache_dir": "data/cache/nantes_250"},  # pre-resized JPEGs on NVMe (scripts/build_nantes_cache.py); HDD starves the GPU
        "model": {"type": "seq_kinetic", "backbone": {"name": "resnet18", "pretrained": True}, "head": {"name": "lstm", "hidden": 1024, "layers": 2},
                  "num_classes": 16, "dropout": 0.5},
        "loss": {"name": "ce"},
        "optimizer": {"name": "sgd", "lr": 0.001, "momentum": 0.9, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "training": {"epochs": 10, "batch_size": 10, "eval_len": 150, "eval_batch_size": 150, "num_workers": 8, "amp": True, "monitor": "p_v"},
        "decoder": {"name": "viterbi"},
    },
}

ADAMW = {"name": "adamw", "lr": 0.0001, "weight_decay": 0.05}
# id -> (tags, override dict merged onto BASE["cfg"])
EXPERIMENTS: dict[str, tuple[list[str], dict]] = {
    "resnet18_lstm_L4_split0": (["official", "reference"], {}),
    "resnet18_lstm_L4_grouped_v1": (["official", "leakage_free_split"], {"data": {"split": "data/splits/nantes_grouped_v1.json"}}),
    "resnet18_none_split0": (["per_frame_reference"], {"data": {"dataset": "nantes_frames"}, "model": {"head": {"name": "none"}}, "training": {"batch_size": 40}}),
    "resnet18_lstm_L16_split0": (["clip_len"], {"data": {"clip_len": 16}, "training": {"batch_size": 4}}),
    "resnet18_gru_L8_split0": (["head"], {"data": {"clip_len": 8}, "training": {"batch_size": 8}, "model": {"head": {"name": "gru", "hidden": 1024, "layers": 2}}}),
    "resnet18_tcn_L16_split0": (["head"], {"data": {"clip_len": 16}, "training": {"batch_size": 4}, "model": {"head": {"name": "tcn", "hidden": 512, "layers": 4}}}),
    "resnet18_transformer_L16_split0": (["head"], {"data": {"clip_len": 16}, "training": {"batch_size": 4}, "model": {"head": {"name": "transformer", "hidden": 512, "layers": 2}}}),
    "resnet18_lstm_L4_multifocal3_split0": (["protocol", "multifocal"], {"data": {"planes": ["embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15"]}}),
    "resnet18_lstm_L4_ceweighted_split0": (["loss"], {"loss": {"name": "ce_weighted"}}),
    "resnet18_lstm_L4_ordinal_split0": (["loss"], {"loss": {"name": "ce_ordinal", "lam": 0.1}}),
    "resnet50_lstm_L8_split0": (["backbone"], {"data": {"clip_len": 8}, "training": {"batch_size": 6}, "model": {"backbone": {"name": "resnet50", "pretrained": True}}}),
    "convnext_tiny_lstm_L8_split0": (["backbone"], {"data": {"clip_len": 8}, "training": {"batch_size": 6}, "model": {"backbone": {"name": "convnext_tiny", "pretrained": True}}, "optimizer": ADAMW}),
    "swin_t_lstm_L8_split0": (["backbone"], {"data": {"clip_len": 8}, "training": {"batch_size": 6}, "model": {"backbone": {"name": "swin_t", "pretrained": True}}, "optimizer": ADAMW}),
    "efficientnet_b0_lstm_L8_split0": (["backbone", "compact"], {"data": {"clip_len": 8}, "training": {"batch_size": 8}, "model": {"backbone": {"name": "efficientnet_b0", "pretrained": True}}}),
    "r2plus1d_L8_split0": (["3dcnn", "official_resnet3d"], {"data": {"clip_len": 8, "resize": 125, "crop": 112, "cache_dir": None}, "training": {"batch_size": 8, "eval_len": 64},  # 112 px: read originals
                                                             "model": {"type": "r2plus1d", "backbone": {"name": "resnet18", "pretrained": True}}}),
    "kinetic_cnn_lstm_L4_split0": (["ablation", "no_pretraining"], {"model": {"backbone": {"name": "kinetic_cnn", "base_channels": 32, "pretrained": False}},
                                                                    "optimizer": {"name": "adamw", "lr": 0.0003, "weight_decay": 0.0001}}),
}


ALL7 = ["embryo_dataset_F-45", "embryo_dataset_F-30", "embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15", "embryo_dataset_F30", "embryo_dataset_F45"]
SPLIT = lambda k: {"data": {"split": f"data/splits/nantes_official/split{k}.csv"}}
# v2: seeds for the promising models, remaining folds for reference + best novel model,
# transformer with its proper optimiser, 7-plane multi-focal with a longer schedule. Entries: (id, tags, override, seeds)
EXPERIMENTS_V2: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_lstm_L4_split0", ["official", "reference", "seeds"], {}, [1, 2]),
    ("resnet50_lstm_L8_split0", ["backbone", "seeds"], EXPERIMENTS["resnet50_lstm_L8_split0"][1], [1, 2]),
    ("resnet18_lstm_L4_multifocal3_split0", ["multifocal", "seeds"], EXPERIMENTS["resnet18_lstm_L4_multifocal3_split0"][1], [1, 2]),
    ("resnet18_tcn_L16_split0", ["head", "seeds"], EXPERIMENTS["resnet18_tcn_L16_split0"][1], [1, 2]),
    ("resnet18_transformer_L16_adamw_split0", ["head", "optim_fix"], {"data": {"clip_len": 16}, "training": {"batch_size": 4},
        "model": {"head": {"name": "transformer", "hidden": 512, "layers": 2}}, "optimizer": {"name": "adamw", "lr": 0.0001, "weight_decay": 0.05},
        "scheduler": {"name": "warmup_cosine", "warmup_epochs": 1}}, [0]),
    ("resnet18_lstm_L4_multifocal7_e20_split0", ["multifocal", "long"], {"data": {"planes": ALL7}, "training": {"epochs": 20}}, [0]),
] + [(f"resnet18_lstm_L4_split{k}", ["official", "reference", "folds"], SPLIT(k), [0]) for k in (1, 2, 3, 4)] \
  + [(f"resnet18_lstm_L4_multifocal3_split{k}", ["multifocal", "folds"],
      # deep-merge: a shallow {**a, **b} here silently dropped data.planes (SPLIT(k) replaced the whole "data" dict), so the
      # four fold runs trained 2026-09-10 were single-plane duplicates of the reference (identical numbers) -- found 2026-09-15
      {"data": {**EXPERIMENTS["resnet18_lstm_L4_multifocal3_split0"][1]["data"], **SPLIT(k)["data"]}}, [0]) for k in (1, 2, 3, 4)]


# v3 (proposed 2026-09-09, NOT launched): the transformer head was evaluated with 150-frame chunks although it only ever
# saw positions 0..15 in training (learned absolute positions) -> p_t 0.37. Re-evaluating the v2 checkpoint with 16-frame
# windows gives p_t 0.79 on 8 test videos (scripts/reeval_eval_len.py), so the head is competitive once eval_len == clip_len.
# v3 = transformer family with eval_len == clip_len, stride L/2 (overlap-averaged), plus the strongest combinations.
TR = lambda L, extra=None: {"data": {"clip_len": L}, "training": {"batch_size": max(2, 64 // L), "eval_len": L, "eval_stride": L // 2},
                            "model": {"head": {"name": "transformer", "hidden": 512, "layers": 2}, **(extra or {})},
                            "optimizer": {"name": "adamw", "lr": 0.0001, "weight_decay": 0.05}, "scheduler": {"name": "warmup_cosine", "warmup_epochs": 1}}
EXPERIMENTS_V3: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L16_evalfix_split0", ["head", "transformer", "evalfix"], TR(16), [0, 1, 2]),
    ("resnet18_transformer_L32_evalfix_split0", ["head", "transformer", "evalfix"], TR(32), [0]),
    ("resnet18_transformer_L64_evalfix_split0", ["head", "transformer", "evalfix"], TR(64), [0]),
    ("resnet50_transformer_L16_evalfix_split0", ["head", "transformer", "backbone"], TR(16, {"backbone": {"name": "resnet50", "pretrained": True}}), [0]),
    ("resnet18_transformer_L16_multifocal3_evalfix_split0", ["head", "transformer", "multifocal"],
        {**TR(16), "data": {"clip_len": 16, "planes": ["embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15"]}}, [0]),
    ("resnet50_lstm_L8_multifocal3_split0", ["backbone", "multifocal"],
        {"data": {"clip_len": 8, "planes": ["embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15"]}, "model": {"backbone": {"name": "resnet50", "pretrained": True}}, "training": {"batch_size": 6}}, [0]),
]


def write_v3(out: Path) -> None:
    V3 = out / "v3"; V3.mkdir(exist_ok=True)
    for old in V3.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V3:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V3 / f"{exp_id}.yaml").write_text(f"# H7 v3 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v3/{exp_id}.yaml", "seed": s} for s in seeds]
    (V3 / "sweep_v3.yaml").write_text("# v3 queue: transformer head with eval_len == clip_len.\nexperiments:\n"
                                      + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    for gpu in (0, 1):
        (V3 / f"sweep_v3_gpu{gpu}.yaml").write_text(f"# half of sweep_v3 for GPU{gpu} (generated)\nexperiments:\n"
                                                    + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[gpu::2]))
    print(f"v3: {len(entries)} runs -> {V3}")


# v4 "reproduce": complete the Gomez baseline table — every official
# baseline on all 5 folds + 3 seeds on fold 0 — and approximate the follow-ups whose code is not public (Misaghi 2025
# EfficientNet-V2-L per frame incl. its image-level split as a leakage demo). ResNet-LSTM folds/seeds are already in v1/v2.
PF = EXPERIMENTS["resnet18_none_split0"][1]; R2 = EXPERIMENTS["r2plus1d_L8_split0"][1]
EXPERIMENTS_V4: list[tuple[str, list[str], dict, list[int]]] = (
    [("resnet18_none_split0", ["official", "per_frame_reference", "seeds"], PF, [1, 2])]
    + [(f"resnet18_none_split{k}", ["official", "per_frame_reference", "folds"], {**PF, **SPLIT(k)}, [0]) for k in (1, 2, 3, 4)]
    + [("r2plus1d_L8_split0", ["official", "official_resnet3d", "seeds"], R2, [1, 2])]
    + [(f"r2plus1d_L8_split{k}", ["official", "official_resnet3d", "folds"], {**R2, **SPLIT(k)}, [0]) for k in (1, 2, 3, 4)]
    + [("efficientnet_v2_l_none_split0", ["followup", "misaghi2025"], {**PF, "model": {"backbone": {"name": "efficientnet_v2_l", "pretrained": True}, "head": {"name": "none"}}, "training": {"batch_size": 24}}, [0]),
       ("efficientnet_v2_l_none_imagesplit", ["followup", "misaghi2025", "leakage_demo"], {**PF, "data": {"dataset": "nantes_frames", "split_mode": "image"}, "model": {"backbone": {"name": "efficientnet_v2_l", "pretrained": True}, "head": {"name": "none"}}, "training": {"batch_size": 24}}, [0]),
       ("resnet18_none_imagesplit", ["leakage_demo"], {**PF, "data": {"dataset": "nantes_frames", "split_mode": "image"}}, [0])]
)


def write_v4(out: Path) -> None:
    V4 = out / "v4"; V4.mkdir(exist_ok=True)
    for old in V4.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V4:
        if (out / f"{exp_id}.yaml").exists():
            path = f"configs/h7/{exp_id}.yaml"
        else:
            doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
            (V4 / f"{exp_id}.yaml").write_text(f"# H7 v4 (reproduce) experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
            path = f"configs/h7/v4/{exp_id}.yaml"
        entries += [{"config": path, "seed": s} for s in seeds]
    (V4 / "sweep_v4.yaml").write_text("# v4 queue (reproduce the Gomez baseline table on 5 folds + follow-up approximations)\nexperiments:\n"
                                      + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    for gpu in (0, 1):
        (V4 / f"sweep_v4_gpu{gpu}.yaml").write_text(f"# half of sweep_v4 for GPU{gpu} (generated)\nexperiments:\n"
                                                    + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[gpu::2]))
    print(f"v4: {len(entries)} runs -> {V4}")


# v5 "modern heads" (proposed 2026-09-09, user asked for state-of-the-art ideas beyond LSTM; not launched):
# selective state-space (Mamba-style, linear-time whole-video context, immune to the T1 eval-window bug by construction),
# a Neural-ODE continuous-time head (motivated by Bechar et al. 2026 RB-NODE — a natural fit for irregular/mismatched
# frame rates, our own-video domain-gap problem-I), and a lightweight diffusion-refinement head
# (approximates EmbryoDiff/DiffAct, literature.md §2, the strongest published model on this benchmark).
EXPERIMENTS_V5: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_mamba_L16_split0", ["head", "ssm", "modern"], {"data": {"clip_len": 16}, "training": {"batch_size": 4},
        "model": {"head": {"name": "mamba", "hidden": 512, "layers": 4}}}, [0, 1, 2]),
    ("resnet18_node_L16_split0", ["head", "neural_ode", "modern"], {"data": {"clip_len": 16}, "training": {"batch_size": 4},
        "model": {"head": {"name": "node", "hidden": 256, "n_substeps": 4}}}, [0, 1, 2]),
    ("resnet18_diffactlite_L16_split0", ["head", "diffusion", "modern"], {"data": {"clip_len": 16}, "training": {"batch_size": 4},
        "model": {"head": {"name": "diffact_lite", "hidden": 384, "steps": 3, "nhead": 6}}}, [0, 1, 2]),
    ("resnet50_mamba_L16_split0", ["head", "ssm", "modern", "backbone"], {"data": {"clip_len": 16}, "training": {"batch_size": 4},
        "model": {"backbone": {"name": "resnet50", "pretrained": True}, "head": {"name": "mamba", "hidden": 512, "layers": 4}}}, [0]),
    ("resnet18_mamba_L16_multifocal3_split0", ["head", "ssm", "modern", "multifocal"],
        {"data": {"clip_len": 16, "planes": ["embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15"]},
         "training": {"batch_size": 4}, "model": {"head": {"name": "mamba", "hidden": 512, "layers": 4}}}, [0]),
]


def write_v5(out: Path) -> None:
    V5 = out / "v5"; V5.mkdir(exist_ok=True)
    for old in V5.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V5:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V5 / f"{exp_id}.yaml").write_text(f"# H7 v5 (modern heads) experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v5/{exp_id}.yaml", "seed": s} for s in seeds]
    (V5 / "sweep_v5.yaml").write_text("# v5 queue (Mamba / Neural-ODE / diffusion-lite heads)\nexperiments:\n"
                                      + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    for gpu in (0, 1):
        (V5 / f"sweep_v5_gpu{gpu}.yaml").write_text(f"# half of sweep_v5 for GPU{gpu} (generated)\nexperiments:\n"
                                                    + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[gpu::2]))
    print(f"v5: {len(entries)} runs -> {V5}")


# v6 "T5 / relpos / embryodiff-protocol" (proposed 2026-09-11, user asked to implement and queue everything discussed):
# (a) relative-position transformer (RoPE) as an alternative fix to T1's sliding-window work-around -- compare against
#     both T1 (absolute position, fixed window) and T8 (Mamba, also position-table-free);
# (b) boundary-weighted CE loss (T5a) vs the plain-CE reference (resnet18_lstm_L4_split0, already 3 seeds in v1/v2);
# (c) our best two recipes run under the EmbryoDiff-described protocol (15 classes, random 7:3 video-level split, not
#     patient-grouped) for a same-data, same-protocol comparison point (track1_TEMPO/protocol_audit.md §4).
EMBRYODIFF_SPLIT = {"data": {"split": "data/splits/nantes_random_73_seed0.json", "merge_last_class": True}}
EXPERIMENTS_V6: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_relpos_L16_split0", ["head", "relpos", "modern"], {"data": {"clip_len": 16}, "training": {"batch_size": 4},
        "model": {"head": {"name": "transformer_relpos", "hidden": 512, "layers": 2}},
        "optimizer": {"name": "adamw", "lr": 0.0001, "weight_decay": 0.05}, "scheduler": {"name": "warmup_cosine", "warmup_epochs": 1}}, [0, 1, 2]),
    ("resnet50_transformer_relpos_L16_split0", ["head", "relpos", "modern", "backbone"], {"data": {"clip_len": 16}, "training": {"batch_size": 4},
        "model": {"backbone": {"name": "resnet50", "pretrained": True}, "head": {"name": "transformer_relpos", "hidden": 512, "layers": 2}},
        "optimizer": {"name": "adamw", "lr": 0.0001, "weight_decay": 0.05}, "scheduler": {"name": "warmup_cosine", "warmup_epochs": 1}}, [0]),
    ("resnet18_lstm_L4_boundaryloss_split0", ["loss", "boundary"], {"loss": {"name": "ce_boundary_weighted", "bonus": 2.0, "sigma": 1.5}}, [0, 1, 2]),
    ("resnet18_lstm_L4_embryodiff_split", ["followup", "embryodiff_protocol"], EMBRYODIFF_SPLIT, [0, 1]),
    ("resnet50_lstm_L8_embryodiff_split", ["followup", "embryodiff_protocol", "backbone"],
        {**EMBRYODIFF_SPLIT, "data": {**EMBRYODIFF_SPLIT["data"], "clip_len": 8}, "model": {"backbone": {"name": "resnet50", "pretrained": True}}, "training": {"batch_size": 6}}, [0]),
]


def write_v6(out: Path) -> None:
    V6 = out / "v6"; V6.mkdir(exist_ok=True)
    for old in V6.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V6:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V6 / f"{exp_id}.yaml").write_text(f"# H7 v6 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v6/{exp_id}.yaml", "seed": s} for s in seeds]
    (V6 / "sweep_v6.yaml").write_text("# v6 queue (relpos transformer, boundary loss, EmbryoDiff-protocol comparison)\nexperiments:\n"
                                      + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    for gpu in (0, 1):
        (V6 / f"sweep_v6_gpu{gpu}.yaml").write_text(f"# half of sweep_v6 for GPU{gpu} (generated)\nexperiments:\n"
                                                    + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[gpu::2]))
    print(f"v6: {len(entries)} runs -> {V6}")


# ---- v8 (TEMPO rung T8, 2026-09-13): representation ideas chosen from track1_TEMPO/diagnostics_findings.md ----
# (a) photometric augmentation + per-frame instance normalisation (finding 6: dim / low-contrast embryos unrecoverable;
#     official recipe augments with flips only); (b) auxiliary ordinal cell-count head (finding 4: t3..t7 recall < 0.3,
#     t7|t8 probe below chance). Each on the two best recipes so far (ResNet-50-LSTM L8, transformer L16 multifocal
#     evalfix), 3 seeds each for the combined setting, 1 seed for the single-factor ablations.
PHOTO = {"photometric": {"brightness": 0.2, "contrast": 0.3, "gamma": 0.3, "p": 0.8}, "instance_norm": True}
AUX = {"model": {"aux": {"name": "cellcount_ordinal", "levels": 9}}, "loss": {"name": "ce_cellcount", "lam": 0.5}}
R50 = EXPERIMENTS["resnet50_lstm_L8_split0"][1]
MF3 = {**TR(16), "data": {"clip_len": 16, "planes": ["embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15"]}}


def _merge(*dicts: dict) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


EXPERIMENTS_V8: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet50_lstm_L8_photo_split0", ["T8a", "photometric", "backbone"], _merge(R50, {"data": PHOTO}), [0]),
    ("resnet50_lstm_L8_cellcount_split0", ["T8b", "cellcount", "backbone"], _merge(R50, AUX), [0]),
    ("resnet50_lstm_L8_photo_cellcount_split0", ["T8", "photometric", "cellcount", "backbone"], _merge(R50, {"data": PHOTO}, AUX), [0, 1, 2]),
    ("resnet18_transformer_L16_multifocal3_photo_evalfix_split0", ["T8a", "photometric", "transformer", "multifocal"], _merge(MF3, {"data": PHOTO}), [0]),
    ("resnet18_transformer_L16_multifocal3_cellcount_evalfix_split0", ["T8b", "cellcount", "transformer", "multifocal"], _merge(MF3, AUX), [0]),
    ("resnet18_transformer_L16_multifocal3_photo_cellcount_evalfix_split0", ["T8", "photometric", "cellcount", "transformer", "multifocal"], _merge(MF3, {"data": PHOTO}, AUX), [0, 1, 2]),
]


def write_v8(out: Path) -> None:
    V8 = out / "v8"; V8.mkdir(exist_ok=True)
    for old in V8.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V8:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V8 / f"{exp_id}.yaml").write_text(f"# H7 v8 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v8/{exp_id}.yaml", "seed": s} for s in seeds]
    (V8 / "sweep_v8.yaml").write_text("# v8 queue (T8 representation: photometric aug + instance norm, ordinal cell-count auxiliary head)\nexperiments:\n"
                                      + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    for gpu in (0, 1):
        (V8 / f"sweep_v8_gpu{gpu}.yaml").write_text(f"# half of sweep_v8 for GPU{gpu} (generated)\nexperiments:\n"
                                                    + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[gpu::2]))
    print(f"v8: {len(entries)} runs -> {V8}")


# ---- v9 (TEMPO rung T9, 2026-09-13 evening): the two input-side directions left after T7/T8 came back null ----
# (a) resolution: 500-px frames (native), 448 crop, from the nantes_500 cache -- thin blastomere membranes at 224 px from a
#     500-px frame are the diagnosed failure; (b) all 7 focal planes fused by cross-focal attention (shared pretrained CNN
#     per plane + learned-query attention) instead of 3 stacked channels.
PLANES7 = ["embryo_dataset_F-45", "embryo_dataset_F-30", "embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15", "embryo_dataset_F30", "embryo_dataset_F45"]
HIRES = {"data": {"resize": 500, "crop": 448, "cache_dir": "data/cache/nantes_500"}}
CF7 = {"data": {"planes": PLANES7}, "model": {"backbone": {"name": "crossfocal", "cnn": "resnet18", "pretrained": True, "nhead": 4}}}
EXPERIMENTS_V9: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_lstm_L8_hires448_split0", ["T9a", "hires"], _merge({"data": {"clip_len": 8}, "training": {"batch_size": 6}}, HIRES), [0]),
    ("resnet50_lstm_L8_hires448_split0", ["T9a", "hires", "backbone"], _merge(R50, HIRES, {"training": {"batch_size": 4}}), [0]),
    ("resnet18_transformer_L16_crossfocal7_evalfix_split0", ["T9b", "crossfocal", "transformer"], _merge(TR(16), CF7, {"training": {"batch_size": 2}}), [0]),
    ("resnet18_lstm_L8_crossfocal7_split0", ["T9b", "crossfocal"], _merge({"data": {"clip_len": 8}, "training": {"batch_size": 4}}, CF7), [0]),
    ("resnet18_lstm_L8_hires448_crossfocal7_split0", ["T9", "hires", "crossfocal"], _merge({"data": {"clip_len": 8}, "training": {"batch_size": 2}}, HIRES, CF7, {"data": {"cache_dir": None}}), [0]),
]


def write_v9(out: Path) -> None:
    V9 = out / "v9"; V9.mkdir(exist_ok=True)
    for old in V9.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V9:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V9 / f"{exp_id}.yaml").write_text(f"# H7 v9 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v9/{exp_id}.yaml", "seed": s} for s in seeds]
    (V9 / "sweep_v9.yaml").write_text("# v9 queue (T9: 448-px input; 7-plane cross-focal attention). User go 2026-09-13.\nexperiments:\n"
                                      + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    (V9 / "sweep_v9_gpu0.yaml").write_text("# hires half (GPU0, after the reference seed-0 retrain)\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries if "hires" in e["config"] and "crossfocal" not in e["config"]))
    (V9 / "sweep_v9_gpu1.yaml").write_text("# crossfocal half (GPU1)\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries if "crossfocal" in e["config"]))
    print(f"v9: {len(entries)} runs -> {V9}")


# ---- v10 (2026-09-14): confirm the one T9 configuration every diagnostic agreed on -- cross-focal 7-plane transformer:
#      seeds 1,2 (seed 0 exists) + a ResNet-50 per-plane CNN variant; then ensemble + flip-TTA + HSMM are retrain-free.
CF7_TR = _merge(TR(16), CF7, {"training": {"batch_size": 2}})
EXPERIMENTS_V10: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L16_crossfocal7_evalfix_split0", ["T9b", "crossfocal", "transformer", "seeds"], CF7_TR, [1, 2]),
    ("resnet50_transformer_L16_crossfocal7_evalfix_split0", ["T9b", "crossfocal", "transformer", "backbone"],
        _merge(CF7_TR, {"model": {"backbone": {"name": "crossfocal", "cnn": "resnet50", "pretrained": True, "nhead": 4}}, "training": {"batch_size": 1}}), [0]),
]


def write_v10(out: Path) -> None:
    V = out / "v10"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V10:
        src = out / "v9" / f"{exp_id}.yaml"
        if not src.exists():
            doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
            (V / f"{exp_id}.yaml").write_text(f"# H7 v10 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
            path = f"configs/h7/v10/{exp_id}.yaml"
        else:
            path = f"configs/h7/v9/{exp_id}.yaml"
        entries += [{"config": path, "seed": s} for s in seeds]
    (V / "sweep_v10_gpu0.yaml").write_text("# cross-focal transformer seeds 1,2 (GPU0). User go 2026-09-14.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[:2]))
    (V / "sweep_v10_gpu1.yaml").write_text("# cross-focal ResNet-50 CNN variant (GPU1)\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[2:]))
    print(f"v10: {len(entries)} runs -> {V}")


# ---- v11 (2026-09-14): 5-fold coverage of the headline configuration (cross-focal 7-plane transformer), folds 1-4, seed 0.
EXPERIMENTS_V11: list[tuple[str, list[str], dict, list[int]]] = [
    (f"resnet18_transformer_L16_crossfocal7_evalfix_split{k}", ["T9b", "crossfocal", "transformer", "folds"],
     _merge(CF7_TR, {"data": {"split": f"data/splits/nantes_official/split{k}.csv"}}), [0]) for k in (1, 2, 3, 4)
]


def write_v11(out: Path) -> None:
    V = out / "v11"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V11:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v11 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v11/{exp_id}.yaml", "seed": s} for s in seeds]
    for gpu in (0, 1):
        (V / f"sweep_v11_gpu{gpu}.yaml").write_text(f"# cross-focal transformer folds (GPU{gpu}). User go 2026-09-14.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[gpu::2]))
    print(f"v11: {len(entries)} runs -> {V}")


# ---- v12 (2026-09-15): internal-SOTA claim support. (a) Re-implemented follow-up baselines missing from the sweep --
#      ASFormer-lite and MS-TCN heads -- on the same ResNet-18 features/protocol as the transformer (evalfix windows);
#      (b) seeds 1,2 on folds 1-4 for the cross-focal transformer AND the reference LSTM (3 seeds x 5 folds each).
def _EVALFIX(L, head, bs):
    return {"data": {"clip_len": L}, "training": {"batch_size": bs, "eval_len": L, "eval_stride": L // 2},
            "model": {"head": head}, "optimizer": {"name": "adamw", "lr": 0.0001, "weight_decay": 0.05}, "scheduler": {"name": "warmup_cosine", "warmup_epochs": 1}}
EXPERIMENTS_V12A: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_asformer_L16_evalfix_split0", ["baseline_reimpl", "asformer"], _EVALFIX(16, {"name": "asformer_lite", "hidden": 256, "layers": 6, "n_decoders": 2}, 4), [0]),
    ("resnet18_asformer_L64_evalfix_split0", ["baseline_reimpl", "asformer"], _EVALFIX(64, {"name": "asformer_lite", "hidden": 256, "layers": 6, "n_decoders": 2}, 1), [0]),
    ("resnet18_mstcn_L64_evalfix_split0", ["baseline_reimpl", "mstcn"], _EVALFIX(64, {"name": "mstcn", "hidden": 64, "layers": 10, "stages": 4}, 1), [0]),
]
EXPERIMENTS_V12B: list[tuple[str, list[str], dict, list[int]]] = (
    [(f"resnet18_transformer_L16_crossfocal7_evalfix_split{k}", ["T9b", "crossfocal", "folds", "seeds"], _merge(CF7_TR, SPLIT(k)), [1, 2]) for k in (1, 2, 3, 4)]
    + [(f"resnet18_lstm_L4_split{k}", ["official", "reference", "folds", "seeds"], SPLIT(k), [1, 2]) for k in (1, 2, 3, 4)])


def write_v12(out: Path) -> None:
    V = out / "v12"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    ea = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V12A:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v12 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        ea += [{"config": f"configs/h7/v12/{exp_id}.yaml", "seed": s} for s in seeds]
    (V / "sweep_v12a_baselines.yaml").write_text("# v12a: re-implemented ASFormer / MS-TCN baselines (GPU0, after the mf3 fold retrain)\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in ea))
    eb = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V12B:
        path = f"configs/h7/v11/{exp_id}.yaml" if "crossfocal" in exp_id else f"configs/h7/v2/{exp_id}.yaml"
        eb += [{"config": path, "seed": s} for s in seeds]
    # interleave so each GPU gets 2 cross-focal folds x 2 seeds + 2 reference folds x 2 seeds
    for gpu in (0, 1):
        (V / f"sweep_v12b_gpu{gpu}.yaml").write_text(f"# v12b: seeds 1,2 on folds 1-4 (GPU{gpu} half)\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in eb[gpu::2]))
    print(f"v12: {len(ea)} baseline runs, {len(eb)} seed/fold runs -> {V}")


# ---- v13 (2026-09-15): T10 Focus-Adaptive Cross-Plane Attention -- ablation ladder on the headline recipe (transformer L16,
#      7 planes, evalfix): cross-focal (pooled, learned query; v9) -> +sharpness prior -> spatial fusion -> spatial + sharpness.
def _FA(spatial, sharp):
    return _merge(TR(16), {"data": {"planes": PLANES7}, "training": {"batch_size": 2},
                           "model": {"backbone": {"name": "focalattn", "cnn": "resnet18", "pretrained": True, "spatial": spatial, "sharpness": sharp}}})
EXPERIMENTS_V13: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L16_focalattn_sharp_evalfix_split0", ["T10", "focalattn", "sharpness"], _FA(False, True), [0]),
    ("resnet18_transformer_L16_focalattn_spatial_evalfix_split0", ["T10", "focalattn", "spatial"], _FA(True, False), [0]),
    ("resnet18_transformer_L16_focalattn_spatial_sharp_evalfix_split0", ["T10", "focalattn", "spatial", "sharpness"], _FA(True, True), [0]),
]


def write_v13(out: Path) -> None:
    V = out / "v13"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V13:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v13 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v13/{exp_id}.yaml", "seed": s} for s in seeds]
    (V / "sweep_v13_gpu0.yaml").write_text("# T10 ablation (GPU0 half): sharpness-only, spatial+sharpness\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in [entries[0], entries[2]]))
    (V / "sweep_v13_gpu1.yaml").write_text("# T10 ablation (GPU1 half): spatial-only\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in [entries[1]]))
    print(f"v13: {len(entries)} runs -> {V}")


# ---- v14 (2026-09-15): T11 backbone fair test. Same fusion (cross-focal, 7 planes) + transformer L16, but each backbone gets a
#      fine-tuning recipe suited to it instead of the ResNet one: backbone lr x0.1 with layer-wise decay 0.75, AdamW, grad-accum
#      to an effective batch of 8 clips, 20 epochs with 2 warm-up. ResNet-18 under the SAME recipe is the control that separates
#      'bigger/transformer backbone' from 'better recipe'.
FAIR = {"optimizer": {"name": "adamw", "lr": 0.0001, "weight_decay": 0.05, "backbone_lr_mult": 0.1, "layer_decay": 0.75},
        "scheduler": {"name": "warmup_cosine", "warmup_epochs": 2}, "training": {"epochs": 20, "grad_accum": 4}}
def _FAIRCF(cnn, bs):
    return _merge(TR(16), {"data": {"planes": PLANES7}, "training": {"batch_size": bs},
                           "model": {"backbone": {"name": "crossfocal", "cnn": cnn, "pretrained": True, "nhead": 4}}}, FAIR)
EXPERIMENTS_V14: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L16_crossfocal7_fair_evalfix_split0", ["T11", "fair", "control"], _FAIRCF("resnet18", 2), [0]),
    ("dinov2b_transformer_L16_crossfocal7_fair_evalfix_split0", ["T11", "fair", "vit", "dinov2"], _FAIRCF("dinov2_vitb14", 1), [0]),
    ("swin_t_transformer_L16_crossfocal7_fair_evalfix_split0", ["T11", "fair", "swin"], _FAIRCF("swin_t", 2), [0]),
    ("convnext_tiny_transformer_L16_crossfocal7_fair_evalfix_split0", ["T11", "fair", "convnext"], _FAIRCF("convnext_tiny", 2), [0]),
]


def write_v14(out: Path) -> None:
    V = out / "v14"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V14:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v14 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v14/{exp_id}.yaml", "seed": s} for s in seeds]
    for gpu in (0, 1):
        (V / f"sweep_v14_gpu{gpu}.yaml").write_text(f"# T11 backbone fair test (GPU{gpu} half). Queue after v13.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[gpu::2]))
    print(f"v14: {len(entries)} runs -> {V}")


# ---- v15: E1/E2 required for Route A (headline models on the patient-grouped
#      CLEAVE split), E3/E4 controls for DA-C1 (single-plane and 3-plane transformers on folds 1-4, so the 5-fold gain can be
#      attributed to cross-plane fusion rather than to transformer + window fix), E7 rotary transformer WITH the window fix.
GROUPED = {"data": {"split": "data/splits/nantes_grouped_v1.json"}}
RELPOS_FIX = {"data": {"clip_len": 16}, "training": {"batch_size": 4, "eval_len": 16, "eval_stride": 8},
              "model": {"head": {"name": "transformer_relpos", "hidden": 512, "layers": 2}},
              "optimizer": {"name": "adamw", "lr": 0.0001, "weight_decay": 0.05}, "scheduler": {"name": "warmup_cosine", "warmup_epochs": 1}}
EXPERIMENTS_V15_GPU1: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L16_crossfocal7_evalfix_grouped_v1", ["E2", "crossfocal", "grouped"], _merge(CF7_TR, GROUPED), [0, 1, 2]),
    ("resnet18_lstm_L4_grouped_v1", ["E1", "reference", "grouped"], GROUPED, [1, 2]),
    ("resnet18_transformer_relpos_L16_evalfix_split0", ["E7", "relpos", "evalfix"], RELPOS_FIX, [0, 1, 2]),
]
EXPERIMENTS_V15_GPU0: list[tuple[str, list[str], dict, list[int]]] = (
    [(f"resnet18_transformer_L16_evalfix_split{k}", ["E3", "transformer", "folds", "control"], _merge(TR(16), SPLIT(k)), [0]) for k in (1, 2, 3, 4)]
    + [(f"resnet18_transformer_L16_multifocal3_evalfix_split{k}", ["E4", "transformer", "multifocal", "folds", "control"],
        _merge(TR(16), {"data": {"clip_len": 16, "planes": ["embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15"]}}, SPLIT(k)), [0]) for k in (1, 2, 3, 4)])


def write_v15(out: Path) -> None:
    V = out / "v15"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    for gpu, exps in ((1, EXPERIMENTS_V15_GPU1), (0, EXPERIMENTS_V15_GPU0)):
        entries = []
        for exp_id, tags, over, seeds in exps:
            src_v1 = out / f"{exp_id}.yaml"
            if src_v1.exists():
                path = f"configs/h7/{exp_id}.yaml"
            else:
                doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
                (V / f"{exp_id}.yaml").write_text(f"# H7 v15 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
                path = f"configs/h7/v15/{exp_id}.yaml"
            entries += [{"config": path, "seed": s} for s in seeds]
        (V / f"sweep_v15_gpu{gpu}.yaml").write_text(f"# review-round-1 experiments (GPU{gpu}). User go 2026-09-16.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
        print(f"v15 gpu{gpu}: {len(entries)} runs")


# ---- v16 (2026-09-16, "SOTA push", pre-registered insec 12.y): Stage 1 on fold 0, 3 seeds each. Success criterion
#      for any module claim: delta p_t >= +0.03 vs the single-plane window-fixed transformer control (E3) on 5 folds x 3 seeds,
#      pooled per-video CI excluding zero, and consistent t3/t5/t7 recall gains. Below that = reported as a small improvement.
FA_SPATIAL = {"data": {"planes": PLANES7}, "training": {"batch_size": 2},
              "model": {"backbone": {"name": "focalattn", "cnn": "resnet18", "pretrained": True, "spatial": True, "sharpness": False}}}
EXPERIMENTS_V16: list[tuple[str, list[str], dict, list[int]]] = [
    # E8: 7 planes stacked as channels under the same transformer -- the control that isolates the fusion *mechanism*
    ("resnet18_transformer_L16_multifocal7_evalfix_split0", ["v16E8", "multifocal7", "control"], _merge(TR(16), {"data": {"planes": PLANES7}, "training": {"batch_size": 2}}), [0, 1, 2]),
    # A: spatial focus-adaptive attention, seeds 1,2 (seed 0 = v13)
    ("resnet18_transformer_L16_focalattn_spatial_evalfix_split0", ["v16A", "focalattn", "spatial"], _merge(TR(16), FA_SPATIAL), [1, 2]),
    # C: cross-focal trained 20 epochs (train-longer control; 12/15 cross-focal runs peaked at epoch 9-10 of 10)
    ("resnet18_transformer_L16_crossfocal7_e20_evalfix_split0", ["v16C", "crossfocal", "epochs20"], _merge(CF7_TR, {"training": {"epochs": 20}, "scheduler": {"name": "warmup_cosine", "warmup_epochs": 2}}), [0, 1, 2]),
    # D: long context -- cross-focal + rotary transformer, 64-frame clips, window fix, eff. batch 4 via grad accumulation
    ("resnet18_relpos_L64_crossfocal7_evalfix_split0", ["v16D", "crossfocal", "relpos", "longcontext"],
        _merge({"data": {"clip_len": 64, "planes": PLANES7}, "training": {"batch_size": 1, "grad_accum": 4, "eval_len": 64, "eval_stride": 32},
                "model": {"backbone": {"name": "crossfocal", "cnn": "resnet18", "pretrained": True, "nhead": 4}, "head": {"name": "transformer_relpos", "hidden": 512, "layers": 2}},
                "optimizer": {"name": "adamw", "lr": 0.0001, "weight_decay": 0.05}, "scheduler": {"name": "warmup_cosine", "warmup_epochs": 1}}), [0, 1, 2]),
    # B: spatial focus-adaptive + native 448 px, batch 1 x grad_accum 4, 20 epochs (reads the nantes_500 cache for the central plane only -> uncached planes: originals)
    ("resnet18_transformer_L16_focalattn_spatial_hires448_e20_evalfix_split0", ["v16B", "focalattn", "spatial", "hires", "epochs20"],
        _merge(TR(16), FA_SPATIAL, {"data": {"resize": 500, "crop": 448, "cache_dir": None}, "training": {"batch_size": 1, "grad_accum": 4, "epochs": 20},
                                     "scheduler": {"name": "warmup_cosine", "warmup_epochs": 2}}), [0, 1, 2]),
]


def write_v16(out: Path) -> None:
    V = out / "v16"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V16:
        src13 = out / "v13" / f"{exp_id}.yaml"
        if src13.exists():
            path = f"configs/h7/v13/{exp_id}.yaml"
        else:
            doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
            (V / f"{exp_id}.yaml").write_text(f"# H7 v16 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
            path = f"configs/h7/v16/{exp_id}.yaml"
        entries += [{"config": path, "seed": s} for s in seeds]
    # GPU0: A (2) + D (3); GPU1: C (3) + B (3)  -- roughly balanced wall-clock
    g0 = [e for e in entries if "multifocal7" in e["config"] or "focalattn_spatial_evalfix" in e["config"] or "relpos_L64" in e["config"]]
    g1 = [e for e in entries if "_e20_" in e["config"]]
    (V / "sweep_v16_gpu0.yaml").write_text("# v16 Stage 1 (GPU0): A spatial focus-adaptive seeds 1,2; D long-context rotary cross-focal x3. User go 2026-09-16.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in g0))
    (V / "sweep_v16_gpu1.yaml").write_text("# v16 Stage 1 (GPU1): C cross-focal 20 epochs x3; B spatial focus-adaptive + 448 px x3.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in g1))
    print(f"v16: {len(entries)} runs ({len(g0)} GPU0, {len(g1)} GPU1) -> {V}")


# ---- v17 (2026-09-17, after the v16 post-mortem): the cross-focal attention is a *static* plane weighting (240-frame probe:
#      residual variance 15%, entropy 1.64/1.95), all 10-epoch runs are under-trained, and longer training memorises the rare
#      short phases (t3 = 1.7% of frames, median 4-frame segments) without generalising. So v17 attacks data efficiency, not
#      capacity: (a) planes as augmentation + plane-TTA for a single-plane model, (b) rotation augmentation (label-preserving),
#      (c) temporal label smoothing at boundaries, (d) SWA over the last 5 epochs, all at 20 epochs. Control: transformer
#      1-plane 20 epochs (E) and cross-focal 20 epochs (v16 C). Fold 0 x 3 seeds; pre-registered rule unchanged.
E20 = {"training": {"epochs": 20}, "scheduler": {"name": "warmup_cosine", "warmup_epochs": 2}}
RECIPE = {"data": {"rotation": {"p": 0.75, "deg": "any"}}, "loss": {"name": "ce_tls", "eps": 0.3}, "training": {"swa_last": 5, "monitor": "p_t"}}
PLANEAUG = {"data": {"planes": PLANES7, "plane_mode": "random_single", "eval_plane": "embryo_dataset"}}
TR16 = _merge(TR(16), {"training": {"batch_size": 4}})
EXPERIMENTS_V17: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L16_e20_evalfix_split0", ["v17E", "control", "epochs20"], _merge(TR16, E20), [0, 1, 2]),
    ("resnet18_transformer_L16_planeaug_e20_evalfix_split0", ["v17A", "planeaug", "epochs20"], _merge(TR16, E20, PLANEAUG), [0, 1, 2]),
    ("resnet18_transformer_L16_recipe_e20_evalfix_split0", ["v17B", "rotation", "tls", "swa", "epochs20"], _merge(TR16, E20, RECIPE), [0, 1, 2]),
    ("resnet18_transformer_L16_planeaug_recipe_e20_evalfix_split0", ["v17D", "planeaug", "rotation", "tls", "swa", "epochs20"], _merge(TR16, E20, PLANEAUG, RECIPE), [0, 1, 2]),
    ("resnet18_transformer_L16_crossfocal7_recipe_e20_evalfix_split0", ["v17C", "crossfocal", "rotation", "tls", "swa", "epochs20"], _merge(CF7_TR, E20, RECIPE), [0, 1, 2]),
]


def write_v17(out: Path) -> None:
    V = out / "v17"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V17:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v17 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v17/{exp_id}.yaml", "seed": s} for s in seeds]
    # GPU0: E (3) + A (3) + B seed 0..1 ; GPU1: B seed 2 + D (3) + C (3)  -- roughly balanced (C is 2x slower)
    g0 = entries[0:8]; g1 = entries[8:]
    (V / "sweep_v17_gpu0.yaml").write_text("# v17 (GPU0): control 20ep x3, plane-aug x3, recipe seeds 0-1. User go 2026-09-17.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in g0))
    (V / "sweep_v17_gpu1.yaml").write_text("# v17 (GPU1): recipe seed 2, plane-aug+recipe x3, cross-focal+recipe x3.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in g1))
    print(f"v17: {len(entries)} runs ({len(g0)} GPU0, {len(g1)} GPU1) -> {V}")


# ---- v18 (2026-09-18, after the v17 post-mortem): plane_mode=random_single gave the eval plane only
#      ~1/7 of a dedicated single-plane model's training exposure over 20 epochs (history.json train_loss stays well above
#      control throughout; 7-plane TTA does not recover it -- ruled out as an eval artifact). New hypothesis: biasing the
#      plane draw toward the eval plane (plane_center_weight=0.5, remaining 0.5 split over the other 6 planes) restores most
#      of that exposure at the SAME total epoch budget/compute as v17 A, while keeping some cross-plane augmentation for
#      7-plane TTA to still have something to average over. Falsifiable: if p_t stays well below control even with this
#      bias and TTA stays flat/negative, planes-as-augmentation is dead for a single-plane-deployable model. Fold 0 x 3
#      seeds, 20 epochs, same claim rule.
PLANEAUG_BIASED50 = _merge(PLANEAUG, {"data": {"plane_center_weight": 0.5}})
EXPERIMENTS_V18: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L16_planeaugbias50_e20_evalfix_split0", ["v18A2", "planeaug", "biased_sampling", "epochs20"], _merge(TR16, E20, PLANEAUG_BIASED50), [0, 1, 2]),
]


def write_v18(out: Path) -> None:
    V = out / "v18"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V18:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v18 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v18/{exp_id}.yaml", "seed": s} for s in seeds]
    (V / "sweep_v18_gpu0.yaml").write_text("# v18 (GPU0): plane-aug with plane_center_weight=0.5 (biased toward eval plane) x3 seeds. User go 2026-09-18.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    print(f"v18: {len(entries)} runs -> {V}")


# ---- v19 (2026-09-19): two independent things queued together.
#      (a) Stage 2 of the pre-registered protocol for the Stage-1 winner, C (cross-focal + recipe): folds 1-4 x 3 seeds,
#          to pool with the existing fold-0 numbers (12.dd) into the confirmatory 5-fold comparison. GPU1 (slow, ~18 min/
#          epoch): this is the expensive, unavoidable part of the two-stage protocol.
#      (b) Stage-1 screening (fold 0 only, cheap) of a NEW brainstormed direction: rare_phase_boost -- bias the clip-start
#          sampler toward windows containing t3/t5/t7 (1.7-3.6% of frames, 4-8 frame segments) instead of attacking the
#          same bottleneck indirectly via loss smoothing (ce_tls) or longer training. Tested alone against the plain
#          control (not stacked with the recipe yet) to isolate its own effect. GPU0 (fast, ~4.5 min/epoch).
RAREBOOST = {"data": {"rare_phase_boost": {"phases": ["t3", "t5", "t7"], "weight": 3.0}}}
EXPERIMENTS_V19_C_STAGE2: list[tuple[str, list[str], dict, list[int]]] = [
    (f"resnet18_transformer_L16_crossfocal7_recipe_e20_evalfix_split{k}",
     ["v17C", "crossfocal", "rotation", "tls", "swa", "epochs20", "stage2", f"fold{k}"],
     _merge(CF7_TR, E20, RECIPE, {"data": {"split": f"data/splits/nantes_official/split{k}.csv"}}), [0, 1, 2])
    for k in (1, 2, 3, 4)
]
EXPERIMENTS_V19_RAREBOOST_STAGE1: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L16_rareboost_e20_evalfix_split0", ["v19", "rare_phase_boost", "epochs20", "stage1"],
     _merge(TR16, E20, RAREBOOST), [0, 1, 2]),
]


def write_v19(out: Path) -> None:
    V = out / "v19"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries_c, entries_rb = [], []
    for exp_id, tags, over, seeds in EXPERIMENTS_V19_C_STAGE2:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v19 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries_c += [{"config": f"configs/h7/v19/{exp_id}.yaml", "seed": s} for s in seeds]
    for exp_id, tags, over, seeds in EXPERIMENTS_V19_RAREBOOST_STAGE1:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v19 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries_rb += [{"config": f"configs/h7/v19/{exp_id}.yaml", "seed": s} for s in seeds]
    (V / "sweep_v19_gpu1_stage2_C.yaml").write_text("# v19 Stage 2 (GPU1): arm C (cross-focal+recipe) on folds 1-4 x 3 seeds, to pool with fold 0 (12.dd) for the 5-fold confirmatory comparison. User go 2026-09-19.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries_c))
    (V / "sweep_v19_gpu0_stage1_rareboost.yaml").write_text("# v19 Stage 1 (GPU0): rare_phase_boost screening on fold 0 x 3 seeds vs control.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries_rb))
    print(f"v19: {len(entries_c)} Stage-2 C runs (GPU1) + {len(entries_rb)} Stage-1 rareboost runs (GPU0) -> {V}")


# ---- v20 (2026-09-20, GPU0 idle while v19 Stage 2 grinds on GPU1): next Stage-1 (fold 0, 3 seeds) screening
#      candidate from the still-untried list in 12.gg -- L32 absolute-position. D (rotary + L64) failed in v16, but
#      that confounded three things at once (rotary itself, half the optimiser steps from grad-accum, and L64 clips
#      being dominated by long phases). This isolates just "more context, same position scheme that already works at
#      L16" -- TR(32) already gives the right batch_size/eval_len/eval_stride, no code changes needed. Tested alone
#      against the L16 control (not stacked with the recipe) to isolate its own effect.
EXPERIMENTS_V20_L32_STAGE1: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L32_e20_evalfix_split0", ["v20", "L32", "abspos", "epochs20", "stage1"],
     _merge(TR(32), E20), [0, 1, 2]),
]


def write_v20(out: Path) -> None:
    V = out / "v20"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V20_L32_STAGE1:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v20 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v20/{exp_id}.yaml", "seed": s} for s in seeds]
    (V / "sweep_v20_gpu0_stage1_L32.yaml").write_text("# v20 Stage 1 (GPU0): L32 absolute-position screening on fold 0 x 3 seeds vs L16 control. User go 2026-09-20.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    print(f"v20: {len(entries)} Stage-1 L32 runs (GPU0) -> {V}")


# ---- v21 (2026-09-20): dual-branch cell-counting backbone Stage-1 screening (fold 0, 3 seeds).
#      Oracle finding (12.hh): perfectly classifying only the mid-cleavage block (t3-t8, 32% of frames) would raise
#      frame accuracy by +0.14, dwarfing every architecture/recipe lever tried (+0.03 at best). The internal
#      auxiliary cell-count head (ED Table 2) failed because its target is derived from the very label the shared
#      backbone predicts -- no new information. This gives counting its own backbone at its own (tightly-cropped,
#      higher-effective-resolution) input: `dualbranch` = resnet18 (low-res, unchanged from control) concatenated
#      with a YOLOv8-derived branch warm-started from N3's embryo-domain-pretrained weights (measurably better than
#      COCO init on an 8-class Nantes cell-count probe, scripts/train_cellcount_branch.py: +4-5 accuracy points at
#      every epoch). Requires data.hires_crop_dir (scripts/build_nantes_embryo_crop_cache.py). Tested alone against
#      the plain L16 control (not stacked with the recipe) to isolate its own effect, same as every other Stage-1
#      screen this cycle.
HIRES_CROP_DIR = "data/cache/nantes_embryo_crop_224"
DUALBRANCH = {"data": {"hires_crop_dir": HIRES_CROP_DIR},
              "model": {"backbone": {"name": "dualbranch", "low_res_cnn": "resnet18", "pretrained": True, "hires_weights": "n3"}}}
EXPERIMENTS_V21_STAGE1: list[tuple[str, list[str], dict, list[int]]] = [
    ("resnet18_transformer_L16_dualbranch_e20_evalfix_split0", ["v21", "dualbranch", "cellcount", "epochs20", "stage1"],
     _merge(TR16, E20, DUALBRANCH), [0, 1, 2]),
]


def write_v21(out: Path) -> None:
    V = out / "v21"; V.mkdir(exist_ok=True)
    for old in V.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V21_STAGE1:
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (V / f"{exp_id}.yaml").write_text(f"# H7 v21 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
        entries += [{"config": f"configs/h7/v21/{exp_id}.yaml", "seed": s} for s in seeds]
    (V / "sweep_v21_gpu0_stage1_dualbranch.yaml").write_text("# v21 Stage 1 (GPU0): dualbranch cell-counting backbone screening on fold 0 x 3 seeds vs L16 control. User go 2026-09-20.\nexperiments:\n" + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    print(f"v21: {len(entries)} Stage-1 dualbranch runs (GPU0) -> {V}")


def main() -> None:
    import sys
    OUT.mkdir(exist_ok=True)
    if "--only-v3" in sys.argv:  # v1/v2 files are left untouched (a running queue reads them)
        write_v3(OUT); return
    if "--only-v4" in sys.argv:
        write_v4(OUT); return
    if "--only-v5" in sys.argv:
        write_v5(OUT); return
    if "--only-v6" in sys.argv:
        write_v6(OUT); return
    if "--only-v8" in sys.argv:
        write_v8(OUT); return
    if "--only-v9" in sys.argv:
        write_v9(OUT); return
    if "--only-v10" in sys.argv:
        write_v10(OUT); return
    if "--only-v11" in sys.argv:
        write_v11(OUT); return
    if "--only-v12" in sys.argv:
        write_v12(OUT); return
    if "--only-v13" in sys.argv:
        write_v13(OUT); return
    if "--only-v14" in sys.argv:
        write_v14(OUT); return
    if "--only-v15" in sys.argv:
        write_v15(OUT); return
    if "--only-v16" in sys.argv:
        write_v16(OUT); return
    if "--only-v17" in sys.argv:
        write_v17(OUT); return
    if "--only-v18" in sys.argv:
        write_v18(OUT); return
    if "--only-v19" in sys.argv:
        write_v19(OUT); return
    if "--only-v20" in sys.argv:
        write_v20(OUT); return
    if "--only-v21" in sys.argv:
        write_v21(OUT); return
    for old in OUT.glob("*.yaml"):
        old.unlink()
    (OUT / "_base.yaml").write_text("# Base = official Nantes ResNet-LSTM protocol (arXiv 2203.00531). Experiments inherit and override.\n"
                                    "# Generated by scripts/make_h7_configs.py — edit that file, not this one.\n" + yaml.safe_dump(BASE, sort_keys=False))
    for exp_id, (tags, over) in EXPERIMENTS.items():
        doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
        (OUT / f"{exp_id}.yaml").write_text(f"# H7 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n"
                                            f"# Run: PYTHONPATH=src python scripts/run_pipeline.py --pipeline_config configs/h7/{exp_id}.yaml\n" + yaml.safe_dump(doc, sort_keys=False))
    ids = list(EXPERIMENTS)
    (OUT / "sweep_v1.yaml").write_text("# Sequential sweep, priority order; finished runs (results.json) are skipped; interrupted runs resume from last.pt.\n"
                                       "experiments:\n" + "".join(f"  - configs/h7/{e}.yaml\n" for e in ids))
    for gpu, sel in ((0, ids[0::2]), (1, ids[1::2])):  # two GPUs: interleaved halves, priority order preserved
        (OUT / f"sweep_v1_gpu{gpu}.yaml").write_text(f"# half of sweep_v1 for GPU{gpu} (generated)\nexperiments:\n" + "".join(f"  - configs/h7/{e}.yaml\n" for e in sel))
    # ---- v2 ----
    V2 = OUT / "v2"; V2.mkdir(exist_ok=True)
    for old in V2.glob("*.yaml"):
        old.unlink()
    entries = []
    for exp_id, tags, over, seeds in EXPERIMENTS_V2:
        if not (OUT / f"{exp_id}.yaml").exists():  # new experiment definition (v1 ids reuse the v1 file)
            doc = {"inherits": "configs/h7/_base.yaml", "cfg": {"experiment": {"id": exp_id, "tags": tags}, **over}}
            (V2 / f"{exp_id}.yaml").write_text(f"# H7 v2 experiment `{exp_id}` — overrides on configs/h7/_base.yaml. Generated by scripts/make_h7_configs.py.\n" + yaml.safe_dump(doc, sort_keys=False))
            path = f"configs/h7/v2/{exp_id}.yaml"
        else:
            path = f"configs/h7/{exp_id}.yaml"
        entries += [{"config": path, "seed": s} for s in seeds]
    (V2 / "sweep_v2.yaml").write_text("# v2 queue: seeds, folds, transformer optimiser fix, 7-plane long run.\nexperiments:\n"
                                      + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries))
    for gpu in (0, 1):
        (V2 / f"sweep_v2_gpu{gpu}.yaml").write_text(f"# half of sweep_v2 for GPU{gpu} (generated)\nexperiments:\n"
                                                    + "".join(f"  - {{config: {e['config']}, seed: {e['seed']}}}\n" for e in entries[gpu::2]))
    print(f"v2: {len(entries)} runs -> {V2}")
    write_v3(OUT)
    write_v4(OUT)
    write_v5(OUT)
    write_v6(OUT)
    print(f"wrote _base.yaml + {len(EXPERIMENTS)} experiment files + sweep_v1.yaml -> {OUT}")


if __name__ == "__main__":
    main()
