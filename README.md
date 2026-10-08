# CLEAVE

**CLeavage-Event Annotation Verification and Evaluation** — an audit of the public benchmark for embryo
morphokinetic phase recognition, and the corrected protocol that came out of it.

Code and data artefacts for:

> Duy Tan Nguyen, Phuong Huy Tran, The Bao Pham, Ngoc Thanh Sang Vu. *CLEAVE: protocol limitations, suspected label
> mismatches and a cell-counting ceiling in a public benchmark for embryo morphokinetic phase recognition.* Submitted to *Medical Image Analysis*, 2026.

The benchmark audited here is the Nantes time-lapse dataset (704 EmbryoScope videos, seven focal planes, sixteen
annotated developmental events; Gomez et al., *Data in Brief* 42, 108258, 2022). This repository does not
redistribute any of its images; it releases the protocol, the splits, the issue log, the code and every per-run
result needed to regenerate the paper's numbers.

## Contents

| What | Where | Produced by |
|---|---|---|
| Released (video-level) folds, as distributed with the dataset | `data/splits/nantes_official/split{0..4}.csv` | — |
| Code-grouped split, cycle code (every grouped result in the paper) | `data/splits/nantes_grouped_v1.json` | `scripts/make_nantes_split.py` |
| Code-grouped split, code level (**recommended**) | `data/splits/nantes_grouped_v2.json` | `scripts/make_nantes_split_v2.py` |
| Cleaned training splits (flagged videos removed from training only) | `data/splits/nantes_grouped_v2_trainclean_v{1,2,3}.json` | `scripts/make_trainclean_split.py --version k` |
| Paired code-overlap splits (no overlap / overlap) | `data/splits/nantes_siblingleak_{clean,leaky}_v1.json` | `scripts/make_sibling_leak_split.py` |
| Out-of-fold timeline-audit folds | `data/splits/nantes_oof_audit_v1_k{0..4}.json` | `scripts/make_oof_audit_splits.py` |
| Random video-level split matching a follow-up study's description | `data/splits/nantes_random_73_seed0.json` | `scripts/make_embryodiff_split.py` |
| Per-video issue log, tiers A/B/C (Supplementary Table 4) | `data/qc/nantes_video_defects_v4.json`, `results/defect_log.{csv,md}` | `scripts/make_defect_log_v4.py`, `scripts/render_defect_log.py` |
| Label cut-offs for wells that empty during recording | `data/qc/nantes_label_cutoff_v4.json` | `scripts/make_defect_log_v4.py` |
| Earlier, frozen versions of the log (traceability) | `data/qc/nantes_video_defects_v{1,2,3}.json` | `scripts/make_defect_log_v{2,3}.py` |
| Image–annotation timeline alignment per video | `results/frame_timing/alignment{,_oof}.csv` | `scripts/audit_frame_timing.py [--oof]` |
| Model-free timeline mechanism (time-file signature) | `results/frame_timing/mechanism.json` | `scripts/timeline_mechanism.py` |
| Every code-overlap number in the paper | `results/protocol_audit.json` | `scripts/audit_protocol_splits.py` |
| 288 definitions of temporal accuracy (Fig. 2d) | `results/pt_definitions.json` | `scripts/pt_definitions.py` |
| Release manifest: excluded videos, issue log, SHA-256 of every split | `results/tempo_bench_release.json` | `scripts/make_release_manifest.py` |
| Per-run results of 328 runs | `runs/h7/<run>/{results,config.resolved,per_video_test}.json` | `scripts/run_pipeline.py` |
| One row per run (Supplementary Data 1) | `results/runs_master.csv` | `scripts/build_results_master.py` |
| Rendered figures of the article and supplement | `figures/` | `scripts/make_paper_figures.py`, `make_fig_*.py`, `figdata_*.py` |

File names keep their original words (`defect_log`, `nantes_video_defects_*`, `siblingleak`) so that the checksums
and the paths used by the scripts stay valid; the article calls the same objects the issue log and the
code-overlap splits.

The 52 videos that are JPEG-truncated in the released archive are listed under `excluded_videos` in the release
manifest; they are excluded everywhere, leaving 652 videos.

## Installation

```bash
git clone https://github.com/dyytnn/CLEAVE.git && cd CLEAVE
pip install -e ".[dev]"          # Python >= 3.11; developed with torch 2.0.1 / torchvision 0.15.2
```

Every script is run from the repository root with `PYTHONPATH=src` (some also need `scripts` on the path, as noted
in their headers).

## Reproducing the paper

### 1. Every number and table — no GPU, no images needed

```bash
PYTHONPATH=src python scripts/build_results_master.py
```

Reads the per-run result files under `runs/h7/` and writes `results/runs_master.csv`, the article's tables as LaTeX
under `outputs/tables/` (protocol comparison, five-fold comparison, issue log and sensitivity, and the supplementary
configuration, negative-result and semi-Markov tables) and `outputs/numbers.tex` (one macro per in-text number).
Re-running it on a clean clone reproduces all 382 numbers quoted in the manuscript exactly.

```bash
PYTHONPATH=src python scripts/audit_protocol_splits.py      # -> results/protocol_audit.json (leakage counts)
```

### 2. Data

Download the Nantes dataset from Zenodo ([10.5281/zenodo.7912264](https://doi.org/10.5281/zenodo.7912264)) under
its original licence and place (or symlink) it at `data/raw/nantes_embryo_dataset/` with this layout (the two
archives extracted into `ann/` and `time_elapsed/`):

```
data/raw/nantes_embryo_dataset/
├── embryo_dataset/                 # central plane
├── embryo_dataset_F-45/ … embryo_dataset_F45/   # the six other focal planes
├── ann/embryo_dataset_annotations/
├── time_elapsed/embryo_dataset_time_elapsed/
└── embryo_dataset_grades.csv
```

The frame manifest used by every config, `data/derived/nantes_manifest_F0.csv`, is shipped; it can be rebuilt with
`scripts/nantes_preprocess.py`. Training reads pre-resized frames:

```bash
PYTHONPATH=src python scripts/build_nantes_cache.py --planes embryo_dataset embryo_dataset_F15 embryo_dataset_F-15 \
    embryo_dataset_F30 embryo_dataset_F-30 embryo_dataset_F45 embryo_dataset_F-45       # -> data/cache/nantes_250
```

### 3. Training and evaluation

One YAML is one experiment; components are looked up by name in registries (`src/stseg/kinetic/`).

```bash
PYTHONPATH=src python scripts/run_pipeline.py --list                                   # registered components
PYTHONPATH=src python scripts/run_pipeline.py --pipeline_config <config.yaml> --seed 0 # -> runs/h7/<id>_seed0/
PYTHONPATH=src python scripts/run_pipeline.py --pipeline_config <config.yaml> --smoke  # 2-epoch smoke test
```

| Model (Table 2) | Fold 0 | Folds 1–4 | Code-grouped split |
|---|---|---|---|
| Released reference, ResNet-18-LSTM | `configs/h7/resnet18_lstm_L4_split0.yaml` | `configs/h7/v2/resnet18_lstm_L4_split{k}.yaml` | `configs/h7/v35/resnet18_lstm_L4_grouped_v2.yaml` |
| Seven-plane cross-focal transformer | `configs/h7/v9/resnet18_transformer_L16_crossfocal7_evalfix_split0.yaml` | `configs/h7/v11/resnet18_transformer_L16_crossfocal7_evalfix_split{k}.yaml` | `configs/h7/v35/resnet18_transformer_L16_crossfocal7_evalfix_grouped_v2.yaml` |

The configuration of every other run is in `configs/h7/` and, fully resolved, in `runs/h7/<run>/config.resolved.json`.
Each run writes `results.json` and `per_video_test.json`; step 1 then folds them into the tables.

### 4. Audit and figures

```bash
PYTHONPATH=src python scripts/audit_frame_timing.py --oof        # timeline audit (needs the out-of-fold predictions)
PYTHONPATH=src python scripts/timeline_mechanism.py              # model-free check from the release's time files
PYTHONPATH=src:scripts python scripts/make_defect_log_v4.py      # issue log
PYTHONPATH=src python scripts/render_defect_log.py
PYTHONPATH=src:scripts python scripts/make_paper_figures.py --skip_gpu
```

Figures drawn from stored results need nothing else; panels that show frames need the dataset, and panels computed
from a model need its checkpoint.

## Benchmark card (CLEAVE)

- **Intended use:** comparing models for morphokinetic phase recognition on the Nantes data.
- **Split:** `nantes_grouped_v2`, grouped by the code in the video name, which most likely identifies one couple;
  no code is shared between partitions.
  `nantes_grouped_v1` (cycle level) is kept frozen for comparability with the paper.
- **Classes:** the released sixteen.
- **Window rule:** evaluation window = training clip length, stride half, probabilities averaged over overlaps.
- **Metrics:** frame accuracy *p*, Viterbi accuracy *p_v*, temporal accuracy *p_t* with the skipped-phase rule and
  per-event tolerances stated in the paper; per-event onset error in hours; tolerance curve at 0.5/1/2/4 h; segmental
  edit score; F1@{10,25,50}. Implementations: `src/stseg/eval/kinetic_metrics.py`.
- **Mandatory baselines:** the released decoder and a monotone uniform-cost decoder.
- **Tracks:** any-plane and single-plane.
- **Issue log:** report headline numbers on all test videos and, as a sensitivity check, without tier A; train on the
  cleaned split.

## Checkpoints

Trained checkpoints of the models in Table 2 (three seeds, all folds) and the cached frame log-probabilities used by the
retrain-free decoders are available at: **[link to be added]**

Unpack them so that each checkpoint sits at `runs/h7/<run>/best.pt`.

## Tests

```bash
PYTHONPATH=src python -m pytest tests -q
```

Tests that read real frames are skipped until the dataset is in `data/raw/`.

## Licence

Code: MIT (`LICENSE`). Splits, issue log and other data artefacts in `data/` and `results/`: CC BY 4.0
(`LICENSE-DATA`). The Nantes images and annotations remain under their original licence and are not redistributed.

## Citation

See `CITATION.cff`. Please also cite the dataset:

> Gomez T., Feyeux M., Boulant J., Normand N., David L., Paul-Gilloteaux P., Fréour T., Mouchère H. A time-lapse embryo
> dataset for morphokinetic parameter prediction. *Data in Brief* 42, 108258 (2022). doi:10.1016/j.dib.2022.108258
