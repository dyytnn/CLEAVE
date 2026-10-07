"""WP9/H7: per-frame kinetic-stage classification dataset for the Nantes morphokinetic dataset.

Follows the official benchmark protocol (Gomez et al., arXiv 2203.00531; bench_mk_pred/load_data.py),
see "Official benchmark code check":

* **16 classes, no background** — the 16 kinetic events in developmental order (same index order as the
  official ``getLabels()``). Frames outside the annotated window (before tPB2, or after the last
  annotated phase) are *dropped*, exactly as the official loader skips ``gt == -1`` frames and never
  loads frames past the last phase. Blank empty-well frames are always dropped as well.
* **Frame ↔ image alignment**: ours is ``frame_index k ↔ RUNk`` (visually validated on t2 onsets). The
  official loader is off by one (positional 0-based indexing, ``frame k ↔ RUN(k+1)``); pass
  ``official_offset=True`` only to reproduce their numbers exactly.
* **Preprocessing** (official): resize 500→``resize`` (250), then a random ``crop``×``crop`` (224) crop with
  horizontal+vertical flips (p=0.5) for training, centre crop for evaluation; grayscale replicated to 3
  channels and ImageNet-normalised (the backbone is ImageNet-pretrained).

Reads the manifest from ``scripts/nantes_preprocess.py`` (one row per frame: video, plane, frame_index,
path, time_h, phase, is_tail, focus, is_blank, grade_TE, grade_ICM) and a split that is either our
patient-grouped JSON (``scripts/make_nantes_split.py``) or an official ``split{k}.csv`` (rows
``video,partition``, 5 folds, video-level, copied to ``data/splits/nantes_official/``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

PHASES = [
    "tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+",
    "tM", "tSB", "tB", "tEB", "tHB",
]
CLASS_NAMES = list(PHASES)  # index == official getLabels() index
PHASE_TO_CLASS = {name: i for i, name in enumerate(PHASES)}
NUM_CLASSES = len(PHASES)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MIN_PER_FRAME_H = 10.0 / 60.0  # fallback spacing when time_h is missing
EMFIT_PLANES = [
    "embryo_dataset_F-45",
    "embryo_dataset_F-30",
    "embryo_dataset_F-15",
    "embryo_dataset",
    "embryo_dataset_F15",
    "embryo_dataset_F30",
    "embryo_dataset_F45",
]


def load_split_videos(split_path: str | Path, partition: str) -> set[str]:
    p = Path(split_path)
    if partition not in {"train", "val", "test"}:
        raise ValueError(f"unknown partition {partition!r}")
    if p.suffix == ".json":
        return set(json.loads(p.read_text())["videos"][partition])
    with open(p) as f:
        return {row[0] for row in csv.reader(f) if len(row) >= 2 and row[1].strip() == partition}


def _fill_time(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing time_h within each video over frame_index; fallback to 10 min/frame."""
    def fix(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("frame_index")
        t = g.time_h.astype(float)
        if t.notna().sum() >= 2:
            t = t.interpolate(limit_direction="both")
        elif t.notna().sum() == 1:
            k0 = g.frame_index[t.notna()].iloc[0]
            t = t.fillna(t.dropna().iloc[0] + (g.frame_index - k0) * MIN_PER_FRAME_H)
        else:
            t = g.frame_index * MIN_PER_FRAME_H
        return g.assign(time_h=t.to_numpy())
    return df.groupby("video", group_keys=False)[df.columns].apply(fix)


class NantesKineticFrames(Dataset):
    """One row per labelled, non-blank frame; label in 0..15 (official class order).

    Args:
        manifest_csv: per-plane manifest, or nantes_manifest_all.csv (then pass ``plane``).
        split: JSON (patient-grouped) or official split CSV.
        partition: train | val | test.
        mode: "train" (random crop + flips) | "eval" (centre crop). Independent of ``partition`` so a
            train-partition set can be evaluated deterministically.
        plane: manifest plane folder name (F0 = "embryo_dataset"); None keeps all rows.
        max_frames_per_video: optional even subsample per video (training-time speed knob).
        official_offset: reproduce the official loader's one-frame shift (label of RUNk := phase(k-1)).
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str | Path,
        partition: str,
        mode: str = "train",
        plane: str | None = "embryo_dataset",
        resize: int = 250,
        crop: int = 224,
        max_frames_per_video: int | None = None,
        official_offset: bool = False,
        seed: int = 0,
        planes: list[str] | None = None,
        cache_dir: str | Path | None = None,
        split_mode: str = "video",
        merge_last_class: bool = False,
        photometric: dict | None = None,
        instance_norm: bool = False,
        plane_mode: str | None = None,
        eval_plane: str | None = None,
        rotation: dict | None = None,
        plane_center_weight: float | None = None,
        hires_crop_dir: str | Path | None = None,
        division_score_path: str | Path | None = None,
        division_score_strict: bool = False,
        label_cutoff: str | Path | None = None,
    ) -> None:
        """``photometric`` (TEMPO rung T8a, train mode only): {"brightness": 0.2, "contrast": 0.3, "gamma": 0.3, "p": 0.8}
        -- per clip, with probability p, draw brightness shift U(-b, b), contrast factor U(1-c, 1+c) and gamma
        exp(U(-g, g)) and apply them identically to every frame (and every focal plane) of the clip. Motivated by
        diagnostics_findings.md finding 6: three dim / low-contrast test embryos are unrecoverable for every model
        trained with the official flip-only augmentation. ``instance_norm``: standardise every frame to zero mean /
        unit variance over its own pixels (per channel) instead of the fixed dataset mean/std -- removes absolute
        illumination as a cue at train *and* eval time.

        ``planes``: focal-plane folder names stacked as input channels, e.g.
        ["embryo_dataset_F-15", "embryo_dataset", "embryo_dataset_F15"] (same RUN file name under each plane folder);
        None = the manifest plane replicated to 3 channels (official protocol)."""
        if mode not in {"train", "eval"}:
            raise ValueError(f"mode must be train|eval, got {mode!r}")
        df = pd.read_csv(manifest_csv)
        if plane is not None:
            df = df[df.plane == plane]
        if split_mode == "video":
            df = df[df.video.isin(load_split_videos(split, partition))]
        elif split_mode == "image":
            # CLEAVE defect demo: frames of *every* video are split 70/10/20 at random (as an image-level protocol
            # would), so neighbouring frames of the same embryo sit in train and test. Deterministic hash, seed-free.
            allv = set().union(*(load_split_videos(split, p) for p in ("train", "val", "test")))
            df = df[df.video.isin(allv)]
            h = pd.util.hash_pandas_object(df.video.astype(str) + "/" + df.frame_index.astype(str), index=False).to_numpy() % 100
            lo, hi = {"train": (0, 70), "val": (70, 80), "test": (80, 100)}[partition]
            df = df[(h >= lo) & (h < hi)]
        else:
            raise ValueError(f"split_mode must be video|image, got {split_mode!r}")
        df = df.sort_values(["video", "frame_index"]).reset_index(drop=True)
        if official_offset:
            df = df.assign(phase=df.groupby("video").phase.shift(1))
        if label_cutoff:
            #: for embryos removed from the well (transfer or freezing) the release extends the last
            # annotated phase to the end of the recording; frames after the listed hour carry no embryo and are dropped.
            # Passed for the training partition only, so evaluation partitions stay identical to the uncut split.
            cut = json.loads(Path(label_cutoff).read_text())["cutoff_h"]
            late = df.video.map(cut).astype(float)
            df = df[~(late.notna() & (df.time_h > late))]
        df = df[df.phase.notna() & (df.is_blank != True)]  # noqa: E712 — drop unlabelled + blank
        df = df.assign(label=df.phase.map(PHASE_TO_CLASS).astype(int))
        if merge_last_class:  # EmbryoDiff-style 15-class scheme (Sun et al. 2025): tHB merged into tEB (rare class,
            # dropped by them "due to rarity"); implemented as a label remap rather than shrinking NUM_CLASSES, so the
            # rest of the pipeline (evaluate_videos, CLASS_NAMES-indexed code) needs no change — class 15 (tHB) simply
            # never occurs. See track1_TEMPO/protocol_audit.md.
            df = df.assign(label=df.label.replace(NUM_CLASSES - 1, NUM_CLASSES - 2))
        df = _fill_time(df)
        if max_frames_per_video:
            df = df.groupby("video", group_keys=False)[df.columns].apply(
                lambda g: g.iloc[np.linspace(0, len(g) - 1, min(len(g), max_frames_per_video)).round().astype(int)]
            )
        self.rows = df.sort_values(["video", "frame_index"]).reset_index(drop=True)
        self.mode, self.resize, self.crop = mode, int(resize), int(crop)
        self.planes = list(planes) if planes else None
        self.base_plane = plane or "embryo_dataset"
        # optional cache of pre-resized (resize x resize) JPEGs on fast storage, built by scripts/build_nantes_cache.py:
        # <cache_dir>/<plane>/<video>/<file>; falls back to the original path when a cached file is missing.
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.in_channels = len(self.planes) if self.planes else 3
        if self.in_channels == 3:
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        else:  # channel-agnostic grey statistics
            self.mean = torch.full((self.in_channels, 1, 1), 0.45); self.std = torch.full((self.in_channels, 1, 1), 0.225)
        self.rng = np.random.default_rng(seed)
        self.photometric = dict(photometric) if photometric else None
        self.instance_norm = bool(instance_norm)
        # TEMPO v17: planes as *augmentation* instead of fusion. plane_mode="random_single": ``planes`` lists the candidate
        # folders, each training clip is loaded from ONE randomly drawn plane (grey replicated to 3 channels, so the model
        # stays a single-plane model), evaluation uses ``eval_plane`` (default: the manifest plane). Motivated by the finding
        # that the cross-focal attention learned a static plane weighting, i.e. the 7-plane gain is information/diversity,
        # not fusion -- if so a single-plane model trained on all planes (+ plane-TTA at test) should match it and remain
        # deployable on single-plane systems.
        self.plane_mode = plane_mode
        self.plane_pool = list(planes) if (plane_mode == "random_single" and planes) else None
        if self.plane_pool:
            self.planes = None; self.in_channels = 3; self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        self.eval_plane = eval_plane
        # TEMPO v18 (post-mortem of v17 A/D): plane_mode="random_single" gave the eval plane only ~1/7 of the training
        # exposure a dedicated single-plane model gets in the same epoch budget (history.json: train_loss stays well above
        # control at every epoch) -- a data-efficiency cost, not an eval artifact (7-plane TTA did not recover it either,
        # see). ``plane_center_weight``: probability of drawing the eval plane on any given clip: the
        # remaining probability mass is split uniformly over the other planes in the pool, so the eval plane's training
        # exposure over the run approaches ``plane_center_weight`` * (epochs) instead of 1/len(pool) * (epochs), while
        # augmentation on the other planes is retained (weaker). None = uniform (original v17 behaviour).
        self.plane_center_weight = float(plane_center_weight) if plane_center_weight is not None else None
        self.plane_probs = None
        if self.plane_pool and self.plane_center_weight is not None:
            n = len(self.plane_pool)
            center = self.eval_plane or self.base_plane
            if center not in self.plane_pool:
                raise ValueError(f"plane_center_weight set but eval_plane {center!r} not in plane pool {self.plane_pool}")
            w = np.array([self.plane_center_weight if p == center else (1.0 - self.plane_center_weight) / (n - 1)
                          for p in self.plane_pool])
            self.plane_probs = w / w.sum()
        # rotation augmentation (label-preserving: round embryo in a round well): {"p": 0.75, "deg": "any" | 90}
        self.rotation = dict(rotation) if rotation else None
        # TEMPO v21: dual-branch cell-counting backbone. hires_crop_dir points at a cache built by
        # scripts/build_nantes_embryo_crop_cache.py (one embryo-centred crop per frame, independent of plane_mode/planes)
        # appended as one extra channel; the dualbranch backbone (kinetic/backbones.py) splits it back off and feeds it
        # to its own YOLOv8-derived sub-network. Extra channel gets mean=0/std=1 (no-op) since that branch normalises
        # internally with its own /255-only convention, not this dataset's ImageNet/grey stats.
        self.hires_crop_dir = Path(hires_crop_dir) if hires_crop_dir else None
        if self.hires_crop_dir is not None:
            self.in_channels += 1
            self.mean = torch.cat([self.mean, torch.zeros(1, 1, 1)])
            self.std = torch.cat([self.std, torch.ones(1, 1, 1)])
        # follow-up: a per-frame division-event score (sigmoid output of the N3-warm-started
        # `scripts/train_division_detector.py` model, precomputed by scripts/build_division_score_cache.py over EVERY
        # frame of EVERY video, train+val+test -- no leakage, since the detector itself was only ever trained on
        # train-partition videos) appended as one extra CONSTANT-valued channel, exactly like ``hires_crop_dir``
        # (mean=0/std=1 no-op passthrough since the raw [0,1] score is already the informative value). A constant
        # channel through ``instance_norm`` would collapse to ~0/undefined; not combined with that option here.
        self.division_score_path = Path(division_score_path) if division_score_path else None
        self.division_score_strict = bool(division_score_strict)
        self.division_score: dict[tuple[str, int], float] | None = None
        if self.division_score_strict and (self.division_score_path is None or self.instance_norm):
            raise ValueError("strict event scores require a cache and forbid instance_norm")
        if self.division_score_path is not None:
            if self.division_score_strict:
                from stseg.data.event_scores import load_event_scores, require_event_coverage

                self.division_score = load_event_scores(self.division_score_path)
                require_event_coverage(self.division_score, self.rows)
            else:  # Preserve the frozen v22 lookup semantics.
                sc = pd.read_csv(self.division_score_path)
                self.division_score = dict(zip(zip(sc.video, sc.frame_index.astype(int)), sc.div_score.astype(float)))
            self.in_channels += 1
            self.mean = torch.cat([self.mean, torch.zeros(1, 1, 1)])
            self.std = torch.cat([self.std, torch.ones(1, 1, 1)])

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.rows)

    def class_counts(self) -> dict[str, int]:
        c = self.rows.label.value_counts().to_dict()
        return {CLASS_NAMES[k]: int(v) for k, v in sorted(c.items())}

    def videos(self) -> list[str]:
        return list(self.rows.video.unique())

    def label_sequences(self) -> list[np.ndarray]:
        """Per-video label sequences in frame order (for the empirical transition matrix)."""
        return [g.label.to_numpy() for _, g in self.rows.groupby("video", sort=False)]

    # ------------------------------------------------------------------ items
    def sample_aug(self) -> tuple[int, int, bool, bool]:
        """Crop offset + flips: random in train mode, centre/no-flip in eval mode."""
        m = self.resize - self.crop
        if self.mode == "train":
            return int(self.rng.integers(0, m + 1)), int(self.rng.integers(0, m + 1)), self.rng.random() < 0.5, self.rng.random() < 0.5
        return m // 2, m // 2, False, False

    def sample_photo(self) -> tuple[float, float, float] | None:
        """(brightness shift, contrast factor, gamma) for one clip, or None (identity). Train mode only."""
        ph = self.photometric
        if ph is None or self.mode != "train" or self.rng.random() >= float(ph.get("p", 0.8)):
            return None
        b, c, g = float(ph.get("brightness", 0.2)), float(ph.get("contrast", 0.3)), float(ph.get("gamma", 0.3))
        return (float(self.rng.uniform(-b, b)), float(self.rng.uniform(1 - c, 1 + c)), float(np.exp(self.rng.uniform(-g, g))))

    def sample_extra(self) -> dict:
        """Per-clip augmentation draws that are not crop/flip: photometric params, rotation angle (deg) and, in
        plane_mode='random_single', the plane folder to read from. Identity in eval mode."""
        ex: dict = {"photo": self.sample_photo(), "angle": 0.0, "plane": None}
        if self.mode == "train":
            if self.rotation and self.rng.random() < float(self.rotation.get("p", 0.75)):
                deg = self.rotation.get("deg", "any")
                ex["angle"] = float(self.rng.uniform(0, 360)) if deg == "any" else float(self.rng.integers(1, 4) * 90)
            if self.plane_pool:
                if self.plane_probs is not None:
                    ex["plane"] = str(self.rng.choice(self.plane_pool, p=self.plane_probs))
                else:
                    ex["plane"] = self.plane_pool[int(self.rng.integers(0, len(self.plane_pool)))]
        return ex

    @staticmethod
    def apply_photo(a: np.ndarray, photo: tuple[float, float, float] | None) -> np.ndarray:
        """``a`` in [0, 1]: contrast about mid-grey, brightness shift, then gamma; clipped back to [0, 1]."""
        if photo is None:
            return a
        b, c, g = photo
        return np.clip((a - 0.5) * c + 0.5 + b, 0.0, 1.0) ** g

    def _plane_path(self, path: str, plane: str) -> str:
        return path.replace(f"/{self.base_plane}/", f"/{plane}/", 1)

    def _cached(self, path: str) -> str:
        if self.cache_dir is None:
            return path
        parts = Path(path).parts  # .../<plane>/<video>/<file>
        c = self.cache_dir / parts[-3] / parts[-2] / (Path(parts[-1]).stem + ".jpg")
        return str(c) if c.exists() else path

    def _hires_path(self, path: str) -> str:
        """<hires_crop_dir>/<video>/<file>.jpg, matching scripts/build_nantes_embryo_crop_cache.py's layout."""
        parts = Path(path).parts  # .../<plane>/<video>/<file>
        return str(self.hires_crop_dir / parts[-2] / (Path(parts[-1]).stem + ".jpg"))

    def load_frame(self, path: str, y: int, x: int, fh: bool, fv: bool, photo: tuple[float, float, float] | None = None,
                   angle: float = 0.0, plane: str | None = None, video: str | None = None, frame_index: int | None = None) -> torch.Tensor:
        """(C, crop, crop) normalised tensor; C = 3 replicated grey (default) or one channel per focal plane, plus one
        more channel if ``hires_crop_dir`` is set.
        ``plane``: read this focal-plane folder instead of the manifest plane (plane_mode='random_single' / plane-TTA);
        ``angle``: rotate the resized frame about its centre before cropping (degrees)."""
        if self.planes:
            paths = [self._plane_path(path, pl) for pl in self.planes]
        else:
            pl = plane or self.eval_plane
            paths = [self._plane_path(path, pl) if pl and pl != self.base_plane else path]
        chans = []
        for pth in paths:
            img = Image.open(self._cached(pth)).convert("L")
            if img.size != (self.resize, self.resize):
                img = img.resize((self.resize, self.resize), Image.Resampling.BILINEAR)
            if angle:
                img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            a = np.asarray(img, dtype=np.float32)[y : y + self.crop, x : x + self.crop] / 255.0
            a = self.apply_photo(a, photo)
            if fv: a = a[::-1]
            if fh: a = a[:, ::-1]
            chans.append(torch.from_numpy(np.ascontiguousarray(a)))
        t = torch.stack(chans) if self.planes else chans[0][None].expand(3, -1, -1)
        if self.hires_crop_dir is not None:
            himg = Image.open(self._hires_path(path)).convert("L")
            if himg.size != (self.crop, self.crop):
                himg = himg.resize((self.crop, self.crop), Image.Resampling.BILINEAR)
            if angle:
                himg = himg.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            ha = np.asarray(himg, dtype=np.float32) / 255.0
            if fv: ha = ha[::-1]
            if fh: ha = ha[:, ::-1]
            hchan = torch.from_numpy(np.ascontiguousarray(ha).copy())
            t = torch.cat([t, hchan[None]], dim=0)
        if self.division_score is not None:
            if getattr(self, "division_score_strict", False) and (video, frame_index) not in self.division_score:
                raise ValueError("strict event lookup requires a cached video/frame_index key")
            score = self.division_score.get((video, frame_index), 0.5) if video is not None and frame_index is not None else 0.5
            dchan = torch.full((self.crop, self.crop), float(score), dtype=torch.float32)
            t = torch.cat([t, dchan[None]], dim=0)
        if self.instance_norm:
            mu = t.mean(dim=(1, 2), keepdim=True); sd = t.std(dim=(1, 2), keepdim=True)
            return (t - mu) / (sd + 1e-3)
        return (t - self.mean) / self.std

    def _load(self, path: str, video: str | None = None, frame_index: int | None = None) -> torch.Tensor:
        ex = self.sample_extra()
        return self.load_frame(path, *self.sample_aug(), photo=ex["photo"], angle=ex["angle"], plane=ex["plane"],
                               video=video, frame_index=frame_index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.rows.iloc[i]
        return {
            "image": self._load(r.path, r.video, int(r.frame_index)),
            "label": int(r.label),
            "video": r.video,
            "frame_index": int(r.frame_index),
            "time_h": float(r.time_h),
        }


class _EMFiTGaussianNoise:
    """Official EMFiT additive Gaussian image noise followed by `[0,1]` clipping."""

    def __init__(self, std: float) -> None:
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + torch.randn_like(x) * self.std, 0.0, 1.0)


class NantesEMFiTFrames(NantesKineticFrames):
    """One seven-plane frame with the exact public EMFiT preprocessing contract.

    Every focal plane draws its training transform independently, matching the
    authors' released dataset loop.  Images have shape `(3,7,224,224)` and the
    scalar frame input is the un-clipped `frame_index / frame_index_divisor`.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str | Path,
        partition: str,
        mode: str,
        plane: str = "embryo_dataset",
        planes: list[str] | None = None,
        max_frames_per_video: int | None = None,
        official_offset: bool = False,
        seed: int = 0,
        frame_index_divisor: float = 415.0,
        emfit_augmentation: dict[str, float] | None = None,
        cache_dir: str | Path | None = None,
        emfit_cache_strict: bool = False,
    ) -> None:
        super().__init__(
            manifest_csv,
            split,
            partition,
            mode=mode,
            plane=plane,
            resize=224,
            crop=224,
            max_frames_per_video=max_frames_per_video,
            official_offset=official_offset,
            seed=seed,
            planes=None,
            cache_dir=cache_dir,
            split_mode="video",
        )
        from torchvision import transforms

        self.emfit_planes = list(planes or EMFIT_PLANES)
        if self.emfit_planes != EMFIT_PLANES:
            raise ValueError(
                "EMFiT reproduction requires the ordered F-45..F45 seven-plane stack"
            )
        if frame_index_divisor <= 0:
            raise ValueError("frame_index_divisor must be positive")
        self.frame_index_divisor = float(frame_index_divisor)
        self.emfit_cache_strict = bool(emfit_cache_strict)
        if self.emfit_cache_strict and self.cache_dir is None:
            raise ValueError("strict EMFiT cache mode requires cache_dir")
        aug = {
            "crop_scale_min": 0.8,
            "rotation_deg": 45.0,
            "brightness": 0.2,
            "contrast": 0.2,
            "noise_std": 0.04,
        }
        if emfit_augmentation:
            unknown = set(emfit_augmentation) - set(aug)
            if unknown:
                raise ValueError(f"unknown EMFiT augmentation keys: {sorted(unknown)}")
            aug.update({key: float(value) for key, value in emfit_augmentation.items()})
        if not 0 < aug["crop_scale_min"] <= 1 or any(
            aug[key] < 0 for key in ("rotation_deg", "brightness", "contrast", "noise_std")
        ):
            raise ValueError(f"invalid EMFiT augmentation: {aug}")
        if mode == "train":
            self.emfit_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224,
                        scale=(aug["crop_scale_min"], 1.0),
                    ),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomRotation(aug["rotation_deg"]),
                    transforms.ColorJitter(
                        brightness=aug["brightness"],
                        contrast=aug["contrast"],
                    ),
                    transforms.ToTensor(),
                    _EMFiTGaussianNoise(aug["noise_std"]),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        else:
            self.emfit_transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        self.in_channels = 3

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows.iloc[i]
        focal = []
        for plane in self.emfit_planes:
            path = self._plane_path(row.path, plane)
            cached = self._cached(path)
            if self.emfit_cache_strict and cached == path:
                raise FileNotFoundError(f"missing strict EMFiT cache entry for {path}")
            image = Image.open(cached).convert("L")
            focal.append(self.emfit_transform(image).repeat(3, 1, 1))
        image = torch.stack(focal, dim=1)
        frame_input = torch.tensor(
            [float(row.frame_index) / self.frame_index_divisor],
            dtype=torch.float32,
        )
        return {
            "image": image,
            "label": int(row.label),
            "video": row.video,
            "frame_index": int(row.frame_index),
            "frame_input": frame_input,
            "time_h": float(row.time_h),
        }


class NantesKineticClips(Dataset):
    """Clips of ``clip_len`` consecutive labelled frames of one video, for the sequence baselines.

    Official protocol (arXiv 2203.00531 §2.7 / bench_mk_pred load_data.py): a training batch is 10 sequences of 4
    consecutive images; the crop/flip augmentation is drawn once per clip and applied identically to every frame of
    the clip. An epoch is ``clips_per_video`` random windows per video (uniform random start), i.e. sampling with
    replacement as in the paper. Frames come from an already-filtered ``NantesKineticFrames`` (labelled, non-blank).
    Returns image (L, 3, crop, crop), label (L,), video, frame_index (L,), time_h (L,).
    """

    def __init__(self, frames: NantesKineticFrames, clip_len: int = 4, clips_per_video: int = 20, seed: int = 0,
                 rare_phase_boost: dict | None = None) -> None:
        """``rare_phase_boost`` (TEMPO v19, post-mortem of v17/v18): {"phases": ["t3","t5","t7"], "weight": 3.0} --
        biases the random clip-start position (mode="train" only) toward windows that contain at least one frame of
        the listed rare phases, instead of the official uniform-random start. A window containing a target phase gets
        sampling weight (1 + weight) vs. 1 for a window that doesn't, i.e. weight=3.0 makes it 4x as likely to be
        picked as an otherwise-equal window. Motivated by diagnostics_findings.md /: t3/t5/t7 are
        1.7-3.6% of frames with 4-8 frame median segments, and every arm so far attacks this indirectly (loss
        smoothing, longer training); this attacks the sampling distribution directly."""
        self.f, self.L, self.k = frames, int(clip_len), int(clips_per_video)
        self.groups = [g.index.to_numpy() for _, g in frames.rows.groupby("video", sort=False) if len(g) >= self.L]
        self.rng = np.random.default_rng(seed)
        self.start_probs: list[np.ndarray] | None = None
        if rare_phase_boost:
            target = {PHASE_TO_CLASS[p] for p in rare_phase_boost["phases"]}
            boost = float(rare_phase_boost.get("weight", 3.0))
            labels = frames.rows.label.to_numpy()
            self.start_probs = []
            for idx in self.groups:
                lab = labels[idx]
                n_starts = len(idx) - self.L + 1
                is_rare = np.isin(lab, list(target)).astype(np.float64)
                cum = np.concatenate([[0.0], np.cumsum(is_rare)])
                has_rare = (cum[self.L:] - cum[:-self.L]) > 0
                w = np.where(has_rare, 1.0 + boost, 1.0)
                self.start_probs.append(w / w.sum())

    def __len__(self) -> int:
        return len(self.groups) * self.k

    def __getitem__(self, i: int) -> dict[str, Any]:
        gi = i // self.k
        rows_idx = self.groups[gi]
        n_starts = len(rows_idx) - self.L + 1
        if self.start_probs is not None and self.f.mode == "train":
            start = int(self.rng.choice(n_starts, p=self.start_probs[gi]))
        else:
            start = int(self.rng.integers(0, n_starts))
        sel = self.f.rows.iloc[rows_idx[start : start + self.L]]
        y, x, fh, fv = self.f.sample_aug()  # one augmentation per clip (official protocol)
        ex = self.f.sample_extra()  # photometric / rotation / plane drawn once per clip (identity in eval mode)
        imgs = [self.f.load_frame(path, y, x, fh, fv, ex["photo"], ex["angle"], ex["plane"], video, int(fi))
                for path, video, fi in zip(sel.path, sel.video, sel.frame_index)]
        return {"image": torch.stack(imgs), "label": torch.tensor(sel.label.to_numpy(), dtype=torch.long),
                "video": sel.video.iloc[0], "frame_index": torch.tensor(sel.frame_index.to_numpy()),
                "time_h": torch.tensor(sel.time_h.to_numpy(dtype=np.float32))}
