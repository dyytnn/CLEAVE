"""Dataset producers. ``nantes_frames`` = per-frame protocol; ``nantes_clips`` = clip protocol (sequence models).
Both return (train_set, val_frames, test_frames, train_frames_for_transitions)."""

from __future__ import annotations

from stseg.data.nantes_kinetic import (
    NantesEMFiTFrames,
    NantesKineticClips,
    NantesKineticFrames,
)

from .registry import DATASET_REGISTRY


def _frames(d: dict, partition: str, mode: str, seed: int, max_fpv=None, limit_videos=None) -> NantesKineticFrames:
    ds = NantesKineticFrames(d["manifest"], d["split"], partition, mode=mode, plane=d.get("plane", "embryo_dataset"),
                             resize=d.get("resize", 250), crop=d.get("crop", 224), max_frames_per_video=max_fpv,
                             official_offset=bool(d.get("official_offset", False)), seed=seed, planes=d.get("planes"), cache_dir=d.get("cache_dir"),
                             split_mode=d.get("split_mode", "video"), merge_last_class=bool(d.get("merge_last_class", False)),
                             photometric=d.get("photometric"), instance_norm=bool(d.get("instance_norm", False)),
                             plane_mode=d.get("plane_mode"), eval_plane=d.get("eval_plane"), rotation=d.get("rotation"),
                             plane_center_weight=d.get("plane_center_weight"), hires_crop_dir=d.get("hires_crop_dir"),
                             division_score_path=d.get("division_score_path"),
                             division_score_strict=bool(d.get("division_score_strict", False)),
                             label_cutoff=d.get("train_label_cutoff") if partition == "train" else None)
    if limit_videos:
        keep = set(ds.videos()[:limit_videos]); ds.rows = ds.rows[ds.rows.video.isin(keep)].reset_index(drop=True)
    return ds


@DATASET_REGISTRY.register("nantes_frames")
class NantesFramesProducer:
    def __init__(self, d: dict, seed: int, smoke: bool = False) -> None:
        lim = 3 if smoke else None
        # a smoke still needs one whole clip per video, or a clip_len-16 recipe yields zero training clips
        smoke_fpv = max(8, int(d.get("clip_len", 4)))
        self.train_frames = _frames(d, "train", "train", seed, smoke_fpv if smoke else d.get("max_frames_per_video"), lim)
        self.train_full = _frames(d, "train", "eval", seed, None, lim)  # for the transition matrix
        self.val = _frames(d, "val", "eval", seed, 40 if smoke else None, lim)
        self.test = _frames(d, "test", "eval", seed, 40 if smoke else None, lim)
        self.train_set = self.train_frames
        self.in_channels = self.train_frames.in_channels
        self.default_batch = 40


@DATASET_REGISTRY.register("nantes_clips")
class NantesClipsProducer(NantesFramesProducer):
    def __init__(self, d: dict, seed: int, smoke: bool = False) -> None:
        super().__init__(d, seed, smoke)
        L = int(d.get("clip_len", 4))
        cpv = d.get("clips_per_video", "auto")
        if cpv == "auto":  # one epoch ~ N_frames / L clips (paper: sampling with replacement)
            cpv = max(1, round(len(self.train_frames) / (L * max(1, len(self.train_frames.videos())))))
        self.clip_len, self.clips_per_video = L, (2 if smoke else int(cpv))
        self.train_set = NantesKineticClips(self.train_frames, clip_len=L, clips_per_video=self.clips_per_video, seed=seed,
                                             rare_phase_boost=d.get("rare_phase_boost"))
        self.default_batch = 10


@DATASET_REGISTRY.register("nantes_clips_trainval")
class NantesClipsTrainValProducer:
    """Train/validation-only clip producer; never constructs the test partition.

    This is used for patient-safe development rungs whose endpoint must remain
    untouched until a separately authorised confirmatory evaluation.
    """

    def __init__(self, d: dict, seed: int, smoke: bool = False) -> None:
        # One video with at most 20 frames keeps the smoke within the protocol
        # budget while still permitting the inherited L16 clip length.
        lim = 1 if smoke else None
        max_fpv = 20 if smoke else d.get("max_frames_per_video")
        self.train_frames = _frames(d, "train", "train", seed, max_fpv, lim)
        self.train_full = _frames(d, "train", "eval", seed, None, lim)
        self.val = _frames(d, "val", "eval", seed, 20 if smoke else None, lim)
        self.in_channels = self.train_frames.in_channels
        self.default_batch = 10
        length = int(d.get("clip_len", 4))
        clips = d.get("clips_per_video", "auto")
        if clips == "auto":
            clips = max(
                1,
                round(
                    len(self.train_frames)
                    / (length * max(1, len(self.train_frames.videos())))
                ),
            )
        self.clip_len = length
        self.clips_per_video = 2 if smoke else int(clips)
        self.train_set = NantesKineticClips(
            self.train_frames,
            clip_len=length,
            clips_per_video=self.clips_per_video,
            seed=seed,
            rare_phase_boost=d.get("rare_phase_boost"),
        )


@DATASET_REGISTRY.register("nantes_emfit_frames_trainval")
class NantesEMFiTFramesTrainValProducer:
    """Validation-only raw seven-plane producer for the EMFiT reproduction."""

    def __init__(self, d: dict, seed: int, smoke: bool = False) -> None:
        limit_videos = 1 if smoke else None
        max_frames = 8 if smoke else d.get("max_frames_per_video")

        def make(partition: str, mode: str, max_fpv: int | None):
            dataset = NantesEMFiTFrames(
                d["manifest"],
                d["split"],
                partition,
                mode=mode,
                plane=d.get("plane", "embryo_dataset"),
                planes=d.get("planes"),
                max_frames_per_video=max_fpv,
                official_offset=bool(d.get("official_offset", False)),
                seed=seed,
                frame_index_divisor=float(d.get("frame_index_divisor", 415.0)),
                emfit_augmentation=d.get("emfit_augmentation"),
                cache_dir=d.get("cache_dir"),
                emfit_cache_strict=bool(d.get("emfit_cache_strict", False)),
            )
            if limit_videos:
                keep = set(dataset.videos()[:limit_videos])
                dataset.rows = dataset.rows[
                    dataset.rows.video.isin(keep)
                ].reset_index(drop=True)
            return dataset

        self.train_frames = make("train", "train", max_frames)
        self.train_full = make("train", "eval", None)
        self.val = make("val", "eval", 8 if smoke else None)
        self.train_set = self.train_frames
        self.in_channels = 3
        self.default_batch = 4
