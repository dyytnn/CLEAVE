"""Validated scalar event-score transport, independent of images and phase labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_event_scores(path: str | Path) -> dict[tuple[str, int], float]:
    """Read unique (video, integer frame_index) -> finite probability in [0, 1]."""
    table = pd.read_csv(path, dtype={"video": str})
    required = {"video", "frame_index", "div_score"}
    if not required.issubset(table.columns) or table.empty:
        raise ValueError(f"event cache requires nonempty columns {sorted(required)}")
    if table[list(required)].isna().any().any() or table.video.str.strip().eq("").any():
        raise ValueError("event cache contains missing values or empty video IDs")
    frames = pd.to_numeric(table.frame_index, errors="raise").to_numpy(dtype=float)
    scores = pd.to_numeric(table.div_score, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(frames).all() or (frames < 0).any() or (frames != np.floor(frames)).any():
        raise ValueError("event cache frame_index must be a finite nonnegative integer")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("event cache div_score must be finite and in [0, 1]")
    keys = list(zip(table.video, frames.astype(np.int64)))
    if len(set(keys)) != len(keys):
        raise ValueError("event cache contains duplicate (video, frame_index) keys")
    return dict(zip(keys, scores))


def require_event_coverage(scores: dict[tuple[str, int], float], rows: pd.DataFrame) -> None:
    """Fail on missing keys, including unlabelled rows when supplied by the caller."""
    missing = sum((str(v), int(f)) not in scores for v, f in zip(rows.video, rows.frame_index))
    if missing:
        raise ValueError(f"event cache missing {missing} required frame keys")
