"""Dense cleavage targets and one-to-one peak detection metrics (no image IO)."""

from __future__ import annotations

import numpy as np

from .event_audit import CLEAVAGE_FROM, binary_summary


def onset_targets(labels: np.ndarray, indices: np.ndarray, radius: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (T,) targets and validity; gaps/starts/skipped stages are uncertain.

    A window around a known onset covers [b-radius, b+radius), as in v23.
    Late stages remain negatives, not an inference-time ground-truth gate.
    """
    y, ix = np.asarray(labels), np.asarray(indices)
    if y.ndim != 1 or y.shape != ix.shape or not len(y) or radius < 0:
        raise ValueError("invalid onset target arrays/radius")
    if not np.isin(y, np.arange(16)).all() or (np.diff(ix) <= 0).any():
        raise ValueError("labels must be 0..15 and frame indices strictly increasing")
    valid = np.r_[False, np.diff(ix) == 1]
    # A jump/reversal in the cleavage interval has no precisely observed onset.
    change = np.r_[False, y[1:] != y[:-1]]
    ambiguous = change & np.r_[False, np.isin(y[:-1], CLEAVAGE_FROM)] & np.r_[False, y[1:] != y[:-1] + 1]
    valid[ambiguous] = False
    exact = valid & np.r_[False, np.isin(y[:-1], CLEAVAGE_FROM) & (y[1:] == y[:-1] + 1)]
    target = exact.astype(np.float32)
    if radius:
        for b in np.flatnonzero(exact):
            target[(ix >= ix[b] - radius) & (ix < ix[b] + radius)] = 1
    return target, valid


def peak_indices(scores: np.ndarray, indices: np.ndarray, valid: np.ndarray, separation: int) -> np.ndarray:
    """Select local peaks with deterministic plateau tie handling and frame-index NMS."""
    q = np.where(valid, scores, -np.inf)
    # Boundaries across missing observations are independent, not neighbours.
    left = np.r_[-np.inf, q[:-1]]
    right = np.r_[q[1:], -np.inf]
    left[np.r_[True, np.diff(indices) != 1]] = -np.inf
    right[np.r_[np.diff(indices) != 1, True]] = -np.inf
    candidates = np.flatnonzero(valid & (q > left) & (q >= right))
    kept = []
    for i in candidates[np.argsort(-q[candidates], kind="stable")]:
        if all(abs(int(indices[i]) - int(indices[j])) > separation for j in kept):
            kept.append(int(i))
    return np.asarray(kept, dtype=int)


def event_metrics(records: list[dict], tolerance: int, separation: int) -> dict:
    """Rank peaks across videos; a GT event can match at most one prediction.

    AP integrates precision at score-group ends, so ties do not get favourable
    per-example ordering. The reported F1 threshold is selected on these records.
    Timing error is conditional on matching, and must be read with recall.
    """
    if tolerance < 0 or separation < 0 or not records:
        raise ValueError("invalid event metric settings")
    predictions, targets = [], []
    for v, rec in enumerate(records):
        q, ix, valid = rec["scores"], rec["indices"], rec["valid"]
        gt = ix[(rec["onset"] == 1) & valid]
        targets.append(gt)
        for i in peak_indices(q, ix, valid, separation):
            predictions.append((float(q[i]), v, int(ix[i])))
    predictions.sort(key=lambda x: (-x[0], x[1], x[2]))
    total = sum(map(len, targets))
    used = [set() for _ in records]
    scores, hits, errors = [], [], []
    for score, v, ix in predictions:
        available = [j for j, gt in enumerate(targets[v]) if j not in used[v] and abs(int(gt) - ix) <= tolerance]
        hit = bool(available)
        if hit:
            j = min(available, key=lambda j: (abs(int(targets[v][j]) - ix), j))
            used[v].add(j)
            errors.append(abs(int(targets[v][j]) - ix))
        else:
            errors.append(None)
        scores.append(score)
        hits.append(hit)
    if not scores or not total:
        return {"event_ap": None if not total else 0.0, "n_gt": total, "n_peaks": len(scores), "best_f1": 0.0, "threshold": None}
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    tp = np.cumsum(hits)[ends]
    precision, recall = tp / (ends + 1), tp / total
    f1 = 2 * tp / (total + ends + 1)
    best = int(f1.argmax())
    matched_errors = [e for e in errors[: ends[best] + 1] if e is not None]
    return {
        "event_ap": float(np.sum(np.diff(np.r_[0, recall]) * precision)),
        "n_gt": total, "n_peaks": len(scores), "best_f1": float(f1[best]),
        "threshold": scores[ends[best]], "precision": float(precision[best]), "recall": float(recall[best]),
        "false_positives_per_video": float((ends[best] + 1 - tp[best]) / len(records)),
        "median_matched_error_frames": float(np.median(matched_errors)) if matched_errors else None,
        "threshold_selected_on_same_validation": True,
    }


def dense_metrics(records: list[dict]) -> dict:
    """Compare scores with EXACT onset support for both target variants."""
    y = np.concatenate([r["onset"][r["valid"]] for r in records])
    q = np.concatenate([r["scores"][r["valid"]] for r in records])
    result = binary_summary(y, q)
    result["onset_ap"] = result.pop("window_ap")
    result["n_onsets"] = result.pop("n_positive_windows")
    return result
