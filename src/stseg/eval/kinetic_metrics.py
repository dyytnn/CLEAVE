"""Official Nantes-benchmark evaluation for kinetic-stage prediction (arXiv 2203.00531 §2.4, §2.6).

Given per-frame class log-probabilities for a whole video:
1. **Viterbi decoding** with an empirical 16×16 transition matrix estimated on the *training* label
   sequences (biologically impossible transitions have zero count → never predicted), so the decoded
   sequence is a monotone, biologically plausible chain of phases.
2. **Timings**: the time (hours) of the first frame assigned to each phase. A phase present in the
   ground truth but skipped in the prediction is "implicitly predicted" at the timing of the next
   predicted phase (paper §2.6).
3. **Metrics**: ``p`` frame accuracy of the raw argmax, ``p_v`` frame accuracy after Viterbi, ``r``
   Pearson correlation between predicted and true transition times (transitions present in both),
   ``p_t`` temporal accuracy = share of ground-truth transitions whose predicted time is within the
   phase-specific tolerance ``THETA_H`` (inter-operator SD in hours, paper Table 2; averaged over
   videos as in the paper, pooled value also returned). tPB2 (first event, no transition into it) and
   tHB (no tolerance in Table 2) are excluded from ``p_t``, as in the paper's table.
"""

from __future__ import annotations

import numpy as np

from stseg.data.nantes_kinetic import CLASS_NAMES, NUM_CLASSES, PHASE_TO_CLASS

THETA_H = {  # paper Table 2, hours
    "tPNa": 1.13, "tPNf": 0.50, "t2": 0.91, "t3": 1.81, "t4": 1.34, "t5": 1.49, "t6": 1.61,
    "t7": 2.93, "t8": 5.36, "t9+": 4.42, "tM": 5.46, "tSB": 3.78, "tB": 3.29, "tEB": 4.85,
}
THETA_BY_CLASS = {PHASE_TO_CLASS[k]: v for k, v in THETA_H.items()}
FIXED_TOL_H = (0.5, 1.0, 2.0, 4.0)  # CLEAVE: tolerance curve instead of one θ_p
F1_TAUS = (0.10, 0.25, 0.50)  # segmental F1@k (action segmentation; EmbryoDiff reports these)


def transition_log_matrix(label_seqs: list[np.ndarray], n: int = NUM_CLASSES) -> np.ndarray:
    counts = np.zeros((n, n), dtype=np.float64)
    for s in label_seqs:
        s = np.asarray(s)
        if len(s) > 1:
            np.add.at(counts, (s[:-1], s[1:]), 1.0)
    for i in range(n):  # a phase can always persist
        if counts[i].sum() == 0:
            counts[i, i] = 1.0
    probs = counts / counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.log(probs)


def viterbi(log_probs: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """log_probs (T, n) frame class log-probabilities; returns the best path (T,) of class ids."""
    T, n = log_probs.shape
    score = log_probs[0].copy()
    back = np.zeros((T, n), dtype=np.int64)
    for t in range(1, T):
        cand = score[:, None] + log_trans  # (from, to)
        back[t] = cand.argmax(0)
        score = cand.max(0) + log_probs[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def first_times(labels: np.ndarray, times_h: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for c, t in zip(labels, times_h):
        c = int(c)
        if c not in out:
            out[c] = float(t)
    return out


def fill_skipped(pred: dict[int, float], gt_classes: list[int]) -> dict[int, float]:
    """Skipped phases inherit the timing of the next predicted phase (developmental order)."""
    filled = dict(pred)
    later = sorted(pred)
    for c in gt_classes:
        if c in filled:
            continue
        nxt = [k for k in later if k > c]
        if nxt:
            filled[c] = pred[nxt[0]]
    return filled


def refine_onsets_subframe(path: np.ndarray, log_probs: np.ndarray, times_h: np.ndarray, first_t: dict[int, float]) -> dict[int, float]:
    """TEMPO rung T5b: continuous-time onset refinement, no retraining -- applies to any existing checkpoint.

    The frame-level onset of phase ``c`` is only accurate to +-1 sampling interval (10-20 min for Nantes) because
    Viterbi decodes one *discrete* class per frame. Between the last frame of the previous phase and the first frame
    of ``c``, the model's log-probabilities for the two classes cross continuously; we linearly interpolate that
    crossing point using the two neighbouring frames' log-prob *difference* (a first-order estimate of "when between
    frame i-1 and frame i did the evidence for c overtake the evidence for the previous phase"), giving a sub-frame
    estimate of the transition time. Falls back to the frame time unchanged for phase 0 (no previous phase) or a
    degenerate (zero-slope) crossing.
    """
    refined = dict(first_t)
    for c, t in first_t.items():
        idx = int(np.argmin(np.abs(times_h - t)))
        if idx == 0 or c == 0:
            continue
        prev_class = int(path[idx - 1])
        d_prev = log_probs[idx - 1, c] - log_probs[idx - 1, prev_class]
        d_cur = log_probs[idx, c] - log_probs[idx, prev_class]
        denom = d_cur - d_prev
        if denom == 0:
            continue
        frac = float(np.clip(-d_prev / denom, 0.0, 1.0))
        refined[c] = float(times_h[idx - 1] + frac * (times_h[idx] - times_h[idx - 1]))
    return refined


# ----------------------------------------------------------------------------- CLEAVE additions
def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length segments (class, start, end_exclusive) of a label sequence."""
    out, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append((int(labels[s]), s, i)); s = i
    return out


def edit_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Segmental edit score (Lea et al. 2016): 100 * (1 - Levenshtein(pred segments, gt segments) / max(len))."""
    a = [c for c, _, _ in _segments(pred)]; b = [c for c, _, _ in _segments(gt)]
    d = np.arange(len(b) + 1, dtype=float)
    for i in range(1, len(a) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(b) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return 100.0 * (1.0 - d[len(b)] / max(len(a), len(b), 1))


def f1_at_overlap(pred: np.ndarray, gt: np.ndarray, tau: float) -> tuple[int, int, int]:
    """Segmental F1@tau counts (tp, fp, fn) as in MS-TCN: a predicted segment is a TP if its IoU with an unmatched GT
    segment of the same class exceeds tau."""
    P, G = _segments(pred), _segments(gt)
    used = np.zeros(len(G), bool); tp = 0
    for c, s, e in P:
        best, bi = 0.0, -1
        for k, (cg, sg, eg) in enumerate(G):
            if cg != c or used[k]:
                continue
            inter = max(0, min(e, eg) - max(s, sg)); union = max(e, eg) - min(s, sg)
            iou = inter / union if union else 0.0
            if iou > best:
                best, bi = iou, k
        if best >= tau and bi >= 0:
            tp += 1; used[bi] = True
    return tp, len(P) - tp, len(G) - tp


# ----------------------------------------------------------------------------- TEMPO rung T7: semi-Markov (HSMM) decoding
def segment_durations(label_seqs: list[np.ndarray], n: int = NUM_CLASSES) -> list[list[int]]:
    """Per class: durations (frames) of every run-length segment in the training label sequences."""
    out: list[list[int]] = [[] for _ in range(n)]
    for s in label_seqs:
        for c, a, b in _segments(np.asarray(s)):
            out[c].append(b - a)
    return out


def duration_log_probs(label_seqs: list[np.ndarray], max_dur: int = 600, kind: str = "gamma", alpha: float = 0.5,
                       n: int = NUM_CLASSES) -> np.ndarray:
    """(n, max_dur+1) log P(duration = d | class); column 0 is -inf (a segment has >= 1 frame).

    ``kind``: 'gamma' -- method-of-moments gamma fit per class, discretised on d = 1..max_dur (smooth, works for rare
    classes such as t3 with few training segments); 'empirical' -- add-``alpha`` smoothed histogram; 'flat' -- uniform
    (turns the HSMM into a plain segment-level Markov model, useful as a control)."""
    from math import lgamma
    durs = segment_durations(label_seqs, n)
    d = np.arange(1, max_dur + 1, dtype=np.float64)
    out = np.full((n, max_dur + 1), -np.inf)
    for c in range(n):
        x = np.asarray(durs[c], dtype=np.float64)
        if kind == "flat" or len(x) < 2:
            out[c, 1:] = -np.log(max_dur); continue
        if kind == "empirical":
            h = np.bincount(np.clip(x.astype(int), 1, max_dur), minlength=max_dur + 1)[1:].astype(float)
            out[c, 1:] = np.log((h + alpha) / (h.sum() + alpha * max_dur)); continue
        m, v = x.mean(), max(x.var(), 1.0)
        k, theta = m * m / v, v / m  # shape, scale
        lp = (k - 1) * np.log(d) - d / theta - k * np.log(theta) - lgamma(k)
        lp -= np.logaddexp.reduce(lp)  # renormalise on the discrete support
        out[c, 1:] = lp
    return out


def segment_transition_log_matrix(label_seqs: list[np.ndarray], alpha: float = 0.5, n: int = NUM_CLASSES) -> np.ndarray:
    """Segment-level transitions (class of a segment -> class of the next segment), add-``alpha`` smoothed over forward
    moves; self-transitions (handled by the duration model) and backward moves are impossible."""
    counts = np.zeros((n, n))
    for s in label_seqs:
        segs = [c for c, _, _ in _segments(np.asarray(s))]
        for p, c in zip(segs[:-1], segs[1:]):
            counts[p, c] += 1
    fwd = np.triu(np.ones((n, n)), k=1)
    probs = (counts + alpha) * fwd
    probs /= np.maximum(probs.sum(1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore"):
        lt = np.log(probs)
    lt[fwd == 0] = -np.inf
    return lt


def hsmm_viterbi(log_probs: np.ndarray, log_trans_seg: np.ndarray, log_dur: np.ndarray, lam: float = 1.0,
                 log_init: np.ndarray | None = None) -> np.ndarray:
    """Exact MAP segmentation under a hidden semi-Markov model: score = sum of frame log-probs + ``lam`` * log P(duration)
    per segment + segment-transition log-probs. Unlike frame-level Viterbi (geometric implicit durations), a phase that
    the frame model supports only weakly is *not* silently absorbed by its neighbours when its expected duration says a
    segment should be there, and an over-long segment pays for it. O(T * K * D) with cumulative sums."""
    T, K = log_probs.shape
    D = min(log_dur.shape[1] - 1, T)
    cum = np.vstack([np.zeros((1, K)), np.cumsum(log_probs, 0)])  # cum[t] = sum of frames [0, t)
    ld = (lam * log_dur[:, :D + 1]).T if lam > 0 else np.where(np.isneginf(log_dur[:, :D + 1]), -np.inf, 0.0).T  # (D+1, K); lam=0 keeps d>=1 only
    delta = np.full((T + 1, K), -np.inf); back_d = np.zeros((T + 1, K), dtype=np.int64); back_p = np.full((T + 1, K), -1, dtype=np.int64)
    best_in = np.full((T + 1, K), -np.inf); best_in[0] = 0.0 if log_init is None else log_init
    lt = log_trans_seg.copy(); np.fill_diagonal(lt, -np.inf)
    for t in range(1, T + 1):
        nd = min(D, t); ds = np.arange(1, nd + 1)
        cand = (cum[t][None, :] - cum[t - ds]) + ld[1:nd + 1] + best_in[t - ds]  # (nd, K)
        j = cand.argmax(0); delta[t] = cand[j, np.arange(K)]; back_d[t] = ds[j]
        inc = delta[t][:, None] + lt  # (from, to)
        back_p[t] = inc.argmax(0); best_in[t] = inc.max(0)
    path = np.empty(T, dtype=np.int64)
    t, c = T, int(delta[T].argmax())
    while t > 0:
        d = int(back_d[t, c]); path[t - d:t] = c
        t -= d
        if t > 0:
            c = int(back_p[t, c])
    return path


def timing_errors(pred_t: dict[int, float], gt_t: dict[int, float]) -> dict[int, float]:
    """Signed error (pred - gt, hours) for every phase present in both."""
    return {c: pred_t[c] - gt_t[c] for c in gt_t if c in pred_t}


def evaluate_videos(videos: list[dict], log_trans: np.ndarray, subframe: bool = False, decode=None) -> dict:
    """videos: [{"labels": (T,), "log_probs": (T,n), "times_h": (T,)}, ...] in frame order.
    ``subframe``: apply ``refine_onsets_subframe`` (T5b) to the decoded onsets before computing timing metrics.
    ``decode``: optional ``callable(log_probs) -> path`` replacing frame-level Viterbi (e.g. ``hsmm_viterbi``); ``p_v``
    then reports the accuracy of that decoder."""
    n_ok = n_ok_v = n_tot = 0
    pt_per_video, pred_t_all, gt_t_all = [], [], []
    per_video = []
    fixed_hits = {tol: 0 for tol in FIXED_TOL_H}; err_by_class: dict[int, list[float]] = {}; edits: list[float] = []
    f1_counts = {tau: [0, 0, 0] for tau in F1_TAUS}
    for v in videos:
        y, lp, th = np.asarray(v["labels"]), np.asarray(v["log_probs"]), np.asarray(v["times_h"], dtype=float)
        raw = lp.argmax(1)
        path = viterbi(lp, log_trans) if decode is None else decode(lp)
        n_tot += len(y); n_ok += int((raw == y).sum()); n_ok_v += int((path == y).sum())
        gt_t, pr_t = first_times(y, th), first_times(path, th)
        if subframe:
            pr_t = refine_onsets_subframe(path, lp, th, pr_t)
        gt_classes = sorted(gt_t)
        pr_filled = fill_skipped(pr_t, gt_classes)
        trans = [c for c in gt_classes[1:] if c in THETA_BY_CLASS]  # transitions into phases with a tolerance
        far = 0
        for c in trans:
            if c in pr_t:  # correlation only on transitions present in both (paper)
                pred_t_all.append(pr_t[c]); gt_t_all.append(gt_t[c])
            if c not in pr_filled or abs(pr_filled[c] - gt_t[c]) > THETA_BY_CLASS[c]:
                far += 1
        if trans:
            pt_per_video.append((len(trans) - far) / len(trans))
        # CLEAVE additions: fixed-tolerance hits, signed timing errors, segmental metrics
        for c in trans:
            for tol in FIXED_TOL_H:
                fixed_hits[tol] += int(c in pr_filled and abs(pr_filled[c] - gt_t[c]) <= tol)
        for c, e in timing_errors(pr_filled, gt_t).items():
            if c in THETA_BY_CLASS:
                err_by_class.setdefault(c, []).append(e)
        edits.append(edit_score(path, y))
        for tau in F1_TAUS:
            tp, fp, fn = f1_at_overlap(path, y, tau); f1_counts[tau][0] += tp; f1_counts[tau][1] += fp; f1_counts[tau][2] += fn
        per_video.append({"video": v.get("video"), "n_frames": int(len(y)), "acc": float((raw == y).mean()),
                          "acc_viterbi": float((path == y).mean()), "n_transitions": len(trans), "n_far": far,
                          "edit": edits[-1],
                          "gt_times": {CLASS_NAMES[c]: gt_t[c] for c in gt_t},
                          "pred_times": {CLASS_NAMES[c]: pr_filled[c] for c in pr_filled}})
    r = float(np.corrcoef(pred_t_all, gt_t_all)[0, 1]) if len(pred_t_all) > 2 else float("nan")
    tot_trans = sum(pv["n_transitions"] for pv in per_video); tot_far = sum(pv["n_far"] for pv in per_video)
    return {
        "p": n_ok / max(n_tot, 1), "p_v": n_ok_v / max(n_tot, 1), "r": r,
        "p_t": float(np.mean(pt_per_video)) if pt_per_video else float("nan"),
        "p_t_pooled": (tot_trans - tot_far) / max(tot_trans, 1),
        "p_t_fixed": {f"{tol:g}h": fixed_hits[tol] / max(tot_trans, 1) for tol in FIXED_TOL_H},
        "timing_error_h": {CLASS_NAMES[c]: {"n": len(e), "median_abs": float(np.median(np.abs(e))), "mae": float(np.mean(np.abs(e))),
                                             "bias": float(np.mean(e))} for c, e in sorted(err_by_class.items())},
        "mae_h_all": float(np.mean([abs(x) for e in err_by_class.values() for x in e])) if err_by_class else float("nan"),
        "edit": float(np.mean(edits)) if edits else float("nan"),
        "f1": {f"{int(tau * 100)}": (2 * tp / max(2 * tp + fp + fn, 1)) * 100 for tau, (tp, fp, fn) in f1_counts.items()},
        "n_videos": len(videos), "n_frames": n_tot, "n_transitions": tot_trans,
        "per_video": per_video,
    }
