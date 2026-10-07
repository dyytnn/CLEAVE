"""Sequence decoders expressed as the transition matrix handed to stseg.eval.kinetic_metrics.evaluate_videos."""

from __future__ import annotations

import numpy as np

from stseg.eval.kinetic_metrics import transition_log_matrix

from .interfaces import AbsDecoder
from .registry import DECODER_REGISTRY


@DECODER_REGISTRY.register("argmax")
class ArgmaxDecoder(AbsDecoder):
    """Zero transition costs: Viterbi degenerates to the per-frame argmax (no temporal constraint)."""

    def log_transition(self, seqs, n_classes):
        return np.zeros((n_classes, n_classes))


@DECODER_REGISTRY.register("viterbi")
class ViterbiDecoder(AbsDecoder):
    """Official protocol: empirical 16x16 transition matrix from the training labels; impossible moves get -inf."""

    def log_transition(self, seqs, n_classes):
        return transition_log_matrix(seqs, n_classes)


@DECODER_REGISTRY.register("monotonic")
class MonotonicDecoder(AbsDecoder):
    """Order-only prior: any forward move (or staying) costs 0, any backward move is impossible. Unlike ``viterbi`` it
    learns nothing from the training label statistics (no self-transition bias, no forbidden skips), so the comparison
    viterbi vs monotonic isolates what the empirical transition counts contribute (CLEAVE rung T3)."""

    def log_transition(self, seqs, n_classes):
        m = np.zeros((n_classes, n_classes))
        m[np.tril_indices(n_classes, k=-1)] = -np.inf  # row = from, col = to; to < from forbidden
        return m
