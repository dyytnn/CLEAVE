"""Losses over per-frame logits (B, L, K) and labels (B, L)."""

from __future__ import annotations

import torch
from torch import nn

from stseg.data.nantes_kinetic import CLASS_NAMES

from .interfaces import AbsLoss
from .registry import LOSS_REGISTRY

# Ordinal cell-count level per phase (0-based): tPB2/tPNa/tPNf = 1 cell (0), t2..t8 = 2..8 cells (1..7),
# t9+ and everything after (morula, blastocyst stages: compacted / >16 cells) = "9 or more" (8).
_COUNT_LEVEL = {"tPB2": 0, "tPNa": 0, "tPNf": 0, "t2": 1, "t3": 2, "t4": 3, "t5": 4, "t6": 5, "t7": 6, "t8": 7,
                "t9+": 8, "tM": 8, "tSB": 8, "tB": 8, "tEB": 8, "tHB": 8}
CELL_COUNT_LEVEL = torch.tensor([_COUNT_LEVEL[c] for c in CLASS_NAMES])


@LOSS_REGISTRY.register("ce")
class CrossEntropy(AbsLoss):
    def __init__(self, class_counts: torch.Tensor | None = None, num_classes: int = 16) -> None:
        self.ce = nn.CrossEntropyLoss()

    def __call__(self, logits, y):
        return self.ce(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@LOSS_REGISTRY.register("emfit_multitask")
class EMFiTMultiTaskLoss(AbsLoss):
    """Published EMFiT objective: phase CE plus weighted frame-index MSE."""

    needs_outputs = True

    def __init__(
        self,
        class_counts: torch.Tensor | None = None,
        num_classes: int = 16,
        frame_weight: float = 5.0,
    ) -> None:
        del class_counts, num_classes
        if frame_weight < 0:
            raise ValueError("frame_weight must be nonnegative")
        self.frame_weight = float(frame_weight)
        self.ce = nn.CrossEntropyLoss()

    def __call__(self, out, y):
        if not isinstance(out, dict) or not {
            "logits",
            "frame_regression",
            "frame_target",
        }.issubset(out):
            raise ValueError("emfit_multitask requires EMFiT output fields")
        logits = out["logits"]
        classification = self.ce(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
        regression = nn.functional.mse_loss(
            out["frame_regression"].float(), out["frame_target"].float()
        )
        return classification + self.frame_weight * regression


@LOSS_REGISTRY.register("ce_weighted")
class WeightedCrossEntropy(AbsLoss):
    """Class weights = sqrt(inverse frequency) of the training labels, normalised to mean 1 (long phases dominate)."""

    def __init__(self, class_counts: torch.Tensor, num_classes: int = 16) -> None:
        w = (class_counts.sum() / class_counts.clamp(min=1)).sqrt(); w = w / w.mean()
        self.weight = w
        self.ce = None

    def __call__(self, logits, y):
        if self.ce is None:
            self.ce = nn.CrossEntropyLoss(weight=self.weight.to(logits.device))
        return self.ce(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@LOSS_REGISTRY.register("ce_ordinal")
class OrdinalCrossEntropy(AbsLoss):
    """CE + lambda * |E[class index] - y| / (K-1): phases are ordered, so far-off confusions cost more."""

    def __init__(self, class_counts: torch.Tensor | None = None, num_classes: int = 16, lam: float = 0.1) -> None:
        self.ce, self.lam, self.K = nn.CrossEntropyLoss(), float(lam), int(num_classes)

    def __call__(self, logits, y):
        lg = logits.reshape(-1, logits.shape[-1]); yy = y.reshape(-1)
        idx = torch.arange(self.K, device=lg.device, dtype=torch.float32)
        expect = (torch.softmax(lg.float(), 1) * idx).sum(1)
        return self.ce(lg, yy) + self.lam * (expect - yy.float()).abs().mean() / (self.K - 1)


@LOSS_REGISTRY.register("ce_boundary_weighted")
class BoundaryWeightedCrossEntropy(AbsLoss):
    """TEMPO rung T5a: up-weights frames near a phase transition (a boundary is where ``y[t] != y[t-1]`` within the
    clip -- computed straight from the label sequence, no extra data needed). Motivation: clinically, the exact hour
    of a transition matters more than getting an interior frame right, and boundary frames are the hardest (visually
    ambiguous, minority of frames) -- symmetric to how object-detection losses up-weight boundary pixels.

    ``weight(t) = 1 + bonus * exp(-d(t)^2 / (2*sigma^2))``, ``d(t)`` = distance in frames to the nearest transition
    *within the same clip* (edge frames of the clip cannot see further away -- a known, documented approximation for
    short clips; negligible for whole-video eval which uses much longer windows). Per-frame CE is weighted then
    averaged (not summed), so the overall loss scale stays comparable to plain CE.
    """

    def __init__(self, class_counts: torch.Tensor | None = None, num_classes: int = 16, bonus: float = 2.0, sigma: float = 1.5) -> None:
        self.bonus, self.sigma = float(bonus), float(sigma)

    def __call__(self, logits, y):
        B, L = y.shape[:2] if y.dim() > 1 else (1, y.shape[0])
        y2 = y.reshape(B, -1)
        t = torch.arange(y2.shape[1], device=y.device, dtype=torch.float32)
        w = torch.ones_like(y2, dtype=torch.float32)
        for b in range(y2.shape[0]):
            trans = torch.where(y2[b, 1:] != y2[b, :-1])[0].float() + 0.5  # boundary sits between frame i and i+1
            if len(trans) == 0:
                continue
            d = (t[:, None] - trans[None, :]).abs().min(1).values
            w[b] = 1.0 + self.bonus * torch.exp(-(d ** 2) / (2 * self.sigma ** 2))
        ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
        return (ce * w.reshape(-1).to(logits.device)).mean()


@LOSS_REGISTRY.register("ce_cellcount")
class CellCountAuxiliaryLoss(AbsLoss):
    """TEMPO rung T8b: CE on the 16 phases + ``lam`` * ordinal BCE on the auxiliary cell-count head (``model.aux``).

    The phase label is mapped to a cell-count level (1 cell, 2, ..., 8, 9+), and the ``levels-1`` auxiliary logits are
    trained as cumulative binary targets ``[level > k]`` (Frank & Hall / CORAL-style ordinal regression), so a t7 frame
    predicted as t8 is penalised by exactly one threshold while t7 -> t2 costs five. Requires the model to return
    ``aux_logits`` (``needs_outputs``: the training loop passes the whole output dict)."""

    needs_outputs = True

    def __init__(self, class_counts: torch.Tensor | None = None, num_classes: int = 16, lam: float = 0.5) -> None:
        self.ce, self.lam = nn.CrossEntropyLoss(), float(lam)
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, out, y):
        logits = out["logits"] if isinstance(out, dict) else out
        lg = logits.reshape(-1, logits.shape[-1]); yy = y.reshape(-1)
        loss = self.ce(lg, yy)
        if isinstance(out, dict) and "aux_logits" in out and self.lam > 0:
            al = out["aux_logits"].reshape(-1, out["aux_logits"].shape[-1]).float()
            level = CELL_COUNT_LEVEL.to(yy.device)[yy]
            ks = torch.arange(al.shape[1], device=yy.device)
            target = (level[:, None] > ks[None, :]).float()
            loss = loss + self.lam * self.bce(al, target)
        return loss


@LOSS_REGISTRY.register("ce_divconsistency")
class DivisionConsistencyLoss(AbsLoss):
    """ follow-up: CE + ``lam`` * BCE between the model's OWN predicted transition probability at
    consecutive frames and an externally-computed division-event score (``data.division_score_path``, the
    N3-warm-started detector of ``scripts/train_division_detector.py``, echoed back via ``model.echo_div_score``
    into ``out["div_score"]`` -- see models.py).

    ``trans_prob(t) = 1 - sum_c softmax(logits[t-1])_c * softmax(logits[t])_c`` (Bhattacharyya-style overlap between
    consecutive frames' predicted distributions -- 0 when frame t-1 and t agree, up to 1 when they disagree
    completely; fully differentiable, no hard argmax). Compared via BCE against ``div_score[t]`` treated as a soft
    pseudo-target. Known approximation: ``div_score`` was cached against a *centred* (t-K, t+K) window (K=2) while
    ``trans_prob`` is a *consecutive*-frame (t-1, t) measure -- division events last several frames, so the two
    should still overlap near a real transition, but this is a soft prior, not an exact alignment; requires
    ``clip_len >= 2`` (needs at least one consecutive pair)."""

    needs_outputs = True

    def __init__(self, class_counts: torch.Tensor | None = None, num_classes: int = 16, lam: float = 0.5) -> None:
        self.ce, self.lam = nn.CrossEntropyLoss(), float(lam)
        self.bce = nn.BCEWithLogitsLoss()  # BCELoss on post-sigmoid probs is unsafe under torch.autocast; take logits instead

    def __call__(self, out, y):
        logits = out["logits"] if isinstance(out, dict) else out
        lg = logits.reshape(-1, logits.shape[-1]); yy = y.reshape(-1)
        loss = self.ce(lg, yy)
        if isinstance(out, dict) and "div_score" in out and self.lam > 0 and logits.shape[1] >= 2:
            p = torch.softmax(logits.float(), -1)  # (B, L, K)
            overlap = (p[:, :-1] * p[:, 1:]).sum(-1).clamp(1e-6, 1 - 1e-6)  # (B, L-1)
            trans_prob_logit = torch.logit(1.0 - overlap, eps=1e-6)  # BCEWithLogitsLoss expects logits, not probabilities
            target = out["div_score"][:, 1:].float().clamp(0, 1)
            loss = loss + self.lam * self.bce(trans_prob_logit, target)
        return loss


@LOSS_REGISTRY.register("ce_tls")
class TemporalLabelSmoothingCE(AbsLoss):
    """TEMPO v17: cross-entropy with *temporal* label smoothing at phase boundaries. Diagnostics showed the training loss
    on boundary frames stays ~3x the interior loss and stops decreasing (irreducible: onset annotations are only accurate to
    +-1 frame), while test errors sit in the interior of short phases. For the two frames on either side of a within-clip
    label change, ``eps`` of the target mass is moved to the neighbouring phase's label (soft target); all other frames
    keep a hard one-hot target. Per-frame model (1-D labels): plain CE."""

    def __init__(self, class_counts: torch.Tensor | None = None, num_classes: int = 16, eps: float = 0.3) -> None:
        self.eps, self.K = float(eps), int(num_classes)

    def __call__(self, logits, y):
        lg = logits.reshape(-1, logits.shape[-1]).float()
        if y.dim() == 1:
            return nn.functional.cross_entropy(lg, y.reshape(-1))
        B, L = y.shape
        target = nn.functional.one_hot(y, self.K).float()  # (B, L, K)
        change = torch.zeros(B, L, dtype=torch.bool, device=y.device); change[:, 1:] = y[:, 1:] != y[:, :-1]
        # frame t (first of new phase) leans towards y[t-1]; frame t-1 (last of old phase) leans towards y[t]
        idx_b, idx_t = torch.where(change)
        if idx_t.numel():
            prev_lbl = y[idx_b, idx_t - 1]; new_lbl = y[idx_b, idx_t]
            target[idx_b, idx_t] *= (1 - self.eps); target[idx_b, idx_t, prev_lbl] += self.eps
            target[idx_b, idx_t - 1] *= (1 - self.eps); target[idx_b, idx_t - 1, new_lbl] += self.eps
        logp = nn.functional.log_softmax(lg, 1)
        return -(target.reshape(-1, self.K) * logp).sum(1).mean()


@LOSS_REGISTRY.register("multistage_ce_tmse")
class MultiStageTemporalLoss(AbsLoss):
    """Stage-wise CE plus truncated temporal smoothing for full-video models.

    Every stage is directly supervised. The smoothing term is the truncated
    squared difference between consecutive log-probabilities, with the previous
    frame detached as in MS-TCN training. Inputs contain no padding in v26
    (one full video or fixed L16 clips per sample), so every temporal position is
    valid and no padding mask is required.
    """

    needs_outputs = True

    def __init__(
        self,
        class_counts: torch.Tensor | None = None,
        num_classes: int = 16,
        lam: float = 0.15,
        truncation: float = 4.0,
    ) -> None:
        del class_counts, num_classes
        if lam < 0 or truncation <= 0:
            raise ValueError("lam must be nonnegative and truncation must be positive")
        self.lam = float(lam)
        self.truncation = float(truncation)

    def __call__(self, out, y):
        if not isinstance(out, dict) or "stage_logits" not in out:
            raise ValueError("multistage_ce_tmse requires model stage_logits")
        stages = out["stage_logits"]
        if stages.ndim != 4 or stages.shape[1:3] != y.shape:
            raise ValueError(
                "stage_logits must be (S,B,T,K) aligned with labels (B,T)"
            )
        losses = []
        for logits in stages:
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
            if logits.shape[1] > 1 and self.lam > 0:
                logp = nn.functional.log_softmax(logits.float(), dim=-1)
                delta = logp[:, 1:] - logp[:, :-1].detach()
                tmse = delta.square().clamp(max=self.truncation**2).mean()
                ce = ce + self.lam * tmse
            losses.append(ce)
        return torch.stack(losses).mean()
