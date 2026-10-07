"""TEMPO v29: whole-video milestone queries with local refinement.

Each of the ``K-1`` phase onsets (phase ``k`` = first frame whose ordinal phase is >= k) is a milestone
with a categorical location over ``T+1`` slots: slot ``t < T`` means the onset is at frame ``t`` (slot 0 also
absorbs onsets before the recording starts), slot ``T`` means the phase is never reached (right-censored).
Annotated skips (a phase with zero frames) are onsets that coincide with the next one, so no event is forced.
Milestone locations are turned into frame-level phase probabilities through their cumulative distributions
(monotone across milestones by construction) and combined with the SCE framewise head by a fixed product of
experts, so the unchanged Viterbi decoder and metrics consume them. No clock or positional encoding is used:
localisation is content- and local-context-based only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .embryodiff import _ConditionEncoder
from .interfaces import AbsKineticModel, AbsLoss
from .registry import LOSS_REGISTRY, MODEL_REGISTRY

VARIANTS = ("onset_linear", "query_global", "query_refine")
FUSIONS = ("product", "gated")           # v31: fixed product of experts (v29) | per-frame learned gate
PARAMETERISATIONS = ("onset", "hazard")  # v31: onset slot logits (v29) | duration/hazard, monotone by construction
_NEG = -1.0e4
_EPS = 1e-6


def milestone_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Onset slot per milestone.

    Args:
        y: Ordinal phase labels ``(B, T)``.
        num_classes: Number of phases ``K``.

    Returns:
        ``(B, K-1)`` long slots in ``[0, T]``: the first frame with phase >= k, or ``T`` if never reached.
    """
    if y.ndim != 2:
        raise ValueError(f"labels must be (B,T), got {tuple(y.shape)}")
    t_len = y.shape[1]
    ks = torch.arange(1, num_classes, device=y.device)
    reached = y[:, None, :] >= ks[None, :, None]
    first = reached.float().argmax(-1)
    return torch.where(reached.any(-1), first, torch.full_like(first, t_len))


def milestone_soft_targets(targets: torch.Tensor, t_len: int, sigma: float) -> torch.Tensor:
    """Gaussian (sigma frames) target over frame slots for observed onsets; one-hot on the censored slot."""
    b, m = targets.shape
    out = targets.new_zeros(b, m, t_len + 1, dtype=torch.float32)
    observed = targets < t_len
    if sigma > 0:
        pos = torch.arange(t_len, device=targets.device, dtype=torch.float32)
        g = torch.exp(-(pos[None, None] - targets[..., None].float()) ** 2 / (2.0 * sigma * sigma))
        g = g / g.sum(-1, keepdim=True).clamp_min(_EPS)
        out[..., :t_len] = g * observed[..., None]
    else:
        out[..., :t_len] = F.one_hot(targets.clamp(max=t_len - 1), t_len).float() * observed[..., None]
    out[..., t_len] = (~observed).float()
    return out


def milestone_frame_log_probs(slot_logits: torch.Tensor) -> torch.Tensor:
    """Frame-level phase log-probabilities from milestone slot logits.

    Args:
        slot_logits: ``(B, M, T+1)`` logits over onset slots, milestones in phase order.

    Returns:
        ``(B, T, M+1)`` log-probabilities. ``P(phase=k | t) = F_k(t) - F_{k+1}(t)`` with ``F_k`` the onset CDF,
        made monotone across milestones by a cumulative minimum so every frame distribution is valid.
    """
    p = torch.softmax(slot_logits.float(), dim=-1)
    cdf = torch.cumsum(p[..., :-1], dim=-1).clamp(0.0, 1.0)
    cdf = torch.cummin(cdf, dim=1).values
    b, _, t_len = cdf.shape
    upper = torch.cat([cdf.new_ones(b, 1, t_len), cdf], dim=1)
    lower = torch.cat([cdf, cdf.new_zeros(b, 1, t_len)], dim=1)
    probs = (upper - lower).clamp_min(0.0)
    return torch.log_softmax(torch.log(probs.clamp_min(_EPS)), dim=1).transpose(1, 2)


def hazard_onset_log_probs(scores: torch.Tensor) -> torch.Tensor:
    """Milestone onset log-probabilities from per-frame hazard scores, monotone by construction (v31).

    Args:
        scores: ``(B, M, T)`` hazard logits; ``h_m(t) = sigmoid(scores[m, t])`` is the probability that milestone
            ``m`` fires at frame ``t`` given it has not fired yet and milestone ``m-1`` has already fired.

    Returns:
        ``(B, M, T+1)`` log-probabilities over onset slots (frame ``t`` or ``T`` = never reached). Milestone ``m``
        can fire only at or after milestone ``m-1`` (gap 0 = the phase is skipped), so onset CDFs are ordered
        without any projection, "never" mass propagates to later milestones, and each row sums to one.
    """
    b, m, t_len = scores.shape
    log_h = F.logsigmoid(scores.float())
    log_s = F.logsigmoid(-scores.float())
    surv = torch.cat([log_s.new_zeros(b, m, 1), log_s.cumsum(-1)], dim=-1)  # L(t) = sum_{u<t} log(1-h), L(T) at index T
    prev = scores.new_full((b, t_len + 1), _NEG, dtype=torch.float32)
    prev[:, 0] = 0.0  # milestone "0" = recording start
    rows = []
    for k in range(m):
        a = prev[:, :t_len] - surv[:, k, :t_len]                      # log P_{k-1}(s) - L(s)
        c = torch.logcumsumexp(a, dim=-1)                              # log sum_{s<=t} ...
        frames = log_h[:, k] + surv[:, k, :t_len] + c                  # log P_k(t)
        never = torch.logaddexp(prev[:, t_len], surv[:, k, t_len] + c[:, -1])
        prev = torch.cat([frames, never[:, None]], dim=-1)
        rows.append(prev)
    return torch.stack(rows, dim=1)


class _QueryDecoderLayer(nn.Module):
    """Pre-norm self-attention among milestones, cross-attention to the video, feed-forward."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.norm_self(q)
        q = q + self.drop(self.self_attn(h, h, h, need_weights=False)[0])
        h = self.norm_cross(q)
        q = q + self.drop(self.cross_attn(h, memory, memory, need_weights=False)[0])
        return q + self.drop(self.ffn(self.norm_ffn(q)))


@MODEL_REGISTRY.register("milestone_query_adapter")
class MilestoneQueryAdapter(AbsKineticModel):
    """SCE framewise head plus milestone onset localisation over complete cached-feature videos.

    Variants (one change each): ``onset_linear`` scores every frame for every onset with a linear head (no global
    queries); ``query_global`` replaces it with learned milestone queries decoded against the whole video;
    ``query_refine`` adds a zero-initialised local residual that sharpens each query's distribution within
    ``refine_radius`` frames of its (detached) coarse argmax. Input ``(B, T, D)``, equal-length batches only.
    """

    is_sequence = True
    requires_labels = False

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        variant: str,
        hidden: int,
        layers: int,
        intermediate_layers: tuple[int, ...],
        attention_reduction: int,
        dropout: float,
        query_dim: int,
        query_layers: int,
        query_heads: int,
        query_dropout: float,
        local_kernel: int,
        refine_radius: int,
        combine_weight: float,
        fusion: str = "product",
        parameterisation: str = "onset",
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown milestone variant {variant!r}; expected {VARIANTS}")
        if fusion not in FUSIONS or parameterisation not in PARAMETERISATIONS:
            raise ValueError(f"fusion must be one of {FUSIONS} and parameterisation one of {PARAMETERISATIONS}")
        if parameterisation == "hazard" and variant != "onset_linear":
            raise ValueError("the hazard parameterisation is defined for the onset_linear variant only")
        if local_kernel < 1 or local_kernel % 2 == 0:
            raise ValueError("local_kernel must be a positive odd integer")
        if min(query_dim, query_layers, query_heads) < 1 or query_dim % query_heads:
            raise ValueError("query_dim/query_layers/query_heads must be positive and query_dim divisible by heads")
        if refine_radius < 1 or combine_weight < 0:
            raise ValueError("refine_radius must be >= 1 and combine_weight >= 0")
        self.variant = variant
        self.num_classes = int(num_classes)
        self.num_milestones = self.num_classes - 1
        self.combine_weight = float(combine_weight)
        self.fusion, self.parameterisation = fusion, parameterisation
        self.refine_radius = int(refine_radius)
        self.scale = query_dim**-0.5

        self.encoder = _ConditionEncoder(input_dim, hidden, layers, intermediate_layers, attention_reduction, dropout)
        self.frame_classifier = nn.Linear(hidden, num_classes)
        c = self.encoder.output_dim
        self.local = nn.Sequential(
            nn.Conv1d(c, query_dim, local_kernel, padding=local_kernel // 2),
            nn.GELU(),
            nn.Conv1d(query_dim, query_dim, 3, padding=1),
        )
        self.memory_norm = nn.LayerNorm(query_dim)
        m = self.num_milestones
        if variant == "onset_linear":
            self.onset_head = nn.Linear(query_dim, m)
            if parameterisation == "onset":
                self.never_head = nn.Linear(query_dim, m)  # hazard mode derives the never-reached mass instead
        else:
            self.queries = nn.Parameter(torch.randn(m, query_dim) * 0.02)
            self.decoder = nn.ModuleList([_QueryDecoderLayer(query_dim, query_heads, query_dropout) for _ in range(query_layers)])
            self.query_norm = nn.LayerNorm(query_dim)
            self.loc_q = nn.Linear(query_dim, query_dim)
            self.loc_k = nn.Linear(query_dim, query_dim)
            self.never_head = nn.Linear(query_dim, 1)
        if variant == "query_refine":
            self.refine = nn.Sequential(
                nn.Conv1d(query_dim, query_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv1d(query_dim, query_dim, 3, padding=2, dilation=2),
                nn.GELU(),
            )
            self.refine_q = nn.Linear(query_dim, query_dim)
            self.refine_k = nn.Linear(query_dim, query_dim)
            nn.init.zeros_(self.refine_q.weight)
            nn.init.zeros_(self.refine_q.bias)
        if fusion == "gated":
            # per-frame trust in the milestone evidence; starts near the v29 fixed product (sigmoid(3) = 0.953)
            self.gate = nn.Linear(query_dim, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 3.0)

    @classmethod
    def from_config(cls, cfg: dict, in_channels: int) -> "MilestoneQueryAdapter":
        if cfg.get("backbone", {}).get("name") != "cached_features":
            raise ValueError("milestone_query_adapter requires cached_features backbone")
        if cfg.get("head", {}).get("name") != "none":
            raise ValueError("milestone_query_adapter requires model.head.name=none")
        mcfg = cfg.get("milestone")
        if not isinstance(mcfg, dict):
            raise ValueError("milestone_query_adapter requires model.milestone mapping")
        return cls(
            input_dim=int(in_channels),
            num_classes=int(cfg.get("num_classes", 16)),
            variant=str(mcfg.get("variant", "query_refine")),
            hidden=int(mcfg.get("hidden", 96)),
            layers=int(mcfg.get("layers", 6)),
            intermediate_layers=tuple(mcfg.get("intermediate_layers", [2, 4, 6])),
            attention_reduction=int(mcfg.get("attention_reduction", 2)),
            dropout=float(mcfg.get("dropout", cfg.get("dropout", 0.5))),
            query_dim=int(mcfg.get("query_dim", 128)),
            query_layers=int(mcfg.get("query_layers", 2)),
            query_heads=int(mcfg.get("query_heads", 4)),
            query_dropout=float(mcfg.get("query_dropout", 0.1)),
            local_kernel=int(mcfg.get("local_kernel", 5)),
            refine_radius=int(mcfg.get("refine_radius", 8)),
            combine_weight=float(mcfg.get("combine_weight", 1.0)),
            fusion=str(mcfg.get("fusion", "product")),
            parameterisation=str(mcfg.get("parameterisation", "onset")),
        )

    def _coarse(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Slot logits ``(B, M, T+1)`` and the decoded queries (None for the linear control)."""
        if self.variant == "onset_linear":
            frame_scores = self.onset_head(memory).transpose(1, 2)
            if self.parameterisation == "hazard":
                return hazard_onset_log_probs(frame_scores), None
            never = self.never_head(memory.mean(dim=1))[..., None]
            return torch.cat([frame_scores, never], dim=-1), None
        q = self.queries[None].expand(memory.shape[0], -1, -1)
        for layer in self.decoder:
            q = layer(q, memory)
        q = self.query_norm(q)
        scores = torch.einsum("bmd,btd->bmt", self.loc_q(q), self.loc_k(memory)) * self.scale
        return torch.cat([scores, self.never_head(q)], dim=-1), q

    def _refined(self, coarse: torch.Tensor, memory: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        t_len = memory.shape[1]
        centre = coarse[..., :t_len].detach().argmax(-1, keepdim=True)
        pos = torch.arange(t_len, device=memory.device)
        window = ((pos[None, None] - centre).abs() <= self.refine_radius).to(coarse.dtype)
        r = self.refine(memory.transpose(1, 2)).transpose(1, 2)
        delta = torch.einsum("bmd,btd->bmt", self.refine_q(q), self.refine_k(r)) * self.scale
        frames = coarse[..., :t_len] + window * delta
        return torch.cat([frames, coarse[..., t_len:]], dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"milestone input must be (B,T,D), got {tuple(x.shape)}")
        condition, final = self.encoder(x)
        frame_logits = self.frame_classifier(final)
        memory = self.memory_norm(self.local(condition.transpose(1, 2)).transpose(1, 2))
        coarse, q = self._coarse(memory)
        out = {"frame_logits": frame_logits, "milestone_coarse_logits": coarse}
        used = coarse
        if self.variant == "query_refine":
            used = self._refined(coarse, memory, q)
            out["milestone_refined_logits"] = used
        ms_logp = milestone_frame_log_probs(used)
        trust = self.combine_weight
        if self.fusion == "gated":
            gate = torch.sigmoid(self.gate(memory).float())  # (B, T, 1)
            out["fusion_gate"] = gate.squeeze(-1)
            trust = self.combine_weight * gate
        combined = torch.log_softmax(frame_logits.float(), dim=-1) + trust * ms_logp
        out["milestone_frame_log_probs"] = ms_logp
        out["logits"] = torch.log_softmax(combined, dim=-1)
        return out


@LOSS_REGISTRY.register("milestone_objective")
class MilestoneObjective(AbsLoss):
    """SCE parent objective (0.8 CE + 0.3 truncated smoothing) plus event-normalised milestone CE and combined CE.

    The milestone term averages one soft cross-entropy per milestone per video (so long phases do not dominate),
    over the coarse and, when present, refined heads. The combined term is CE on the product-of-experts output.
    """

    needs_outputs = True

    def __init__(
        self,
        class_counts: torch.Tensor | None = None,
        num_classes: int = 16,
        semantic_weight: float = 0.8,
        smooth_weight: float = 0.3,
        milestone_weight: float = 1.0,
        combined_weight: float = 0.8,
        truncation: float = 4.0,
        target_sigma: float = 1.0,
        combined_smooth_weight: float = 0.0,
    ) -> None:
        del class_counts
        weights = (semantic_weight, smooth_weight, milestone_weight, combined_weight, combined_smooth_weight)
        if any(w < 0 for w in weights) or truncation <= 0 or target_sigma < 0:
            raise ValueError("milestone loss weights/sigma must be nonnegative and truncation positive")
        self.num_classes = int(num_classes)
        self.semantic_weight, self.smooth_weight = float(semantic_weight), float(smooth_weight)
        self.milestone_weight, self.combined_weight = float(milestone_weight), float(combined_weight)
        self.truncation, self.target_sigma = float(truncation), float(target_sigma)
        self.combined_smooth_weight = float(combined_smooth_weight)  # v31: SCE smoothing on the combined output

    def __call__(self, out, y):
        if not isinstance(out, dict) or "frame_logits" not in out or "milestone_coarse_logits" not in out:
            raise ValueError("milestone_objective requires frame_logits and milestone_coarse_logits")
        frame = out["frame_logits"]
        if frame.shape[:2] != y.shape:
            raise ValueError("frame logits must align with (B,T) labels")
        loss = self.semantic_weight * F.cross_entropy(frame.reshape(-1, frame.shape[-1]).float(), y.reshape(-1))
        if y.shape[1] > 1 and self.smooth_weight > 0:
            logp = F.log_softmax(frame.float(), dim=-1)
            delta = logp[:, 1:] - logp[:, :-1].detach()
            loss = loss + self.smooth_weight * delta.square().clamp(max=self.truncation**2).mean()
        if self.milestone_weight > 0:
            soft = milestone_soft_targets(milestone_targets(y, self.num_classes), y.shape[1], self.target_sigma)
            heads = [out["milestone_coarse_logits"]]
            if "milestone_refined_logits" in out:
                heads.append(out["milestone_refined_logits"])
            terms = [-(soft * F.log_softmax(h.float(), dim=-1)).sum(-1).mean() for h in heads]
            loss = loss + self.milestone_weight * torch.stack(terms).mean()
        if self.combined_weight > 0:
            comb = out["logits"]
            loss = loss + self.combined_weight * F.cross_entropy(comb.reshape(-1, comb.shape[-1]).float(), y.reshape(-1))
        if self.combined_smooth_weight > 0 and y.shape[1] > 1:
            logp = out["logits"].float()  # already log-probabilities
            delta = logp[:, 1:] - logp[:, :-1].detach()
            loss = loss + self.combined_smooth_weight * delta.square().clamp(max=self.truncation**2).mean()
        return loss


def milestone_parameter_count(model: nn.Module) -> dict[str, float]:
    """Parameter counts (millions) of the shared SCE part vs the milestone module, for reporting."""
    shared = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("encoder.", "frame_classifier.")))
    total = sum(p.numel() for p in model.parameters())
    return {"sce_M": shared / 1e6, "milestone_M": (total - shared) / 1e6, "total_M": total / 1e6}


__all__ = [
    "FUSIONS",
    "PARAMETERISATIONS",
    "MilestoneObjective",
    "MilestoneQueryAdapter",
    "VARIANTS",
    "hazard_onset_log_probs",
    "milestone_frame_log_probs",
    "milestone_parameter_count",
    "milestone_soft_targets",
    "milestone_targets",
]
