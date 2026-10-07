"""Temporal heads over per-frame features: (B, L, D) -> (B, L, d_out)."""

from __future__ import annotations

import torch
from torch import nn

from .interfaces import AbsTemporalHead
from .registry import HEAD_REGISTRY


@HEAD_REGISTRY.register("none")
class IdentityHead(AbsTemporalHead):
    def __init__(self, d_in: int) -> None:
        super().__init__(); self.d_out = d_in

    def forward(self, f):
        return f


class _RNNHead(AbsTemporalHead):
    kind = "lstm"

    def __init__(self, d_in: int, hidden: int = 1024, layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        cls = nn.LSTM if self.kind == "lstm" else nn.GRU
        self.rnn = cls(d_in, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.d_out = 2 * hidden

    def forward(self, f):
        return self.rnn(f)[0]


@HEAD_REGISTRY.register("lstm")
class LSTMHead(_RNNHead):
    """Official Nantes ResNet-LSTM head: bidirectional, 2 layers, hidden 1024."""
    kind = "lstm"


@HEAD_REGISTRY.register("gru")
class GRUHead(_RNNHead):
    kind = "gru"


@HEAD_REGISTRY.register("tcn")
class TCNHead(AbsTemporalHead):
    """Residual dilated 1-D convs (dilations 1,2,4,8 -> receptive field 31 frames), non-causal."""

    def __init__(self, d_in: int, hidden: int = 512, layers: int = 4, kernel: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.inp = nn.Conv1d(d_in, hidden, 1)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            dil = 2 ** i
            self.blocks.append(nn.Sequential(nn.Conv1d(hidden, hidden, kernel, padding=dil * (kernel - 1) // 2, dilation=dil),
                                             nn.ReLU(inplace=True), nn.Conv1d(hidden, hidden, 1), nn.Dropout(dropout)))
        self.d_out = hidden

    def forward(self, f):
        h = self.inp(f.transpose(1, 2))
        for blk in self.blocks:
            h = h + blk(h)
        return h.transpose(1, 2)


@HEAD_REGISTRY.register("transformer")
class TransformerHead(AbsTemporalHead):
    def __init__(self, d_in: int, hidden: int = 512, layers: int = 2, nhead: int = 8, dropout: float = 0.1, max_len: int = 1024) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden)); nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=hidden * 2, dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.d_out = hidden

    def forward(self, f):
        return self.enc(self.proj(f) + self.pos[:, : f.shape[1]])


@HEAD_REGISTRY.register("mamba")
class SelectiveSSMHead(AbsTemporalHead):
    """Simplified selective state-space head (Mamba-style, arXiv 2312.00752), pure PyTorch — no ``mamba-ssm`` CUDA
    kernel dependency, so it runs on CPU for tests and on any GPU. TEMPO rung T8: linear-time whole-video context
    (unlike the O(L^2) transformer head), and — because the decay/gate is *input-dependent* rather than a fixed learned
    absolute position table — it does not suffer the eval-window bug found in ``TransformerHead`` (2026-09-09): the same
    weights generalise from a short training clip to a whole video.

    Per channel c, a diagonal selective scan ``h_t = a_t * h_{t-1} + b_t * x_t`` with input-dependent gates
    ``a_t = sigmoid(W_a x_t) in (0,1)``, ``b_t = softplus(W_b x_t)``, computed by a numerically-stable cumulative-sum
    (log-space) parallel scan instead of a Python loop over time, so training is vectorised like the paper's parallel scan.
    Output is the elementwise product of the scan state with an input-dependent gate ``y_t = h_t * silu(W_y x_t)``
    (SSM output-gating, as in Mamba's block), stacked in ``layers`` residual blocks.
    """

    def __init__(self, d_in: int, hidden: int = 512, layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([_SSMBlock(hidden, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.d_out = hidden

    def forward(self, f):
        h = self.inp(f)
        for blk in self.blocks:
            h = h + blk(h)
        return self.norm(h)


class _SSMBlock(nn.Module):
    def __init__(self, d: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)  # pre-norm (standard in stacked SSM/transformer blocks, e.g. Mamba's RMSNorm):
        # without it, un-normalised residual stacking of several _SSMBlock layers over a 16-64-frame recurrence can
        # grow unboundedly with depth x time even though each block's own recurrence is locally stable (a_t in (0,1)),
        # which is what produced the nan divergence fixed alongside the log-space overflow above.
        self.a = nn.Linear(d, d); self.b = nn.Linear(d, d); self.y = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        # h_t = a_t * h_{t-1} + b_t * x_t, a_t = sigmoid(-softplus_logit) in (0,1), b_t = softplus(...).
        # 2026-09-11: this was previously computed with a vectorised log-space cumulative-ratio identity
        # (h_t = exp(A_t) * cumsum(bx_s * exp(-A_s)), A_t = cumsum(log a)) to avoid a Python loop over time. That
        # identity is mathematically correct but numerically unsafe: exp(-A_s) alone reaches ~1e9 by frame 16 even
        # in float32 (measured), overflowing fp16's ~65504 range and producing inf * 0 = nan -- this is exactly what
        # caused every v5 Mamba-head training run to diverge to nan loss from epoch 1. Fixed by an explicit
        # sequential recurrence: L is small (16-64 frames per clip), so a Python loop over time here costs a few ms
        # and every intermediate value stays within [0,1] x |x|, with no cumulative product ever computed.
        log_a = -nn.functional.softplus(self.a(x))                     # (B,L,D), log a_t in (-inf, 0]
        a = torch.exp(log_a)                                           # a_t in (0,1), no cumulative product taken
        bx = nn.functional.softplus(self.b(x)) * x                     # b_t * x_t
        B, L, D = x.shape
        h_t = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        out = []
        for t in range(L):
            h_t = a[:, t] * h_t + bx[:, t]
            out.append(h_t)
        h = torch.stack(out, dim=1)
        return self.drop(h * torch.nn.functional.silu(self.y(x)))


@HEAD_REGISTRY.register("node")
class NeuralODEHead(AbsTemporalHead):
    """Continuous-time latent dynamics head (Neural ODE, Chen et al. 2018), following the spirit of Bechar et al. 2026
    (RB-NODE, arXiv/JARG — literature.md §2): the hidden state evolves continuously as ``dh/dt = g(h)`` between frames
    instead of jumping frame-to-frame like an RNN, which is the natural fit for *irregularly or differently sampled*
    time-lapse video — exactly our domain-gap problem (Nantes ~18 min/frame vs our own videos 10–60 min/frame,
    `` §12-I). Frame features re-inject into the state at every step (``dh/dt = g(h) + Linear(x_t)``, an
    ODE-RNN, Rubanova et al. 2019), integrated with a fixed-step RK4 solver (no ``torchdiffeq`` dependency).

    Scope note (2026-09-09): ``n_substeps`` subdivides the *frame-index* interval uniformly; real irregular
    inter-frame ``time_h`` deltas are not yet threaded into this head (would need ``AbsTemporalHead.forward`` to accept an
    optional ``dt`` tensor and the data pipeline to carry it end-to-end) — tracked as follow-up T10b. As implemented,
    this head already changes behaviour vs a plain RNN: the dynamics are shared and continuous, so a checkpoint trained
    with ``clip_len`` frames extrapolates by design to any evaluation window (same motivation as T1's window fix).
    """

    def __init__(self, d_in: int, hidden: int = 256, n_substeps: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.dynamics = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.n_substeps = int(n_substeps)
        self.drop = nn.Dropout(dropout)
        self.d_out = hidden

    def _step(self, h, x_t, dt):
        def g(hh):
            return self.dynamics(hh) + x_t
        k1 = g(h); k2 = g(h + dt / 2 * k1); k3 = g(h + dt / 2 * k2); k4 = g(h + dt * k3)
        return h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, f):
        x = self.inp(f)  # (B, L, H)
        B, L, H = x.shape
        h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        dt = 1.0 / self.n_substeps
        out = []
        for t in range(L):
            for _ in range(self.n_substeps):
                h = self._step(h, x[:, t], dt)
            out.append(h)
        return self.drop(torch.stack(out, dim=1))


@HEAD_REGISTRY.register("diffact_lite")
class DiffusionRefinementHead(AbsTemporalHead):
    """Simplified diffusion-style iterative refinement head (TEMPO rung T9), following the coarse-to-fine idea of
    DiffAct (Liu et al. 2023) and EmbryoDiff (Sun et al. 2025, literature.md §2, the strongest published model on this
    dataset: 82.8-83.1% accuracy with a full diffusion action-segmentation decoder). **Scope note**: this is a
    *lightweight* approximation — a fixed number of self-attention refinement stages conditioned on a noise-step
    embedding, trained with the same per-frame classification loss as the other heads (no full forward/reverse
    diffusion process over label sequences, no separate boundary/temporal losses) — a reduced-cost stand-in to compare
    against MS-TCN++/ASFormer/DiffAct at full scale (rung T2) if this simplified version already helps.

    Each of ``steps`` stages sees the current refined features plus a learned embedding of "how many stages remain"
    (coarse -> fine, akin to a denoising schedule) and refines them with one local-window self-attention block.
    """

    def __init__(self, d_in: int, hidden: int = 384, steps: int = 3, nhead: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.step_embed = nn.Embedding(steps, hidden)
        self.stages = nn.ModuleList([nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=hidden * 2,
                                                                  dropout=dropout, batch_first=True, norm_first=True)
                                      for _ in range(steps)])
        self.steps = steps
        self.d_out = hidden

    def forward(self, f):
        h = self.inp(f)
        for i, stage in enumerate(self.stages):
            h = stage(h + self.step_embed.weight[i][None, None])
        return h


def _rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding (Su et al. 2021): rotates pairs of channels by an angle proportional to absolute
    position, so dot products between two positions depend only on their *relative* offset. ``x``: (B, L, H, Dh)."""
    B, L, H, Dh = x.shape
    half = Dh // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    pos = torch.arange(L, device=x.device, dtype=torch.float32)
    ang = pos[:, None] * freqs[None, :]  # (L, half)
    cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class _RelPosSelfAttnBlock(nn.Module):
    """Pre-norm self-attention with RoPE (relative position, no learned absolute table) + a feed-forward block --
    one stage of ``RelPosTransformerHead``."""

    def __init__(self, d: int, nhead: int, dropout: float) -> None:
        super().__init__()
        assert d % nhead == 0
        self.nhead, self.dh = nhead, d // nhead
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, L, 3, self.nhead, self.dh)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, L, nhead, dh)
        q, k = _rope(q), _rope(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, nhead, L, dh)
        att = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        att = att.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.proj(att))
        return x + self.drop(self.ff(self.norm2(x)))


@HEAD_REGISTRY.register("transformer_relpos")
class RelPosTransformerHead(AbsTemporalHead):
    """Transformer head with rotary (relative) position embeddings instead of ``TransformerHead``'s learned absolute
    position table. TEMPO rung T5 alternative to T1: relative position naturally generalises to any evaluation
    window (no position index ever exceeds what was trained on, because there IS no absolute index), so it should not
    need the eval-window fix that ``transformer`` (T1) required, and is a more standard fix than the sliding-window
    work-around -- worth comparing against both T1 (fixed window) and T8 (Mamba, also position-table-free)."""

    def __init__(self, d_in: int, hidden: int = 512, layers: int = 2, nhead: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([_RelPosSelfAttnBlock(hidden, nhead, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.d_out = hidden

    def forward(self, f):
        h = self.proj(f)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)


class _LocalAttn(nn.Module):
    """Single-head self-attention restricted to a window of +-w frames (ASFormer's local attention), (B, L, D)."""

    def __init__(self, d: int, w: int, dropout: float) -> None:
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.w, self.drop = int(w), nn.Dropout(dropout)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        B, L, D = x.shape
        att = torch.einsum("bld,bmd->blm", self.q(x), self.k(ctx)) / D ** 0.5
        idx = torch.arange(L, device=x.device)
        mask = (idx[:, None] - idx[None, :]).abs() > self.w
        att = att.masked_fill(mask[None], float("-inf")).softmax(-1)
        return self.drop(torch.einsum("blm,bmd->bld", att, self.v(ctx)))


class _ASBlock(nn.Module):
    """ASFormer block: dilated temporal conv -> local attention (+ optional cross-attention to encoder) -> 1x1 -> residual."""

    def __init__(self, d: int, dilation: int, window: int, dropout: float, cross: bool) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation), nn.ReLU(inplace=True))
        self.norm = nn.InstanceNorm1d(d, affine=True)
        self.attn = _LocalAttn(d, window, dropout)
        self.cross = cross
        self.out = nn.Sequential(nn.Linear(d, d), nn.Dropout(dropout))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, enc=None):  # x (B, L, D)
        h = self.norm(self.conv(x.transpose(1, 2))).transpose(1, 2)
        a = self.attn(h, enc if (self.cross and enc is not None) else None)
        return x + self.out(h + self.alpha * a)


@HEAD_REGISTRY.register("asformer_lite")
class ASFormerLiteHead(AbsTemporalHead):
    """Re-implementation of ASFormer (Yi et al. 2021) adapted to per-frame CNN features and the clip-level training of
    this benchmark: an encoder of ``layers`` dilated-conv + local-attention blocks (dilation and attention window
    doubling per layer), followed by ``n_decoders`` refinement decoders that take the softmax of the previous stage's
    class prediction as input and cross-attend to the encoder features. The stage classifiers are internal (the
    pipeline's own classifier sits on the final decoder features), so the multi-stage loss of the original is
    approximated by supervising the last stage only -- documented deviation, same as ``diffact_lite``. ASFormer is
    the strongest published model on this dataset for the edit score (89.0 under its authors' protocol), which our
    sweep had no counterpart for until now."""

    def __init__(self, d_in: int, hidden: int = 256, layers: int = 6, n_decoders: int = 2, num_classes: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.enc = nn.ModuleList([_ASBlock(hidden, 2 ** i, 2 ** i, dropout, cross=False) for i in range(layers)])
        self.enc_cls = nn.Linear(hidden, num_classes)
        self.dec_in = nn.ModuleList([nn.Linear(num_classes, hidden) for _ in range(n_decoders)])
        self.dec = nn.ModuleList([nn.ModuleList([_ASBlock(hidden, 2 ** i, 2 ** i, dropout, cross=True) for i in range(layers)]) for _ in range(n_decoders)])
        self.dec_cls = nn.ModuleList([nn.Linear(hidden, num_classes) for _ in range(n_decoders - 1)])
        self.d_out = hidden

    def forward(self, f):
        h = self.inp(f)
        for blk in self.enc:
            h = blk(h)
        enc = h
        logits = self.enc_cls(enc)
        for s, blocks in enumerate(self.dec):
            x = self.dec_in[s](logits.softmax(-1))
            for blk in blocks:
                x = blk(x, enc)
            if s < len(self.dec_cls):
                logits = self.dec_cls[s](x)
        return x


@HEAD_REGISTRY.register("mstcn")
class MSTCNHead(AbsTemporalHead):
    """MS-TCN (Farha & Gall 2019) re-implementation: ``stages`` single-stage TCNs, each refining the softmax of the previous
    stage's class prediction; final-stage features are returned (last-stage supervision only, see ``asformer_lite``)."""

    def __init__(self, d_in: int, hidden: int = 64, layers: int = 10, stages: int = 4, num_classes: int = 16, dropout: float = 0.5) -> None:
        super().__init__()
        def stage(d):
            return nn.ModuleDict({"inp": nn.Conv1d(d, hidden, 1),
                                  "blocks": nn.ModuleList([nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=2 ** i, dilation=2 ** i), nn.ReLU(inplace=True),
                                                                          nn.Conv1d(hidden, hidden, 1), nn.Dropout(dropout)) for i in range(layers)])})
        self.stages = nn.ModuleList([stage(d_in)] + [stage(num_classes) for _ in range(stages - 1)])
        self.cls = nn.ModuleList([nn.Conv1d(hidden, num_classes, 1) for _ in range(stages - 1)])
        self.d_out = hidden

    def forward(self, f):
        x = f.transpose(1, 2)
        for s, st in enumerate(self.stages):
            h = st["inp"](x)
            for blk in st["blocks"]:
                h = h + blk(h)
            if s < len(self.cls):
                x = self.cls[s](h).softmax(1)
        return h.transpose(1, 2)
