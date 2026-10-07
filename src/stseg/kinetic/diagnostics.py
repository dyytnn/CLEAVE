"""Training/evaluation diagnostics for the kinetic pipeline (added 2026-09-12 after the TEMPO sweeps showed that
head-swapping alone does not move the benchmark and that we had no view into *why*).

Everything here is cheap enough to run every epoch by default except embeddings/attention probes, which are opt-in
(``logging.embeddings``). All outputs go to ``<run>/diagnostics/`` and are designed so that the three questions that
decide the next design step can be answered from files alone:

1. representation vs temporal bottleneck  -> confusion matrices (argmax vs Viterbi), embeddings + linear probe
2. bias vs variance of timing errors       -> per-event signed error stats per epoch
3. does the model know when it is unsure   -> entropy at boundary vs interior frames

Config (all optional, under ``logging:``)::

    logging:
      frame_probs: true      # dump val/test per-frame log-probs of the best model (T x 16, float16)
      embeddings: false      # dump backbone features of the best model (T x D, float16) -- separable models only
      grad_norms: true       # per-epoch mean gradient norm, global and per module group
      confusion: true        # per-epoch 16x16 confusion (argmax and Viterbi) on val
      per_phase_loss: true   # per-epoch mean CE per phase and boundary-vs-interior frames
      worst_k: 10            # per-epoch k worst val videos by Viterbi accuracy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from stseg.data.nantes_kinetic import CLASS_NAMES, NUM_CLASSES
from stseg.eval.kinetic_metrics import viterbi

DEFAULTS = {"frame_probs": True, "embeddings": False, "grad_norms": True, "confusion": True, "per_phase_loss": True, "worst_k": 10}


def param_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Split parameters into backbone / head / classifier / other by attribute name (SeqKinetic layout)."""
    groups: dict[str, list[nn.Parameter]] = {"backbone": [], "head": [], "cls": [], "other": []}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        groups[top if top in groups else "other"].append(p)
    return {k: v for k, v in groups.items() if v}


def grad_norms(model: nn.Module) -> dict[str, float]:
    """L2 norm of the (unscaled) gradients, global and per group. Call after ``scaler.unscale_(opt)``."""
    out = {}
    total = 0.0
    for g, params in param_groups(model).items():
        sq = sum(float(p.grad.detach().float().pow(2).sum()) for p in params if p.grad is not None)
        out[f"grad_norm_{g}"] = sq ** 0.5
        total += sq
    out["grad_norm_total"] = total ** 0.5
    return out


class PhaseLossMeter:
    """Accumulates per-frame CE by phase and by boundary/interior position within the clip (label changes at t or
    t-1 inside the clip => boundary frame)."""

    def __init__(self) -> None:
        self.sum = np.zeros(NUM_CLASSES); self.cnt = np.zeros(NUM_CLASSES)
        self.b_sum = self.b_cnt = self.i_sum = self.i_cnt = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        lg = logits.detach().float()
        if y.dim() == 1:  # per-frame model: (B, K) / (B,)
            lg, y2 = lg.reshape(-1, lg.shape[-1]), y.reshape(1, -1)
            per = nn.functional.cross_entropy(lg, y2.reshape(-1), reduction="none").reshape(1, -1)
            boundary = torch.zeros_like(y2, dtype=torch.bool)  # no temporal context in a frame batch
        else:
            B, L = y.shape
            per = nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").reshape(B, L)
            y2 = y
            change = torch.zeros_like(y, dtype=torch.bool)
            change[:, 1:] = y[:, 1:] != y[:, :-1]
            boundary = change.clone(); boundary[:, :-1] |= change[:, 1:]
        yc, pc = y2.reshape(-1).cpu().numpy(), per.reshape(-1).cpu().numpy()
        np.add.at(self.sum, yc, pc); np.add.at(self.cnt, yc, 1)
        bm = boundary.reshape(-1).cpu().numpy()
        self.b_sum += pc[bm].sum(); self.b_cnt += bm.sum(); self.i_sum += pc[~bm].sum(); self.i_cnt += (~bm).sum()

    def summary(self) -> dict[str, Any]:
        with np.errstate(invalid="ignore", divide="ignore"):
            per_phase = {CLASS_NAMES[k]: float(self.sum[k] / self.cnt[k]) if self.cnt[k] else None for k in range(NUM_CLASSES)}
        return {"loss_per_phase": per_phase,
                "loss_boundary": float(self.b_sum / self.b_cnt) if self.b_cnt else None,
                "loss_interior": float(self.i_sum / self.i_cnt) if self.i_cnt else None}


def confusion_matrices(seqs: list[dict], log_trans: np.ndarray) -> dict[str, list[list[int]]]:
    """16x16 confusion (rows = truth, cols = prediction) for raw argmax and for Viterbi decoding."""
    cm_a = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64); cm_v = cm_a.copy()
    for s in seqs:
        y = np.asarray(s["labels"]); lp = np.asarray(s["log_probs"])
        a = lp.argmax(1); v = viterbi(lp, log_trans)
        np.add.at(cm_a, (y, a), 1); np.add.at(cm_v, (y, v), 1)
    return {"argmax": cm_a.tolist(), "viterbi": cm_v.tolist()}


def boundary_entropy(seqs: list[dict], radius: int = 2) -> dict[str, float]:
    """Mean predictive entropy (nats) at frames within ``radius`` of a ground-truth transition vs elsewhere."""
    b, i = [], []
    for s in seqs:
        y = np.asarray(s["labels"]); lp = np.asarray(s["log_probs"], dtype=np.float64)
        ent = -(np.exp(lp) * lp).sum(1)
        change = np.zeros(len(y), bool); change[1:] = y[1:] != y[:-1]
        near = np.zeros(len(y), bool)
        for t in np.where(change)[0]:
            near[max(0, t - radius): t + radius + 1] = True
        b.append(ent[near]); i.append(ent[~near])
    b = np.concatenate(b) if b else np.array([]); i = np.concatenate(i) if i else np.array([])
    return {"entropy_boundary": float(b.mean()) if b.size else None, "entropy_interior": float(i.mean()) if i.size else None,
            "n_boundary_frames": int(b.size), "n_interior_frames": int(i.size)}


def worst_videos(per_video: list[dict], k: int) -> list[dict]:
    ranked = sorted(per_video, key=lambda v: v.get("acc_viterbi", 1.0))[:k]
    return [{"video": v["video"], "acc_viterbi": v["acc_viterbi"], "acc": v["acc"], "n_frames": v["n_frames"],
             "n_transitions": v["n_transitions"], "n_far": v["n_far"]} for v in ranked]


def dump_frame_probs(seqs: list[dict], path: Path) -> None:
    """One npz: per video ``<vid>__lp`` (T,16) float16, ``<vid>__y`` (T,) int16, ``<vid>__t`` (T,) float32."""
    arrs = {}
    for s in seqs:
        v = str(s["video"])
        arrs[f"{v}__lp"] = np.asarray(s["log_probs"], dtype=np.float16)
        arrs[f"{v}__y"] = np.asarray(s["labels"], dtype=np.int16)
        arrs[f"{v}__t"] = np.asarray(s["times_h"], dtype=np.float32)
    np.savez_compressed(path, **arrs)


@torch.no_grad()
def extract_embeddings(model: nn.Module, frames_by_video: list[tuple[str, torch.Tensor, np.ndarray]], device, amp: bool) -> dict[str, np.ndarray] | None:
    """Backbone features per frame for separable models (has .backbone). Returns None otherwise."""
    if not hasattr(model, "backbone"):
        return None
    arrs = {}
    for vid, frames, y in frames_by_video:
        fs = []
        for i in range(0, len(frames), 256):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                fs.append(model.backbone(frames[i:i + 256].to(device, non_blocking=True)).float().cpu())
        arrs[f"{vid}__emb"] = torch.cat(fs).numpy().astype(np.float16)
        arrs[f"{vid}__y"] = np.asarray(y, dtype=np.int16)
    return arrs


class DiagnosticsWriter:
    def __init__(self, out_dir: Path, cfg: dict | None) -> None:
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.dir = out_dir / "diagnostics"; self.dir.mkdir(parents=True, exist_ok=True)
        self.epochs: list[dict] = []

    def log_epoch(self, rec: dict) -> None:
        self.epochs.append(rec)
        (self.dir / "epochs.json").write_text(json.dumps(self.epochs, indent=1, default=float))

    def save_json(self, name: str, obj: Any) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=1, default=float))
