"""The runnable product: one H7 experiment (train on Nantes, select on val, evaluate once on test)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from stseg.data.nantes_kinetic import CLASS_NAMES, NUM_CLASSES
from stseg.eval.kinetic_metrics import evaluate_videos

from .interfaces import AbsProcess
from . import diagnostics as diag


def _configure_data_worker(_worker_id: int) -> None:
    """Prevent each DataLoader process from spawning a full CPU thread pool."""
    torch.set_num_threads(1)


def _commit(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def sequence_log_probs(model, frames: torch.Tensor, eval_len: int, eval_stride: int, device, amp: bool) -> np.ndarray:
    """Whole-video inference for a sequence model. ``frames`` (T, C, H, W) on CPU.

    Windows of ``eval_len`` frames start every ``eval_stride`` frames (the last window is aligned to the end); log-softmax
    outputs are averaged where windows overlap. ``eval_stride == eval_len`` is the paper's non-overlapping chunking.
    Heads with learned absolute positions (transformer) must be evaluated with ``eval_len == clip_len``: they only saw
    positions 0..clip_len-1 during training (2026-09-09: p_t 0.44 -> 0.79 on 8 test videos by fixing this).
    For ``SeqKinetic`` the backbone runs once over all frames and only the head is re-run per window.
    """
    T = len(frames)
    starts = list(range(0, max(T - eval_len, 0) + 1, eval_stride))
    if starts[-1] + eval_len < T:
        starts.append(T - eval_len)
    acc = torch.zeros(T, NUM_CLASSES, dtype=torch.float32); cnt = torch.zeros(T, 1)
    separable = hasattr(model, "backbone") and hasattr(model, "head") and hasattr(model, "cls")
    feats = None
    if separable:
        fs = []
        for i in range(0, T, 256):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                fs.append(model.backbone(frames[i:i + 256].to(device, non_blocking=True)).float())
        feats = torch.cat(fs)
    for s in starts:
        e = min(s + eval_len, T)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            if separable:
                out = model.cls(model.head(feats[s:e][None])).float()[0]
            else:
                out = model(frames[s:e][None].to(device, non_blocking=True))["logits"].float()[0]
        acc[s:e] += torch.log_softmax(out, 1).cpu(); cnt[s:e] += 1
    return (acc / cnt).numpy()


class KineticTrainProcess(AbsProcess):
    def __init__(self, cfg: dict, data, model, criterion, optimizer, scheduler, log_trans: np.ndarray, out_dir: Path,
                 seed: int, smoke: bool, device: torch.device, resume: bool = True,
                 evaluate_test: bool = True) -> None:
        self.cfg, self.data, self.model, self.criterion = cfg, data, model, criterion
        self.opt, self.sched, self.log_trans, self.out_dir = optimizer, scheduler, log_trans, out_dir
        self.seed, self.smoke, self.device, self.resume = seed, smoke, device, resume
        self.evaluate_test = bool(evaluate_test)
        if not self.evaluate_test and hasattr(data, "test"):
            raise ValueError("validation-only process must not receive a test dataset")
        t = cfg["training"]
        self.epochs = (2 if int(t.get("swa_last", 0)) else 1) if smoke else int(t.get("epochs", 10))  # smoke: 2 epochs exercise the SWA path
        self.bs = int(t.get("batch_size", data.default_batch)); self.ebs = int(t.get("eval_batch_size", 150))
        self.eval_len = int(t.get("eval_len", 150)); self.eval_stride = int(t.get("eval_stride", self.eval_len)); self.nw = int(t.get("num_workers", 8))
        self.amp = bool(t.get("amp", True)) and device.type == "cuda"; self.monitor = t.get("monitor", "p_v")
        self.diag = diag.DiagnosticsWriter(out_dir, cfg.get("logging"))

    def _forward_train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        frame_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a train forward, supplying labels only to generative models."""
        if getattr(self.model, "requires_frame_index", False):
            if frame_input is None:
                raise ValueError("model requires frame_input but the dataset did not provide it")
            return self.model(x, frame_input)
        if getattr(self.model, "requires_labels", False):
            return self.model(x, y)
        return self.model(x)

    # ---------------------------------------------------------------- evaluation
    @torch.no_grad()
    def _collect(self, ds) -> list[dict]:
        self.model.eval()
        seqs = []
        if self.model.is_sequence:  # whole video in windows of eval_len frames every eval_stride (paper: 150, non-overlapping)
            for vid, g in ds.rows.groupby("video", sort=False):
                loader = DataLoader(Subset(ds, g.index.to_numpy()), batch_size=self.ebs, shuffle=False, num_workers=self.nw, pin_memory=True,
                                    worker_init_fn=_configure_data_worker)
                frames = torch.cat([b["image"] for b in loader])
                lp = sequence_log_probs(self.model, frames, self.eval_len, self.eval_stride, self.device, self.amp)
                seqs.append({"video": vid, "labels": g.label.to_numpy(), "log_probs": lp, "times_h": g.time_h.to_numpy(dtype=float)})
            return seqs
        loader = DataLoader(ds, batch_size=self.ebs, shuffle=False, num_workers=self.nw, pin_memory=True,
                            worker_init_fn=_configure_data_worker)
        lps, labels, vids, times = [], [], [], []
        for b in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                images = b["image"].to(self.device, non_blocking=True)
                if getattr(self.model, "requires_frame_index", False):
                    frame_input = b["frame_input"].to(self.device, non_blocking=True)
                    out = self.model(images, frame_input)["logits"].float()
                else:
                    out = self.model(images)["logits"].float()
            lps.append(torch.log_softmax(out.reshape(-1, out.shape[-1]), 1).cpu().numpy())
            labels.append(b["label"].numpy()); vids += list(b["video"]); times.append(b["time_h"].numpy())
        lp, y, t, v = np.concatenate(lps), np.concatenate(labels), np.concatenate(times), np.asarray(vids)
        for vid in dict.fromkeys(v):
            m = v == vid
            seqs.append({"video": vid, "labels": y[m], "log_probs": lp[m], "times_h": t[m]})
        return seqs

    def evaluate(self, ds) -> dict:
        return self.evaluate_with_seqs(ds)[0]

    def evaluate_with_seqs(self, ds) -> tuple[dict, list[dict]]:
        seqs = self._collect(ds)
        return evaluate_videos(seqs, self.log_trans), seqs

    # ---------------------------------------------------------------- training
    def run(self) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.out_dir / "transition_log_matrix.npy", self.log_trans)
        (self.out_dir / "config.resolved.json").write_text(json.dumps(self.cfg, indent=2, default=str))
        bs = max(2, min(self.bs, len(self.data.train_set))) if self.smoke else self.bs
        loader = DataLoader(self.data.train_set, batch_size=bs, shuffle=True, num_workers=self.nw, drop_last=not self.smoke,
                            pin_memory=True, persistent_workers=self.nw > 0,
                            worker_init_fn=_configure_data_worker)
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        best, history, start = -1.0, [], 1
        last = self.out_dir / "last.pt"
        swa_last = int(self.cfg.get("training", {}).get("swa_last", 0)); swa_state, swa_n = None, 0
        if self.resume and last.exists() and not self.smoke:  # interrupted run (shared GPU): continue from the last epoch
            ck = torch.load(last, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ck["model"]); self.opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
            self.sched.load_state_dict(ck.get("scheduler", {})); best, history, start = ck["best"], ck["history"], ck["epoch"] + 1
            print(f"resumed from {last} (epoch {ck['epoch']}, best {self.monitor}={best:.4f})", flush=True)
        dcfg = self.diag.cfg
        for epoch in range(start, self.epochs + 1):
            self.model.train(); t0 = time.time(); running = 0.0; steps = 0
            meter = diag.PhaseLossMeter() if dcfg["per_phase_loss"] else None
            gsum: dict[str, float] = {}
            accum = max(1, int(self.cfg.get("training", {}).get("grad_accum", 1)))  # T11: effective batch = batch_size * accum
            self.opt.zero_grad(set_to_none=True)
            for i, b in enumerate(loader):
                x, y = b["image"].to(self.device, non_blocking=True), b["label"].to(self.device, non_blocking=True)
                frame_input = b.get("frame_input")
                if frame_input is not None:
                    frame_input = frame_input.to(self.device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    out = self._forward_train(x, y, frame_input)
                    logits = out["logits"]
                    loss = self.criterion(out if getattr(self.criterion, "needs_outputs", False) else logits, y)
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum != 0 and i + 1 < len(loader):
                    running += float(loss); steps += 1
                    if meter is not None:
                        meter.update(logits, y)
                    continue
                if dcfg["grad_norms"]:
                    scaler.unscale_(self.opt)  # norms of the true gradients; GradScaler.step skips re-unscaling
                    for k, v in diag.grad_norms(self.model).items():
                        gsum[k] = gsum.get(k, 0.0) + v
                scaler.step(self.opt); scaler.update(); self.opt.zero_grad(set_to_none=True)
                if meter is not None:
                    meter.update(logits, y)
                running += float(loss); steps += 1
                if self.smoke and steps >= 5:
                    break
            self.sched.step()
            if hasattr(self.model, "use_selection_inference"):
                self.model.use_selection_inference()
            val, val_seqs = self.evaluate_with_seqs(self.data.val)
            rec = {"epoch": epoch, "train_loss": running / max(steps, 1), "val_p": val["p"], "val_p_v": val["p_v"], "val_r": val["r"],
                   "val_p_t": val["p_t"], "minutes": (time.time() - t0) / 60}
            history.append(rec)
            drec: dict[str, Any] = {"epoch": epoch, "lr": float(self.opt.param_groups[0]["lr"]),
                                    "weight_norm": float(sum(p.detach().float().pow(2).sum() for p in self.model.parameters()) ** 0.5)}
            if gsum:
                drec.update({k: v / max(steps, 1) for k, v in gsum.items()})
            if meter is not None:
                drec.update(meter.summary())
            if dcfg["confusion"]:
                drec["confusion_val"] = diag.confusion_matrices(val_seqs, self.log_trans)
            drec["timing_error_val"] = val.get("timing_error_h"); drec["p_t_fixed_val"] = val.get("p_t_fixed")
            drec.update(diag.boundary_entropy(val_seqs))
            if dcfg["worst_k"]:
                drec["worst_val_videos"] = diag.worst_videos(val["per_video"], int(dcfg["worst_k"]))
            self.diag.log_epoch(drec)
            print(f"epoch {epoch}/{self.epochs} loss={rec['train_loss']:.4f} val p={val['p']:.4f} p_v={val['p_v']:.4f} r={val['r']:.3f} p_t={val['p_t']:.4f} ({rec['minutes']:.1f} min)", flush=True)
            if val[self.monitor] > best:
                best = val[self.monitor]
                torch.save({"model": self.model.state_dict(), "epoch": epoch, f"val_{self.monitor}": best}, self.out_dir / "best.pt")
            if swa_last and epoch > self.epochs - swa_last:  # TEMPO v17: running average of the last k epochs' weights
                sd = {k: v.detach().float().cpu() for k, v in self.model.state_dict().items()}
                if swa_state is None:
                    swa_state, swa_n = sd, 1
                else:
                    swa_n += 1
                    for k in swa_state:
                        if swa_state[k].dtype.is_floating_point:
                            swa_state[k] += (sd[k] - swa_state[k]) / swa_n
                        else:
                            swa_state[k] = sd[k]
            (self.out_dir / "history.json").write_text(json.dumps(history, indent=2))
            torch.save({"model": self.model.state_dict(), "optimizer": self.opt.state_dict(), "scaler": scaler.state_dict(),
                        "scheduler": self.sched.state_dict(), "epoch": epoch, "best": best, "history": history}, last)

        state = torch.load(self.out_dir / "best.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        if hasattr(self.model, "use_final_inference"):
            self.model.use_final_inference()
        val, val_seqs = self.evaluate_with_seqs(self.data.val)
        test = test_seqs = None
        if self.evaluate_test:
            test, test_seqs = self.evaluate_with_seqs(self.data.test)
        swa_results = None
        if swa_state is not None and swa_n > 1:
            # averaged weights need fresh BatchNorm statistics: reset and re-estimate with forward passes over training clips
            best_sd = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict({k: v.to(self.device) if v.dtype.is_floating_point else v for k, v in swa_state.items()})
            bn = [m for m in self.model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
            if bn:
                mom = [m.momentum for m in bn]
                for m in bn: m.reset_running_stats(); m.momentum = None
                self.model.train()
                with torch.no_grad():
                    for i, b in enumerate(loader):
                        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                            bx = b["image"].to(self.device, non_blocking=True)
                            by = b["label"].to(self.device, non_blocking=True)
                            frame_input = b.get("frame_input")
                            if frame_input is not None:
                                frame_input = frame_input.to(self.device, non_blocking=True)
                            self._forward_train(bx, by, frame_input)
                        if i >= 60: break
                for m, mo in zip(bn, mom): m.momentum = mo
            self.model.eval()
            sval, sval_seqs = self.evaluate_with_seqs(self.data.val)
            stest = stest_seqs = None
            if self.evaluate_test:
                stest, stest_seqs = self.evaluate_with_seqs(self.data.test)
            torch.save({"model": self.model.state_dict(), "epochs_averaged": swa_n}, self.out_dir / "swa.pt")
            swa_results = {"n_epochs": swa_n, "val": {k: v for k, v in sval.items() if k != "per_video"}}
            if self.evaluate_test:
                swa_results["test"] = {k: v for k, v in stest.items() if k != "per_video"}
            message = f"swa({swa_n} epochs): val {self.monitor}={sval[self.monitor]:.4f} vs best-ckpt {val[self.monitor]:.4f}"
            if self.evaluate_test:
                message += f" | test p_t {stest['p_t']:.4f} vs {test['p_t']:.4f}"
            print(message, flush=True)
            if sval[self.monitor] >= val[self.monitor]:  # selected on validation only
                val, val_seqs = sval, sval_seqs
                if self.evaluate_test:
                    test, test_seqs = stest, stest_seqs
                state = {"epoch": f"swa{swa_n}"}
            else:
                self.model.load_state_dict(best_sd)
        if dcfg["frame_probs"]:
            diag.dump_frame_probs(val_seqs, self.diag.dir / "frame_probs_val.npz")
            if self.evaluate_test:
                diag.dump_frame_probs(test_seqs, self.diag.dir / "frame_probs_test.npz")
        if self.evaluate_test:
            self.diag.save_json("confusion_test.json", diag.confusion_matrices(test_seqs, self.log_trans))
            self.diag.save_json("boundary_entropy_test.json", diag.boundary_entropy(test_seqs))
        if dcfg["embeddings"]:
            partitions = [("val", self.data.val)]
            if self.evaluate_test:
                partitions.append(("test", self.data.test))
            for name, ds in partitions:
                fbv = []
                for vid, g in ds.rows.groupby("video", sort=False):
                    ld = DataLoader(Subset(ds, g.index.to_numpy()), batch_size=self.ebs, shuffle=False, num_workers=self.nw,
                                    worker_init_fn=_configure_data_worker)
                    fbv.append((vid, torch.cat([bb["image"] for bb in ld]), g.label.to_numpy()))
                emb = diag.extract_embeddings(self.model, fbv, self.device, self.amp)
                if emb is not None:
                    np.savez_compressed(self.diag.dir / f"embeddings_{name}.npz", **emb)
        strip = lambda m: {k: v for k, v in m.items() if k != "per_video"}
        accumulation = max(1, int(self.cfg.get("training", {}).get("grad_accum", 1)))
        batches_per_epoch = int(np.ceil(len(self.data.train_set) / max(self.bs, 1)))
        accounting = {
            "train_samples_per_epoch": int(len(self.data.train_set)),
            "epochs": int(self.epochs),
            "frame_exposures": int(len(self.data.train_set) * self.epochs),
            "micro_batch_size": int(self.bs),
            "gradient_accumulation": accumulation,
            "effective_batch_size": int(self.bs * accumulation),
            "optimizer_steps": int(np.ceil(batches_per_epoch / accumulation) * self.epochs),
        }
        results = {"best_epoch": state["epoch"], "val": strip(val), "test_evaluated": self.evaluate_test,
                   "training_accounting": accounting}
        if self.evaluate_test:
            results["test"] = strip(test)
        if swa_results is not None:
            results["swa"] = swa_results; results["final_weights"] = "swa" if isinstance(state["epoch"], str) else "best_ckpt"
        (self.out_dir / "results.json").write_text(json.dumps(results, indent=2))
        (self.out_dir / "per_video_val.json").write_text(json.dumps(val["per_video"], indent=1))
        if self.evaluate_test:
            (self.out_dir / "per_video_test.json").write_text(json.dumps(test["per_video"], indent=1))
        (self.out_dir / "meta.json").write_text(json.dumps({
            "experiment": self.cfg.get("experiment", {}), "seed": self.seed, "smoke": self.smoke, "stseg_commit": _commit(Path(__file__).resolve().parents[3]),
            "class_names": CLASS_NAMES, "protocol": "arXiv 2203.00531 (16 classes, decoder + p/p_v/r/p_t)",
            "model": self.cfg["model"], "loss": self.cfg.get("loss"), "data": self.cfg["data"], "decoder": self.cfg.get("decoder"),
            "n_params_M": sum(p.numel() for p in self.model.parameters()) / 1e6,
            "training_accounting": accounting}, indent=2, default=str))
        last.unlink(missing_ok=True)  # finished: keep best.pt only
        if self.evaluate_test:
            print(f"done; best epoch {state['epoch']} | TEST p={test['p']:.4f} p_v={test['p_v']:.4f} r={test['r']:.3f} p_t={test['p_t']:.4f} | {self.out_dir}")
        else:
            print(f"done; best epoch {state['epoch']} | VAL p_t={val['p_t']:.4f} | test NOT CONSTRUCTED | {self.out_dir}")
        return results
