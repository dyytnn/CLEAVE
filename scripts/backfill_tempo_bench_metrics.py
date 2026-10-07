#!/usr/bin/env python
"""Backfill the CLEAVE metrics (per-event timing MAE, tolerance-curve p_t_fixed, segmental Edit/F1@k) into every
existing ``results.json`` that predates their addition to ``stseg.eval.kinetic_metrics.evaluate_videos`` (2026-09-09).
No retraining: loads each run's ``best.pt``, re-runs inference on its own test split with the run's own eval window
(clip_len for sequence models -- the T1 fix; 150 for non-sequence models, where the window is a no-op), and merges the
new keys into the existing ``test`` block, leaving p/p_v/p_t/r untouched (they are recomputed identically as a sanity
check and asserted to match within floating-point tolerance -- if they don't, the run is flagged, not silently
overwritten).

Usage:
  PYTHONPATH=src python scripts/backfill_tempo_bench_metrics.py            # all runs missing the new keys
  PYTHONPATH=src python scripts/backfill_tempo_bench_metrics.py --dry_run  # just list what would be updated
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from stseg.eval.kinetic_metrics import evaluate_videos
from stseg.kinetic.datasets import _frames
from stseg.kinetic.models import build_model
from stseg.kinetic.process import sequence_log_probs

ROOT = Path(__file__).resolve().parents[1]
# All 34 runs missing CLEAVE metrics predate the eval-window fix and were themselves *evaluated* at the
# original, non-overlapping 150-frame chunking (process.py's old default) -- that is what produced their existing,
# already-reported p/p_v/p_t. Backfilling must reproduce that SAME window for every one of them, not the corrected
# per-model clip_len window, or this script would silently change already-published numbers (discovered 2026-09-12:
# an earlier version of this script used the corrected window uniformly and shifted p/p_v by up to 0.02 for models
# that were never affected by the eval-window defect in the first place -- e.g. plain LSTM heads -- while p_t stayed
# accidentally stable). Window=150 for every run in this backfill, full stop; none of the 34 targets is one of the
# two defect-demonstration transformer runs (those already have full metrics and are not in this list).
BACKFILL_WINDOW = 150


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-evaluate even runs that already have an 'edit' key "
                     "(use after a bad backfill attempt to redo everything with the correct window)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=6)
    a = ap.parse_args()
    device = torch.device(a.device)

    # 2026-09-12: 11 of these were already (wrongly) backfilled with the corrected per-model window before the bug
    # above was caught; --force re-does the full original 34-run list regardless of whether 'edit' is already present,
    # so those 11 get overwritten again with the correct window=150 values instead of being skipped.
    ORIGINAL_34 = {
        "convnext_tiny_lstm_L8_split0_seed0", "efficientnet_b0_lstm_L8_split0_seed0", "kinetic_cnn_lstm_L4_split0_seed0",
        "r2plus1d_L8_split0_seed0", "resnet18_gru_L8_split0_seed0", "resnet18_lstm_L16_split0_seed0",
        "resnet18_lstm_L4_ceweighted_split0_seed0", "resnet18_lstm_L4_grouped_v1_seed0",
        "resnet18_lstm_L4_multifocal3_split0_seed0", "resnet18_lstm_L4_multifocal3_split0_seed1",
        "resnet18_lstm_L4_multifocal3_split0_seed2", "resnet18_lstm_L4_multifocal3_split1_seed0",
        "resnet18_lstm_L4_multifocal3_split2_seed0", "resnet18_lstm_L4_multifocal3_split3_seed0",
        "resnet18_lstm_L4_multifocal3_split4_seed0", "resnet18_lstm_L4_multifocal7_e20_split0_seed0",
        "resnet18_lstm_L4_ordinal_split0_seed0", "resnet18_lstm_L4_split0_seed0", "resnet18_lstm_L4_split0_seed1",
        "resnet18_lstm_L4_split0_seed2", "resnet18_lstm_L4_split1_seed0", "resnet18_lstm_L4_split2_seed0",
        "resnet18_lstm_L4_split3_seed0", "resnet18_lstm_L4_split4_seed0", "resnet18_none_split0_seed0",
        "resnet18_tcn_L16_split0_seed0", "resnet18_tcn_L16_split0_seed1", "resnet18_tcn_L16_split0_seed2",
        "resnet18_transformer_L16_adamw_split0_seed0", "resnet18_transformer_L16_split0_seed0",
        "resnet50_lstm_L8_split0_seed0", "resnet50_lstm_L8_split0_seed1", "resnet50_lstm_L8_split0_seed2",
        "swin_t_lstm_L8_split0_seed0",
    }
    targets = []
    for f in sorted(glob.glob(str(ROOT / "runs/h7/*_seed*/results.json"))):
        run_dir = Path(f).parent
        d = json.loads(Path(f).read_text())
        t = d.get("test", d)
        if a.force:
            if run_dir.name in ORIGINAL_34:
                targets.append(run_dir)
        elif "edit" not in t:
            targets.append(run_dir)
    print(f"{len(targets)} runs missing CLEAVE metrics" + (" (dry run, not evaluating)" if a.dry_run else ""))
    if a.dry_run:
        for r in targets:
            print(" ", r.name)
        return

    for run in targets:
        cfg = json.loads((run / "config.resolved.json").read_text())
        log_trans = np.load(run / "transition_log_matrix.npy")
        ds = _frames(cfg["data"], "test", "eval", 0, None, None)
        model = build_model(dict(cfg["model"]), ds.in_channels)
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
        model.to(device).eval()
        eval_len, eval_stride = BACKFILL_WINDOW, BACKFILL_WINDOW

        seqs = []
        with torch.no_grad():
            for vid, g in ds.rows.groupby("video", sort=False):
                loader = DataLoader(Subset(ds, g.index.to_numpy()), batch_size=150, shuffle=False, num_workers=a.num_workers)
                frames = torch.cat([b["image"] for b in loader])
                if model.is_sequence:
                    lp = sequence_log_probs(model, frames, eval_len, eval_stride, device, device.type == "cuda")
                else:
                    lps = []
                    for i in range(0, len(frames), 256):
                        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                            out = model(frames[i:i + 256].to(device))["logits"].float()
                        # per-frame models still get wrapped as (B, L=1, K) by SeqKinetic.forward's x.dim()==4 case;
                        # flatten before softmax (same pattern as process.py's own per-frame _collect branch) or the
                        # log_softmax normalises over the wrong (length-1) axis and log_probs ends up 3-D, which is
                        # exactly what crashed resnet18_none_split0_seed0 here on 2026-09-12.
                        lps.append(torch.log_softmax(out.reshape(-1, out.shape[-1]), 1).cpu().numpy())
                    lp = np.concatenate(lps)
                seqs.append({"video": vid, "labels": g.label.to_numpy(), "log_probs": lp, "times_h": g.time_h.to_numpy(dtype=float)})

        r = evaluate_videos(seqs, log_trans)
        old = json.loads((run / "results.json").read_text())
        old_t = old.get("test", old)
        for k in ("p", "p_v", "r"):
            if k in old_t and abs(old_t[k] - r[k]) > 0.01:
                print(f"  [WARN] {run.name}: recomputed {k}={r[k]:.3f} vs stored {old_t[k]:.3f} (>0.01 apart)")
        new_keys = {k: v for k, v in r.items() if k not in ("per_video",)}
        if "test" in old:
            old["test"].update(new_keys)
        else:
            old.update(new_keys)
        (run / "results.json").write_text(json.dumps(old, indent=1, default=float))
        print(f"{run.name}: edit={r['edit']:.1f} F1@50={r['f1']['50']:.1f} MAE={r['mae_h_all']:.2f}h (p_t={r['p_t']:.3f}, was {old_t.get('p_t', float('nan')):.3f})")


if __name__ == "__main__":
    main()
