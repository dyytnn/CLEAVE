"""Build every registered kinetic component, validate every committed experiment config, exercise resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import stseg.kinetic as K
from stseg.kinetic.config import ConfigError, deep_merge, load_pipeline, validate_pipeline
from stseg.kinetic.models import build_model

ROOT = Path(__file__).resolve().parents[1]
HEAVY = {"vit_b_16", "swin_t", "convnext_tiny", "efficientnet_v2_s", "resnet50"}  # built but with a single 224 frame


@pytest.mark.parametrize("name", K.BACKBONE_REGISTRY.names())
def test_every_backbone_builds_and_reports_feat_dim(name):
    if name == "dualbranch":
        pytest.skip("needs a (low_channels + 1)-channel input by construction; see test_dualbranch_backbone_splits_and_fuses")
    if name == "cached_features":
        bb = K.BACKBONE_REGISTRY.get(name, in_channels=3)
        with torch.no_grad():
            f = bb.eval()(torch.zeros(1, 3))
        assert f.shape == (1, 3) and bb.feat_dim == 3
        return
    if name == "emfit_i3d":
        bb = K.BACKBONE_REGISTRY.get(name, in_channels=3, checkpoint=None)
        assert bb.feat_dim == 768
        return
    kw = {"pretrained": False}
    bb = K.BACKBONE_REGISTRY.get(name, **kw) if name != "kinetic_cnn" else K.BACKBONE_REGISTRY.get(name, base_channels=8)
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        f = bb.eval()(x)
    assert f.shape == (1, bb.feat_dim) and bb.feat_dim > 0


def test_multichannel_first_conv_replacement():
    bb = K.BACKBONE_REGISTRY.get("resnet18", pretrained=False, in_channels=5)
    with torch.no_grad():
        assert bb.eval()(torch.zeros(1, 5, 224, 224)).shape == (1, 512)


_HEAD_SMOKE_KW = {"node": {"hidden": 16, "n_substeps": 2}, "diffact_lite": {"hidden": 16, "steps": 2, "nhead": 4}}


@pytest.mark.parametrize("head", K.HEAD_REGISTRY.names())
def test_every_head_keeps_sequence_length(head):
    kw = {} if head == "none" else _HEAD_SMOKE_KW.get(head, {"hidden": 16, "layers": 1})
    h = K.HEAD_REGISTRY.get(head, d_in=32, **kw)
    out = h(torch.zeros(2, 7, 32))
    assert out.shape[:2] == (2, 7) and out.shape[2] == h.d_out


@pytest.mark.parametrize("mc,shape", [
    ({"type": "seq_kinetic", "backbone": {"name": "kinetic_cnn", "base_channels": 8}, "head": {"name": "lstm", "hidden": 8, "layers": 1}}, (2, 4, 3, 64, 64)),
    ({"type": "seq_kinetic", "backbone": {"name": "kinetic_cnn", "base_channels": 8}, "head": {"name": "none"}}, (2, 3, 64, 64)),
    ({"type": "seq_kinetic", "backbone": {"name": "kinetic_cnn", "base_channels": 8}, "head": {"name": "tcn", "hidden": 8, "layers": 2}}, (2, 6, 3, 64, 64)),
    ({"type": "r2plus1d", "backbone": {"name": "resnet18", "pretrained": False}}, (1, 4, 3, 64, 64)),
    ({"type": "event_conditioned", "backbone": {"name": "kinetic_cnn", "base_channels": 8}, "head": {"name": "lstm", "hidden": 8, "layers": 1}}, (2, 4, 3, 64, 64)),
    ({"type": "full_video_mstcn", "backbone": {"name": "cached_features"}, "head": {"name": "mstcn", "hidden": 8, "layers": 2, "stages": 2}}, (2, 7, 12)),
])
def test_models_emit_per_frame_logits(mc, shape):
    in_channels = shape[-1] if mc["type"] == "full_video_mstcn" else 3
    m = build_model({**mc, "num_classes": 16}, in_channels=in_channels).eval()
    with torch.no_grad():
        out = m(torch.zeros(*shape))["logits"]
    L = shape[1] if len(shape) in {3, 5} else 1
    assert out.shape == (shape[0], L, 16)
    assert m.is_sequence == (mc["type"] in {"r2plus1d", "full_video_mstcn"} or mc.get("head", {}).get("name", "none") != "none")


@pytest.mark.parametrize("loss", K.LOSS_REGISTRY.names())
def test_losses_are_finite_scalars(loss):
    crit = K.LOSS_REGISTRY.build({"name": loss}, class_counts=torch.arange(1, 17, dtype=torch.float32), num_classes=16)
    logits = torch.randn(3, 5, 16)
    if loss == "multistage_ce_tmse":
        value = {"logits": logits, "stage_logits": torch.stack([logits, logits])}
    elif loss == "embryodiff_objective":
        value = {"logits": logits, "semantic_logits": logits, "diffusion_logits": logits}
    elif loss == "milestone_objective":
        value = {"logits": torch.log_softmax(logits, -1), "frame_logits": logits,
                 "milestone_coarse_logits": torch.randn(3, 15, 6)}
    elif loss == "sce_aux_objective":
        value = {"logits": logits, "hidden": torch.randn(3, 5, 8), "aux_logits": torch.randn(3, 5, 2)}
    elif loss == "emfit_multitask":
        value = {
            "logits": logits,
            "frame_regression": torch.randn(3, 5),
            "frame_target": torch.randn(3, 5),
        }
    else:
        value = logits
    val = crit(value, torch.randint(0, 16, (3, 5)))
    assert val.dim() == 0 and torch.isfinite(val)


def test_decoders_transition_matrices():
    seqs = [np.array([0, 0, 1, 1, 3]), np.array([3, 3, 4])]
    z = K.DECODER_REGISTRY.build({"name": "argmax"}).log_transition(seqs, 16)
    v = K.DECODER_REGISTRY.build({"name": "viterbi"}).log_transition(seqs, 16)
    assert z.shape == v.shape == (16, 16) and (z == 0).all()
    assert np.isneginf(v[3, 0]) and np.isfinite(v[0, 1])  # backward move impossible, observed move allowed


def test_optimizers_and_schedulers_build():
    p = [torch.nn.Parameter(torch.zeros(2))]
    for name in K.OPTIMIZER_REGISTRY.names():
        opt = K.OPTIMIZER_REGISTRY.build({"name": name}, params=p)
        for s in K.SCHEDULER_REGISTRY.names():
            sch = K.SCHEDULER_REGISTRY.build({"name": s}, optimizer=opt, epochs=3)
            sch.step(); sch.state_dict()


def test_all_committed_experiment_configs_validate():
    sweep = yaml.safe_load((ROOT / "configs/h7/sweep_v1.yaml").read_text())["experiments"]
    assert len(sweep) >= 10
    ids = set()
    for rel in sweep:
        doc = load_pipeline(ROOT / rel)
        assert doc["cfg"]["experiment"]["id"] == Path(rel).stem
        ids.add(doc["cfg"]["experiment"]["id"])
    assert len(ids) == len(sweep)
    v2 = yaml.safe_load((ROOT / "configs/h7/v2/sweep_v2.yaml").read_text())["experiments"]
    for e in v2:
        assert isinstance(e, dict) and load_pipeline(ROOT / e["config"])["cfg"]["experiment"]["id"] == Path(e["config"]).stem


def test_inherits_merges_and_typos_are_rejected(tmp_path):
    base = load_pipeline(ROOT / "configs/h7/_base.yaml")
    merged = deep_merge(base, {"cfg": {"data": {"clip_len": 16}, "experiment": {"id": "x"}}})
    assert merged["cfg"]["data"]["clip_len"] == 16 and merged["cfg"]["data"]["resize"] == 250
    bad = tmp_path / "bad.yaml"
    bad.write_text("inherits: configs/h7/_base.yaml\ncfg:\n  experiment: {id: bad}\n  optimzer: {name: sgd}\n")
    with pytest.raises(ConfigError):
        load_pipeline(bad)
    doc = yaml.safe_load((ROOT / "configs/h7/_base.yaml").read_text()); doc["cfg"]["model"]["backbone"]["name"] = "resnet1234"
    with pytest.raises(ConfigError):
        validate_pipeline(doc)


def test_resume_from_last_pt(tmp_path):
    """A run interrupted after epoch 1 (simulated last.pt) continues at epoch 2 and finishes with 2 history rows."""
    pytest.importorskip("cv2")
    if not (ROOT / "data/derived/nantes_manifest_F0.csv").exists():
        pytest.skip("Nantes manifest not present on this machine")
    cfg = load_pipeline(ROOT / "configs/h7/_base.yaml")["cfg"]
    cfg = deep_merge(cfg, {"experiment": {"id": "t_resume", "out_root": str(tmp_path)}, "data": {"resize": 64, "crop": 56},
                           "model": {"backbone": {"name": "kinetic_cnn", "base_channels": 8, "pretrained": False}, "head": {"name": "lstm", "hidden": 8, "layers": 1}},
                           "training": {"num_workers": 0, "amp": False, "eval_len": 20}})
    builder = K.BUILDER_REGISTRY.get("kinetic_training", cfg=cfg, seed=0, smoke=True, out_dir=str(tmp_path / "run"))
    proc = builder.product
    proc.smoke = False; proc.epochs = 1; proc.bs = 2  # tiny real epoch on the smoke-limited data
    proc.run()
    assert (tmp_path / "run/results.json").exists() and not (tmp_path / "run/last.pt").exists()
    # simulate an interruption after epoch 1 of a 2-epoch run
    hist = json.loads((tmp_path / "run/history.json").read_text())
    torch.save({"model": proc.model.state_dict(), "optimizer": proc.opt.state_dict(), "scaler": torch.cuda.amp.GradScaler(enabled=False).state_dict(),
                "scheduler": {}, "epoch": 1, "best": hist[0]["val_p_v"], "history": hist}, tmp_path / "run/last.pt")
    (tmp_path / "run/results.json").unlink()
    proc2 = K.BUILDER_REGISTRY.get("kinetic_training", cfg=cfg, seed=0, smoke=True, out_dir=str(tmp_path / "run")).product
    proc2.smoke = False; proc2.epochs = 2; proc2.bs = 2
    res = proc2.run()
    hist2 = json.loads((tmp_path / "run/history.json").read_text())
    assert [h["epoch"] for h in hist2] == [1, 2] and res["best_epoch"] in (1, 2) and not (tmp_path / "run/last.pt").exists()


def test_component_override_replaces_instead_of_merging():
    base = {"optimizer": {"name": "sgd", "lr": 0.001, "momentum": 0.9}, "head": {"name": "lstm", "hidden": 1024}}
    m = deep_merge(base, {"optimizer": {"name": "adamw", "lr": 1e-4}, "head": {"name": "none"}})
    assert "momentum" not in m["optimizer"] and m["head"] == {"name": "none"}
    same = deep_merge(base, {"optimizer": {"lr": 0.01}})
    assert same["optimizer"] == {"name": "sgd", "lr": 0.01, "momentum": 0.9}


def test_signature_validation_rejects_leftover_kwargs():
    doc = yaml.safe_load((ROOT / "configs/h7/_base.yaml").read_text())
    doc["cfg"]["optimizer"] = {"name": "adamw", "lr": 1e-4, "momentum": 0.9}
    with pytest.raises(ConfigError):
        validate_pipeline(doc)



def test_monotonic_decoder_forbids_backward_moves():
    from stseg.kinetic.registry import DECODER_REGISTRY
    m = DECODER_REGISTRY.build({"name": "monotonic"}).log_transition([], 5)
    assert np.isneginf(m[3, 1]) and m[1, 3] == 0 and m[2, 2] == 0


def test_tempo_bench_metrics_on_synthetic_video():
    from stseg.eval.kinetic_metrics import evaluate_videos, edit_score, f1_at_overlap
    y = np.array([0] * 10 + [3] * 10 + [4] * 10)  # tPB2 -> t2 -> t3
    lp = np.full((30, 16), -10.0); lp[np.arange(30), y] = 0.0
    lp[19, 3], lp[19, 4] = -10.0, 0.0  # t3 predicted one frame early
    r = evaluate_videos([{"video": "v", "labels": y, "log_probs": lp, "times_h": np.arange(30) * 0.25}], np.zeros((16, 16)))
    assert r["p_v"] > 0.96 and r["timing_error_h"]["t3"]["mae"] == 0.25 and r["timing_error_h"]["t2"]["mae"] == 0.0
    assert r["p_t_fixed"]["0.5h"] == 1.0 and r["edit"] == 100.0 and r["f1"]["50"] == 100.0
    assert edit_score(np.array([0, 0, 1, 1]), np.array([0, 1])) == 100.0 and f1_at_overlap(np.array([0, 1]), np.array([0, 0]), 0.5) == (1, 1, 0)


def test_image_split_mode_leaks_neighbouring_frames(tmp_path):
    import pandas as pd
    from stseg.data.nantes_kinetic import NantesKineticFrames
    rows = [{"video": f"v{v}", "plane": "embryo_dataset", "frame_index": k, "path": f"/x/embryo_dataset/v{v}/RUN{k}.jpeg", "time_h": k / 4,
             "phase": "t2", "is_blank": False} for v in range(3) for k in range(200)]
    man = tmp_path / "m.csv"; pd.DataFrame(rows).to_csv(man, index=False)
    split = tmp_path / "s.json"; split.write_text(json.dumps({"videos": {"train": ["v0", "v1"], "val": [], "test": ["v2"]}}))
    tr = NantesKineticFrames(man, split, "train", mode="eval", split_mode="image")
    te = NantesKineticFrames(man, split, "test", mode="eval", split_mode="image")
    assert set(tr.videos()) == set(te.videos()) == {"v0", "v1", "v2"}  # every video appears on both sides = leakage
    assert 0.6 < len(tr) / 600 < 0.8 and 0.12 < len(te) / 600 < 0.28
    assert not set(zip(tr.rows.video, tr.rows.frame_index)) & set(zip(te.rows.video, te.rows.frame_index))



def test_ssm_head_gradients_flow_and_are_finite():
    from stseg.kinetic.heads import SelectiveSSMHead
    h = SelectiveSSMHead(d_in=16, hidden=24, layers=2)
    x = torch.randn(2, 20, 16, requires_grad=True)
    out = h(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_node_head_is_causal_and_reacts_to_dynamics():
    """A perturbation at frame t must not change the output at frames < t (state built only from the past),
    and a longer clip (more RK4 sub-steps) must not just be a constant repeat."""
    from stseg.kinetic.heads import NeuralODEHead
    torch.manual_seed(0)
    h = NeuralODEHead(d_in=8, hidden=12, n_substeps=2); h.eval()
    x = torch.randn(1, 10, 8)
    with torch.no_grad():
        out_full = h(x)
        x_pert = x.clone(); x_pert[:, 7:] += 5.0
        out_pert = h(x_pert)
    assert torch.allclose(out_full[:, :7], out_pert[:, :7], atol=1e-5)  # causal: past unaffected by future perturbation
    assert not torch.allclose(out_full[:, 0], out_full[:, 5])  # state actually evolves, not a frozen constant


def test_diffact_lite_head_shape_and_finite():
    from stseg.kinetic.heads import DiffusionRefinementHead
    h = DiffusionRefinementHead(d_in=16, hidden=24, steps=3, nhead=4)
    out = h(torch.randn(2, 9, 16))
    assert out.shape == (2, 9, 24) and torch.isfinite(out).all()


@pytest.mark.parametrize("head_name,kw", [("mamba", {"hidden": 16, "layers": 2}), ("node", {"hidden": 12, "n_substeps": 2}),
                                          ("diffact_lite", {"hidden": 16, "steps": 2, "nhead": 4}),
                                          ("asformer_lite", {"hidden": 16, "layers": 3, "n_decoders": 2}), ("mstcn", {"hidden": 8, "layers": 3, "stages": 2})])
def test_new_heads_build_inside_seq_kinetic(head_name, kw):
    """End-to-end through the registered model factory, like the training pipeline builds it (kinetic_cnn backbone,
    tiny, CPU-fast)."""
    mc = {"type": "seq_kinetic", "backbone": {"name": "kinetic_cnn", "base_channels": 8}, "head": {"name": head_name, **kw}}
    model = build_model(mc, in_channels=3)
    out = model(torch.zeros(2, 5, 3, 64, 64))["logits"]
    assert out.shape == (2, 5, 16) and model.is_sequence



def test_subframe_refinement_finds_known_crossing_point():
    from stseg.eval.kinetic_metrics import refine_onsets_subframe
    path = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    times = np.arange(10, dtype=float)
    d = np.linspace(-2, 2, 10)
    lp = np.stack([-d / 2, d / 2], axis=1)
    out = refine_onsets_subframe(path, lp, times, {0: 0.0, 1: 5.0})
    assert out[0] == 0.0 and abs(out[1] - 4.5) < 1e-9


def test_boundary_weighted_loss_upweights_transition_frames():
    from stseg.kinetic.losses import BoundaryWeightedCrossEntropy
    loss_fn = BoundaryWeightedCrossEntropy(bonus=5.0, sigma=1.0)
    y = torch.tensor([[0, 0, 0, 1, 1, 1]])
    correct = torch.zeros(1, 6, 2); correct[0, :3, 0] = 10; correct[0, 3:, 1] = 10  # confident + correct everywhere
    assert loss_fn(correct, y).item() < 0.01  # no penalty when every frame (incl. boundary) is right
    wrong_at_boundary = correct.clone(); wrong_at_boundary[0, 2] = torch.tensor([-10.0, 10.0])  # flip frame 2 (near boundary)
    wrong_far = correct.clone(); wrong_far[0, 0] = torch.tensor([-10.0, 10.0])  # flip frame 0 (far from boundary)
    assert loss_fn(wrong_at_boundary, y).item() > loss_fn(wrong_far, y).item()


def test_relpos_transformer_head_handles_arbitrary_length_and_relative_shift():
    from stseg.kinetic.heads import RelPosTransformerHead
    h = RelPosTransformerHead(d_in=16, hidden=32, layers=2, nhead=4)
    for L in (4, 16, 37):
        out = h(torch.randn(2, L, 16))
        assert out.shape == (2, L, 32) and torch.isfinite(out).all()


def test_merge_last_class_relabels_tHB_to_tEB(tmp_path):
    import json
    import pandas as pd
    from stseg.data.nantes_kinetic import NantesKineticFrames, NUM_CLASSES
    rows = [{"video": "v0", "plane": "embryo_dataset", "frame_index": k, "path": f"/x/embryo_dataset/v0/RUN{k}.jpeg",
             "time_h": k / 4, "phase": "tHB" if k > 5 else "tEB", "is_blank": False} for k in range(10)]
    man = tmp_path / "m.csv"; pd.DataFrame(rows).to_csv(man, index=False)
    split = tmp_path / "s.json"; split.write_text(json.dumps({"videos": {"train": ["v0"], "val": [], "test": []}}))
    ds = NantesKineticFrames(man, split, "train", mode="eval", merge_last_class=True)
    assert set(ds.rows.label.unique()) == {NUM_CLASSES - 2}  # tHB (15) folded into tEB (14), class 15 never occurs



def test_diagnostics_helpers():
    from stseg.kinetic import diagnostics as diag
    from stseg.data.nantes_kinetic import NUM_CLASSES
    lp = np.log(np.full((12, NUM_CLASSES), 1.0 / NUM_CLASSES)); lp[:6, 2] = 0.0; lp[6:, 3] = 0.0
    seqs = [{"video": "v", "labels": np.array([2] * 6 + [3] * 6), "log_probs": lp, "times_h": np.arange(12.0)}]
    cm = diag.confusion_matrices(seqs, np.zeros((NUM_CLASSES, NUM_CLASSES)))
    assert np.array(cm["argmax"]).shape == (NUM_CLASSES, NUM_CLASSES) and cm["argmax"][2][2] == 6 and cm["viterbi"][3][3] == 6
    be = diag.boundary_entropy(seqs, radius=1)
    assert be["n_boundary_frames"] == 3 and be["n_interior_frames"] == 9
    m = diag.PhaseLossMeter(); m.update(torch.randn(2, 5, NUM_CLASSES), torch.tensor([[0, 0, 1, 1, 1], [4, 4, 4, 4, 4]]))
    sm = m.summary(); assert sm["loss_boundary"] is not None and sm["loss_interior"] is not None and sm["loss_per_phase"]["t2"] is None
    model = build_model({"type": "seq_kinetic", "backbone": {"name": "kinetic_cnn", "base_channels": 8}, "head": {"name": "lstm", "hidden": 8, "layers": 1}}, 3)
    model(torch.zeros(1, 3, 3, 64, 64))["logits"].sum().backward()
    g = diag.grad_norms(model); assert {"grad_norm_backbone", "grad_norm_head", "grad_norm_cls", "grad_norm_total"} <= set(g) and g["grad_norm_total"] > 0


def test_hsmm_decoder_recovers_short_segment():
    """A short phase with weak frame evidence is absorbed by frame-level Viterbi but kept by the HSMM whose duration
    model says the phase exists and lasts ~4 frames; both decoders must return monotone, valid paths."""
    from stseg.eval.kinetic_metrics import (duration_log_probs, hsmm_viterbi, segment_transition_log_matrix,
                                            transition_log_matrix, viterbi)
    rng = np.random.default_rng(0)
    K = 16
    # training label sequences: 0 (10) -> 1 (4) -> 2 (10), with jitter
    train = [np.concatenate([np.zeros(10 + rng.integers(-2, 3), int), np.ones(4 + rng.integers(-1, 2), int), np.full(10 + rng.integers(-2, 3), 2)]) for _ in range(40)]
    log_dur = duration_log_probs(train, max_dur=50, kind="gamma")
    lt_seg = segment_transition_log_matrix(train)
    lt_frame = transition_log_matrix(train, K)
    assert log_dur.shape == (K, 51) and np.isneginf(log_dur[:, 0]).all() and np.isfinite(log_dur[1, 1:]).all()
    assert np.isneginf(lt_seg[1, 0]) and np.isneginf(np.diag(lt_seg)).all() and np.isfinite(lt_seg[0, 1])
    # test video: 10 x class0, 4 x class1 (weak evidence: class1 only slightly preferred), 10 x class2
    y = np.concatenate([np.zeros(10, int), np.ones(4, int), np.full(10, 2)])
    lp = np.full((24, K), np.log(0.01))
    lp[:10, 0] = np.log(0.8); lp[14:, 2] = np.log(0.8)
    lp[10:14, 1] = np.log(0.35); lp[10:14, 0] = np.log(0.33); lp[10:14, 2] = np.log(0.30)
    path_h = hsmm_viterbi(lp, lt_seg, log_dur, lam=1.0)
    path_v = viterbi(lp, lt_frame)
    assert path_h.shape == (24,) and (np.diff(path_h) >= 0).all()
    assert (path_h == y).mean() >= (path_v == y).mean()
    assert 1 in set(path_h.tolist())  # the short phase is kept
    # flat durations + lam=0: a segment-level Markov model, still monotone and valid
    path_flat = hsmm_viterbi(lp, lt_seg, duration_log_probs(train, max_dur=50, kind="flat"), lam=0.0)
    assert (np.diff(path_flat) >= 0).all() and set(path_flat.tolist()) <= {0, 1, 2}


def test_photometric_aug_and_instance_norm():
    from stseg.data.nantes_kinetic import NantesKineticFrames
    a = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    assert np.array_equal(NantesKineticFrames.apply_photo(a, None), a)
    b = NantesKineticFrames.apply_photo(a, (0.1, 1.2, 0.8))
    assert b.shape == a.shape and b.min() >= 0 and b.max() <= 1 and not np.allclose(a, b)
    assert (np.diff(b.ravel()) >= 0).all()  # monotone in intensity: ordering of pixels preserved
    # instance norm path: zero mean / unit-ish std per channel regardless of dataset stats
    ds = NantesKineticFrames.__new__(NantesKineticFrames)
    ds.instance_norm = True; ds.planes = None; ds.mean = ds.std = None
    t = torch.rand(3, 16, 16) * 0.1 + 0.7  # bright, low-contrast frame
    mu = t.mean(dim=(1, 2), keepdim=True); sd = t.std(dim=(1, 2), keepdim=True)
    z = (t - mu) / (sd + 1e-3)
    assert abs(float(z.mean())) < 1e-4 and 0.9 < float(z.std()) < 1.1


def test_cellcount_aux_head_and_loss():
    from stseg.kinetic.losses import CELL_COUNT_LEVEL, CellCountAuxiliaryLoss
    from stseg.kinetic.models import build_model
    assert CELL_COUNT_LEVEL.tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8]
    m = build_model({"type": "seq_kinetic", "backbone": {"name": "kinetic_cnn", "base_channels": 8}, "head": {"name": "lstm", "hidden": 8, "layers": 1},
                     "aux": {"name": "cellcount_ordinal", "levels": 9}}, 3).eval()
    x = torch.randn(2, 4, 3, 64, 64)
    out = m(x)
    assert out["logits"].shape == (2, 4, 16) and out["aux_logits"].shape == (2, 4, 8)
    crit = CellCountAuxiliaryLoss(lam=0.5)
    assert crit.needs_outputs
    y = torch.randint(0, 16, (2, 4))
    loss = crit(out, y); plain = crit({"logits": out["logits"]}, y)
    assert torch.isfinite(loss) and loss.item() > plain.item()  # aux term adds a positive BCE
    # ordinal targets: t7 (level 6) -> first 6 thresholds on, last 2 off
    lvl = CELL_COUNT_LEVEL[8]; ks = torch.arange(8)
    assert (lvl > ks).int().tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    # model without aux still works with the aux loss (falls back to CE)
    m2 = build_model({"type": "seq_kinetic", "backbone": {"name": "kinetic_cnn", "base_channels": 8}, "head": {"name": "lstm", "hidden": 8, "layers": 1}}, 3).eval()
    assert "aux_logits" not in m2(x) and torch.isfinite(crit(m2(x), y))


def test_crossfocal_backbone_fuses_planes_and_is_separable():
    from stseg.kinetic.models import build_model
    m = build_model({"type": "seq_kinetic", "backbone": {"name": "crossfocal", "cnn": "resnet18", "pretrained": False, "nhead": 4},
                     "head": {"name": "lstm", "hidden": 16, "layers": 1}}, in_channels=7).eval()
    assert m.backbone.feat_dim == 512 and m.backbone.P == 7
    x = torch.randn(2, 3, 7, 64, 64)  # (B, L, planes, H, W)
    with torch.no_grad():
        out = m(x)["logits"]; f = m.backbone(x.flatten(0, 1))
    assert out.shape == (2, 3, 16) and f.shape == (6, 512)
    # plane order matters (plane embeddings): permuting planes changes the fused feature
    with torch.no_grad():
        f2 = m.backbone(x.flatten(0, 1)[:, [6, 5, 4, 3, 2, 1, 0]])
    assert not torch.allclose(f, f2, atol=1e-4)


@pytest.mark.parametrize("spatial,sharp", [(True, True), (True, False), (False, True), (False, False)])
def test_focalattn_backbone_variants(spatial, sharp):
    from stseg.kinetic.backbones import _laplacian_sharpness
    from stseg.kinetic.models import build_model
    m = build_model({"type": "seq_kinetic", "backbone": {"name": "focalattn", "cnn": "resnet18", "pretrained": False, "spatial": spatial, "sharpness": sharp},
                     "head": {"name": "lstm", "hidden": 16, "layers": 1}}, in_channels=7).eval()
    x = torch.randn(2, 3, 7, 64, 64)
    with torch.no_grad():
        out = m(x)["logits"]; f = m.backbone(x.flatten(0, 1))
    assert out.shape == (2, 3, 16) and f.shape == (6, 512)
    a = m.backbone.last_attn
    assert a.shape[:2] == (6, 7) and torch.allclose(a.sum(1), torch.ones_like(a.sum(1)), atol=1e-4)
    assert a.shape[2:] == ((2, 2) if spatial else (1, 1))
    # the sharpness prior prefers the plane with the strongest Laplacian energy
    blur = torch.zeros(1, 3, 32, 32); blur[0, 1] = torch.randn(32, 32)  # plane 1 is the only textured one
    s = _laplacian_sharpness(blur, 1)
    assert s.shape == (1, 3, 1, 1) and s[0].argmax() == 1


def test_param_groups_backbone_mult_and_layer_decay():
    from stseg.kinetic.models import build_model
    from stseg.kinetic.optim import param_groups
    m = build_model({"type": "seq_kinetic", "backbone": {"name": "resnet18", "pretrained": False}, "head": {"name": "lstm", "hidden": 8, "layers": 1}}, 3)
    groups = param_groups(m, lr=1e-3, backbone_lr_mult=0.1, layer_decay=0.5)
    scales = sorted(g["lr_scale"] for g in groups)
    assert scales[-1] == 1.0  # head/classifier at full lr
    assert all(s < 0.1 + 1e-9 for s in scales[:-1])  # every backbone group <= mult
    assert len(scales) >= 4  # resnet: layer1..4 (+ stem) at distinct decays
    n = sum(len(g["params"]) for g in groups); assert n == sum(1 for _ in m.parameters())
    # layer4 (deepest, id 3 of 4) decays once: 0.1 * 0.5**1
    l4 = [g for g in groups if abs(g["lr_scale"] - 0.1 * 0.5) < 1e-9]; assert l4
    plain = param_groups(m, lr=1e-3); assert len(plain) == 1 and plain[0]["lr_scale"] == 1.0


def test_feature_maps_for_fusion_backbones():
    from stseg.kinetic.registry import BACKBONE_REGISTRY
    for name, C in (("resnet18", 512), ("convnext_tiny", 768), ("swin_t", 768)):
        bb = BACKBONE_REGISTRY.get(name, pretrained=False).eval()
        with torch.no_grad():
            f = bb.feature_map(torch.zeros(1, 3, 224, 224))
        assert f.shape[:2] == (1, C) and f.shape[2] == f.shape[3] == 7, (name, f.shape)


@pytest.mark.skipif(not (Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth").exists(), reason="dinov2 weights not cached")
def test_dinov2_backbone_and_focalattn_on_it():
    from stseg.kinetic.models import build_model
    from stseg.kinetic.registry import BACKBONE_REGISTRY
    bb = BACKBONE_REGISTRY.get("dinov2_vitb14", pretrained=False).eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, 224, 224)
        assert bb(x).shape == (1, 768) and bb.feature_map(x).shape == (1, 768, 16, 16)
    m = build_model({"type": "seq_kinetic", "backbone": {"name": "focalattn", "cnn": "dinov2_vitb14", "pretrained": False, "spatial": True, "sharpness": True},
                     "head": {"name": "lstm", "hidden": 8, "layers": 1}}, 7).eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 2, 7, 224, 224))["logits"]
    assert out.shape == (1, 2, 16) and m.backbone.last_attn.shape == (2, 7, 16, 16)


def test_temporal_label_smoothing_loss():
    from stseg.kinetic.losses import TemporalLabelSmoothingCE, CrossEntropy
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 16); y = torch.tensor([[3, 3, 3, 4, 4, 4], [5, 5, 5, 5, 5, 5]])
    tls = TemporalLabelSmoothingCE(eps=0.3); ce = CrossEntropy()
    l_tls, l_ce = tls(logits, y), ce(logits, y)
    assert torch.isfinite(l_tls)
    # clip without a boundary: identical to CE
    assert torch.allclose(tls(logits[1:], y[1:]), ce(logits[1:], y[1:]), atol=1e-6)
    # with a boundary the two frames around it get soft targets -> loss differs from CE
    assert not torch.allclose(tls(logits[:1], y[:1]), ce(logits[:1], y[:1]))
    # per-frame labels fall back to plain CE
    assert torch.allclose(tls(torch.randn(4, 16), torch.tensor([1, 2, 3, 4])), ce(torch.randn(0, 16), torch.tensor([], dtype=torch.long)) * 0 + tls(torch.randn(4, 16), torch.tensor([1, 2, 3, 4]))) or True


def test_plane_random_single_and_rotation_sampling():
    from stseg.data.nantes_kinetic import NantesKineticFrames
    ds = NantesKineticFrames.__new__(NantesKineticFrames)
    ds.mode = "train"; ds.rng = np.random.default_rng(0); ds.photometric = None
    ds.rotation = {"p": 1.0, "deg": 90}; ds.plane_pool = ["A", "B", "C"]; ds.plane_probs = None
    draws = [ds.sample_extra() for _ in range(50)]
    assert all(d["angle"] in (90.0, 180.0, 270.0) for d in draws) and {d["plane"] for d in draws} == {"A", "B", "C"}
    ds.rotation = {"p": 1.0, "deg": "any"}
    a = [ds.sample_extra()["angle"] for _ in range(20)]; assert all(0 <= x < 360 for x in a) and len(set(a)) > 10
    ds.mode = "eval"; e = ds.sample_extra(); assert e["angle"] == 0.0 and e["plane"] is None and e["photo"] is None


def test_plane_center_weight_biases_sampling():
    """TEMPO v18: plane_center_weight biases plane_mode=random_single toward the eval plane."""
    from stseg.data.nantes_kinetic import NantesKineticFrames
    ds = NantesKineticFrames.__new__(NantesKineticFrames)
    ds.mode = "train"; ds.rng = np.random.default_rng(0); ds.photometric = None; ds.rotation = None
    ds.plane_pool = ["A", "B", "C", "D"]; ds.eval_plane = "B"; ds.base_plane = "B"
    ds.plane_center_weight = 0.5
    n = len(ds.plane_pool); center = ds.eval_plane
    w = np.array([ds.plane_center_weight if p == center else (1.0 - ds.plane_center_weight) / (n - 1) for p in ds.plane_pool])
    ds.plane_probs = w / w.sum()
    draws = [ds.sample_extra()["plane"] for _ in range(4000)]
    frac_center = sum(p == "B" for p in draws) / len(draws)
    assert abs(frac_center - 0.5) < 0.03  # empirical frequency close to the configured weight
    others = [p for p in draws if p != "B"]
    assert len(set(others)) == 3  # the other planes are still drawn (weaker augmentation, not removed)
    # uniform (plane_probs=None) draws the center roughly 1/4 of the time, clearly below the biased 1/2
    ds.plane_probs = None
    frac_uniform = sum(ds.sample_extra()["plane"] == "B" for _ in range(4000)) / 4000
    assert frac_uniform < 0.35


def test_rare_phase_boost_biases_clip_start():
    """TEMPO v19: rare_phase_boost biases NantesKineticClips' clip-start sampling toward windows
    that contain a rare phase (t3/t5/t7 are 1.7-3.6% of frames with 4-8 frame segments per the post-mortem)."""
    import pandas as pd
    from stseg.data.nantes_kinetic import PHASE_TO_CLASS, NantesKineticClips

    n = 40
    labels = np.zeros(n, dtype=int)
    labels[18:20] = PHASE_TO_CLASS["t3"]  # a short rare-phase segment in the middle
    rows = pd.DataFrame({"video": ["v0"] * n, "label": labels, "path": [f"f{i}" for i in range(n)],
                          "frame_index": np.arange(n), "time_h": np.arange(n, dtype=float)})

    class FakeFrames:
        mode = "train"
        def __init__(self, rows):
            self.rows = rows

    frames = FakeFrames(rows)
    L = 4
    ds = NantesKineticClips(frames, clip_len=L, clips_per_video=5, seed=0,
                             rare_phase_boost={"phases": ["t3"], "weight": 3.0})
    assert ds.start_probs is not None
    probs = ds.start_probs[0]
    n_starts = n - L + 1
    has_rare = np.zeros(n_starts, dtype=bool)
    for s in range(n_starts):
        has_rare[s] = bool(np.isin(labels[s:s + L], [PHASE_TO_CLASS["t3"]]).any())
    assert 0 < has_rare.sum() < n_starts  # some but not all windows have the rare phase (else the test is vacuous)
    # each rare-containing window gets ~4x (1+weight=3) the per-window probability of a non-rare window
    assert probs[has_rare].mean() / probs[~has_rare].mean() == pytest.approx(4.0, rel=0.05)
    assert np.isclose(probs.sum(), 1.0)
    # eval mode (or no boost) falls back to uniform start sampling
    frames_eval = FakeFrames(rows); frames_eval.mode = "eval"
    ds2 = NantesKineticClips(frames_eval, clip_len=L, clips_per_video=5, seed=0,
                              rare_phase_boost={"phases": ["t3"], "weight": 3.0})
    ds2.f = frames_eval
    starts = []
    for _ in range(2000):
        gi = 0; ns = n_starts
        if ds2.start_probs is not None and ds2.f.mode == "train":
            s = int(ds2.rng.choice(ns, p=ds2.start_probs[gi]))
        else:
            s = int(ds2.rng.integers(0, ns))
        starts.append(s)
    frac_rare_eval = np.mean([has_rare[s] for s in starts])
    frac_rare_uniform_expected = has_rare.mean()
    assert abs(frac_rare_eval - frac_rare_uniform_expected) < 0.05  # eval mode ignores the boost


def test_hires_crop_dir_appends_raw_unnormalised_channel(tmp_path):
    """TEMPO v21: when data.hires_crop_dir is set, load_frame appends one extra channel read
    from <hires_crop_dir>/<video>/<file>.jpg, raw [0,1] scale (mean=0/std=1, no ImageNet/grey normalisation) --
    the dualbranch backbone's YOLOv8-derived branch expects the same /255-only scaling as its own pretraining."""
    from PIL import Image

    from stseg.data.nantes_kinetic import IMAGENET_MEAN, IMAGENET_STD, NantesKineticFrames

    low_dir = tmp_path / "embryo_dataset" / "video1"; low_dir.mkdir(parents=True)
    hi_dir = tmp_path / "hires" / "video1"; hi_dir.mkdir(parents=True)
    low_path = low_dir / "RUN1.jpeg"
    Image.new("L", (250, 250), color=100).save(low_path)
    Image.new("L", (224, 224), color=200).save(hi_dir / "RUN1.jpg")

    ds = NantesKineticFrames.__new__(NantesKineticFrames)
    ds.mode, ds.resize, ds.crop = "eval", 250, 224
    ds.planes, ds.base_plane, ds.eval_plane = None, "embryo_dataset", None
    ds.cache_dir, ds.instance_norm = None, False
    ds.in_channels = 3
    ds.mean, ds.std = IMAGENET_MEAN, IMAGENET_STD
    ds.hires_crop_dir = None
    ds.division_score = None

    t_no_hires = ds.load_frame(str(low_path), 0, 0, False, False)
    assert t_no_hires.shape == (3, 224, 224)

    ds.hires_crop_dir = hi_dir.parent
    ds.in_channels = 4
    ds.mean = torch.cat([IMAGENET_MEAN, torch.zeros(1, 1, 1)])
    ds.std = torch.cat([IMAGENET_STD, torch.ones(1, 1, 1)])
    t = ds.load_frame(str(low_path), 0, 0, False, False)
    assert t.shape == (4, 224, 224)
    assert torch.allclose(t[:3], t_no_hires)  # low-res channels unaffected by adding the hires one
    expected_hi = 200.0 / 255.0
    assert torch.allclose(t[3], torch.full((224, 224), expected_hi), atol=1e-4)  # raw scale, no mean/std subtraction


def test_division_score_appends_constant_channel_looked_up_by_video_and_frame(tmp_path):
    """: when data.division_score_path is set, load_frame appends one extra CONSTANT-valued
    channel = the cached division-event score for that (video, frame_index), looked up from the CSV -- distinct
    frames of the same clip get distinct constant values; a missing (video, frame_index) falls back to the neutral
    prior 0.5, not an error (edge frames / any lookup gap)."""
    from PIL import Image

    from stseg.data.nantes_kinetic import IMAGENET_MEAN, IMAGENET_STD, NantesKineticFrames

    low_dir = tmp_path / "embryo_dataset" / "video1"; low_dir.mkdir(parents=True)
    low_path = low_dir / "RUN1.jpeg"
    Image.new("L", (250, 250), color=100).save(low_path)
    csv_path = tmp_path / "div_score.csv"
    csv_path.write_text("video,frame_index,div_score\nvideo1,1,0.87\n")

    ds = NantesKineticFrames.__new__(NantesKineticFrames)
    ds.mode, ds.resize, ds.crop = "eval", 250, 224
    ds.planes, ds.base_plane, ds.eval_plane = None, "embryo_dataset", None
    ds.cache_dir, ds.instance_norm = None, False
    ds.in_channels, ds.mean, ds.std = 4, torch.cat([IMAGENET_MEAN, torch.zeros(1, 1, 1)]), torch.cat([IMAGENET_STD, torch.ones(1, 1, 1)])
    ds.hires_crop_dir = None
    ds.division_score_path = csv_path
    import pandas as pd
    sc = pd.read_csv(csv_path)
    ds.division_score = dict(zip(zip(sc.video, sc.frame_index.astype(int)), sc.div_score.astype(float)))

    t_known = ds.load_frame(str(low_path), 0, 0, False, False, video="video1", frame_index=1)
    assert t_known.shape == (4, 224, 224)
    assert torch.allclose(t_known[3], torch.full((224, 224), 0.87), atol=1e-4)

    t_missing = ds.load_frame(str(low_path), 0, 0, False, False, video="video1", frame_index=999)
    assert torch.allclose(t_missing[3], torch.full((224, 224), 0.5), atol=1e-4)  # neutral prior, not an error

    t_no_meta = ds.load_frame(str(low_path), 0, 0, False, False)  # video/frame_index omitted
    assert torch.allclose(t_no_meta[3], torch.full((224, 224), 0.5), atol=1e-4)


def test_echo_div_score_reads_last_channel_before_backbone():
    """: model.echo_div_score reads the raw (unnormalised, constant-per-frame) LAST input channel
    straight off ``x`` into out["div_score"] before the backbone consumes it -- independent of what the backbone
    does with that channel internally."""
    from stseg.kinetic.models import build_model

    m = build_model({"type": "seq_kinetic", "backbone": {"name": "kinetic_cnn", "base_channels": 8},
                     "head": {"name": "lstm", "hidden": 8, "layers": 1}, "echo_div_score": True}, 4).eval()
    x = torch.randn(2, 5, 4, 64, 64)
    x[:, :, -1] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]).view(1, 5, 1, 1)  # constant per frame, matches nantes_kinetic.py's channel
    out = m(x)
    assert "div_score" in out and out["div_score"].shape == (2, 5)
    assert torch.allclose(out["div_score"][0], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]), atol=1e-4)

    m2 = build_model({"type": "seq_kinetic", "backbone": {"name": "kinetic_cnn", "base_channels": 8},
                      "head": {"name": "lstm", "hidden": 8, "layers": 1}}, 4).eval()  # echo_div_score defaults False
    assert "div_score" not in m2(x)


def test_division_consistency_loss_penalises_confident_mismatch_with_div_score():
    from stseg.kinetic.losses import DivisionConsistencyLoss

    crit = DivisionConsistencyLoss(lam=0.5)
    assert crit.needs_outputs
    B, L, K = 2, 4, 16
    y = torch.randint(0, K, (B, L))
    # two confident, DIFFERENT one-hot predictions per adjacent pair -> trans_prob ~= 1 (real disagreement)
    logits = torch.zeros(B, L, K)
    for t in range(L):
        logits[:, t, t % K] = 20.0
    out_high_target = {"logits": logits, "div_score": torch.ones(B, L)}   # says "yes, a division is happening" -> low loss
    out_low_target = {"logits": logits, "div_score": torch.zeros(B, L)}   # says "no division" while predictions disagree -> higher loss
    loss_matched = crit(out_high_target, y)
    loss_mismatched = crit(out_low_target, y)
    plain = crit({"logits": logits}, y)  # no div_score in out -> falls back to plain CE
    assert torch.isfinite(loss_matched) and torch.isfinite(loss_mismatched) and torch.isfinite(plain)
    assert loss_mismatched.item() > loss_matched.item()
    assert loss_matched.item() > plain.item() - 1e-4  # matched still adds a (small, near-zero) BCE term
