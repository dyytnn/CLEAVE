"""Locks for the CLEAVE patient-grouping audit (paper Section 'defects', defect 3).

Every leakage number quoted in the paper is produced by scripts/audit_protocol_splits.py from
stseg.data.patient; these tests pin the parsing rule, the counts and the construction of the paired
sibling-leak splits so that a silent change to any of them fails CI instead of the manuscript.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stseg.data.patient import (  # noqa: E402
    exposed_test_videos, group_videos, leaking_patients, patient_of,
)


def released_names() -> list[str]:
    return [r[0] for r in csv.reader(open(ROOT / "data/splits/nantes_official/split0.csv")) if r]


def partition_of(fold: int) -> dict[str, str]:
    return {v: p for v, p in csv.reader(open(ROOT / f"data/splits/nantes_official/split{fold}.csv")) if v}


def json_partition(name: str) -> dict[str, str]:
    doc = json.loads((ROOT / f"data/splits/{name}.json").read_text())
    return {v: part for part, vids in doc["videos"].items() for v in vids}


# --------------------------------------------------------------------------- parsing rule
@pytest.mark.parametrize("video,couple,cycle", [
    ("AA83-7", "AA83", "AA83"),                      # the common form
    ("PMDPI029-1-10", "PMDPI029", "PMDPI029-1"),     # second treatment cycle of one couple
    ("GF667-2-6", "GF667", "GF667-2"),               # the pair that v1's cycle grouping separates
    ("GS490-_6", "GS490", "GS490"),                  # stray underscore
    ("MRA165-7T", "MRA165", "MRA165"),               # letter-suffixed embryo number
    ("RV146N2-6", "RV146N2", "RV146N2"),             # letter+digit inside the code
])
def test_patient_of_handles_every_irregular_name(video, couple, cycle):
    assert patient_of(video, "couple") == couple
    assert patient_of(video, "cycle") == cycle


def test_every_released_name_parses():
    for v in released_names():
        assert patient_of(v)  # raises on an unparseable name


def test_couple_level_matches_the_simple_rule_on_the_released_names():
    """The regex is defensive; on the released archive it is exactly `name.split('-')[0]`."""
    for v in released_names():
        assert patient_of(v, "couple") == v.split("-")[0]


def test_unparseable_name_raises_rather_than_disabling_grouping():
    with pytest.raises(ValueError):
        patient_of("no_embryo_number")
    with pytest.raises(ValueError):
        patient_of("AA83-7", level="video")


def test_cycle_level_is_strictly_finer_than_couple_level():
    names = released_names()
    assert len(set(map(patient_of, names))) < len({patient_of(v, "cycle") for v in names})


# --------------------------------------------------------------------------- released folds
def test_released_folds_leak_at_both_grouping_levels():
    counts = {lv: [len(leaking_patients(partition_of(k), lv)) for k in range(5)] for lv in ("couple", "cycle")}
    assert counts["couple"] == [39, 42, 50, 35, 37]
    assert counts["cycle"] == [38, 39, 46, 34, 36]   # the number the first draft quoted (38 of 549)


def test_exposed_test_videos_is_the_quantity_that_contaminates_a_score():
    exposed = [len(exposed_test_videos(partition_of(k))) for k in range(5)]
    assert exposed == [20, 21, 30, 17, 22]
    assert sum(exposed) / 350 == pytest.approx(0.3143, abs=5e-4)
    # ... and it is several times larger than the leaking-patient fraction the first draft quoted
    assert sum(exposed) / 350 > 4 * (39 / len({patient_of(v) for v in released_names()}))


# --------------------------------------------------------------------------- this repository's splits
def test_grouped_v1_residual_couple_level_leak_is_disclosed():
    """v1 groups by cycle, so four couples straddle a partition; the paper must keep saying so."""
    leaks = leaking_patients(json_partition("nantes_grouped_v1"), "couple")
    assert set(leaks) == {"BS648", "GF1042", "LGA881", "LC161"}
    assert leaks["LGA881"] == ["test", "train"]                   # the only train/test one
    assert len(exposed_test_videos(json_partition("nantes_grouped_v1"))) == 1


def test_grouped_v2_has_no_leak_at_either_level():
    part = json_partition("nantes_grouped_v2")
    for level in ("couple", "cycle"):
        assert leaking_patients(part, level) == {}
    assert exposed_test_videos(part) == []
    assert len(part) == 652


# --------------------------------------------------------------------------- paired sibling-leak design
@pytest.fixture(scope="module")
def paired():
    return {a: json.loads((ROOT / f"data/splits/nantes_siblingleak_{a}_v1.json").read_text())
            for a in ("clean", "leaky")}


def test_arms_share_test_and_val_and_train_size(paired):
    a, b = paired["clean"]["videos"], paired["leaky"]["videos"]
    assert a["test"] == b["test"] and a["val"] == b["val"]
    assert len(a["train"]) == len(b["train"]) == 348
    assert len(set(a["train"]) ^ set(b["train"])) == 142   # 71 swapped out, 71 swapped in


def test_only_the_leaky_arm_leaks(paired):
    assert exposed_test_videos({v: p for p, vs in paired["clean"]["videos"].items() for v in vs}) == []
    leaky = {v: p for p, vs in paired["leaky"]["videos"].items() for v in vs}
    exposed = set(exposed_test_videos(leaky))
    g = paired["leaky"]["groups"]
    assert exposed == set(g["L_test_videos"])
    assert exposed.isdisjoint(g["C_test_videos"])


def test_the_two_test_groups_are_matched_in_size_and_sibling_count(paired):
    g = paired["leaky"]["groups"]
    by = group_videos(sorted({v for d in paired.values() for vs in d["videos"].values() for v in vs}
                             | set(g["siblings_in_leaky_train"])), "couple")
    assert len(g["L_test_videos"]) == len(g["C_test_videos"]) == 41
    n_l = sum(len(by[c]) - 1 for c in g["L_leaked_couples"])
    assert n_l == len(g["siblings_in_leaky_train"]) == 71


def test_validation_set_never_leaks_into_test_in_either_arm(paired):
    for arm, doc in paired.items():
        val = {patient_of(v) for v in doc["videos"]["val"]}
        test = {patient_of(v) for v in doc["videos"]["test"]}
        assert val.isdisjoint(test), arm


def test_control_couples_siblings_are_used_by_neither_arm(paired):
    g = paired["leaky"]["groups"]
    used = {v for d in paired.values() for vs in d["videos"].values() for v in vs}
    c_test = set(g["C_test_videos"])
    for v in used:
        if patient_of(v) in set(g["C_control_couples"]):
            assert v in c_test


# --------------------------------------------------------------------------- DiD estimator
def test_did_estimator_recovers_a_planted_effect(tmp_path, monkeypatch):
    import summarize_siblingleak as S

    g = json.loads((ROOT / "data/splits/nantes_siblingleak_leaky_v1.json").read_text())["groups"]
    monkeypatch.setattr(S, "RUNS", tmp_path)
    for seed in (0, 1, 2):
        for arm in ("clean", "leaky"):
            rows = []
            for v in g["L_test_videos"] + g["C_test_videos"]:
                far = 4                                   # p_t = 1 - 4/10 = 0.6 everywhere
                if arm == "leaky":
                    far -= 1                              # +0.10 swap effect on every test video
                    if v in set(g["L_test_videos"]):
                        far -= 2                          # +0.20 extra where siblings are in train
                rows.append({"video": v, "n_transitions": 10, "n_far": far})
            d = tmp_path / f"siblingleak_{arm}_resnet18_seed{seed}"
            d.mkdir()
            (d / "per_video_test.json").write_text(json.dumps(rows))
    deltas, used = S.collect("resnet18", [0, 1, 2])
    assert used == [0, 1, 2]
    mean = {v: sum(x) / len(x) for v, x in deltas.items()}
    dl = sum(mean[v] for v in g["L_test_videos"]) / 41
    dc = sum(mean[v] for v in g["C_test_videos"]) / 41
    assert dc == pytest.approx(0.10)
    assert dl == pytest.approx(0.30)
    assert dl - dc == pytest.approx(0.20)                  # the planted leakage effect


def test_per_video_pt_reproduces_the_reported_aggregate():
    """p_t is the unweighted mean over videos of 1 - n_far/n_transitions; the DiD rests on that."""
    import summarize_siblingleak as S

    run = ROOT / "runs/h7/resnet18_none_imagesplit_seed0"
    if not (run / "per_video_test.json").exists():
        pytest.skip("reference run not present")
    per = S.per_video_pt(run)
    reported = json.loads((run / "results.json").read_text())
    reported = reported.get("test", reported)["p_t"]
    assert sum(per.values()) / len(per) == pytest.approx(reported, abs=1e-6)
