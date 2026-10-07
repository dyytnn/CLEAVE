"""Tests that read real Nantes frames are skipped until the dataset is in data/raw/ (see README)."""
from pathlib import Path

import pytest

NEEDS_DATA = {
    "test_emfit_dataset_shape_clock_and_patient_partition",
    "test_real_train_only_two_epoch_smoke",
    "test_resume_from_last_pt",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test")


def pytest_collection_modifyitems(config, items):
    if Path("data/raw/nantes_embryo_dataset").exists():
        return
    skip = pytest.mark.skip(reason="needs the Nantes dataset in data/raw/nantes_embryo_dataset")
    for item in items:
        if item.originalname in NEEDS_DATA:
            item.add_marker(skip)
