import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(autouse=True)
def _run_log_in_tmp(tmp_path_factory, monkeypatch):
    """A test that calls a script's main() must not append to the real runs/aerial.log."""
    monkeypatch.setenv(
        "AERIAL_LOG_FILE", str(tmp_path_factory.getbasetemp() / "aerial.log")
    )
