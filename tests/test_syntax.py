"""Parse every script under src/ -- catches syntax errors without needing
torch/ultralytics/opencv installed, so it runs in any environment."""

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = sorted(SRC_DIR.glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_parses(script):
    ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
