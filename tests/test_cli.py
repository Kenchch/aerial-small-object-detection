"""Every script must answer --help without the project env installed.

This is what the Dockerfile's default CMD does, and it is the first thing
anyone runs to find out what a script needs. It only holds while the heavy
imports (torch, ultralytics, cv2) stay inside the functions that use them --
a module-scope import anywhere in src/ turns --help into a ModuleNotFoundError
traceback on any machine that has not installed the requirements yet.

Deliberately a subprocess: importing the modules in-process would let an
already-imported torch mask exactly the regression this is guarding.
"""
import subprocess
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = sorted(SRC_DIR.glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_help_exits_zero(script):
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"{script.name} --help exited {proc.returncode}:\n{proc.stderr[-800:]}"
    )
    assert "usage:" in proc.stdout.lower()
