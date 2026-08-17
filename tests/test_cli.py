"""Every script must answer --help without the project env installed.

This is what the Dockerfile's default CMD does, and it is the first thing
anyone runs to find out what a script needs. It only holds while the heavy
imports (torch, ultralytics, cv2) stay inside the functions that use them --
a module-scope import anywhere in src/ turns --help into a ModuleNotFoundError
traceback on any machine that has not installed the requirements yet.

Deliberately a subprocess: importing the modules in-process would let an
already-imported torch mask exactly the regression this is guarding.

The subprocess additionally runs with those packages made UNIMPORTABLE. Without
that, the test could not fail where it matters: in the env the README tells you
to create, torch and ultralytics are installed, so `--help` succeeds whether the
imports are deferred or not, and the guard silently protected nothing. Blocking
them makes it hold in any environment, including the author's own.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = sorted(SRC_DIR.glob("*.py"))
BLOCKED = ("torch", "torchvision", "ultralytics", "cv2")

_SITECUSTOMIZE = textwrap.dedent(f"""
    import sys
    BLOCKED = {BLOCKED!r}
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(
                    f"{{name}} is deliberately unavailable: --help must not need it")
            return None
    sys.meta_path.insert(0, _Blocker())
""")


@pytest.fixture(scope="module")
def bare_env(tmp_path_factory):
    """An environment whose sitecustomize makes the heavy deps unimportable."""
    d = tmp_path_factory.mktemp("bare")
    (d / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    return {"PYTHONPATH": str(d), "PATH": "/usr/bin:/bin", "HOME": str(d)}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_help_exits_zero_without_the_heavy_deps(script, bare_env):
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        env=bare_env,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{script.name} --help exited {proc.returncode} with torch/ultralytics/cv2 "
        f"blocked -- a module-scope import has crept back in:\n{proc.stderr[-800:]}"
    )
    assert "usage:" in proc.stdout.lower()


def test_the_blocker_actually_blocks(bare_env):
    """Guards the guard: if sitecustomize stopped being loaded, every test above
    would start passing for the wrong reason and nothing would say so."""
    proc = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        timeout=60,
        env=bare_env,
        check=False,
    )
    assert proc.returncode != 0
    assert "deliberately unavailable" in proc.stderr
