"""scripts/summarize_tracking.py: reproducible from the runs, and strict about
what counts as a repeat. No test executed it before."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPEATS = ROOT / "reports/tracking_repeats"

spec = importlib.util.spec_from_file_location(
    "summarize_tracking", ROOT / "scripts/summarize_tracking.py"
)
summarize_tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_tracking)


@pytest.fixture
def repeats(tmp_path):
    target = tmp_path / "tracking_repeats"
    shutil.copytree(REPEATS, target)
    (target / "summary.json").unlink()
    return target


def _edit(path: Path, change):
    run = json.loads(path.read_text(encoding="utf-8"))
    change(run)
    path.write_text(json.dumps(run), encoding="utf-8")


def test_the_committed_summary_is_what_the_committed_runs_give(repeats):
    summarize_tracking.main(["--repeats", str(repeats)])
    assert json.loads((repeats / "summary.json").read_text(encoding="utf-8")) == (
        json.loads((REPEATS / "summary.json").read_text(encoding="utf-8"))
    )
    assert not (repeats / "summary.json.tmp").exists()


@pytest.mark.parametrize(
    ("change", "what"),
    [
        (lambda r: r.update(frames=30), "frames"),
        (lambda r: r["source"].update(sha256="0" * 64), "source clip"),
        (lambda r: r["config"].update(imgsz=640), "config"),
        (lambda r: r["environment"].update(device_resolved="cpu"), "environment"),
    ],
)
def test_a_repeat_of_something_else_is_refused(repeats, change, what):
    """A sixth run at another size or on another clip was averaged in, and the
    protocol still said "Same 90-frame ... source"."""
    _edit(repeats / "run_5.json", change)
    with pytest.raises(SystemExit, match=f"repeats differ in {what}.*run_5.json"):
        summarize_tracking.summarize(repeats)


def test_a_different_output_path_is_still_the_same_protocol(repeats):
    _edit(repeats / "run_5.json", lambda r: r["config"].update(out="elsewhere.mp4"))
    assert summarize_tracking.summarize(repeats)["runs"] == 5


def test_a_run_without_every_stage_is_refused(repeats):
    """A --no-write run has no encode stage; sum() over it raised TypeError."""

    def no_write(run):
        run["stage_ms_median"]["encode"] = None
        run["output"] = None

    _edit(repeats / "run_2.json", no_write)
    with pytest.raises(SystemExit, match="run_2.json: not a full"):
        summarize_tracking.summarize(repeats)


def test_a_run_on_a_mismatched_clip_is_refused(repeats):
    _edit(repeats / "run_3.json", lambda r: r["source"].update(ran_with_mismatch=True))
    with pytest.raises(SystemExit, match="run_3.json: ran on a clip"):
        summarize_tracking.summarize(repeats)


def test_the_protocol_counts_the_runs_it_has(repeats):
    for n in (3, 4, 5):
        (repeats / f"run_{n}.json").unlink()
    assert summarize_tracking.summarize(repeats)["protocol"].startswith("Two separate")


def test_the_release_is_claimed_only_from_a_recorded_digest(repeats):
    """The committed runs predate weights_sha256, so they cannot claim it."""
    assert "digest not recorded" in summarize_tracking.summarize(repeats)["protocol"]

    release = summarize_tracking._release_sha256()
    for path in repeats.glob("run_*.json"):
        _edit(path, lambda r: r["config"].update(weights_sha256=release))
    assert (
        "release v1.0 checkpoint" in summarize_tracking.summarize(repeats)["protocol"]
    )
