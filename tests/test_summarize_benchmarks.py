"""scripts/summarize_benchmarks.py: the cross-session range the README quotes."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "summarize_benchmarks", ROOT / "scripts/summarize_benchmarks.py"
)
summarize_benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_benchmarks)

SESSIONS = [
    ROOT / "reports/benchmark.json",
    *sorted((ROOT / "reports/benchmark_repeats").glob("run_*.json")),
]


def test_the_committed_summary_is_what_the_sessions_give():
    committed = json.loads(
        (ROOT / "reports/benchmark_repeats/summary.json").read_text(encoding="utf-8")
    )
    assert summarize_benchmarks.summarize(SESSIONS) == committed


def test_sessions_of_different_checkpoints_are_refused(tmp_path):
    other = json.loads(SESSIONS[1].read_text(encoding="utf-8"))
    other["export"]["weights_sha256"] = "0" * 64
    path = tmp_path / "run_x.json"
    path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(SystemExit, match="different checkpoints"):
        summarize_benchmarks.summarize([SESSIONS[0], path])


def test_an_fp16_run_is_not_an_fp32_session():
    with pytest.raises(SystemExit, match="not an FP32 run"):
        summarize_benchmarks.summarize(
            [SESSIONS[0], ROOT / "reports/benchmark_fp16.json"]
        )
