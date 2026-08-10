"""Unit tests for benchmark.py's pure helpers.

_summarise needs only the standard library, but benchmark.py imports torch at
module scope. A plain import here would fail at *collection* time without the
project env installed, which aborts the entire run -- including the
dependency-free tests in test_syntax.py. importorskip keeps that to a skip.
"""
import pytest

pytest.importorskip("torch", reason="project env not installed (see requirements.txt)")
pytest.importorskip("ultralytics", reason="project env not installed (see requirements.txt)")

from benchmark import _summarise  # noqa: E402  (must follow the skip guards)


def test_summarise_basic_stats():
    times = [10.0, 12.0, 11.0, 100.0, 9.0]  # one outlier, 5 samples
    result = _summarise(times)

    # sorted: [9, 10, 11, 12, 100]
    assert result["min_ms"] == 9.0
    assert result["median_ms"] == 11.0
    # p95 index on 5 samples is int(0.95*5)-1 == 3 -> 12.0, not the outlier
    assert result["p95_ms"] == 12.0
    assert result["fps"] == round(1000.0 / 11.0, 1)


def test_summarise_fps_is_inverse_of_median():
    times = [20.0, 20.0, 20.0, 20.0]
    result = _summarise(times)

    assert result["median_ms"] == 20.0
    assert result["fps"] == 50.0
