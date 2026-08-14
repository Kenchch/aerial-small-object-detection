"""Unit tests for benchmark.py's pure helpers.

benchmark.py defers its torch/ultralytics imports into the functions that use
them, so this module imports cleanly with nothing but the standard library --
no skip guard needed, and these run on a bare clone.
"""
import math

import pytest

from benchmark import TIMED_ITERS, _summarise


def test_summarise_basic_stats():
    times = [10.0, 12.0, 11.0, 100.0, 9.0]  # one outlier, 5 samples
    result = _summarise(times)

    # sorted: [9, 10, 11, 12, 100]
    assert result["min_ms"] == 9.0
    assert result["median_ms"] == 11.0
    # Nearest-rank p95 on 5 samples is ceil(0.95*5) == 5 -> the 5th value.
    # With only 5 samples the 95th percentile IS the outlier; reporting 12.0
    # here would be the tail going unreported, which is the one thing p95 is
    # in the table to prevent.
    assert result["p95_ms"] == 100.0
    assert result["fps"] == round(1000.0 / 11.0, 1)


def test_summarise_fps_is_inverse_of_median():
    times = [20.0, 20.0, 20.0, 20.0]
    result = _summarise(times)

    assert result["median_ms"] == 20.0
    assert result["fps"] == 50.0


@pytest.mark.parametrize("n", [5, 10, 20, 30, 50, TIMED_ITERS, 200])
def test_p95_never_understates_the_tail(n):
    """The old index, int(0.95*n)-1, agreed with nearest-rank only when 0.95*n
    landed on a whole number - true at TIMED_ITERS=100, false at 5/10/30/50,
    where it silently reported the next value down."""
    times = [float(i) for i in range(1, n + 1)]          # 1..n, already sorted
    expected = float(math.ceil(0.95 * n))                # nearest-rank value
    assert _summarise(times)["p95_ms"] == expected
    assert expected >= float(int(0.95 * n))              # never below the old one
