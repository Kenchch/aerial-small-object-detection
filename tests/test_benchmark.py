"""Requires the project env (torch, ultralytics) -- see requirements.txt."""
from benchmark import _summarise


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
