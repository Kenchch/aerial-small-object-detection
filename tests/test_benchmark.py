"""Unit tests for benchmark.py's pure helpers.

benchmark.py defers its torch/ultralytics imports into the functions that use
them, so this module imports cleanly with nothing but the standard library --
no skip guard needed, and these run on a bare clone.
"""

import argparse
import json
import math
from pathlib import Path

import pytest

import benchmark
from benchmark import TIMED_ITERS, _summarise, check_ort_build, onnx_cuda_runnable


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
    times = [float(i) for i in range(1, n + 1)]  # 1..n, already sorted
    expected = float(math.ceil(0.95 * n))  # nearest-rank value
    assert _summarise(times)["p95_ms"] == expected
    assert expected >= float(int(0.95 * n))  # never below the old one


# The provider list onnxruntime-gpu 1.20.2 reports on a machine with no GPU.
GPU_WHEEL = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


@pytest.mark.parametrize(
    ("cuda_device", "providers", "expected"),
    [
        (True, GPU_WHEEL, True),
        # The GPU wheel with no device: the case that used to request CUDA,
        # get the CPU, and raise before the ONNX-CPU row or report existed.
        (False, GPU_WHEEL, False),
        (True, ["CPUExecutionProvider"], False),  # CPU-only onnxruntime wheel
        (False, ["CPUExecutionProvider"], False),
    ],
)
def test_onnx_cuda_rows_need_a_device_and_a_provider(cuda_device, providers, expected):
    assert onnx_cuda_runnable(cuda_device, providers) is expected


def test_a_gpu_machine_with_a_cpu_only_onnxruntime_is_refused():
    """The pinned GPU build overwritten by a CPU one: the CUDA rows would be
    skipped and the report published without them."""
    with pytest.raises(SystemExit, match="onnxruntime-gpu"):
        check_ort_build(True, ["AzureExecutionProvider", "CPUExecutionProvider"])


@pytest.mark.parametrize(
    ("cuda_device", "providers"),
    [(True, GPU_WHEEL), (False, GPU_WHEEL), (False, ["CPUExecutionProvider"])],
)
def test_consistent_ort_builds_pass(cuda_device, providers):
    check_ort_build(cuda_device, providers)


def test_summarise_median_on_even_n_is_the_midpoint():
    """TIMED_ITERS=100 is even: the committed median is the mean of ranks 50
    and 51, not the lower of the two."""
    result = _summarise([1.0, 2.0, 4.0, 100.0])
    assert result["median_ms"] == 3.0
    assert result["fps"] == round(1000.0 / 3.0, 1)


# --- the accuracy gate, placement, cache naming and speedup ------------------ #

PT = {"mAP50": 0.3748, "mAP50_95": 0.2216}


@pytest.mark.parametrize(
    "shift",
    [
        {"mAP50": 0.003},
        {"mAP50": -0.003},
        {"mAP50_95": 0.003},
        {"mAP50_95": -0.003},  # the direction the gate exists for
    ],
)
def test_the_accuracy_gate_refuses_a_shift_either_way(shift):
    onnx = {k: v + shift.get(k, 0.0) for k, v in PT.items()}
    with pytest.raises(SystemExit, match="more than 0.002"):
        benchmark.accuracy_delta(PT, onnx, 0.002)


def test_the_accuracy_gate_passes_small_shifts_and_reports_them():
    onnx = {"mAP50": PT["mAP50"] - 0.0005, "mAP50_95": PT["mAP50_95"] + 0.0015}
    delta = benchmark.accuracy_delta(PT, onnx, 0.002)
    assert delta == {"mAP50": -0.0005, "mAP50_95": 0.0015, "tolerance": 0.002}


def test_the_committed_delta_is_what_the_gate_computes():
    acc = json.loads(
        (Path(__file__).resolve().parents[1] / "reports/benchmark.json").read_text(
            encoding="utf-8"
        )
    )["accuracy"]
    assert benchmark.accuracy_delta(acc["pytorch"], acc["onnx"], 0.002) == acc["delta"]


def test_placement_counts_only_node_events():
    events = [
        {"cat": "Node", "args": {"provider": "CUDAExecutionProvider"}},
        {"cat": "Node", "args": {"provider": "CUDAExecutionProvider"}},
        {"cat": "Node", "args": {"provider": "CUDAExecutionProvider"}},
        {"cat": "Node", "args": {"provider": "CPUExecutionProvider"}},
        {"cat": "Session", "args": {"provider": "CPUExecutionProvider"}},
        {"cat": "Node", "args": {}},
    ]
    got = benchmark.placement_from_events(events)
    assert (got["nodes_total"], got["cpu_fallback_nodes"], got["all_on_cuda"]) == (
        4,
        1,
        False,
    )


def test_an_empty_profile_is_not_all_on_cuda():
    assert benchmark.placement_from_events([])["all_on_cuda"] is False


def test_the_cache_path_carries_stem_size_and_precision(tmp_path):
    weights = tmp_path / "w" / "last.pt"
    assert benchmark.onnx_cache_path(weights, 1024, False).name == "last_1024.onnx"
    assert benchmark.onnx_cache_path(weights, 1024, True).name == "last_1024_fp16.onnx"
    assert benchmark.onnx_cache_path(weights, 640, False, tmp_path / "c") == (
        tmp_path / "c" / "last_640.onnx"
    )


def test_speedup_is_relative_to_the_reference():
    assert benchmark.speedup_pct(10, 12) == 16.7


def test_the_committed_fp16_speedup_is_what_the_formula_gives():
    b = json.loads(
        (Path(__file__).resolve().parents[1] / "reports/benchmark_fp16.json").read_text(
            encoding="utf-8"
        )
    )
    ref = b["fp32_reference"]
    assert (
        benchmark.speedup_pct(
            b["onnx_cuda"]["core"]["median_ms"], ref["onnx_cuda"]["core"]["median_ms"]
        )
        == ref["core_speedup_pct"]
    )


# --- argument validation ----------------------------------------------------- #


@pytest.mark.parametrize("value", [1024, "640", 32])
def test_imgsz_accepts_multiples_of_32(value):
    assert benchmark._imgsz(value) == int(value)


@pytest.mark.parametrize("value", [1000, 0, -32, "abc"])
def test_imgsz_refuses_what_ultralytics_would_silently_round(value):
    """1000 was exported as a 1024 graph while everything else recorded 1000."""
    with pytest.raises(argparse.ArgumentTypeError, match="imgsz must be"):
        benchmark._imgsz_arg(value)


def test_a_bad_map_tolerance_is_explained_on_the_command_line():
    """argparse printed "invalid _map_tolerance value" for a ValueError; the
    explanation only reaches the user as an ArgumentTypeError."""
    with pytest.raises(
        argparse.ArgumentTypeError, match=r"finite fraction in \[0, 1\]"
    ):
        benchmark._map_tolerance_arg("2")
