"""The README's headline numbers must match the JSON they claim to come from.

Every figure in README.md is supposed to be traceable to a file in reports/.
Keeping that true by hand does not work: the tracking profile has been
re-measured several times, and each time a resync script updated the tables and
missed prose. The last one left `the mean (159.3 ms) and the median (32.26 ms)`
standing after a re-run had made them 97.93 and 32.21 - a 63 % overstatement,
sitting sixty lines below a table that said otherwise, because the script's
regex spanned a line break and silently matched nothing.

A regex that matches nothing looks exactly like a regex that had nothing to do.
So the check lives here, where matching nothing is a failure.

These read the committed reports/*.json, so they run anywhere - no GPU, no
dataset, no torch.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _json(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


TRACKING = _json("tracking.json")
BENCHMARK = _json("benchmark.json")


def _one(pattern: str) -> str:
    """The single capture for `pattern`, or a failure naming the pattern.

    Deliberately strict about the count. A pattern that stops matching because
    the prose was reworded is the failure mode this file exists to catch, and a
    silent zero-match would reproduce it exactly.
    """
    found = re.findall(pattern, README)
    assert found, f"README no longer contains anything matching: {pattern}"
    assert len(found) == 1, f"{len(found)} matches for {pattern!r}: {found}"
    return found[0]


# --- the tracking stage table ----------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("decode", "decode"),
        (r"detect \+ track", "detect_and_track"),
        ("annotate", "annotate"),
        ("encode", "encode"),
    ],
)
def test_stage_rows_match_the_tracking_report(label, key):
    row = _one(rf"\| {label} \| ([\d.]+) ms \| \*?\*?([\d.]+) ms\*?\*? \|")
    assert float(row[0]) == TRACKING["stage_ms_median"][key]
    assert float(row[1]) == TRACKING["stage_ms_mean"][key]


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("preprocess", "preprocess"),
        ("forward", "forward"),
        (r"postprocess \(NMS\)", "postprocess_nms"),
        (r"association \+ overhead", "association_and_overhead"),
    ],
)
def test_detect_and_track_breakdown_matches(label, key):
    value = _one(rf"\| [├└] {label} \| ([\d.]+) ms \| \|")
    assert float(value) == TRACKING["detect_and_track_ms_median"][key]


def test_the_reconciliation_row_matches():
    accounted = _one(r"\| \*\*accounted\*\* \| \| \*\*([\d.]+) ms\*\*")
    wall = _one(r"\| \*\*wall per frame\*\* \| \| \*\*([\d.]+) ms\*\*")
    assert float(accounted) == TRACKING["reconciliation"]["accounted_mean_ms"]
    assert float(wall) == TRACKING["per_frame_wall_ms"]


# --- the prose that has drifted before --------------------------------------- #


def test_the_mean_median_gap_sentence_matches():
    """The exact sentence that went stale.

    It restates two numbers the table above it already gives, which is why
    nothing caught the disagreement: both were plausible, and only one was
    measured.
    """
    mean = _one(r"gap between the mean\s*\(([\d.]+) ms\)")
    median = _one(r"and the median \(([\d.]+) ms\)")
    assert float(mean) == TRACKING["stage_ms_mean"]["detect_and_track"]
    assert float(median) == TRACKING["stage_ms_median"]["detect_and_track"]


def test_end_to_end_throughput_matches():
    fps = _one(r"throughput\s*\n?\(\*\*([\d.]+) FPS\*\*\)")
    assert float(fps) == TRACKING["end_to_end_fps"]


def test_the_steady_state_figure_is_the_sum_of_the_stage_medians():
    """Derived, not measured - so it is checked by recomputing it."""
    stated_ms, stated_fps = _one(
        r"which gives \*\*([\d.]+) ms/frame → ([\d.]+) FPS\*\*"
    )
    steady = sum(v for v in TRACKING["stage_ms_median"].values() if v)
    assert float(stated_ms) == pytest.approx(steady, abs=0.01)
    assert float(stated_fps) == pytest.approx(1000 / steady, abs=0.05)


def test_the_first_frame_cost_matches():
    ms, penalty, steady = _one(
        r"frame costs \*\*([\d,]+) ms — ([\d.]+)× the steady-state ([\d.]+) ms\*\*"
    )
    assert float(ms.replace(",", "")) == pytest.approx(
        TRACKING["warmup"]["first_frame_ms"], abs=0.5
    )
    # The README rounds the penalty to a whole number (180x against 180.2),
    # which is right for a figure this unstable - the cold start has ranged
    # 5.6-11.2 s on this machine. Tolerance to match the rounding.
    assert float(penalty) == pytest.approx(
        TRACKING["warmup"]["warmup_penalty_x"], abs=0.5
    )
    assert float(steady) == TRACKING["warmup"]["steady_state_ms"]


# --- the backend latency table ----------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "key"),
    [
        (r"PyTorch \(CUDA, eager\)", "pytorch_cuda"),
        (r"ONNX Runtime \(CUDA\)", "onnx_cuda"),
    ],
)
def test_backend_rows_match_the_benchmark_report(label, key):
    row = _one(
        rf"\| {label} \| \*?\*?([\d.]+) ms · ([\d.]+) FPS\*?\*? \| "
        rf"\*?\*?([\d.]+) ms · ([\d.]+) FPS\*?\*? \|"
    )
    core, transfer = BENCHMARK[key]["core"], BENCHMARK[key]["transfer_inclusive"]
    assert float(row[0]) == core["median_ms"]
    assert float(row[1]) == core["fps"]
    assert float(row[2]) == transfer["median_ms"]
    assert float(row[3]) == transfer["fps"]


def test_the_cpu_row_matches_and_both_columns_agree():
    """On CPU there is no host copy to separate, so the two columns are the
    same measurement and the README says so."""
    row = _one(r"\| ONNX Runtime \(CPU\) \| ([\d.]+) ms · ([\d.]+) FPS \|")
    assert float(row[0]) == BENCHMARK["onnx_cpu"]["core"]["median_ms"]
    assert float(row[1]) == BENCHMARK["onnx_cpu"]["core"]["fps"]
    assert (
        BENCHMARK["onnx_cpu"]["core"]["median_ms"]
        == BENCHMARK["onnx_cpu"]["transfer_inclusive"]["median_ms"]
    )


def test_the_two_accuracy_figures_stay_attributed_to_their_sources():
    """There are two mAP numbers and they are not interchangeable.

    0.373/0.221 is the training run's own final validation pass; 0.3748/0.2216
    is benchmark.py re-running validation to pair accuracy with the latency
    figures. The README distinguishes them deliberately, so this asserts the
    benchmark pair matches benchmark.json and that the two have not been
    collapsed into one.
    """
    m50, m5095 = _one(r"mAP50\s*([\d.]+), mAP50-95 ([\d.]+), in `reports/benchmark")
    assert float(m50) == BENCHMARK["mAP50"]
    assert float(m5095) == BENCHMARK["mAP50_95"]
    assert float(m50) != float(_one(r"\*\*mAP50 ([\d.]+), mAP50-95 [\d.]+\.\*\*"))


def test_the_cuda_placement_claim_matches():
    total, on_cuda = _one(r"\*\*(\d+) of (\d+) nodes on CUDA, 0 on CPU\*\*")
    placement = BENCHMARK["onnx_cuda_placement"]
    assert int(total) == int(on_cuda) == placement["nodes_total"]
    assert placement["cpu_fallback_nodes"] == 0


# --- the provenance chain ---------------------------------------------------- #


def test_the_readme_quotes_the_published_clip_digest():
    """The exact-reproduction command in the README has to name the clip that
    is actually published, or following it produces a different clip and the
    --expected- guards fire on a correct run."""
    record = _json("demo_pan.provenance.json")
    quoted = _one(r"--expected-clip-sha256 ([0-9a-f]{64})")
    assert quoted == record["clip"]["sha256"]
    assert (
        _one(r"--expected-source-sha256 ([0-9a-f]{64})")
        == (record["generator"]["source_sha256"])
    )


def test_the_output_digest_ties_the_gif_to_the_tracking_run():
    gif = _json("tracking_demo.provenance.json")
    assert TRACKING["output"]["sha256"] == gif["source"]["sha256"]
    assert README.count(TRACKING["output"]["sha256"][:16]) >= 1
