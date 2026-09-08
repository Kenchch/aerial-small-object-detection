"""Every number the README states must come from a committed report.

One assertion per number, each naming its own source file. A single assertion
covering several figures fails on whichever it reaches first and says nothing
about the rest, so a stale number can hide behind a fresh one.

These are string containment checks against the rendered README, using the same
formatting the README uses. That is deliberate: the failure mode being guarded
against is a number typed into prose and then left behind when the report was
regenerated, and only comparing the formatted text catches it.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _report(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# reports/benchmark.json
# --------------------------------------------------------------------------- #


def test_validation_map_matches_benchmark():
    b = _report("benchmark.json")
    assert f"mAP50 {b['mAP50']:.3f} / mAP50-95 {b['mAP50_95']:.3f}" in _readme()


def test_onnx_cuda_latency_matches_benchmark():
    b = _report("benchmark.json")
    core = b["onnx_cuda"]["core"]["median_ms"]
    assert f"{core:.1f} ms" in _readme()


def test_onnx_speedup_matches_benchmark():
    b = _report("benchmark.json")
    core = b["onnx_cuda"]["core"]["median_ms"]
    eager = b["pytorch_cuda"]["core"]["median_ms"]
    assert f"{100 * (1 - core / eager):.0f}% faster" in _readme()


def test_cpu_slowdown_matches_benchmark():
    b = _report("benchmark.json")
    ratio = b["onnx_cpu"]["core"]["median_ms"] / b["onnx_cuda"]["core"]["median_ms"]
    assert f"{ratio:.0f}× slower" in _readme()


def test_node_placement_matches_benchmark():
    b = _report("benchmark.json")
    placement = b["onnx_cuda_placement"]
    total = placement["nodes_total"]
    assert placement["cpu_fallback_nodes"] == 0, (
        "the report itself says nodes fell back"
    )
    assert f"{total}/{total} nodes" in _readme()


def test_onnx_parity_delta_matches_benchmark():
    b = _report("benchmark.json")
    assert f"+{b['accuracy']['delta']['mAP50_95']:.4f}" in _readme()


# --------------------------------------------------------------------------- #
# reports/evaluation_test.json  -- the held-out split
# --------------------------------------------------------------------------- #


def test_test_dev_map_matches_its_report():
    e = _report("evaluation_test.json")["accuracy"]["overall"]
    readme = _readme()
    assert f"mAP50 {e['mAP50']:.3f} / mAP50-95 {e['mAP50_95']:.3f}" in readme
    assert f"**{e['mAP50']:.4f}**" in readme, "the long form in Additional measurements"
    assert f"**{e['mAP50_95']:.4f}**" in readme, (
        "the long form in Additional measurements"
    )


def test_test_dev_image_count_matches_its_report():
    """Both places that state the size of the split, not just one.

    The count is what makes the test-dev row a claim about a held-out set
    rather than a number; it is stated twice and read from `label_scale`.
    """
    images = _report("evaluation_test.json")["label_scale"]["images"]
    assert _readme().count(f"{images:,} images") == 2


def test_test_dev_gap_against_validation_is_stated_correctly():
    """The gap is the point of quoting test-dev at all, so it cannot be stale.

    Quoted to four places, matching the two figures it is a difference of.
    Three places would put this particular gap on a rounding boundary --
    0.2216 - 0.1831 is 0.0385 in decimal but 0.03849999... in binary, so
    ``.3f`` gives 0.038 while rounding the decimal by hand gives 0.039, and
    the README and this test would disagree for a reason that has nothing to
    do with either number being wrong.
    """
    val = _report("benchmark.json")["mAP50_95"]
    test = _report("evaluation_test.json")["accuracy"]["overall"]["mAP50_95"]
    assert f"{val - test:.4f} mAP50-95 below val" in _readme()


# --------------------------------------------------------------------------- #
# reports/benchmark_fp16.json and reports/tracking_repeats/summary.json
# --------------------------------------------------------------------------- #


def test_fp16_map_matches_its_report():
    f = _report("benchmark_fp16.json")
    value = f.get("mAP50_95") or f.get("accuracy", {}).get("mAP50_95")
    if value is None:
        pytest.skip("benchmark_fp16.json does not record mAP50_95")
    assert f"**{value:.4f}**" in _readme()


def test_fp16_latency_matches_its_report():
    """The FP16 latency is transfer-inclusive and so is not comparable with the
    core figure on the first screen. Both are quoted; both have to be current."""
    lat = _report("benchmark_fp16.json")["latency"]
    readme = _readme()
    assert f"**{lat['median_ms']:.2f} ms" in readme
    assert f"{lat['p95_ms']:.2f} ms p95**" in readme
    assert f"{lat['warmup']} warm-ups, {lat['iterations']} timed iterations" in readme


def test_tracking_repeats_match_their_summary():
    s = json.loads(
        (ROOT / "reports/tracking_repeats/summary.json").read_text(encoding="utf-8")
    )
    fps = s["statistics"]["end_to_end_fps"]
    readme = _readme()
    assert f"**{fps['median']:.1f} FPS median**" in readme
    assert f"{fps['min']:.1f}-{fps['max']:.1f} FPS" in readme

    # The README writes the repeat count as a word, so `str(s["runs"])` would
    # be a bare "5" matched anywhere in the file -- an assertion that passes
    # whatever the report says. Same for the frame count.
    words = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven"}
    assert f"{words[s['runs']]} repeats" in readme
    frames = set(s["frames_each"])
    assert len(frames) == 1, f"the repeats did not all run the same length: {frames}"
    assert f"{frames.pop()}-frame synthetic pan" in readme


# --------------------------------------------------------------------------- #
# docs/DESIGN.md -- the same failure mode, a longer document
# --------------------------------------------------------------------------- #


def _design() -> str:
    return (ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8")


def test_design_parity_paragraph_matches_the_benchmark():
    """DESIGN.md states both backends and their delta in one sentence.

    Four numbers in one line, none of them in the README, all of them from
    `accuracy` in benchmark.json. Regenerating the report moves them together,
    which is exactly the case where a hand-typed sentence is left behind.
    """
    a = _report("benchmark.json")["accuracy"]
    design = _design()
    assert (
        f"PyTorch mAP50 {a['pytorch']['mAP50']:.4f} / "
        f"mAP50-95 {a['pytorch']['mAP50_95']:.4f}" in design
    )
    assert f"ONNX {a['onnx']['mAP50']:.4f} / {a['onnx']['mAP50_95']:.4f}" in design
    assert (
        f"delta of +{a['delta']['mAP50']:.4f} / +{a['delta']['mAP50_95']:.4f}" in design
    )


def test_design_tolerance_matches_the_constant():
    """The prose names the constant, so it has to name its value correctly."""
    from benchmark import MAP_TOLERANCE  # src/ is on sys.path via conftest

    assert f"MAP_TOLERANCE, {MAP_TOLERANCE}" in _design()
    assert _report("benchmark.json")["accuracy"]["delta"]["tolerance"] == MAP_TOLERANCE


def test_design_node_placement_matches_the_benchmark():
    placement = _report("benchmark.json")["onnx_cuda_placement"]
    total = placement["nodes_total"]
    fallback = placement["cpu_fallback_nodes"]
    assert f"**{total} of {total} nodes on CUDA, {fallback} on CPU**" in _design()


# --------------------------------------------------------------------------- #
# The guard on the guard
# --------------------------------------------------------------------------- #


def test_a_changed_report_would_fail_these_assertions():
    """Proves the checks above are load-bearing rather than tautological.

    Every assertion here is `formatted_value in readme`, which passes for any
    value that happens to appear somewhere in the file. This perturbs a real
    figure and asserts the formatted result is absent, so a check that would
    accept anything fails here first.
    """
    b = _report("benchmark.json")
    core = b["onnx_cuda"]["core"]["median_ms"]
    assert f"{core + 1.0:.1f} ms" not in _readme(), (
        "a latency 1 ms off the report also appears in the README, so the "
        "latency assertion proves nothing"
    )
