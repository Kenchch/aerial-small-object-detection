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


def test_fp16_accuracy_matches_its_report():
    """Both sides of the comparison, because the FP16 figure means nothing on
    its own -- the claim is what half precision cost, not what it scored."""
    f = _report("benchmark_fp16.json")
    assert f["precision"] == "FP16"
    readme = _readme()
    assert f"mAP50-95 **{f['accuracy']['onnx']['mAP50_95']:.4f}**" in readme
    assert f"baseline's {f['accuracy']['pytorch']['mAP50_95']:.4f}" in readme
    assert f"**{f['accuracy']['delta']['mAP50_95']:+.4f}**" in readme


def test_fp16_passed_the_same_accuracy_gate_as_the_fp32_export():
    """The README says "inside the same 0.002 gate". That is only true while
    the two reports carry the same tolerance and the delta is under it."""
    from benchmark import MAP_TOLERANCE  # src/ is on sys.path via conftest

    f = _report("benchmark_fp16.json")["accuracy"]["delta"]
    assert (
        f["tolerance"]
        == MAP_TOLERANCE
        == _report("benchmark.json")["accuracy"]["delta"]["tolerance"]
    )
    assert max(abs(f["mAP50"]), abs(f["mAP50_95"])) <= f["tolerance"]
    assert f"same {f['tolerance']} gate" in _readme()


def test_fp16_speedup_is_measured_within_one_session():
    """The ratio has to come from one process.

    This repository measured an 11% spread between sessions on this machine,
    and the FP16 speedup is 16.8% -- so a cross-session comparison is the same
    size as the effect. Taking the committed FP32 figure instead of the
    same-session one turns 16.8% into 26%, which is why `--half` re-benchmarks
    the FP32 graph rather than reading benchmark.json.
    """
    report = _report("benchmark_fp16.json")
    reference = report["fp32_reference"]
    readme = _readme()

    core16 = report["onnx_cuda"]["core"]["median_ms"]
    core32 = reference["onnx_cuda"]["core"]["median_ms"]
    stated = reference["core_speedup_pct"]

    assert f"**{core16:.2f} ms**" in readme
    assert f"against **{core32:.2f} ms**" in readme
    assert f"**{stated}% faster**" in readme
    assert (
        f"it was {report['onnx_cuda']['transfer_inclusive']['median_ms']:.2f} ms"
        in readme
    )

    # The percentage is the two milliseconds, not a third number typed beside
    # them.
    assert stated == round(100 * (1 - core16 / core32), 1)


def test_the_cross_session_figure_is_named_as_the_wrong_one():
    """The README quotes what the comparison WOULD have read against the
    committed FP32 report, to show the size of the trap. That figure has to
    stay correct too, or the warning becomes its own stale number."""
    report = _report("benchmark_fp16.json")
    other_session = _report("benchmark.json")["onnx_cuda"]["core"]["median_ms"]
    core16 = report["onnx_cuda"]["core"]["median_ms"]
    readme = _readme()

    assert f"the {other_session:.2f} ms in the" in readme
    assert f"would read {100 * (1 - core16 / other_session):.0f}%" in readme


def test_fp16_graph_size_and_placement_match_their_reports():
    # Both sizes from the FP16 run's own report: it records the FP32 graph it
    # benchmarked against, so the pair cannot come from two different exports.
    fp16 = _report("benchmark_fp16.json")
    fp32 = fp16["fp32_reference"]
    placement = fp16["onnx_cuda_placement"]
    total = placement["nodes_total"]
    readme = _readme()
    assert placement["cpu_fallback_nodes"] == 0, "the report says nodes fell back"
    assert f"**{total}/{total} nodes ran on CUDA**" in readme
    assert f"{fp16['onnx_size_mb']:.2f} MB against {fp32['onnx_size_mb']:.2f}" in readme


def test_tracking_repeats_match_their_summary():
    s = json.loads(
        (ROOT / "reports/tracking_repeats/summary.json").read_text(encoding="utf-8")
    )
    fps = s["statistics"]["end_to_end_fps"]
    readme = _readme()
    assert f"**{fps['median']:.1f} FPS median**" in readme
    assert f"{fps['min']:.1f}-{fps['max']:.1f} FPS" in readme

    steady = s["statistics"]["steady_state_fps"]
    assert f"{steady['median']:.1f} FPS steady-state" in readme
    assert f"({steady['min']:.1f}-{steady['max']:.1f})" in readme

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


def test_design_states_every_repeat_and_the_median_it_took():
    """DESIGN.md lists all five per-run figures, not just the summary.

    The paragraph's argument is that the published headline sat outside the
    spread, which a reader can only check if the spread is on the page. Five
    numbers typed into prose is five chances to leave one behind.
    """
    s = json.loads(
        (ROOT / "reports/tracking_repeats/summary.json").read_text(encoding="utf-8")
    )
    stats = s["statistics"]
    design = _design()

    per_run = sorted(
        round(
            sum(
                json.loads(p.read_text(encoding="utf-8"))["stage_ms_median"][stage]
                for stage in ("decode", "detect_and_track", "annotate", "encode")
            ),
            2,
        )
        for p in (ROOT / "reports/tracking_repeats").glob("run_*.json")
    )
    assert ", ".join(f"{ms:.2f}" for ms in per_run[:-1]) in design
    assert f"and {per_run[-1]:.2f} ms/frame" in design

    # The claim wraps across a line, so match its two halves rather than
    # pinning the document's line breaks into a test.
    ms, fps = stats["steady_state_frame_ms"], stats["steady_state_fps"]
    assert f"median of **{ms['median']:.2f} ms," in design
    assert f"{fps['median']:.1f} FPS**" in design
    assert f"range of {fps['min']:.1f} to {fps['max']:.1f} FPS" in design


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


def test_the_steady_state_headline_is_not_a_single_run():
    """The first screen quoted 25.8 FPS, derived from one run, and that run was
    faster than all five repeats -- the headline was the best result rather
    than the typical one. Nothing may re-enter the README that sits outside the
    measured spread.
    """
    s = json.loads(
        (ROOT / "reports/tracking_repeats/summary.json").read_text(encoding="utf-8")
    )
    steady = s["statistics"]["steady_state_fps"]
    single = _report("tracking.json")["stage_ms_median"]
    derived = 1000 / sum(
        single[stage] for stage in ("decode", "detect_and_track", "annotate", "encode")
    )
    assert derived > steady["max"], (
        "the single committed run is no longer faster than every repeat, so "
        "this test's premise has changed and the DESIGN.md paragraph that "
        "states it needs rechecking"
    )
    assert f"{derived:.1f} FPS steady-state" not in _readme(), (
        "the first screen is quoting the single run's rate again"
    )


def test_the_summary_reports_the_steady_state_it_can_derive():
    """summarize_tracking.py sums the four per-frame stage medians. If a repeat
    report stops carrying one of them, the summary would silently describe a
    different quantity under the same name."""
    s = json.loads(
        (ROOT / "reports/tracking_repeats/summary.json").read_text(encoding="utf-8")
    )
    runs = sorted((ROOT / "reports/tracking_repeats").glob("run_*.json"))
    assert len(runs) == s["runs"]
    for path in runs:
        stages = json.loads(path.read_text(encoding="utf-8"))["stage_ms_median"]
        assert set(stages) == {"decode", "detect_and_track", "annotate", "encode"}, (
            f"{path.name} carries stages the summary does not sum: {sorted(stages)}"
        )
