"""Unit tests for the metric cores that produce numbers the README cites.

These functions used to be inline inside `per_class_table` / `main`, which made
them reachable only through ultralytics, a GPU, the VisDrone dataset and a
trained checkpoint - i.e. unreachable by any test. A README table nobody can
test is a README table nobody can trust, so the arithmetic was pulled out into
pure functions and is pinned here with synthetic inputs.
"""

import json

import numpy as np
import pytest

import benchmark
from evaluate import error_split, letterbox_scale
from track import association_remainders

# --- confusion-matrix decomposition ---------------------------------------- #


def _cm(rows):
    """Ultralytics layout: rows=Predicted, cols=True, last row/col = background."""
    return np.array(rows, dtype=float)


def test_error_split_components_sum_to_one_per_class():
    """Each column is normalised by its own total, so a true class's outcome
    splits exactly three ways. If it does not, one of the three is wrong."""
    names = {0: "car", 1: "bus"}
    cm = _cm(
        [
            [70, 10, 0],  # predicted car
            [20, 60, 0],  # predicted bus
            [10, 30, 0],
        ]
    )  # predicted background (i.e. missed)
    got = error_split(cm, names)
    for cls in ("car", "bus"):
        total = (
            got[cls]["correct"]
            + got[cls]["missed_as_background"]
            + got[cls]["misclassified"]
        )
        assert total == pytest.approx(1.0, abs=1e-3), f"{cls} splits to {total}"


def test_error_split_values_are_column_normalised():
    names = {0: "car", 1: "bus"}
    cm = _cm([[70, 10, 0], [20, 60, 0], [10, 30, 0]])
    got = error_split(cm, names)
    assert got["car"]["correct"] == pytest.approx(0.70)  # 70/100
    assert got["car"]["missed_as_background"] == pytest.approx(0.10)
    assert got["car"]["misclassified"] == pytest.approx(0.20)  # the 20 predicted bus
    assert got["bus"]["correct"] == pytest.approx(0.60)  # 60/100
    assert got["bus"]["missed_as_background"] == pytest.approx(0.30)


def test_error_split_names_the_dominant_confusion_not_background():
    """`top_confusion` answers "when it is wrong, what does it say instead?".
    Background is reported separately as missed_as_background, so including it
    here would let a missing-limited class report 'background' and hide the
    class it is actually being confused with."""
    names = {0: "car", 1: "bus", 2: "truck"}
    cm = _cm(
        [
            [10, 0, 5],  # predicted car
            [0, 10, 25],  # predicted bus   <- truck mostly goes here
            [0, 0, 10],  # predicted truck
            [0, 0, 60],
        ]
    )  # background      <- numerically the largest cell
    got = error_split(cm, names)
    assert got["truck"]["top_confusion"] == "bus"
    assert got["truck"]["missed_as_background"] == pytest.approx(0.60)


def test_error_split_skips_classes_with_no_true_instances():
    """A class absent from the split has an all-zero column; dividing by it
    would emit nan into a JSON file written with allow_nan=False."""
    names = {0: "car", 1: "ghost"}
    cm = _cm([[50, 0, 0], [0, 0, 0], [10, 0, 0]])
    got = error_split(cm, names)
    assert "ghost" not in got and "car" in got
    assert all(np.isfinite(v) for v in got["car"].values() if isinstance(v, float))


def test_error_split_reproduces_the_committed_report():
    """Guards the layout convention (rows=Predicted, cols=True). Transposing it
    still produces plausible-looking numbers that sum to 1, so only a fixed
    expectation catches it."""
    names = {0: "car", 1: "van"}
    cm = _cm([[713, 194, 0], [194, 341, 0], [93, 465, 0]])
    got = error_split(cm, names)
    assert got["car"]["correct"] == pytest.approx(0.713, abs=1e-3)
    assert got["van"]["correct"] == pytest.approx(0.341, abs=1e-3)
    assert got["van"]["top_confusion"] == "car"


# --- letterbox geometry ----------------------------------------------------- #


@pytest.mark.parametrize(
    "W,H,imgsz,expected",
    [
        (1920, 1080, 640, 640 / 1920),  # 16:9, the whole val split
        (2000, 1500, 640, 640 / 2000),  # 4:3, present in train
        (1360, 765, 1024, 1024 / 1360),
        (640, 640, 640, 1.0),  # already square
        (480, 960, 640, 640 / 960),  # portrait: long side is H
    ],
)
def test_letterbox_scale_uses_the_long_side(W, H, imgsz, expected):
    assert letterbox_scale(W, H, imgsz) == pytest.approx(expected)


def test_letterbox_never_upscales_the_short_side_independently():
    """The bug this pins: scaling each axis to imgsz separately (a stretch)
    instead of both by one factor (a letterbox). For a 16:9 frame the two differ
    by 1.78x on the vertical, which is what made the naive `sqrt(area)*imgsz`
    box-size estimate overstate small objects."""
    W, H, imgsz = 1920, 1080, 640
    s = letterbox_scale(W, H, imgsz)
    assert W * s == pytest.approx(640)  # long side reaches the target
    assert H * s == pytest.approx(360)  # short side is padded, not stretched
    assert H * s < imgsz


def test_letterbox_area_matches_the_committed_median_box():
    """reports/evaluation.json records a median box of 11.3 px at 640 and
    18.0 px at 1024, from a median area of 0.0552% of frame on a 16:9 split."""
    W, H = 1920, 1080
    area_frac = 0.000552
    for imgsz, expected_side in ((640, 11.3), (1024, 18.0)):
        s = letterbox_scale(W, H, imgsz)
        px_area = (np.sqrt(area_frac) * W * s) * (np.sqrt(area_frac) * H * s)
        assert np.sqrt(px_area) == pytest.approx(expected_side, abs=0.1)


# --- association remainder --------------------------------------------------- #


def test_association_remainder_is_per_frame_not_a_difference_of_medians():
    """The case that motivated the change. Each frame has exactly one cheap
    phase, so every series' median is high while every frame's total is
    identical - and the sum of the three medians then exceeds the median of the
    total. Every frame's own remainder is +2.0 ms; the old formula reports
    -8.0 ms of association, a negative duration no frame exhibited."""
    t_pre = [0.0, 10.0, 10.0]
    t_fwd = [10.0, 0.0, 10.0]
    t_post = [10.0, 10.0, 0.0]
    t_infer = [p + f + po + 2.0 for p, f, po in zip(t_pre, t_fwd, t_post)]

    rem = association_remainders(t_infer, t_pre, t_fwd, t_post)
    assert rem == [2.0, 2.0, 2.0]
    assert float(np.median(rem)) == 2.0

    old = np.median(t_infer) - (np.median(t_pre) + np.median(t_fwd) + np.median(t_post))
    assert old < 0, "the old formula should be visibly wrong on this input"


def test_association_remainder_rejects_ragged_inputs():
    """These four series come off the same per-frame loop and must be the same
    length; silently zipping to the shortest would drop frames from the profile
    rather than report that something went wrong collecting it."""
    with pytest.raises(ValueError):
        association_remainders([1.0, 2.0], [0.1], [0.1, 0.2], [0.1, 0.2])


# --- export cache ---------------------------------------------------------- #


def _stub_export(tmp_path):
    """A weights file and an .onnx that stand in for the real pair."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights-bytes")
    onnx = tmp_path / "best_1024.onnx"
    onnx.write_bytes(b"onnx-bytes")
    return weights, onnx


def _write_stamp(onnx, weights, imgsz):
    benchmark._MANIFEST_STAMP(onnx).write_text(
        json.dumps(benchmark._export_manifest(onnx, weights, imgsz), indent=2),
        encoding="utf-8",
    )


def test_an_unchanged_export_is_reused(tmp_path):
    """The writer stamped .onnx.sha256 while the reader looked for
    .onnx.manifest.json, so the check never found a stamp, always returned
    False, and every run re-exported. The cache existed and could not be hit."""
    weights, onnx = _stub_export(tmp_path)
    _write_stamp(onnx, weights, 1024)
    assert benchmark._export_is_current(onnx, weights, 1024) is True


@pytest.mark.parametrize(
    "field",
    [
        "weights_sha256",
        "onnx_sha256",
        "imgsz",
        "opset",
        "simplify",
        "dynamic",
        "ultralytics",
        "onnx",
        "onnxslim",
    ],
)
def test_any_manifest_field_changing_invalidates_the_cache(tmp_path, field):
    """A weights digest alone missed a re-run at another --imgsz, a different
    opset, and a toolchain upgrade emitting a different graph from identical
    inputs - each a stale cache hit benchmarking one model and reporting
    another's accuracy."""
    weights, onnx = _stub_export(tmp_path)
    _write_stamp(onnx, weights, 1024)

    stamp = benchmark._MANIFEST_STAMP(onnx)
    recorded = json.loads(stamp.read_text(encoding="utf-8"))
    recorded[field] = "changed" if isinstance(recorded[field], str) else 999
    stamp.write_text(json.dumps(recorded), encoding="utf-8")

    assert benchmark._export_is_current(onnx, weights, 1024) is False


def test_the_manifest_records_the_flags_the_export_actually_used(tmp_path, monkeypatch):
    """The manifest is the cache key, so it has to follow the export flags.

    It repeated `"simplify": True` as its own literal beside the export's own
    `simplify=True`. Toggling the export would have left every cached graph
    still stamped with the old value and therefore still looking current - the
    exact class of stale hit the manifest was introduced to prevent. Both now
    read one constant.
    """
    weights, onnx = _stub_export(tmp_path)
    monkeypatch.setattr(benchmark, "ONNX_SIMPLIFY", False)
    monkeypatch.setattr(benchmark, "ONNX_DYNAMIC", True)

    recorded = benchmark._export_manifest(onnx, weights, 1024)
    assert recorded["simplify"] is False
    assert recorded["dynamic"] is True


def test_a_missing_or_unreadable_stamp_forces_a_re_export(tmp_path):
    weights, onnx = _stub_export(tmp_path)
    assert benchmark._export_is_current(onnx, weights, 1024) is False

    benchmark._MANIFEST_STAMP(onnx).write_text("{not json", encoding="utf-8")
    assert benchmark._export_is_current(onnx, weights, 1024) is False


def test_export_writes_nothing_into_the_weights_directory(tmp_path):
    """The weights mount is read-only in the container, and the first run has no
    cached graph, so the export path must not touch it.

    Ultralytics writes the .onnx beside the .pt it loaded - verified by
    exporting a checkpoint from a temp directory and finding probe.onnx there -
    so pointing only the *destination* at a cache dir is not enough. The
    checkpoint is copied into the cache and exported from the copy.
    """
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    weights = weights_dir / "best.pt"
    weights.write_bytes(b"checkpoint")
    before = sorted(p.name for p in weights_dir.iterdir())

    cache = tmp_path / "cache"
    onnx_path = cache / "best_1024.onnx"

    def fake_export(src):
        # Stands in for ultralytics: writes beside whatever .pt it was given.
        assert src.parent == cache, f"exported from {src.parent}, not the cache"
        produced = src.with_suffix(".onnx")
        produced.write_bytes(b"graph")
        return produced

    benchmark.export_onnx(weights, onnx_path, 1024, exporter=fake_export)

    assert onnx_path.read_bytes() == b"graph"
    assert sorted(p.name for p in weights_dir.iterdir()) == before, (
        "the export path wrote into the weights directory"
    )
    assert sorted(p.name for p in cache.iterdir()) == ["best_1024.onnx"], (
        "the staged checkpoint copy was left behind"
    )


def test_the_staged_copy_is_removed_even_when_the_export_fails(tmp_path):
    """A failed export must not leave a duplicate checkpoint in the cache."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    cache = tmp_path / "cache"

    def boom(src):
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        benchmark.export_onnx(weights, cache / "best_1024.onnx", 1024, exporter=boom)

    assert list(cache.iterdir()) == []


def test_a_cache_dir_already_holding_that_filename_is_refused(tmp_path):
    """The export stages the checkpoint under its own basename, then deletes it.

    `best.pt` is ultralytics' default output name and is gitignored here, so
    two different checkpoints called best.pt is the normal case, not a strange
    one. Point --cache-dir at a directory already holding one and the sequence
    was: copy2 over it, export, then `finally: staged_weights.unlink()`. The
    stranger's checkpoint was overwritten and then removed, and the benchmark
    exited 0 - measured, with the victim's bytes gone and no message anywhere.

    A file that is not ours to touch means stop, not "clean up afterwards".
    """
    weights_dir = tmp_path / "runs"
    weights_dir.mkdir()
    weights = weights_dir / "best.pt"
    weights.write_bytes(b"THE MODEL BEING EXPORTED")

    cache = tmp_path / "cache"
    cache.mkdir()
    victim = cache / "best.pt"
    victim.write_bytes(b"A DIFFERENT CHECKPOINT SOMEONE CARES ABOUT")

    def fake_export(src):
        produced = src.with_suffix(".onnx")
        produced.write_bytes(b"graph")
        return produced

    with pytest.raises(FileExistsError, match="already exists and is not"):
        benchmark.export_onnx(
            weights, cache / "best_1024.onnx", 1024, exporter=fake_export
        )

    assert victim.read_bytes() == b"A DIFFERENT CHECKPOINT SOMEONE CARES ABOUT", (
        "the stranger's checkpoint was modified"
    )
    assert weights.read_bytes() == b"THE MODEL BEING EXPORTED"


def test_re_exporting_over_our_own_staged_copy_still_works(tmp_path):
    """The guard is about a DIFFERENT file, not about the path being occupied.

    Exporting a checkpoint that already lives in the cache directory - the same
    file, reached by the same path - is the ordinary re-export, and samefile()
    is what tells the two apart. Refusing it would break the cache hit path
    this whole function exists to serve.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    weights = cache / "best.pt"
    weights.write_bytes(b"checkpoint")

    def fake_export(src):
        produced = src.with_suffix(".onnx")
        produced.write_bytes(b"graph")
        return produced

    benchmark.export_onnx(weights, cache / "best_1024.onnx", 1024, exporter=fake_export)

    assert (cache / "best_1024.onnx").read_bytes() == b"graph"
    assert weights.read_bytes() == b"checkpoint", "the source checkpoint was deleted"


# --- CUDA placement --------------------------------------------------------- #


def test_a_graph_fully_on_cuda_passes():
    benchmark.check_placement(
        {
            "nodes_total": 238,
            "by_provider": {"CUDAExecutionProvider": 238},
            "cpu_fallback_nodes": 0,
            "all_on_cuda": True,
        }
    )


def test_cpu_fallback_stops_the_run_instead_of_being_published():
    """`all_on_cuda` was recorded in benchmark.json and never read.

    A run with part of its graph on the CPU still produced a row labelled
    "ONNX CUDA" and published it - the milliseconds would be real and the label
    on them would not. Registering CUDAExecutionProvider does not mean the
    nodes went there: ORT silently places whatever that provider cannot run on
    the CPU, and an unsupported operator is the ordinary way it happens.
    """
    partial = {
        "nodes_total": 238,
        "by_provider": {"CUDAExecutionProvider": 234, "CPUExecutionProvider": 4},
        "cpu_fallback_nodes": 4,
        "all_on_cuda": False,
    }
    with pytest.raises(SystemExit, match="4 of 238"):
        benchmark.check_placement(partial)

    # The opt-out is explicit, and says so in --help rather than being a
    # silent default.
    benchmark.check_placement(partial, allow_cpu_fallback=True)


def test_a_profile_that_saw_no_nodes_is_not_treated_as_success():
    """all_on_cuda is False when nodes_total is 0, and a profiling run that
    recorded nothing is a broken measurement, not a clean one."""
    with pytest.raises(SystemExit):
        benchmark.check_placement(
            {
                "nodes_total": 0,
                "by_provider": {},
                "cpu_fallback_nodes": 0,
                "all_on_cuda": False,
            }
        )


def test_a_perfectly_classified_column_has_no_top_confusion():
    """max() over a generator returns its FIRST element when every value ties.

    An all-zero off-diagonal column - a class nothing was confused with - was
    therefore reported as "mostly confused with <whichever class is listed
    first>". `default=` only fires for an EMPTY sequence, which this never is,
    so it never caught it. Latent on the committed data, where all ten columns
    have real confusion.
    """
    names = {0: "car", 1: "bus", 2: "truck"}
    # Column 1 (bus) is perfect: every bus was called a bus.
    cm = np.array(
        [
            [8.0, 0.0, 3.0],
            [2.0, 5.0, 1.0],
            [1.0, 0.0, 9.0],
            [0.0, 0.0, 0.0],  # background row
        ]
    )
    table = error_split(cm, names)

    assert table["bus"]["top_confusion"] is None
    assert table["bus"]["correct"] == 1.0
    # The classes that DO have confusion still name it.
    assert table["car"]["top_confusion"] == "bus"
    assert table["truck"]["top_confusion"] == "car"
