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
