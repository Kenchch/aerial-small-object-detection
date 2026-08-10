"""Unit tests for track.py's pure helpers.

track_colour needs only numpy, but track.py imports cv2 and ultralytics at
module scope -- see test_benchmark.py for why those imports are guarded
rather than left to fail at collection time.
"""
import pytest

pytest.importorskip("cv2", reason="project env not installed (see requirements.txt)")
pytest.importorskip("numpy", reason="project env not installed (see requirements.txt)")
pytest.importorskip("ultralytics", reason="project env not installed (see requirements.txt)")

from track import track_colour  # noqa: E402  (must follow the skip guards)


def test_colour_is_deterministic_per_track_id():
    assert track_colour(7) == track_colour(7)


def test_colour_is_a_bgr_triple_in_range():
    b, g, r = track_colour(42)
    for channel in (b, g, r):
        assert 70 <= channel < 255


def test_different_ids_usually_differ():
    colours = {track_colour(i) for i in range(20)}
    assert len(colours) > 1
