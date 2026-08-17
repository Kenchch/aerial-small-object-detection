"""Unit tests for track.py's pure helpers.

track.py defers its cv2/ultralytics imports into the functions that use them,
so importing this module needs neither. track_colour does still call into
numpy, which is the one dependency guarded below.
"""

import pytest

pytest.importorskip("numpy", reason="track_colour draws from numpy's RNG")

from track import track_colour


def test_colour_is_deterministic_per_track_id():
    assert track_colour(7) == track_colour(7)


def test_colour_is_a_bgr_triple_in_range():
    b, g, r = track_colour(42)
    for channel in (b, g, r):
        assert 70 <= channel < 255


def test_different_ids_usually_differ():
    colours = {track_colour(i) for i in range(20)}
    assert len(colours) > 1
