"""track.build_report's arithmetic, on hand-computed inputs.

Every derived number in reports/tracking.json comes from build_report, and none
of its formulas was asserted anywhere: the end-to-end test checked one frame
count, and the README tests compare the README with the JSON rather than
recomputing either. A wrong formula would regenerate a wrong report that still
agreed with itself.
"""

import argparse
from pathlib import Path

import pytest

import track

INPUT = {
    "t_decode": [1.0, 2.0, 3.0, 10.0],
    "t_infer": [100.0, 20.0, 30.0, 40.0],
    "t_draw": [1.0, 1.0, 1.0, 5.0],
    "t_write": [2.0, 2.0, 2.0, 6.0],
    "t_pre": [10.0, 2.0, 3.0, 4.0],
    "t_fwd": [80.0, 10.0, 20.0, 30.0],
    "t_post": [5.0, 1.0, 1.0, 1.0],
    "track_frames": {1: 4, 2: 1, 3: 2, 10: 3},
    "n_frames": 4,
    "wall": 0.4,
    "setup_ms": 8.0,
    "flush_ms": 4.0,
    "output": None,
}


@pytest.fixture
def report(monkeypatch, tmp_path):
    monkeypatch.setattr(track, "_environment", lambda _: {})
    monkeypatch.setattr(track, "_source_provenance", lambda _: {})
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    args = argparse.Namespace(
        weights=weights,
        out=Path("o.mp4"),
        no_write=False,
        imgsz=1024,
        conf=0.25,
        tracker="bytetrack.yaml",
        device="0",
        source=Path("clip.mp4"),
    )

    def build(source_matches=True):
        return track.build_report(**INPUT, source_matches=source_matches, args=args)

    return build


def test_throughput(report):
    r = report()
    assert r["per_frame_wall_ms"] == 100.0
    assert r["end_to_end_fps"] == 10.0


def test_stage_medians_and_means(report):
    r = report()
    assert r["stage_ms_median"] == {
        "decode": 2.5,
        "detect_and_track": 35.0,
        "annotate": 1.0,
        "encode": 2.0,
    }
    assert r["stage_ms_mean"] == {
        "decode": 4.0,
        "detect_and_track": 47.5,
        "annotate": 2.0,
        "encode": 3.0,
    }


def test_reconciliation_counts_setup_and_flush_once(report):
    # 4 + 47.5 + 2 + 3 per frame, plus (8 + 4) / 4 amortised = 59.5 of 100 ms
    assert report()["reconciliation"] == {
        "accounted_mean_ms": 59.5,
        "unaccounted_ms": 40.5,
        "coverage_pct": 59.5,
    }


def test_association_is_the_median_of_per_frame_remainders(report):
    # infer - pre - fwd - post per frame: [5, 7, 6, 5] -> median 5.5
    assert report()["detect_and_track_ms_median"]["association_and_overhead"] == 5.5


def test_warmup(report):
    assert report()["warmup"] == {
        "first_frame_ms": 100.0,
        "steady_state_ms": 35.0,
        "warmup_penalty_x": 2.9,
    }


def test_track_statistics(report):
    t = report()["tracks"]
    assert (t["unique_ids"], t["highest_id_seen"], t["id_churn_ratio_min"]) == (
        4,
        10,
        1.5,
    )
    assert (t["single_frame_tracks"], t["single_frame_pct"]) == (1, 25.0)
    assert (t["mean_track_len_frames"], t["max_track_len_frames"]) == (2.5, 4)
    assert t["mean_boxes_per_frame"] == 2.5


def test_a_mismatched_source_is_recorded_as_such(report):
    assert report(source_matches=False)["source"]["ran_with_mismatch"] is True
    assert report(source_matches=True)["source"]["ran_with_mismatch"] is False


def test_steady_state_is_per_frame_totals_after_the_first(report):
    """Per-frame totals [104, 25, 36, 61]; without the cold first frame,
    122 ms over 3 frames. The sum of per-stage medians - 2.5 + 35 + 1 + 2 =
    40.5 ms - describes no frame and reads faster than the run was."""
    assert report()["steady_state"] == {
        "frames_excluded": 1,
        "frame_ms_median": 36.0,
        "frame_ms_mean": 40.67,
        "fps": 24.6,
    }


def test_no_frames_after_the_first_means_no_steady_state():
    assert track.steady_state([1.0], [5.0], [1.0], [1.0]) is None
