"""Unit tests for track.py's pure helpers.

track.py defers its cv2/ultralytics imports into the functions that use them,
so importing this module needs neither. track_colour does still call into
numpy, which is the one dependency guarded below.
"""

import json

import pytest

pytest.importorskip("numpy", reason="track_colour draws from numpy's RNG")

import track
from track import natural_key, track_colour


def test_colour_is_deterministic_per_track_id():
    assert track_colour(7) == track_colour(7)


def test_colour_is_a_bgr_triple_in_range():
    b, g, r = track_colour(42)
    for channel in (b, g, r):
        assert 70 <= channel < 255


def test_different_ids_usually_differ():
    colours = {track_colour(i) for i in range(20)}
    assert len(colours) > 1


# --- frame ordering --------------------------------------------------------- #


def test_frames_sort_numerically_not_lexicographically():
    """`sorted()` on filenames puts frame10 between frame1 and frame2.

    Nothing errors: the sequence is simply fed to the tracker in the order
    1, 10, 11, 2, 3. It associates across jumps that never happened, so track
    lengths, the churn ratio and the single-frame-track percentage all describe
    a sequence that does not exist.
    """
    names = [f"frame{i}.jpg" for i in (1, 2, 3, 10, 11, 20, 100)]
    shuffled = sorted(names)  # lexicographic - the bug
    assert shuffled != names, "the fixture no longer demonstrates the problem"
    assert sorted(names, key=natural_key) == names


def test_the_sort_key_is_total():
    """Two names can put a number and a word at the same position, and
    comparing int with str raises. The key has to order them, not crash."""
    names = ["1.jpg", "a.jpg", "1a.jpg", "a1.jpg", "img_007.png", "img_7.png"]
    assert len(sorted(names, key=natural_key)) == len(names)


def test_zero_padding_and_case_do_not_change_the_order():
    assert sorted(["f9.jpg", "f10.jpg"], key=natural_key) == ["f9.jpg", "f10.jpg"]
    assert sorted(["F2.jpg", "f1.jpg"], key=natural_key) == ["f1.jpg", "F2.jpg"]


# --- report paths ----------------------------------------------------------- #


def test_report_paths_are_repo_relative_and_posix():
    """These reports are committed, so neither the absolute path nor the path
    separator of whoever generated them should be baked into the file."""
    inside = track.PROJECT_ROOT / "reports" / "track_out.mp4"
    assert track._for_report(inside) == "reports/track_out.mp4"
    assert chr(92) not in track._for_report(inside)


# --- source provenance ------------------------------------------------------ #


def test_a_clip_with_a_generator_record_is_identified_by_it(tmp_path):
    """reports/demo_pan.mp4 is build output and is not in the repo, so "re-run
    make_demo_clip.py" does not by itself reproduce the clip a number was
    measured on - the source frame is picked by label count against whatever
    dataset revision is installed, and the crop, pan and fps are arguments.

    The clip therefore identifies itself, and the profile embeds that record.
    """
    clip = tmp_path / "demo_pan.mp4"
    clip.write_bytes(b"pretend this is an mp4")
    digest = track._sha256(clip)
    (tmp_path / "demo_pan.provenance.json").write_text(
        json.dumps(
            {
                "clip": {"sha256": digest, "fps": 15, "frames": 90},
                "generator": {"source_image": "0000295.jpg", "crop_fraction": 0.62},
            }
        ),
        encoding="utf-8",
    )

    info = track._source_provenance(clip)
    assert info["sha256"] == digest
    assert info["generator"]["source_image"] == "0000295.jpg"
    assert info["matches_generator_record"] is True
    assert (info["fps"], info["generated_frames"]) == (15, 90)


def test_a_clip_that_is_not_the_one_the_record_describes_says_so(tmp_path):
    """A stale record beside a regenerated clip is the failure this guards
    against, so the mismatch is reported rather than assumed away."""
    clip = tmp_path / "demo_pan.mp4"
    clip.write_bytes(b"a different clip entirely")
    (tmp_path / "demo_pan.provenance.json").write_text(
        json.dumps({"clip": {"sha256": "0" * 64}, "generator": {}}), encoding="utf-8"
    )
    assert track._source_provenance(clip)["matches_generator_record"] is False


def test_a_clip_with_no_record_still_carries_its_digest(tmp_path):
    """Any video can be passed to --source. Without a generator record the
    profile still says exactly which bytes it profiled."""
    clip = tmp_path / "somebody_elses.mp4"
    clip.write_bytes(b"x")
    info = track._source_provenance(clip)
    assert info["sha256"] == track._sha256(clip)
    assert "generator" not in info


def test_an_unreadable_record_does_not_take_the_run_down(tmp_path):
    clip = tmp_path / "demo_pan.mp4"
    clip.write_bytes(b"x")
    (tmp_path / "demo_pan.provenance.json").write_text("{not json", encoding="utf-8")
    assert track._source_provenance(clip)["sha256"] == track._sha256(clip)
