"""Tests for the demo clip's publish path.

The clip is the footage every tracking number is measured on, and it is build
output that is not in the repo - so the only thing tying a published number to
a reproducible input is this script's provenance record and the checks around
it. Those checks had none of their own.

`probe_clip` takes its capture factory as an argument and the rest is plain
filesystem work, so all of this runs without OpenCV - which CI deliberately
does not install.
"""

import json
import pathlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the fake frames are numpy arrays")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import make_demo_clip as mdc


class FakeCapture:
    """Stands in for cv2.VideoCapture: yields frames, then stops."""

    def __init__(self, frames, fps=15.0, opened=True):
        self._frames = list(frames)
        self._fps = fps
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def get(self, prop):
        assert prop == mdc.CAP_PROP_FPS
        return self._fps

    def release(self):
        self.released = True


def _frames(n, w=8, h=4, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


# --- reading the file back -------------------------------------------------- #


def test_the_probe_reports_what_is_in_the_file():
    """Everything else in the script describes what was handed to the encoder.
    That is not the same thing, and only reading it back tells them apart."""
    cap = FakeCapture(_frames(90), fps=15.0)
    probe = mdc.probe_clip(Path("x.mp4"), open_capture=lambda _: cap)

    assert probe["frames"] == 90
    assert (probe["width"], probe["height"]) == (8, 4)
    assert probe["fps"] == 15.0
    assert len(probe["decoded_frames_sha256"]) == 64
    assert cap.released


def test_the_decoded_digest_is_a_function_of_the_pixels():
    same = mdc.probe_clip(Path("a"), open_capture=lambda _: FakeCapture(_frames(5)))
    again = mdc.probe_clip(Path("a"), open_capture=lambda _: FakeCapture(_frames(5)))
    other = mdc.probe_clip(
        Path("b"), open_capture=lambda _: FakeCapture(_frames(5, seed=1))
    )
    assert same["decoded_frames_sha256"] == again["decoded_frames_sha256"]
    assert same["decoded_frames_sha256"] != other["decoded_frames_sha256"]


def test_a_file_that_will_not_decode_is_a_failure_not_a_zero_frame_clip():
    with pytest.raises(SystemExit, match="cannot be decoded"):
        mdc.probe_clip(
            Path("x.mp4"), open_capture=lambda _: FakeCapture([], opened=False)
        )


def test_an_encoder_that_silently_drops_frames_is_caught():
    """A VideoWriter accepts every frame and reports nothing. A codec that
    drops the last few leaves a shorter clip while the script prints the count
    it INTENDED, and the tracking profile then reports on however many frames
    it found with nothing saying some were missing."""
    probe = mdc.probe_clip(Path("x"), open_capture=lambda _: FakeCapture(_frames(87)))
    with pytest.raises(SystemExit, match="frames: wrote 90, file has 87"):
        mdc.check_encoded(probe, frames=90, width=8, height=4)


def test_wrong_dimensions_are_caught():
    probe = mdc.probe_clip(Path("x"), open_capture=lambda _: FakeCapture(_frames(2)))
    with pytest.raises(SystemExit, match="width: wrote 640, file has 8"):
        mdc.check_encoded(probe, frames=2, width=640, height=4)


def test_a_healthy_clip_passes():
    probe = mdc.probe_clip(Path("x"), open_capture=lambda _: FakeCapture(_frames(90)))
    mdc.check_encoded(probe, frames=90, width=8, height=4)


# --- the staged publish ----------------------------------------------------- #


def _staged_pair(tmp_path):
    out = tmp_path / "demo_pan.mp4"
    out.write_bytes(b"the published clip")
    record = tmp_path / "demo_pan.provenance.json"
    record.write_text(json.dumps({"clip": {"sha256": "published"}}), encoding="utf-8")
    staged = tmp_path / "demo_pan.tmp.mp4"
    staged.write_bytes(b"the new clip")
    return staged, out, record


def test_matching_digests_replace_the_published_clip(tmp_path):
    staged, out, _ = _staged_pair(tmp_path)
    mdc.publish_staged(staged, out, [("clip", "abc", "abc"), ("decoded", "d", "d")])
    assert out.read_bytes() == b"the new clip"
    assert not staged.exists()


def test_an_unchecked_digest_does_not_block_the_publish(tmp_path):
    """`expected` of None means "not asserted", not "must be empty"."""
    staged, out, _ = _staged_pair(tmp_path)
    mdc.publish_staged(staged, out, [("clip", "abc", None)])
    assert out.read_bytes() == b"the new clip"


def test_a_mismatch_leaves_the_published_clip_and_its_record_untouched(tmp_path):
    """The whole reason for staging.

    Writing straight to --out destroyed the published clip before anything had
    established that the replacement was the same clip - and since the sidecar
    was then written from whatever had just been produced, it always agreed
    with itself and "verified" meant nothing.
    """
    staged, out, record = _staged_pair(tmp_path)
    with pytest.raises(SystemExit, match="untouched"):
        mdc.publish_staged(staged, out, [("clip", "actual", "expected")])

    assert out.read_bytes() == b"the published clip"
    assert (
        json.loads(record.read_text(encoding="utf-8"))["clip"]["sha256"] == "published"
    )
    assert not staged.exists(), "the rejected clip was left behind"


def test_the_first_failing_digest_stops_it(tmp_path):
    staged, out, _ = _staged_pair(tmp_path)
    with pytest.raises(SystemExit, match="pre-encode frame sha256 is x"):
        mdc.publish_staged(
            staged,
            out,
            [("clip", "a", "a"), ("pre-encode frame", "x", "y"), ("decoded", "b", "c")],
        )
    assert out.read_bytes() == b"the published clip"


# --- nothing is left behind ------------------------------------------------- #


class _FakeWriter:
    """A VideoWriter that fails part way through encoding."""

    def __init__(self, path, fail_on=None):
        self.path = pathlib.Path(path)
        self.fail_on = fail_on
        self.written = 0
        self.released = False
        self.path.write_bytes(b"")

    def isOpened(self):
        return True

    def write(self, frame):
        self.written += 1
        if self.fail_on and self.written >= self.fail_on:
            raise RuntimeError(f"the encoder gave up on frame {self.written}")
        self.path.write_bytes(b"x" * self.written)

    def release(self):
        self.released = True


def _fake_cv2(monkeypatch, *, fail_on=None, frames_back=None):
    """Put a stand-in cv2 in sys.modules so main() runs without OpenCV.

    main() imports cv2 inside itself, so this substitutes the real module for
    the duration of the call - which means the production code path is what
    runs, rather than a copy of it written in the test.
    """
    import types

    made = {}
    module = types.ModuleType("cv2")
    module.VideoWriter_fourcc = lambda *a: 0
    module.CAP_PROP_FPS = mdc.CAP_PROP_FPS
    module.imread = lambda _: np.zeros((400, 600, 3), dtype=np.uint8)

    def video_writer(path, fourcc, fps, size):
        made["writer"] = _FakeWriter(path, fail_on)
        return made["writer"]

    module.VideoWriter = video_writer
    module.VideoCapture = lambda _: FakeCapture(
        _frames(frames_back if frames_back is not None else 0)
    )
    monkeypatch.setitem(sys.modules, "cv2", module)
    return made


def test_an_interrupted_write_leaves_no_temporary_clip(tmp_path, monkeypatch):
    """The temp file was released but never deleted.

    try/finally around the frame loop closed the writer's handle and stopped
    there, so a failure part way through encoding left demo_pan.tmp.mp4 sitting
    in reports/ - a partial clip with a plausible name, for the next person to
    wonder about. Everything from opening the writer to the publish is now
    inside the guard, and any exception removes it.
    """
    out = tmp_path / "demo_pan.mp4"
    source = tmp_path / "frame.jpg"
    source.write_bytes(b"not really a jpg; imread is stubbed")
    made = _fake_cv2(monkeypatch, fail_on=40)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_demo_clip.py",
            "--source-image",
            str(source),
            "--frames",
            "90",
            "--out",
            str(out),
        ],
    )

    with pytest.raises(RuntimeError, match="frame 40"):
        mdc.main()

    assert made["writer"].released, "the writer handle was left open"
    assert not (tmp_path / "demo_pan.tmp.mp4").exists(), "the temp clip survived"
    assert not out.exists()
    assert not (tmp_path / "demo_pan.provenance.json").exists()


def test_a_run_that_encodes_the_wrong_frame_count_leaves_nothing_behind(
    tmp_path, monkeypatch
):
    """The readback check fires after the writer has closed cleanly, so this is
    the case where nothing raised until the file was examined - and the temp
    clip has to go the same way."""
    out = tmp_path / "demo_pan.mp4"
    source = tmp_path / "frame.jpg"
    source.write_bytes(b"stub")
    _fake_cv2(monkeypatch, frames_back=87)  # the encoder wrote 90, kept 87
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_demo_clip.py",
            "--source-image",
            str(source),
            "--frames",
            "90",
            "--out",
            str(out),
        ],
    )

    with pytest.raises(SystemExit, match="frames: wrote 90, file has 87"):
        mdc.main()

    assert not (tmp_path / "demo_pan.tmp.mp4").exists()
    assert not out.exists()
