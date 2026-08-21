"""Unit tests for track.py's pure helpers.

track.py defers its cv2/ultralytics imports into the functions that use them,
so importing this module needs neither. track_colour does still call into
numpy, which is the one dependency guarded below.
"""

import json
from pathlib import Path

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
    assert info["matches_generator_record"] is True
    # The WHOLE record, not a selection from it. Copying a few fields across
    # meant the report carried a summary of the provenance rather than the
    # provenance, and which fields got copied was decided once and never
    # revisited - frames_sha256 was added to the sidecar and never reached the
    # report.
    assert info["generator_record"] == {
        "clip": {"sha256": digest, "fps": 15, "frames": 90},
        "generator": {"source_image": "0000295.jpg", "crop_fraction": 0.62},
    }


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


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("{not json", "not readable JSON"),
        ("[]", "top level must be a JSON object"),
        ('"a string"', "top level must be a JSON object"),
        ('{"generator": {}}', 'missing "clip"'),
        ('{"clip": {}}', 'missing "generator"'),
        ('{"clip": "not an object", "generator": {}}', '"clip" must be a JSON object'),
        ('{"clip": {}, "generator": []}', '"generator" must be a JSON object'),
    ],
)
def test_a_malformed_record_is_refused_before_any_inference(tmp_path, body, match):
    """Only a JSON *parse* failure was handled.

    A sidecar that parsed but was a list, or an object whose `clip` was a
    string, raised AttributeError from inside the report builder - after ninety
    frames had already been inferred. A minute of GPU time to be told a 2 KB
    JSON file is the wrong shape. read_source_record is called before the model
    loads, and says what is wrong and how to fix it.

    Refused, not ignored: silently dropping a malformed sidecar produces a
    report that looks provenance-free when it is actually provenance-broken,
    and those need different fixes.
    """
    clip = tmp_path / "demo_pan.mp4"
    clip.write_bytes(b"x")
    (tmp_path / "demo_pan.provenance.json").write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        track.read_source_record(clip)
    with pytest.raises(ValueError, match=match):
        track._source_provenance(clip)


def test_no_record_at_all_is_not_an_error(tmp_path):
    """Any video can be passed to --source, and most have no sidecar."""
    clip = tmp_path / "somebody_elses.mp4"
    clip.write_bytes(b"x")
    assert track.read_source_record(clip) is None


# --- device resolution ------------------------------------------------------ #


def _names(index):
    return {0: "NVIDIA GeForce RTX 2070 with Max-Q Design", 1: "NVIDIA A100"}[index]


def test_cpu_is_not_filed_under_a_gpu_that_happens_to_exist():
    """The report recorded get_device_name(0) whenever CUDA was merely
    *available*, so `--device cpu` on a machine with a GPU filed a CPU run
    under an RTX 2070 - a latency figure attributed to hardware that did not
    produce it, in a file whose purpose is to make the numbers checkable."""
    assert track.resolve_device("cpu", True, _names) == ("cpu", None)


def test_the_second_card_is_recorded_as_the_second_card():
    """`--device 1` recorded card 0's name."""
    assert track.resolve_device("1", True, _names) == ("cuda:1", "NVIDIA A100")
    assert track.resolve_device("cuda:1", True, _names) == ("cuda:1", "NVIDIA A100")


def test_the_default_device_resolves_to_the_first_card():
    for requested in ("0", "cuda:0", "cuda", ""):
        assert track.resolve_device(requested, True, _names) == (
            "cuda:0",
            "NVIDIA GeForce RTX 2070 with Max-Q Design",
        )


def test_a_cuda_request_with_no_cuda_records_the_cpu_that_actually_ran():
    """Ultralytics falls back to the CPU, and the CPU is where the numbers came
    from, so that is what the report has to say."""
    assert track.resolve_device("0", False, _names) == ("cpu", None)
    assert track.resolve_device("cpu", False, _names) == ("cpu", None)


def test_a_card_that_cannot_be_named_is_not_guessed_at():
    """Guessing card 0 is exactly how this went wrong. The device is still
    recorded; the name is left null."""

    def out_of_range(index):
        raise RuntimeError("invalid device ordinal")

    assert track.resolve_device("7", True, out_of_range) == ("cuda:7", None)


# --- the source is checked before the run, not after it --------------------- #


def _clip_with_record(tmp_path, body: bytes, recorded_sha: str | None = None):
    clip = tmp_path / "demo_pan.mp4"
    clip.write_bytes(body)
    (tmp_path / "demo_pan.provenance.json").write_text(
        json.dumps(
            {
                "clip": {"sha256": recorded_sha or track._sha256(clip)},
                "generator": {"source_image": "x.jpg"},
            }
        ),
        encoding="utf-8",
    )
    return clip


def test_a_matching_clip_passes_the_up_front_check(tmp_path):
    clip = _clip_with_record(tmp_path, b"the published clip")
    assert track.check_source_matches_record(clip) is True


def test_no_record_is_not_a_mismatch(tmp_path):
    """Any video can be passed to --source; most have no sidecar, and that is
    not something to stop a run over."""
    clip = tmp_path / "somebody_elses.mp4"
    clip.write_bytes(b"x")
    assert track.check_source_matches_record(clip) is True


def test_a_clip_that_does_not_match_its_record_stops_the_run(tmp_path):
    """The comparison used to happen where the report was assembled.

    A clip that did not match its record was therefore profiled in full -
    ninety frames, a minute of GPU time - and the disagreement showed up as
    `"matches_generator_record": false` in a file written at the end. That is a
    fact discovered after paying for it, and a latency figure published on the
    strength of it is a figure about footage nobody meant to measure.
    Digesting the file is milliseconds.
    """
    clip = _clip_with_record(tmp_path, b"a different clip", recorded_sha="0" * 64)
    with pytest.raises(SystemExit, match="not the footage that record is about"):
        track.check_source_matches_record(clip)


def test_the_mismatch_can_be_opted_into_explicitly(tmp_path):
    """Analysing a different input is legitimate; doing it silently is not."""
    clip = _clip_with_record(tmp_path, b"a different clip", recorded_sha="0" * 64)
    assert track.check_source_matches_record(clip, allow_mismatch=True) is False


# --- the output video is evidence too --------------------------------------- #


class _Cap:
    def __init__(self, frames, fps=15.0, opened=True):
        self._frames, self._fps, self._opened = list(frames), fps, opened
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        return (True, self._frames.pop(0)) if self._frames else (False, None)

    def get(self, prop):
        assert prop == track.CAP_PROP_FPS
        return self._fps

    def release(self):
        self.released = True


def _fake_frames(n, w=8, h=4, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_the_output_video_is_read_back():
    """`frames` in the profile counts what was PROCESSED. That is not evidence
    about the file: a VideoWriter accepts every frame and reports nothing, so a
    codec that drops the last few leaves a shorter video while the report says
    ninety."""
    cap = _Cap(_fake_frames(90))
    probe = track.probe_video(Path("track_out.mp4"), open_capture=lambda _: cap)
    assert probe["frames"] == 90
    assert (probe["width"], probe["height"]) == (8, 4)
    assert len(probe["decoded_frames_sha256"]) == 64
    assert cap.released


def test_an_undecodable_output_is_a_failure():
    with pytest.raises(SystemExit, match="cannot be decoded"):
        track.probe_video(
            Path("track_out.mp4"), open_capture=lambda _: _Cap([], opened=False)
        )


def test_the_output_digest_follows_the_pixels():
    a = track.probe_video(Path("a"), open_capture=lambda _: _Cap(_fake_frames(4)))
    b = track.probe_video(
        Path("b"), open_capture=lambda _: _Cap(_fake_frames(4, seed=9))
    )
    assert a["decoded_frames_sha256"] != b["decoded_frames_sha256"]


def test_a_truncated_output_video_is_refused():
    """`frames` in the report counts what was PROCESSED. A codec that drops the
    last few leaves a shorter video while the report still says ninety - and
    the GIF built from it then shows a clip that does not match the numbers
    printed beside it."""
    probe = track.probe_video(Path("x"), open_capture=lambda _: _Cap(_fake_frames(87)))
    with pytest.raises(SystemExit, match="frames: processed 90, file has 87"):
        track.check_output(probe, frames=90, width=8, height=4, name="track_out.mp4")


def test_an_output_at_the_wrong_size_is_refused():
    probe = track.probe_video(Path("x"), open_capture=lambda _: _Cap(_fake_frames(3)))
    with pytest.raises(SystemExit, match="width: processed 842, file has 8"):
        track.check_output(probe, frames=3, width=842, height=4, name="track_out.mp4")


def test_a_complete_output_passes():
    probe = track.probe_video(Path("x"), open_capture=lambda _: _Cap(_fake_frames(90)))
    track.check_output(probe, frames=90, width=8, height=4, name="track_out.mp4")


# --- the staged output leaves nothing behind -------------------------------- #


def _stub_modules(monkeypatch, *, frames_in, frames_back, opened=True):
    """Put stand-in cv2 / ultralytics modules in sys.modules.

    main() imports both inside itself, so this substitutes them for the
    duration of the call and the PRODUCTION code path is what runs - not a
    copy of it written in the test.
    """
    import sys
    import types

    import numpy as np

    made = {}

    class _Writer:
        def __init__(self, path):
            self.path = Path(path)
            self.path.write_bytes(b"staged output")

        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            made["released"] = True

    class _Cap:
        """Playback of the WRITTEN file, for probe_video's read-back."""

        def __init__(self):
            self._left = list(range(frames_back))

        def isOpened(self):
            return opened

        def read(self):
            if not self._left:
                return False, None
            self._left.pop()
            return True, np.zeros((4, 8, 3), dtype=np.uint8)

        def get(self, prop):
            return 15.0

        def release(self):
            pass

    cv2 = types.ModuleType("cv2")
    cv2.__version__ = "5.0.0-stub"
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.LINE_AA = 16
    cv2.INTER_AREA = 3
    cv2.CAP_PROP_FPS = track.CAP_PROP_FPS
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    cv2.VideoWriter_fourcc = lambda *a: 0
    cv2.VideoWriter = lambda path, *a: made.setdefault("writer", _Writer(path))
    cv2.VideoCapture = lambda _: _Cap()
    cv2.rectangle = cv2.putText = lambda *a, **k: None
    cv2.getTextSize = lambda *a, **k: ((10, 10), 0)

    def _frames(source):
        yield None, 15.0, (8, 4)
        for _ in range(frames_in):
            yield np.zeros((4, 8, 3), dtype=np.uint8), None, None

    monkeypatch.setattr(track, "frame_source", _frames)

    # torch too: CI deliberately does not install it, and _environment()
    # imports it to record the device. Without this the test passes locally and
    # fails in exactly the environment the bare-clone guarantee is about.
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-stub"
    torch.version = types.SimpleNamespace(cuda=None)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, get_device_name=lambda i: None
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    ultra = types.ModuleType("ultralytics")

    class _Res:
        def __init__(self):
            self.speed = {"preprocess": 1.0, "inference": 1.0, "postprocess": 1.0}
            self.boxes = None

    ultra.YOLO = lambda *a, **k: type(
        "M", (), {"track": lambda self, *a, **k: [_Res()], "names": {}}
    )()
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)
    return made


def _run(monkeypatch, tmp_path, **kw):
    import sys

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"a source clip")
    out = tmp_path / "track_out.mp4"
    made = _stub_modules(monkeypatch, **kw)
    monkeypatch.setattr(
        sys,
        "argv",
        ["track.py", "--weights", "w.pt", "--source", str(source), "--out", str(out)],
    )
    return source, out, made


def test_a_source_with_no_frames_leaves_no_staged_output(tmp_path, monkeypatch):
    """`no frames read` sat outside every cleanup guard.

    The writer had already created <out>.tmp.mp4 by then, so the exit left a
    partial mp4 under a plausible name with no report describing it - exactly
    the leak make_demo_clip.py has a test for.
    """
    _, out, _ = _run(monkeypatch, tmp_path, frames_in=0, frames_back=0)

    with pytest.raises(SystemExit, match="no frames read"):
        track.main()

    assert not (tmp_path / "track_out.tmp.mp4").exists(), "the staged output survived"
    assert not out.exists()


def test_an_undecodable_output_leaves_no_staged_output(tmp_path, monkeypatch):
    """probe_video raises when the written file will not decode, and that call
    was above the try that cleans up."""
    _, out, _ = _run(monkeypatch, tmp_path, frames_in=3, frames_back=0, opened=False)

    with pytest.raises(SystemExit, match="cannot be decoded"):
        track.main()

    assert not (tmp_path / "track_out.tmp.mp4").exists(), "the staged output survived"
    assert not out.exists()


def test_a_truncated_output_leaves_the_previous_one_untouched(tmp_path, monkeypatch):
    """The control: this path was already guarded, and must stay that way."""
    _, out, _ = _run(monkeypatch, tmp_path, frames_in=5, frames_back=3)
    out.write_bytes(b"the previous run's output")

    with pytest.raises(SystemExit, match="processed 5, file has 3"):
        track.main()

    assert not (tmp_path / "track_out.tmp.mp4").exists()
    assert out.read_bytes() == b"the previous run's output"


def test_a_failing_report_write_leaves_no_staged_output(tmp_path, monkeypatch):
    """The guard has to cover the report write and the publish too.

    Moving the publish to after the report write - so the video is never newer
    than the report describing it - would otherwise have opened a third leak in
    the same place the first two were just closed.
    """
    _, out, _ = _run(monkeypatch, tmp_path, frames_in=4, frames_back=4)

    real = Path.write_text

    def boom(self, *a, **k):
        if self.name == "tracking.json":
            raise OSError("no space left on device")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", boom)

    with pytest.raises(OSError, match="no space"):
        track.main()

    assert not (tmp_path / "track_out.tmp.mp4").exists(), "the staged output survived"
    assert not out.exists(), "the video was published without a report"
