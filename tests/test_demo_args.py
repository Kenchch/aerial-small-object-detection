"""The demo scripts' arguments: checked at parse time, and defaulting to the
settings the committed media were built with."""

import argparse
import json
from pathlib import Path

import pytest

import make_demo_clip
import make_demo_gif

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "ef545c205123b91c1bee517613381b3cd87ab66930186427742f6c4e73b8e87b"


def test_the_gif_defaults_rebuild_the_committed_gif():
    """--every defaulted to 2 while the committed GIF kept every 3rd frame, so
    the documented command did not reproduce the README's figure."""
    recorded = json.loads(
        (ROOT / "reports/tracking_demo.provenance.json").read_text(encoding="utf-8")
    )
    args = make_demo_gif.build_parser().parse_args([])
    assert args.every == recorded["source"]["kept_every"]
    assert args.fps == recorded["gif"]["fps"]
    assert args.width == recorded["gif"]["width"]


@pytest.mark.parametrize("module", [make_demo_clip, make_demo_gif])
@pytest.mark.parametrize(
    "typed", [DIGEST.upper(), f"  {DIGEST}\n", DIGEST], ids=["upper", "padded", "as-is"]
)
def test_a_digest_is_normalised_before_it_is_compared(module, typed):
    """Upper case or a stray space made the same digest read as a mismatch."""
    assert module._sha256_arg(typed) == DIGEST


@pytest.mark.parametrize("module", [make_demo_clip, make_demo_gif])
def test_something_that_is_not_a_digest_is_refused_at_parse_time(module):
    with pytest.raises(argparse.ArgumentTypeError, match="not a sha256"):
        module._sha256_arg("abc123")


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--every", "0"], "--every: must be >= 1"),
        (["--fps", "0"], "--fps: must be in"),
        (["--fps", "60"], "--fps: must be in"),
        (["--width", "1"], "--width: must be >= 2"),
        (["--colours", "1"], "--colours: must be in"),
        (["--colours", "300"], "--colours: must be in"),
    ],
)
def test_gif_settings_that_cannot_work_are_refused(argv, match, capsys):
    with pytest.raises(SystemExit):
        make_demo_gif.build_parser().parse_args(argv)
    assert match in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--frames", "0"], "--frames: must be >= 1"),
        (["--fps", "0"], "--fps: must be in"),
    ],
)
def test_clip_settings_that_cannot_work_are_refused(argv, match, capsys):
    with pytest.raises(SystemExit):
        make_demo_clip.build_parser().parse_args(argv)
    assert match in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--crop", "0"], "--crop must be in (0, 1]"),
        (["--crop", "1.5"], "--crop must be in (0, 1]"),
        (["--out", "clip.avi"], "--out must end in .mp4"),
    ],
)
def test_a_crop_or_container_that_cannot_work_stops_before_any_work(
    argv, match, monkeypatch, capsys
):
    import sys

    monkeypatch.setattr(sys, "argv", ["make_demo_clip.py", *argv])
    with pytest.raises(SystemExit):
        make_demo_clip.main()
    assert match in capsys.readouterr().err
