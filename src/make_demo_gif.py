"""Turn reports/track_out.mp4 into the README's animated GIF.

WHY THIS EXISTS
---------------
`reports/tracking_demo.gif` used to be produced by hand with an ffmpeg
invocation that lived in somebody's shell history. That makes it the one
artefact in the repo with no stated relationship to anything: it looks like
evidence of a tracking run, and nothing said which run, or whether the video it
came from is the video `reports/tracking.json` describes.

So it is a script, and it records the sha256 of the mp4 it read. Comparing that
against `output.sha256` in tracking.json is what turns the GIF from an
illustration into part of the same evidence chain.

Deliberately no ffmpeg dependency: OpenCV is already required, and Pillow -
which is pulled in by ultralytics - writes GIFs. One less thing that has to be
on PATH for the repo to rebuild its own figures.

Usage
-----
    python src/make_demo_gif.py
    python src/make_demo_gif.py --expected-source-sha256 <from tracking.json>
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ranged_int(lo: int, hi: int | None = None):
    """An argparse type: an int in [lo, hi], refused at parse time.

    Unchecked, a bad value either failed after the whole clip had been read,
    with a traceback about something else, or was accepted silently and then
    written into the provenance record as if it meant something.
    """

    def parse(value):
        try:
            n = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
        if n < lo or (hi is not None and n > hi):
            bound = f">= {lo}" if hi is None else f"in [{lo}, {hi}]"
            raise argparse.ArgumentTypeError(f"must be {bound}, got {n}")
        return n

    return parse


def _sha256_arg(value: str) -> str:
    """A sha256 digest as argparse input: trimmed, lower-cased, and checked.

    Compared as typed, the same digest pasted in upper case or with a trailing
    space was reported as a different frame or a different run.
    """
    digest = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise argparse.ArgumentTypeError(f"not a sha256 hex digest: {value!r}")
    return digest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--source", type=Path, default=PROJECT_ROOT / "reports" / "track_out.mp4"
    )
    p.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "reports" / "tracking_demo.gif"
    )
    p.add_argument(
        "--fps",
        # GIF delays are whole hundredths of a second, so above 50 the rate
        # asked for is not the rate played.
        type=_ranged_int(1, 50),
        default=10,
        help="GIF frame rate. Lower keeps the file small. With --every 3 over "
        "the 15 fps clip, 10 plays it back at twice real time.",
    )
    p.add_argument(
        "--width",
        type=_ranged_int(2),
        default=480,
        help="Output width; height follows the aspect. The mp4 is "
        "there for detail; this is a README figure and its size "
        "is the dominant cost - 640 px is 2.9 MB against 1.7 MB "
        "at 480.",
    )
    p.add_argument(
        "--every",
        type=_ranged_int(1),
        # 3, the value the committed GIF was built with: at the old default of
        # 2, the documented command did not reproduce the README's figure.
        default=3,
        help="Keep one frame in N. A GIF of every frame of a 90-frame "
        "clip is several megabytes for no extra information.",
    )
    p.add_argument(
        "--colours",
        type=_ranged_int(2, 256),
        default=32,
        help="Palette size. GIF is indexed, so unquantised truecolour "
        "frames make Pillow choose a palette per frame and the file "
        "balloons: 7.9 MB unquantised against 1.7 MB at 32 colours. "
        "Measured, along with width, as the two levers that matter - "
        "the dither mode changes nothing here.",
    )
    p.add_argument(
        "--expected-source-sha256",
        type=_sha256_arg,
        default=None,
        help="Refuse to run unless track_out.mp4 has this digest. Take "
        "it from `output.sha256` in reports/tracking.json - that is "
        "what ties this GIF to a particular tracking run.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    import cv2
    import PIL
    from PIL import Image

    if not args.source.is_file():
        raise SystemExit(f"{args.source} not found - run src/track.py first")

    digest = sha256(args.source)
    if args.expected_source_sha256 and digest != args.expected_source_sha256:
        raise SystemExit(
            f"{args.source.name} has sha256 {digest}, not "
            f"{args.expected_source_sha256}. This GIF would show a different "
            f"run than the one you meant."
        )

    cap = cv2.VideoCapture(str(args.source))
    try:
        if not cap.isOpened():
            raise SystemExit(f"cannot decode {args.source}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frames, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.every == 0:
                h, w = frame.shape[:2]
                height = int(h * args.width / w)
                # interpolation= by KEYWORD. cv2.resize's third positional
                # parameter is `dst`, not `interpolation`, so passing the
                # constant positionally handed it to `dst`, where OpenCV
                # discarded it without complaint and used the default
                # INTER_LINEAR. Bilinear point-sampling on a 1.75x
                # reduction aliases exactly the few-pixel objects and
                # 2-px track boxes this figure exists to show; INTER_AREA
                # averages the source pixels instead.
                small = cv2.resize(
                    frame, (args.width, height), interpolation=cv2.INTER_AREA
                )
                frames.append(
                    Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).quantize(
                        colors=args.colours, method=Image.Quantize.MEDIANCUT
                    )
                )
            i += 1
    finally:
        cap.release()

    if not frames:
        raise SystemExit(f"{args.source} decoded to no frames")

    staged = args.out.with_name(f"{args.out.stem}.tmp{args.out.suffix}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        frames[0].save(
            staged,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / args.fps),
            loop=0,
            optimize=True,
        )
        staged.replace(args.out)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise

    # Which video this came from, so the GIF is checkable rather than
    # decorative. tracking.json records the same digest under output.sha256.
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "gif": {
                    "path": args.out.name,
                    "sha256": sha256(args.out),
                    "frames": len(frames),
                    "fps": args.fps,
                    "width": args.width,
                    # Everything else that decides the bytes, so the GIF can
                    # be rebuilt rather than only recognised.
                    "colours": args.colours,
                    "pillow": PIL.__version__,
                },
                "source": {
                    "path": args.source.name,
                    "sha256": digest,
                    "kept_every": args.every,
                    "source_frames": i,
                    "source_fps": source_fps,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}  ({len(frames)} of {i} frames @ {args.fps} fps)")
    print(f"from  {args.source.name}  sha256 {digest}")


if __name__ == "__main__":
    main()
