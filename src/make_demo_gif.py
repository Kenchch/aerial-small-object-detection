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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--source", type=Path, default=PROJECT_ROOT / "reports" / "track_out.mp4"
    )
    p.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "reports" / "tracking_demo.gif"
    )
    p.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF frame rate. Lower keeps the file small.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=480,
        help="Output width; height follows the aspect. The mp4 is "
        "there for detail; this is a README figure and its size "
        "is the dominant cost - 640 px is 2.9 MB against 1.7 MB "
        "at 480.",
    )
    p.add_argument(
        "--every",
        type=int,
        default=2,
        help="Keep one frame in N. A GIF of every frame of a 90-frame "
        "clip is several megabytes for no extra information.",
    )
    p.add_argument(
        "--colours",
        type=int,
        default=32,
        help="Palette size. GIF is indexed, so unquantised truecolour "
        "frames make Pillow choose a palette per frame and the file "
        "balloons: 7.9 MB unquantised against 1.7 MB at 32 colours. "
        "Measured, along with width, as the two levers that matter - "
        "the dither mode changes nothing here.",
    )
    p.add_argument(
        "--expected-source-sha256",
        default=None,
        help="Refuse to run unless track_out.mp4 has this digest. Take "
        "it from `output.sha256` in reports/tracking.json - that is "
        "what ties this GIF to a particular tracking run.",
    )
    args = p.parse_args()

    import cv2
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
        frames, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.every == 0:
                h, w = frame.shape[:2]
                height = int(h * args.width / w)
                small = cv2.resize(frame, (args.width, height), cv2.INTER_AREA)
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
                },
                "source": {
                    "path": args.source.name,
                    "sha256": digest,
                    "kept_every": args.every,
                    "source_frames": i,
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
