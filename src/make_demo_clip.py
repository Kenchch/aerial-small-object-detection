"""
Build a pipeline-validation clip by panning a crop window across one real
VisDrone frame.

WHY THIS EXISTS, AND WHAT IT IS NOT
-----------------------------------
VisDrone2019-DET is sampled from video, but the sampling interval is 200 frames
(~8 s) — verified directly from the filenames, whose second field steps
1, 201, 401, 601 … Consecutive frames are simply not in this dataset, so no
honest tracking demo can be built from it. The MOT subset that does contain
them is a separate download not mirrored where the rest of this project's data
comes from.

So this generates synthetic *camera* motion over real imagery. That is a
deliberate, limited substitute:

  It DOES exercise      detection stability frame to frame, ByteTrack's
                        association step, id persistence under translation and
                        scale change, and the full decode→infer→track→encode
                        deployment path with real latency.

  It does NOT exercise  independently moving objects, occlusion between them,
                        re-identification after a full occlusion, or motion
                        blur. Tracking metrics from this clip measure the
                        pipeline, not tracking accuracy — and must not be
                        quoted as MOTA/IDF1-style results.

Camera translation is the dominant motion in most drone footage, so this is a
meaningful smoke test of the deployment path. It is not a tracking benchmark.

Usage
-----
    python src/make_demo_clip.py --frames 90 --out reports/demo_pan.mp4
"""

import argparse
import hashlib
import json
from pathlib import Path

# cv2.CAP_PROP_FPS, spelled out so probe_clip's signature and its tests do not
# drag OpenCV in. The value is fixed by the OpenCV ABI.
CAP_PROP_FPS = 5

# cv2/numpy/ultralytics are imported inside the functions that use them, after
# parse_args(). At module scope they pin `--help` to a fully provisioned
# environment, which makes the CLI undiscoverable exactly when someone is
# trying to find out what it needs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """Digest a file in chunks - the source frames are a few MB, the clip more."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance_path(out: Path) -> Path:
    """Where this clip's provenance sits. Named from the clip, so the pair
    cannot drift apart, and so track.py can find it without being told."""
    return out.with_suffix(".provenance.json")


def probe_clip(path: Path, open_capture=None) -> dict:
    """Decode the finished file and report what is ACTUALLY in it.

    Everything else here describes what was handed to the encoder. That is not
    the same thing: a VideoWriter can accept every frame and write fewer, or
    write a file that will not decode, and both look like success from the
    writing side. Reading it back is the only way to find out, and it is one
    pass over a few megabytes.

    Returns frames, width, height, fps and a digest of the DECODED pixels -
    which is what any consumer of the published clip can recompute for
    themselves, without having the generator or its source frame.

    `open_capture` is injected so this is testable without OpenCV, which CI
    deliberately does not install.
    """
    import hashlib as _hashlib

    if open_capture is None:  # pragma: no cover - the real path needs cv2
        import cv2

        open_capture = cv2.VideoCapture

    cap = open_capture(str(path))
    try:
        if not cap.isOpened():
            raise SystemExit(f"{path} was written but cannot be decoded")
        digest = _hashlib.sha256()
        n = 0
        width = height = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            height, width = frame.shape[:2]
            digest.update(memoryview(frame).tobytes())
        fps = cap.get(CAP_PROP_FPS)
    finally:
        cap.release()
    return {
        "frames": n,
        "width": width,
        "height": height,
        "fps": round(fps, 3) if fps else None,
        "decoded_frames_sha256": digest.hexdigest(),
    }


def check_encoded(probe: dict, *, frames: int, width: int, height: int) -> None:
    """The file has to contain what we thought we wrote.

    A VideoWriter accepts frames and reports nothing; a codec that drops the
    last few, or a container the muxer finalised badly, produces a shorter or
    unreadable clip while the script prints the frame count it INTENDED. The
    tracking profile downstream then reports on however many frames it found,
    with no indication that some were missing.

    fps is deliberately not compared: containers store it as a rational and
    round-trip 15 as 15.0 or 14.999, so an equality check there fails on
    healthy files.
    """
    mismatches = [
        f"{name}: wrote {expected}, file has {probe[name]}"
        for name, expected in (
            ("frames", frames),
            ("width", width),
            ("height", height),
        )
        if probe[name] != expected
    ]
    if mismatches:
        raise SystemExit(
            "the encoded clip does not match what was written - "
            + "; ".join(mismatches)
        )


def publish_staged(staged: Path, out: Path, checks) -> None:
    """Move the staged clip into place, but only if every digest agrees.

    `checks` is (label, actual, expected); an expected of None is not checked.

    This is the whole point of staging. Writing straight to --out destroyed the
    published clip before anything had established that the replacement was the
    same clip - and since the sidecar was then written from whatever had just
    been produced, it always agreed with itself and "verified" meant nothing.
    """
    for label, actual, expected in checks:
        if expected and actual != expected:
            staged.unlink(missing_ok=True)
            raise SystemExit(
                f"{label} sha256 is {actual}, not {expected}. The existing "
                f"{out.name} and its provenance are untouched."
            )
    staged.replace(out)


def densest_val_image() -> tuple[Path, int]:
    """Pick the val frame with the most labelled objects — the hardest case
    for association, and the most legible demo."""
    from ultralytics.utils import SETTINGS

    root = Path(SETTINGS["datasets_dir"]) / "VisDrone"
    labels = root / "labels" / "val"
    images = root / "images" / "val"
    best, best_n = None, -1
    for lbl in labels.glob("*.txt"):
        n = sum(1 for line in lbl.read_text().splitlines() if line.strip())
        if n > best_n:
            img = images / (lbl.stem + ".jpg")
            if img.exists():
                best, best_n = img, n
    if best is None:
        raise SystemExit("no val images found — has the dataset been downloaded?")
    return best, best_n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "reports" / "demo_pan.mp4"
    )
    p.add_argument(
        "--source-image",
        type=Path,
        default=None,
        help="Use this frame instead of searching the val split for "
        "the densest one. The search picks by label count against "
        "whatever dataset revision is installed, so naming the file "
        "is what makes a clip exactly reproducible - see the sha256 "
        "in the clip's .provenance.json.",
    )
    p.add_argument(
        "--expected-source-sha256",
        default=None,
        help="Refuse to run unless the source frame has this digest. "
        "Pair it with --source-image to reproduce a published clip "
        "byte for byte, or to find out that you cannot.",
    )
    p.add_argument(
        "--expected-clip-sha256",
        default=None,
        help="Refuse to publish the result unless the finished mp4 has "
        "this digest. Without it the sidecar is written from the "
        "clip that was just made, so it always agrees with itself "
        "and 'verified' means nothing.",
    )
    p.add_argument(
        "--expected-decoded-frames-sha256",
        default=None,
        help="Same, over the frames DECODED BACK from the finished "
        "file. Independent of how the container was muxed, and "
        "the one digest a consumer can recompute from the "
        "published clip alone.",
    )
    p.add_argument(
        "--expected-pre-encode-frames-sha256",
        default=None,
        help="Same, over the frames handed TO the encoder. This one "
        "is codec-independent: it says the generator produced the "
        "same pixels, whatever the encoder then did with them.",
    )
    p.add_argument(
        "--crop",
        type=float,
        default=0.62,
        help="Crop window as a fraction of the source frame. Smaller "
        "means more apparent motion and more objects entering "
        "and leaving — a harder association test.",
    )
    args = p.parse_args()

    import cv2
    import numpy as np

    if args.source_image is not None:
        src_path = args.source_image
        if not src_path.is_file():
            raise SystemExit(f"{src_path} does not exist")
        # <root>/images/<split>/x.jpg  ->  <root>/labels/<split>/x.txt.
        # None, not a sentinel count: an arbitrary --source-image need not sit
        # in a VisDrone tree at all, and "we did not find labels" is different
        # from "there were none".
        label = (
            src_path.parent.parent.parent
            / "labels"
            / src_path.parent.name
            / (src_path.stem + ".txt")
        )
        n_obj = (
            sum(1 for line in label.read_text().splitlines() if line.strip())
            if label.is_file()
            else None
        )
    else:
        src_path, n_obj = densest_val_image()

    if args.expected_source_sha256:
        # Checked BEFORE the clip is built, so a mismatch costs nothing and is
        # unambiguous. "Re-run the script" is not a way to reproduce a clip if
        # the frame it starts from can quietly differ.
        actual = sha256(src_path)
        if actual != args.expected_source_sha256:
            raise SystemExit(
                f"{src_path.name} has sha256 {actual}, not "
                f"{args.expected_source_sha256}. This is a different frame, so "
                f"the clip would be a different clip."
            )

    img = cv2.imread(str(src_path))
    if img is None:
        raise SystemExit(f"cannot decode {src_path}")
    H, W = img.shape[:2]
    print(
        f"source : {src_path.name}  ({W}x{H}, "
        f"{'unknown' if n_obj is None else n_obj} labelled objects)"
    )

    cw, ch = int(W * args.crop), int(H * args.crop)
    # Even output dimensions keep the H.264/mp4v encoder happy.
    ow, oh = cw - (cw % 2), ch - (ch % 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Written to a temporary file and moved into place only once it has been
    # checked. Writing straight to --out destroys the published clip before
    # anything has established that the replacement is the same clip.
    #
    # The suffix is kept: OpenCV chooses its container from the file
    # EXTENSION, so "demo_pan.mp4.tmp" simply fails to open, with a message
    # about the mp4v codec that points nowhere near the real cause.
    staged = args.out.with_name(f"{args.out.stem}.tmp{args.out.suffix}")
    writer = cv2.VideoWriter(
        str(staged), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (ow, oh)
    )
    # An unopened writer makes write() a silent no-op -- this script would
    # otherwise print "wrote ... (N frames)" and exit 0 with no output file,
    # and the failure would only surface in track.py, the next command in
    # the README's tracking walkthrough, pointing at the wrong script.
    if not writer.isOpened():
        # Release and clear up before raising: an unopened writer still holds a
        # handle, and the zero-byte file it left behind is the next run's
        # confusing leftover.
        writer.release()
        staged.unlink(missing_ok=True)
        raise SystemExit(
            f"cannot open VideoWriter for {staged} "
            f"(mp4v codec unavailable in this OpenCV build?)"
        )

    max_dx, max_dy = W - cw, H - ch
    # Hashed on the way IN, before the encoder sees them. This is the
    # codec-independent identity: it says the generator produced the same
    # pixels, whatever mp4v then did with them. It is NOT the same thing as
    # the frames you get back out - see probe_clip.
    pre_encode = hashlib.sha256()
    try:
        try:
            for i in range(args.frames):
                t = i / max(1, args.frames - 1)
                # Ease-in-out so the pan accelerates and settles like real
                # gimbal motion rather than starting and stopping instantly.
                e = 0.5 - 0.5 * np.cos(np.pi * t)
                x = int(e * max_dx)
                y = int((0.5 - 0.5 * np.cos(2 * np.pi * t)) * max_dy)
                crop = img[y : y + ch, x : x + cw][:oh, :ow]
                pre_encode.update(np.ascontiguousarray(crop).tobytes())
                writer.write(crop)
        finally:
            # Released before anything tries to touch the file: the writer
            # holds an OS handle and, on Windows, a lock, so unlinking first
            # fails for a second, unrelated-looking reason.
            writer.release()

        # Read the file back. Everything above describes what was handed to the
        # encoder, which is not what is in the file - a VideoWriter can accept
        # every frame and write fewer, and that looks like success from the
        # writing side. The tracking profile downstream would then report on
        # however many frames it found, with nothing saying some were missing.
        probe = probe_clip(staged)
        check_encoded(probe, frames=args.frames, width=ow, height=oh)

        publish_staged(
            staged,
            args.out,
            (
                ("clip", sha256(staged), args.expected_clip_sha256),
                (
                    "decoded frame",
                    probe["decoded_frames_sha256"],
                    args.expected_decoded_frames_sha256,
                ),
                (
                    "pre-encode frame",
                    pre_encode.hexdigest(),
                    args.expected_pre_encode_frames_sha256,
                ),
            ),
        )
    except BaseException:
        # Any failure at all leaves nothing behind. Without this the aborted
        # run left demo_pan.tmp.mp4 sitting in reports/ - a partial clip with a
        # plausible name, for the next person to wonder about.
        staged.unlink(missing_ok=True)
        raise

    clip_sha = sha256(args.out)

    # The clip is not in the repo - it is build output - so "re-run
    # make_demo_clip.py" is not by itself a way to get the same clip back.
    # densest_val_image() picks by label count, which depends on which dataset
    # revision is on the machine, and the crop, pan and fps are all arguments.
    # Without this file, a tracking number measured on one clip and a clip
    # regenerated later are two different things that look identical.
    #
    # track.py embeds this into reports/tracking.json, so the profile carries
    # the identity of the footage it profiled.
    provenance = {
        "clip": {
            "path": args.out.name,
            "sha256": clip_sha,
            # Two frame digests, because they answer different questions.
            # decoded_ is what a consumer can recompute from the published
            # clip alone, and survives a remux. pre_encode_ is what the
            # generator produced before the codec touched it, and survives a
            # change of codec. Neither survives both.
            "decoded_frames_sha256": probe["decoded_frames_sha256"],
            "pre_encode_frames_sha256": pre_encode.hexdigest(),
            # Read back from the finished file, not assumed from the arguments.
            "frames": probe["frames"],
            "fps": args.fps,
            "container_fps": probe["fps"],
            "width": probe["width"],
            "height": probe["height"],
        },
        "generator": {
            "script": "src/make_demo_clip.py",
            "source_image": src_path.name,
            "source_sha256": sha256(src_path),
            "source_size": [W, H],
            "source_labelled_objects": n_obj,
            "selection": (
                "named with --source-image"
                if args.source_image is not None
                else "densest val frame by labelled object count"
            ),
            # The dataset's NAME and size, not its path. This file is
            # committed, and where the dataset happens to sit on one machine is
            # not reproducibility information - it is someone's drive letter.
            # The val-split image count is the version signal that matters:
            # the frame is chosen by scanning that split, so a different count
            # means a different selection is possible.
            "dataset": src_path.parent.parent.parent.name,
            "dataset_split": src_path.parent.name,
            "dataset_split_images": len(list(src_path.parent.glob("*.jpg"))),
            "crop_fraction": args.crop,
            "crop_size": [cw, ch],
            "pan": {
                "horizontal": "ease-in-out, 0 -> W-cw, one pass",
                "vertical": "raised cosine, one full sweep, 0 -> H-ch -> 0",
                "max_dx": max_dx,
                "max_dy": max_dy,
            },
        },
        "opencv": cv2.__version__,
    }
    provenance_path(args.out).write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    print(f"wrote  : {args.out}  ({args.frames} frames @ {args.fps} fps, {ow}x{oh})")
    print(f"         {provenance_path(args.out).name}")
    print(f"clip   : sha256 {clip_sha}")
    print(f"decoded: sha256 {probe['decoded_frames_sha256']}")
    print(f"verified {probe['frames']} frames, {probe['width']}x{probe['height']}")
    print(
        "\nNOTE: synthetic camera motion over a real frame. Validates the "
        "pipeline;\n      it is not a tracking accuracy benchmark. See module "
        "docstring."
    )


if __name__ == "__main__":
    main()
