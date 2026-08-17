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
from pathlib import Path

# cv2/numpy/ultralytics are imported inside the functions that use them, after
# parse_args(). At module scope they pin `--help` to a fully provisioned
# environment, which makes the CLI undiscoverable exactly when someone is
# trying to find out what it needs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    src_path, n_obj = densest_val_image()
    img = cv2.imread(str(src_path))
    H, W = img.shape[:2]
    print(f"source : {src_path.name}  ({W}x{H}, {n_obj} labelled objects)")

    cw, ch = int(W * args.crop), int(H * args.crop)
    # Even output dimensions keep the H.264/mp4v encoder happy.
    ow, oh = cw - (cw % 2), ch - (ch % 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (ow, oh)
    )
    # An unopened writer makes write() a silent no-op -- this script would
    # otherwise print "wrote ... (N frames)" and exit 0 with no output file,
    # and the failure would only surface in track.py, the next command in
    # the README's tracking walkthrough, pointing at the wrong script.
    if not writer.isOpened():
        raise SystemExit(
            f"cannot open VideoWriter for {args.out} "
            f"(mp4v codec unavailable in this OpenCV build?)"
        )

    max_dx, max_dy = W - cw, H - ch
    for i in range(args.frames):
        t = i / max(1, args.frames - 1)
        # Ease-in-out so the pan accelerates and settles like real gimbal motion
        # rather than starting and stopping instantaneously.
        e = 0.5 - 0.5 * np.cos(np.pi * t)
        x = int(e * max_dx)
        y = int((0.5 - 0.5 * np.cos(2 * np.pi * t)) * max_dy)  # one vertical sweep
        crop = img[y : y + ch, x : x + cw]
        writer.write(crop[:oh, :ow])

    writer.release()
    print(f"wrote  : {args.out}  ({args.frames} frames @ {args.fps} fps, {ow}x{oh})")
    print(
        "\nNOTE: synthetic camera motion over a real frame. Validates the "
        "pipeline;\n      it is not a tracking accuracy benchmark. See module "
        "docstring."
    )


if __name__ == "__main__":
    main()
