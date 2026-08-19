"""
Video inference with multi-object tracking, and a staged latency profile.

This is the deployment end of the project. Everything upstream measures a model
in isolation; this runs it the way it would actually be used — a frame source in,
annotated frames and persistent track IDs out — and reports where the wall-clock
time actually goes.

The staged breakdown is the point: model latency alone is not end-to-end
latency. Decode, tracking association and encode all sit in the same frame
budget, and reporting only the forward-pass time is how deployment estimates
end up wrong.

Usage
-----
    python src/track.py --weights runs/n_1024/weights/best.pt --source clip.mp4
    python src/track.py --weights runs/n_1024/weights/best.pt --source frames/ --imgsz 1024
    python src/track.py --weights runs/n_1024/weights/best.pt --source clip.mp4 --no-write
"""

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

# cv2/numpy/ultralytics are imported inside the functions that use them, after
# parse_args(). At module scope they pin `--help` to a fully provisioned
# environment, which makes the CLI undiscoverable exactly when someone is
# trying to find out what it needs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

# Colours are per-track, not per-class, because the thing being communicated in
# a tracking overlay is identity persistence.


def association_remainders(t_infer, t_pre, t_fwd, t_post):
    """Per-frame time inside model.track() that Ultralytics does not attribute.

    Returned per frame so callers can take an order statistic of a real
    quantity. Taking `median(t_infer) - (median(t_pre)+median(t_fwd)+median(t_post))`
    instead subtracts four medians that are generally attained on four different
    frames, producing a number no frame ever exhibited - and one that can come
    out negative even when every frame's own remainder is positive.
    """
    return [
        ti - (p + f + po)
        for ti, p, f, po in zip(t_infer, t_pre, t_fwd, t_post, strict=True)
    ]


def track_colour(track_id: int) -> tuple:
    """Stable pseudo-random BGR colour per track id."""
    import numpy as np

    rng = np.random.default_rng(int(track_id) * 9973)
    c = rng.integers(70, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def _for_report(path: Path) -> str:
    """A path fit to be committed.

    Relative to the project root, and as_posix rather than str(): these reports
    are committed for readers to look at, so neither the absolute path nor the
    path separator of whoever generated it should end up baked into the file.
    """
    resolved = Path(path).resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(path)


def natural_key(name: str) -> tuple:
    """Sort key that puts frame2 before frame10.

    `sorted()` on filenames is lexicographic, so a sequence written as
    frame1.jpg ... frame10.jpg is fed to the tracker in the order 1, 10, 11,
    2, 3. Nothing errors: the video comes out re-ordered and the tracker
    associates across jumps that never happened, so track lengths and the
    churn ratio describe a sequence that does not exist.

    Digit runs compare as integers, everything else casefolded. The (0, ...) /
    (1, ...) tags keep the key total - two names can put a number and a word at
    the same position, and comparing int with str raises.
    """
    return tuple(
        (1, int(part), "") if part.isdigit() else (0, 0, part.lower())
        for part in re.split(r"(\d+)", name)
    )


def frame_source(source: Path):
    """Yield (frame, fps, size) from a video file or a directory of images."""
    import cv2

    if source.is_dir():
        files = sorted(
            [
                p
                for p in source.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ],
            key=lambda p: natural_key(p.name),
        )
        if not files:
            raise SystemExit(f"no images in {source}")
        # cv2.imread returns None for an unreadable or corrupt file rather than
        # raising. Reading files[0] purely for its dimensions therefore died
        # with `AttributeError: 'NoneType' object has no attribute 'shape'` -
        # an error naming neither the file nor the reason.
        first = cv2.imread(str(files[0]))
        if first is None:
            raise SystemExit(
                f"cannot decode {files[0]} - it is the first frame, and its dimensions "
                "define the output video. Remove or repair it and re-run."
            )
        h, w = first.shape[:2]
        yield None, 10.0, (w, h)  # header: assume 10 fps for an image sequence
        # `first` is reused rather than decoded a second time: files[0] used to
        # be read twice, once before the loop and once inside it, so one full
        # decode landed outside the profiled region and both wall_s and
        # t_decode[0] under-counted it.
        skipped = []
        for i, f in enumerate(files):
            img = first if i == 0 else cv2.imread(str(f))
            if img is None:
                # Dropping these silently made `frames` disagree with the file
                # count on disk with nothing in the report to explain the gap.
                skipped.append(f.name)
                continue
            yield img, None, None
        if skipped:
            print(
                f"[warn] skipped {len(skipped)} undecodable image(s): "
                f"{', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}"
            )
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        yield None, fps, (w, h)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img, None, None
        cap.release()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Track objects through a video with YOLO + ByteTrack"
    )
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Video file or directory of ordered frames.",
    )
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="ByteTrack keeps low-confidence detections as association "
        "candidates instead of discarding them, which is the right "
        "trade for aerial imagery where small objects sit near the "
        "confidence floor for most of their track.",
    )
    p.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "reports" / "track_out.mp4"
    )
    p.add_argument(
        "--no-write", action="store_true", help="Profile only; write no video."
    )
    p.add_argument(
        "--device",
        default="0",
        help="'0' for GPU, or 'cpu'. Matches evaluate.py and "
        "benchmark.py. CPU works but is not the intended "
        "path here: the staged profile below is only "
        "meaningful against the hardware you plan to deploy "
        "on, and the committed numbers are from a GPU run.",
    )
    args = p.parse_args()

    # Checked before the model is loaded, let alone before either file handle
    # is opened. cv2.VideoWriter truncates its target on open, so pointing
    # --out at --source destroys the input while the reader is still streaming
    # it - and the default --out is reports/track_out.mp4, the very file the
    # README tells you to look at afterwards. resolve() so that
    # reports/track_out.mp4 and ./reports/../reports/track_out.mp4 are
    # recognised as the same file.
    if not args.no_write and args.source.resolve() == args.out.resolve():
        raise SystemExit(
            f"--source and --out are the same file ({args.out}). The writer "
            f"truncates its target on open, so this would destroy the input "
            f"mid-read. Pass a different --out, or --no-write to profile only."
        )

    import cv2
    from ultralytics import YOLO

    font = cv2.FONT_HERSHEY_SIMPLEX
    model = YOLO(str(args.weights))

    # The wall clock starts HERE, not after setup. Opening the capture, reading
    # the header - which for an image sequence is a full decode of frame 0 -
    # and opening the writer are all work the pipeline does, and all of it used
    # to sit outside the measured region while the report claimed no stage went
    # unmeasured.
    wall_start = time.perf_counter()

    src = frame_source(args.source)
    _, fps, (W, H) = next(src)

    writer = None
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )
        # Without this check, a missing mp4v/FFmpeg codec makes write() a
        # silent no-op: the script still exits 0 and prints a frame count,
        # and the failure only surfaces later, in whatever reads the (empty
        # or missing) output file -- pointing at the wrong script entirely.
        if not writer.isOpened():
            raise SystemExit(
                f"cannot open VideoWriter for {args.out} "
                f"(mp4v codec unavailable in this OpenCV build?)"
            )

    # Everything before the first frame: capture open, header read, writer
    # open. Reported rather than folded into t_decode, so the per-frame stage
    # numbers stay per-frame.
    setup_ms = (time.perf_counter() - wall_start) * 1000

    t_infer, t_draw, t_write, t_decode = [], [], [], []
    # Ultralytics fills Results.speed with its own internal split of the call.
    # Without these, "detect + track" is a single number roughly 3x the raw
    # forward pass that benchmark.py reports, with nothing in either report
    # saying where the difference goes.
    t_pre, t_fwd, t_post = [], [], []
    track_frames = defaultdict(int)  # track id -> frames seen
    n_frames = 0

    # Decode has to be timed too. Leaving it out was how the first version of
    # this profile accounted for only a third of wall-clock time.
    t_decode_start = time.perf_counter()
    for frame, _, _ in src:
        t_decode.append((time.perf_counter() - t_decode_start) * 1000)
        n_frames += 1

        t0 = time.perf_counter()
        # persist=True carries tracker state across calls; without it every frame
        # is treated as a new sequence and ids restart from 1.
        res = model.track(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            tracker=args.tracker,
            persist=True,
            verbose=False,
        )[0]
        t_infer.append((time.perf_counter() - t0) * 1000)

        # preprocess = letterbox + BGR->RGB + /255 + HWC->CHW + host-to-device;
        # postprocess = NMS and box rescaling. Association is deliberately not
        # in here -- Ultralytics runs the tracker after postprocess, so it
        # falls out below as the remainder against the wall-clock t_infer.
        speed = res.speed
        t_pre.append(speed["preprocess"])
        t_fwd.append(speed["inference"])
        t_post.append(speed["postprocess"])

        t0 = time.perf_counter()
        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)
            # strict=True: these three come off the same Boxes object and must
            # be the same length. Silently truncating to the shortest would
            # drop detections from the profile rather than report the problem.
            for (x1, y1, x2, y2), tid, cid in zip(xyxy, ids, clss, strict=True):
                track_frames[tid] += 1
                colour = track_colour(tid)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
                label = f"{model.names[cid]} {tid}"
                (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1) - th - 4),
                    (int(x1) + tw + 4, int(y1)),
                    colour,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (int(x1) + 2, int(y1) - 3),
                    font,
                    0.4,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
        t_draw.append((time.perf_counter() - t0) * 1000)

        if writer is not None:
            t0 = time.perf_counter()
            writer.write(frame)
            t_write.append((time.perf_counter() - t0) * 1000)

        t_decode_start = time.perf_counter()

    # release() before the clock stops. It finalises the container - flushing
    # buffered frames and writing the moov atom - which is real encode work,
    # and on a long clip it is not a rounding error. Stopping the timer first
    # was how "encode" could look cheap in a report whose own wall time did not
    # contain the expensive half of it.
    flush_ms = 0.0
    if writer is not None:
        t0 = time.perf_counter()
        writer.release()
        flush_ms = (time.perf_counter() - t0) * 1000
    wall = time.perf_counter() - wall_start

    if not n_frames:
        raise SystemExit("no frames read")

    # --- Tracking quality -------------------------------------------------
    # Mean track length is the honest single number for association quality:
    # a tracker that drops and re-acquires the same object inflates the track
    # count and collapses the mean.
    lengths = list(track_frames.values())
    fragments = sum(1 for v in lengths if v == 1)

    # ByteTrack's id counter is global and monotonic: it increments for every
    # tentative track, including the ones spawned from low-confidence
    # detections that never get confirmed. The ratio of the highest id to the
    # number of ids actually emitted is therefore a churn measure, and it
    # matters downstream — anything keying on track id (a counter, a database,
    # a re-id store) inherits that growth rate.
    #
    # track_frames is only written when a box is drawn, so this is the largest
    # id that reached the output, not ByteTrack's internal counter (not exposed
    # here, and possibly higher — a tentative track created after the last
    # drawn box never appears). This and the churn ratio below are lower bounds.
    # int() because the ids come out of a numpy array and np.int64 is not
    # JSON-serialisable.
    max_id = int(max(track_frames)) if track_frames else 0

    med = lambda xs: statistics.median(xs) if xs else 0.0
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0

    # Median describes the steady state; mean is what the wall clock actually
    # pays, and the two diverge sharply because the first frames carry CUDA
    # context creation and cuDNN autotuning. Report both, and reconcile them
    # against wall time so any unmeasured remainder is visible rather than
    # quietly absorbed.
    # Setup and flush happen once, not per frame, so they are amortised across
    # the run to be comparable with the per-frame means. Left out entirely,
    # they showed up as "unaccounted" and the coverage figure blamed the
    # per-frame stages for time they never spent.
    once_per_run_mean = (setup_ms + flush_ms) / n_frames
    accounted_mean = (
        mean(t_decode)
        + mean(t_infer)
        + mean(t_draw)
        + mean(t_write)
        + once_per_run_mean
    )
    per_frame_wall = wall / n_frames * 1000

    # Association is not timed directly -- Ultralytics runs the tracker inside
    # the same call and does not report it -- so take it as the remainder of
    # the measured call after its own three reported phases. That remainder also
    # absorbs Python-level call overhead, which is why it is named for both
    # rather than presented as a clean ByteTrack number.
    #
    # Compute it per frame and take the median of THAT, not the difference of
    # four separate medians. The medians of four series are generally not
    # attained on the same frame, so their difference is not a quantity any
    # frame exhibited; it can even come out negative while every individual
    # frame's remainder is positive. The per-frame form is a real order
    # statistic of a real quantity, and it degrades safely when Ultralytics
    # reports a phase inconsistently.
    assoc_median = med(association_remainders(t_infer, t_pre, t_fwd, t_post))

    report = {
        "frames": n_frames,
        "wall_s": round(wall, 2),
        "end_to_end_fps": round(n_frames / wall, 1),
        "per_frame_wall_ms": round(per_frame_wall, 2),
        "stage_ms_median": {
            "decode": round(med(t_decode), 2),
            "detect_and_track": round(med(t_infer), 2),
            "annotate": round(med(t_draw), 2),
            "encode": round(med(t_write), 2) if t_write else None,
        },
        # Once per run, not per frame: capture open + header read + writer
        # open, and the writer's release() at the end.
        "open_and_flush_ms": {
            "setup": round(setup_ms, 2),
            "flush": round(flush_ms, 2),
            "amortised_per_frame": round(once_per_run_mean, 3),
        },
        "stage_ms_mean": {
            "decode": round(mean(t_decode), 2),
            "detect_and_track": round(mean(t_infer), 2),
            "annotate": round(mean(t_draw), 2),
            "encode": round(mean(t_write), 2) if t_write else None,
        },
        # What "detect + track" above is actually made of. benchmark.py times
        # only the forward pass on a tensor already resident in VRAM, so it
        # corresponds to `forward` here -- the rest is the cost of feeding a
        # real frame in and turning logits back into tracked boxes.
        "detect_and_track_ms_median": {
            "preprocess": round(med(t_pre), 2),
            "forward": round(med(t_fwd), 2),
            "postprocess_nms": round(med(t_post), 2),
            "association_and_overhead": round(assoc_median, 2),
        },
        "warmup": {
            "first_frame_ms": round(t_infer[0], 1) if t_infer else None,
            "steady_state_ms": round(med(t_infer), 2),
            "warmup_penalty_x": round(t_infer[0] / med(t_infer), 1)
            if t_infer and med(t_infer)
            else None,
        },
        "reconciliation": {
            "accounted_mean_ms": round(accounted_mean, 2),
            "unaccounted_ms": round(per_frame_wall - accounted_mean, 2),
            "coverage_pct": round(100 * accounted_mean / per_frame_wall, 1),
        },
        "tracks": {
            "unique_ids": len(track_frames),
            "highest_id_seen": max_id,
            # Tentative (never-confirmed) ids per confirmed one -- NOT
            # max_id / confirmed, which counts the confirmed ids themselves
            # in the numerator too and would overstate this by exactly 1x.
            "id_churn_ratio_min": round(
                (max_id - len(track_frames)) / len(track_frames), 1
            )
            if track_frames
            else 0,  # lower bound, see above
            "mean_track_len_frames": round(sum(lengths) / len(lengths), 1)
            if lengths
            else 0,
            "max_track_len_frames": max(lengths) if lengths else 0,
            "single_frame_tracks": fragments,
            "single_frame_pct": round(100 * fragments / len(lengths), 1)
            if lengths
            else 0,
            "mean_boxes_per_frame": round(sum(lengths) / n_frames, 1),
        },
        "config": {
            "weights": _for_report(args.weights),
            "source": _for_report(args.source),
            # None rather than a path when nothing was written, so the report
            # cannot name an output file that does not exist.
            "out": None if args.no_write else _for_report(args.out),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "tracker": args.tracker,
            # Every latency figure in this file is a property of the device it
            # ran on, and the report did not say which one that was.
            "device": args.device,
        },
    }

    print(f"\n{'=' * 60}")
    print(
        f"  Tracked {n_frames} frames in {wall:.1f} s "
        f"— {report['end_to_end_fps']} FPS end-to-end"
    )
    print(f"{'=' * 60}")
    md, mn = report["stage_ms_median"], report["stage_ms_mean"]
    print(f"  {'stage':<18}{'median':>10}{'mean':>10}")
    print(f"  {'-' * 38}")
    for k in ("decode", "detect_and_track", "annotate", "encode"):
        if md[k] is None:
            continue
        print(f"  {k:<18}{md[k]:>9.2f} {mn[k]:>9.2f}")
        if k == "detect_and_track":
            for sub, v in report["detect_and_track_ms_median"].items():
                print(f"    {sub:<16}{v:>9.2f}")
    r = report["reconciliation"]
    print(f"  {'-' * 38}")
    print(f"  {'accounted':<18}{'':>9} {r['accounted_mean_ms']:>9.2f}")
    print(f"  {'wall per frame':<18}{'':>9} {report['per_frame_wall_ms']:>9.2f}")
    print(f"  coverage {r['coverage_pct']} %  (unaccounted {r['unaccounted_ms']} ms)")

    w = report["warmup"]
    print(
        f"\n  first frame  : {w['first_frame_ms']} ms  "
        f"({w['warmup_penalty_x']}× steady state)"
    )

    t = report["tracks"]
    print(f"\n  unique tracks      : {t['unique_ids']}")
    print(
        f"  highest id seen    : {t['highest_id_seen']}  "
        f"(churn ≥{t['id_churn_ratio_min']}× — tentative tracks per confirmed one)"
    )
    print(f"  boxes per frame    : {t['mean_boxes_per_frame']}")
    print(f"  mean track length  : {t['mean_track_len_frames']} frames")
    print(f"  longest track      : {t['max_track_len_frames']} frames")
    print(
        f"  single-frame tracks: {t['single_frame_tracks']} ({t['single_frame_pct']} %)"
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_DIR / "tracking.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\n  metrics -> {out_json}")
    if writer is not None:
        print(f"  video   -> {args.out}")


if __name__ == "__main__":
    main()
