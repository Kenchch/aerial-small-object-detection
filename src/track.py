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


def track_colour(track_id: int) -> tuple:
    """Stable pseudo-random BGR colour per track id."""
    import numpy as np

    rng = np.random.default_rng(int(track_id) * 9973)
    c = rng.integers(70, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def frame_source(source: Path):
    """Yield (frame, fps, size) from a video file or a directory of images."""
    import cv2

    if source.is_dir():
        files = sorted(
            [p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if not files:
            raise SystemExit(f"no images in {source}")
        first = cv2.imread(str(files[0]))
        h, w = first.shape[:2]
        yield None, 10.0, (w, h)  # header: assume 10 fps for an image sequence
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                yield img, None, None
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
    p = argparse.ArgumentParser(description="Track objects through a video with YOLO + ByteTrack")
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path,
                   help="Video file or directory of ordered frames.")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="ByteTrack keeps low-confidence detections as association "
                        "candidates instead of discarding them, which is the right "
                        "trade for aerial imagery where small objects sit near the "
                        "confidence floor for most of their track.")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "track_out.mp4")
    p.add_argument("--no-write", action="store_true", help="Profile only; write no video.")
    p.add_argument("--device", default="0",
                   help="'0' for GPU, or 'cpu'. Matches evaluate.py and "
                        "benchmark.py. CPU works but is not the intended "
                        "path here: the staged profile below is only "
                        "meaningful against the hardware you plan to deploy "
                        "on, and the committed numbers are from a GPU run.")
    args = p.parse_args()

    import cv2
    from ultralytics import YOLO

    font = cv2.FONT_HERSHEY_SIMPLEX
    model = YOLO(str(args.weights))

    src = frame_source(args.source)
    _, fps, (W, H) = next(src)

    writer = None
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        # Without this check, a missing mp4v/FFmpeg codec makes write() a
        # silent no-op: the script still exits 0 and prints a frame count,
        # and the failure only surfaces later, in whatever reads the (empty
        # or missing) output file -- pointing at the wrong script entirely.
        if not writer.isOpened():
            raise SystemExit(f"cannot open VideoWriter for {args.out} "
                              f"(mp4v codec unavailable in this OpenCV build?)")

    t_infer, t_draw, t_write, t_decode = [], [], [], []
    track_frames = defaultdict(int)      # track id -> frames seen
    n_frames = 0
    wall_start = time.perf_counter()

    # Decode has to be timed too. Leaving it out was how the first version of
    # this profile accounted for only a third of wall-clock time.
    t_decode_start = time.perf_counter()
    for frame, _, _ in src:
        t_decode.append((time.perf_counter() - t_decode_start) * 1000)
        n_frames += 1

        t0 = time.perf_counter()
        # persist=True carries tracker state across calls; without it every frame
        # is treated as a new sequence and ids restart from 1.
        res = model.track(frame, imgsz=args.imgsz, conf=args.conf, device=args.device,
                          tracker=args.tracker, persist=True, verbose=False)[0]
        t_infer.append((time.perf_counter() - t0) * 1000)

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
                cv2.rectangle(frame, (int(x1), int(y1) - th - 4),
                              (int(x1) + tw + 4, int(y1)), colour, -1)
                cv2.putText(frame, label, (int(x1) + 2, int(y1) - 3), font, 0.4,
                            (0, 0, 0), 1, cv2.LINE_AA)
        t_draw.append((time.perf_counter() - t0) * 1000)

        if writer is not None:
            t0 = time.perf_counter()
            writer.write(frame)
            t_write.append((time.perf_counter() - t0) * 1000)

        t_decode_start = time.perf_counter()

    wall = time.perf_counter() - wall_start
    if writer is not None:
        writer.release()

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
    accounted_mean = mean(t_decode) + mean(t_infer) + mean(t_draw) + mean(t_write)
    per_frame_wall = wall / n_frames * 1000

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
        "stage_ms_mean": {
            "decode": round(mean(t_decode), 2),
            "detect_and_track": round(mean(t_infer), 2),
            "annotate": round(mean(t_draw), 2),
            "encode": round(mean(t_write), 2) if t_write else None,
        },
        "warmup": {
            "first_frame_ms": round(t_infer[0], 1) if t_infer else None,
            "steady_state_ms": round(med(t_infer), 2),
            "warmup_penalty_x": round(t_infer[0] / med(t_infer), 1) if t_infer and med(t_infer) else None,
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
            "id_churn_ratio_min": round((max_id - len(track_frames)) / len(track_frames), 1)
                                  if track_frames else 0,  # lower bound, see above
            "mean_track_len_frames": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "max_track_len_frames": max(lengths) if lengths else 0,
            "single_frame_tracks": fragments,
            "single_frame_pct": round(100 * fragments / len(lengths), 1) if lengths else 0,
            "mean_boxes_per_frame": round(sum(lengths) / n_frames, 1),
        },
        "config": {
            # Relative to the project root, and as_posix rather than str():
            # this report is committed for readers to look at, so neither the
            # absolute path nor the path separator of whoever generated it
            # should end up baked into the file.
            "weights": Path(args.weights).resolve().relative_to(PROJECT_ROOT).as_posix()
                       if Path(args.weights).resolve().is_relative_to(PROJECT_ROOT)
                       else str(args.weights),
            "imgsz": args.imgsz, "conf": args.conf, "tracker": args.tracker,
        },
    }

    print(f"\n{'=' * 60}")
    print(f"  Tracked {n_frames} frames in {wall:.1f} s "
          f"— {report['end_to_end_fps']} FPS end-to-end")
    print(f"{'=' * 60}")
    md, mn = report["stage_ms_median"], report["stage_ms_mean"]
    print(f"  {'stage':<18}{'median':>10}{'mean':>10}")
    print(f"  {'-' * 38}")
    for k in ("decode", "detect_and_track", "annotate", "encode"):
        if md[k] is None:
            continue
        print(f"  {k:<18}{md[k]:>9.2f} {mn[k]:>9.2f}")
    r = report["reconciliation"]
    print(f"  {'-' * 38}")
    print(f"  {'accounted':<18}{'':>9} {r['accounted_mean_ms']:>9.2f}")
    print(f"  {'wall per frame':<18}{'':>9} {report['per_frame_wall_ms']:>9.2f}")
    print(f"  coverage {r['coverage_pct']} %  (unaccounted {r['unaccounted_ms']} ms)")

    w = report["warmup"]
    print(f"\n  first frame  : {w['first_frame_ms']} ms  "
          f"({w['warmup_penalty_x']}× steady state)")

    t = report["tracks"]
    print(f"\n  unique tracks      : {t['unique_ids']}")
    print(f"  highest id seen    : {t['highest_id_seen']}  "
          f"(churn ≥{t['id_churn_ratio_min']}× — tentative tracks per confirmed one)")
    print(f"  boxes per frame    : {t['mean_boxes_per_frame']}")
    print(f"  mean track length  : {t['mean_track_len_frames']} frames")
    print(f"  longest track      : {t['max_track_len_frames']} frames")
    print(f"  single-frame tracks: {t['single_frame_tracks']} ({t['single_frame_pct']} %)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_DIR / "tracking.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\n  metrics -> {out_json}")
    if writer is not None:
        print(f"  video   -> {args.out}")


if __name__ == "__main__":
    main()
