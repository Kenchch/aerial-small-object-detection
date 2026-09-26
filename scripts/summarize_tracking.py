"""Summarize the committed tracking repeats without pooling their frames.

The repeats are only a sample of one thing if they are the same run repeated:
same clip, same frame count, same checkpoint and settings, same machine. That
used to be asserted by a hand-written sentence ("Five separate Python processes
... release v1.0 checkpoint"), which stayed "Five" with two runs present and
stayed "release v1.0" with no checkpoint digest anywhere in the evidence. It is
now checked, and the protocol text is written from what the runs record.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every stage that runs on every frame. Order matches the report.
STAGES = ("decode", "detect_and_track", "annotate", "encode")

NUMBER_WORDS = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
]


def _release_sha256() -> str:
    # The one place the release digest is pinned; evaluate.py imports no
    # third-party package at module scope, so this stays cheap.
    sys.path.insert(0, str(ROOT / "src"))
    from evaluate import RELEASE_SHA256

    return RELEASE_SHA256


def _same(runs, paths, get, what):
    """The value every run shares, or a refusal naming who differs."""
    values = [get(run) for run in runs]
    if len({json.dumps(v, sort_keys=True) for v in values}) != 1:
        detail = ", ".join(
            f"{path.name}={json.dumps(v, sort_keys=True)}"
            for path, v in zip(paths, values, strict=True)
        )
        raise SystemExit(f"repeats differ in {what}: {detail}")
    return values[0]


def _check_complete(path, run):
    """A repeat that skipped a stage, or ran on a clip other than its record's,
    is not a sample of the protocol - and a --no-write run's missing encode
    stage used to surface as a bare TypeError from sum()."""
    if run["source"].get("ran_with_mismatch"):
        raise SystemExit(f"{path.name}: ran on a clip that does not match its record")
    if run.get("output") is None or any(
        run["stage_ms_median"].get(stage) is None for stage in STAGES
    ):
        raise SystemExit(
            f"{path.name}: not a full decode+infer+annotate+encode run "
            f"(was it --no-write?)"
        )


def summarize(repeats: Path) -> dict:
    paths = sorted(repeats.glob("run_*.json"))
    if len(paths) < 2:
        raise SystemExit(f"at least two repeat reports are required in {repeats}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, run in zip(paths, runs, strict=True):
        _check_complete(path, run)

    frames = _same(runs, paths, lambda r: r["frames"], "frames")
    source = _same(runs, paths, lambda r: r["source"]["sha256"], "source clip")
    config = _same(
        runs,
        paths,
        # `out` is where each run wrote its video, which is allowed to differ.
        lambda r: {k: v for k, v in r["config"].items() if k != "out"},
        "config",
    )
    _same(
        runs,
        paths,
        lambda r: [
            r["environment"].get(k)
            for k in ("device_resolved", "gpu", "torch", "ultralytics")
        ],
        "environment",
    )

    digest = config.get("weights_sha256")
    if digest is None:
        checkpoint = f"checkpoint {config['weights']} (digest not recorded)"
    elif digest == _release_sha256():
        checkpoint = "release v1.0 checkpoint"
    else:
        checkpoint = f"checkpoint {config['weights']} (sha256 {digest[:12]})"
    count = NUMBER_WORDS[len(runs)] if len(runs) < len(NUMBER_WORDS) else str(len(runs))

    # Steady state is every stage's median summed, not the largest one: decode,
    # annotate and encode still happen on every frame once the cold start is
    # behind you. Summing medians rather than taking a median of sums is what
    # drops the first frame -- it costs about 180x a normal one, and it moves a
    # mean but not a median over 90 frames.
    # Rounded to the precision the stages themselves carry: summing four
    # 2-decimal figures in binary yields 43.74999999999999, and a report full
    # of that reads as more precision than the measurement has.
    steady_ms = [
        round(sum(run["stage_ms_median"][stage] for stage in STAGES), 2) for run in runs
    ]
    fields = {
        "wall_seconds": [run["wall_s"] for run in runs],
        "end_to_end_fps": [run["end_to_end_fps"] for run in runs],
        "detect_and_track_frame_median_ms": [
            run["stage_ms_median"]["detect_and_track"] for run in runs
        ],
        "steady_state_frame_ms": steady_ms,
        # Per repeat, then summarised -- not 1000 / the summarised ms. The two
        # differ, and the one that means "the throughput half the runs beat" is
        # this one.
        "steady_state_fps": [round(1000 / ms, 1) for ms in steady_ms],
    }
    summary = {
        "runs": len(runs),
        "frames_each": [run["frames"] for run in runs],
        "protocol": (
            f"{count} separate Python processes; decode, inference, annotation "
            f"and encoding enabled. Wall time includes model warm-up within "
            f"tracking; steady_state_* sums per-stage medians and so excludes "
            f"it. Same {frames}-frame source (sha256 {source[:12]}) and "
            f"{checkpoint}."
        ),
        "statistics": {},
    }
    for key, values in fields.items():
        summary["statistics"][key] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "sample_stddev": round(statistics.stdev(values), 4),
        }
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--repeats",
        type=Path,
        default=ROOT / "reports/tracking_repeats",
        help="Directory of run_*.json; summary.json is written there.",
    )
    args = p.parse_args(argv)
    summary = summarize(args.repeats)
    target = args.repeats / "summary.json"
    # Staged, so a failure while writing leaves the previous summary intact.
    staged = target.with_suffix(".json.tmp")
    staged.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    staged.replace(target)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
