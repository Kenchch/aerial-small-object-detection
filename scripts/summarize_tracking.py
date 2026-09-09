"""Summarize the five committed tracking repeats without pooling their frames."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every stage that runs on every frame. Order matches the report.
STAGES = ("decode", "detect_and_track", "annotate", "encode")


def main():
    paths = sorted((ROOT / "reports/tracking_repeats").glob("run_*.json"))
    if len(paths) < 2:
        raise ValueError("At least two repeat reports are required")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
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
        "protocol": "Five separate Python processes; decode, inference, annotation and encoding enabled. Wall time includes model warm-up within tracking; steady_state_* sums per-stage medians and so excludes it. Same 90-frame synthetic pan source and release v1.0 checkpoint.",
        "statistics": {},
    }
    for key, values in fields.items():
        summary["statistics"][key] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "sample_stddev": round(statistics.stdev(values), 4),
        }
    target = ROOT / "reports/tracking_repeats/summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
