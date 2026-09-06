"""Summarize the five committed tracking repeats without pooling their frames."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = sorted((ROOT / "reports/tracking_repeats").glob("run_*.json"))
    if len(paths) < 2:
        raise ValueError("At least two repeat reports are required")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    fields = {
        "wall_seconds": [run["wall_s"] for run in runs],
        "end_to_end_fps": [run["end_to_end_fps"] for run in runs],
        "detect_and_track_frame_median_ms": [
            run["stage_ms_median"]["detect_and_track"] for run in runs
        ],
    }
    summary = {
        "runs": len(runs),
        "frames_each": [run["frames"] for run in runs],
        "protocol": "Five separate Python processes; decode, inference, annotation and encoding enabled. Wall time includes model warm-up within tracking. Same 90-frame synthetic pan source and release v1.0 checkpoint.",
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
