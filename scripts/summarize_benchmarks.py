"""Summarize the FP32 benchmark across separate sessions.

One session's eager-vs-ONNX ratio was the headline, and the repository's own
second measurement of the same pair disagreed with it by 13 points. A ratio
that moves that much between sessions needs its range beside it, and a range
needs more than one session - so benchmark.py is run in several separate
processes and this collects them. Each session's ratios are computed within
that session; nothing is compared across two of them.
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _speedup(new_ms: float, reference_ms: float) -> float:
    return round(100 * (1 - new_ms / reference_ms), 1)


def session(run: dict) -> dict:
    """The within-session figures the README quotes, from one report."""
    med = lambda row, regime: run[row][regime]["median_ms"]
    return {
        "onnx_cuda_core_ms": med("onnx_cuda", "core"),
        "pytorch_cuda_core_ms": med("pytorch_cuda", "core"),
        "onnx_vs_eager_core_pct": _speedup(
            med("onnx_cuda", "core"), med("pytorch_cuda", "core")
        ),
        "onnx_vs_eager_transfer_pct": _speedup(
            med("onnx_cuda", "transfer_inclusive"),
            med("pytorch_cuda", "transfer_inclusive"),
        ),
        # Host-in/host-out on both sides; the CPU row has no separate core.
        "cpu_vs_cuda_x": round(
            med("onnx_cpu", "transfer_inclusive")
            / med("onnx_cuda", "transfer_inclusive"),
            1,
        ),
    }


def summarize(paths: list[Path]) -> dict:
    if len(paths) < 2:
        raise SystemExit("at least two sessions are required for a range")
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    for path, run in zip(paths, runs, strict=True):
        if run.get("precision") != "FP32" or "onnx_cuda" not in run:
            raise SystemExit(f"{path.name}: not an FP32 run with CUDA rows")
    digests = {run["export"]["weights_sha256"] for run in runs}
    if len(digests) != 1:
        raise SystemExit(f"sessions benchmarked different checkpoints: {digests}")
    sessions = [session(run) for run in runs]
    return {
        "sessions": [p.relative_to(ROOT).as_posix() for p in paths],
        "protocol": (
            f"{len(runs)} separate processes of src/benchmark.py, same checkpoint "
            f"(sha256 {next(iter(digests))[:12]}), same machine; every ratio is "
            f"within one session."
        ),
        "per_session": sessions,
        "statistics": {
            key: {
                "median": statistics.median(s[key] for s in sessions),
                "min": min(s[key] for s in sessions),
                "max": max(s[key] for s in sessions),
            }
            for key in sessions[0]
        },
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out", type=Path, default=ROOT / "reports/benchmark_repeats/summary.json"
    )
    p.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="Default: reports/benchmark.json and reports/benchmark_repeats/run_*.json",
    )
    args = p.parse_args(argv)
    paths = [path.resolve() for path in args.runs] or [
        ROOT / "reports/benchmark.json",
        *sorted((ROOT / "reports/benchmark_repeats").glob("run_*.json")),
    ]
    summary = summarize(paths)
    staged = args.out.with_suffix(".json.tmp")
    staged.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    staged.replace(args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
