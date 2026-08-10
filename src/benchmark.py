"""
Export a trained YOLO checkpoint to ONNX and measure inference latency
alongside accuracy, across backends.

Why this script exists
----------------------
A detector is only useful on a drone/edge device if it hits a latency budget.
Reporting mAP alone says nothing about deployability, so this measures the
other half and emits a single table pairing accuracy with latency across
PyTorch (eager) and ONNX Runtime (GPU and CPU).

Measurement notes (these are the parts that are easy to get wrong):
  * GPU work is asynchronous -- torch.cuda.synchronize() is required before
    stopping the clock, otherwise you time the kernel *launch*, not the kernel.
  * The first N iterations are discarded. cuDNN autotunes its algorithm choice
    on first call, so including warmup understates steady-state throughput.
  * Median and p95 are reported rather than the mean. Latency distributions are
    right-skewed, and for a real-time system the tail is what breaks the budget,
    not the average.

Usage
-----
    python src/benchmark.py --weights runs/n_1024/weights/best.pt --imgsz 1024
"""

import argparse
import json
import statistics
import time
from pathlib import Path

# numpy/torch/ultralytics are imported inside the functions that use them,
# after parse_args(). At module scope they pin `--help` to a fully provisioned
# environment, which makes the CLI undiscoverable exactly when someone is
# trying to find out what it needs. Note bench_onnx needs numpy too -- it is
# not only used by the torch paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

WARMUP_ITERS = 20
TIMED_ITERS = 100


def _summarise(times_ms: list[float]) -> dict:
    """Median / p95 / throughput from a list of per-iteration latencies."""
    times_ms = sorted(times_ms)
    median = statistics.median(times_ms)
    return {
        "median_ms": round(median, 2),
        "p95_ms": round(times_ms[int(0.95 * len(times_ms)) - 1], 2),
        "min_ms": round(times_ms[0], 2),
        "fps": round(1000.0 / median, 1),
    }


def bench_pytorch(weights: Path, imgsz: int, device: str = "cuda") -> dict:
    """Latency of the raw PyTorch model -- the pre-optimisation reference."""
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights)).model.fuse().eval().to(device)
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(TIMED_ITERS):
            start = time.perf_counter()
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()  # see module docstring
            times.append((time.perf_counter() - start) * 1000)

    return _summarise(times)


def bench_onnx(onnx_path: Path, imgsz: int, provider: str) -> dict:
    """Latency under ONNX Runtime for one execution provider.

    Asking for a provider is not the same as getting it. If the CUDA EP's
    native library fails to load -- a version mismatch between the
    onnxruntime-gpu build and the installed CUDA runtime is the usual cause --
    ORT logs the failure and silently falls back to CPU. The session then
    reports honest numbers under a dishonest label, which is worse than an
    outright error. So verify what actually got bound and record it.
    """
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=[provider])
    actual = sess.get_providers()
    if provider not in actual:
        raise RuntimeError(
            f"requested {provider} but ORT bound {actual} -- "
            f"refusing to report a CPU measurement as GPU"
        )

    name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(WARMUP_ITERS):
        sess.run(None, {name: dummy})

    times = []
    for _ in range(TIMED_ITERS):
        start = time.perf_counter()
        sess.run(None, {name: dummy})
        times.append((time.perf_counter() - start) * 1000)

    return _summarise(times)


def run_one(weights: Path, imgsz: int, data: str, device: str = "0") -> dict:
    """Accuracy + latency across every available backend."""
    import torch
    from ultralytics import YOLO

    # --- accuracy -------------------------------------------------------
    metrics = YOLO(str(weights)).val(data=data, imgsz=imgsz, device=device, verbose=False)
    row = {
        "imgsz": imgsz,
        "mAP50": round(metrics.box.map50, 4),
        "mAP50_95": round(metrics.box.map, 4),
    }
    print(f"  mAP50={row['mAP50']}  mAP50-95={row['mAP50_95']}")

    # --- latency --------------------------------------------------------
    # This specific comparison is CUDA-vs-CUDA-vs-CPU by design (that is the
    # whole point of the table), so it does not follow --device -- but it
    # must not crash a CPU-only machine outright, since the ONNX-CPU row
    # below is still meaningful there.
    if torch.cuda.is_available():
        row["pytorch_cuda"] = bench_pytorch(weights, imgsz, "cuda")
        print(f"  PyTorch  CUDA : {row['pytorch_cuda']}")
    else:
        print("  PyTorch  CUDA : skipped (no CUDA device on this machine)")

    # weights.stem, not a hardcoded "best": otherwise running against last.pt
    # after best.pt at the same imgsz finds best_<imgsz>.onnx already on disk,
    # skips export, and silently benchmarks best.pt's latency against
    # last.pt's freshly computed accuracy in the same output row.
    onnx_path = weights.parent / f"{weights.stem}_{imgsz}.onnx"
    if not onnx_path.exists():
        exported = YOLO(str(weights)).export(
            format="onnx", imgsz=imgsz, opset=13,
            simplify=True,  # onnxslim folds constants; smaller graph, faster load
            dynamic=False,  # static shapes let ORT pick better kernels
        )
        Path(exported).rename(onnx_path)
    row["onnx_size_mb"] = round(onnx_path.stat().st_size / 1024 ** 2, 2)

    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        row["onnx_cuda"] = bench_onnx(onnx_path, imgsz, "CUDAExecutionProvider")
        print(f"  ONNXRuntime GPU: {row['onnx_cuda']}")
    row["onnx_cpu"] = bench_onnx(onnx_path, imgsz, "CPUExecutionProvider")
    print(f"  ONNXRuntime CPU: {row['onnx_cpu']}")

    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--data", default="VisDrone.yaml")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default="0",
                   help="Device for the accuracy .val() pass, e.g. '0' or "
                        "'cpu'. The PyTorch-CUDA and ONNX-CUDA latency rows "
                        "always need an actual CUDA device regardless of "
                        "this setting -- that comparison is CUDA vs CPU by "
                        "definition -- and are skipped, not forced, when "
                        "one isn't available.")
    p.add_argument("--out", type=Path, default=REPORTS_DIR / "benchmark.json")
    args = p.parse_args()

    row = run_one(args.weights, args.imgsz, args.data, args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2))

    torch_cuda = row.get("pytorch_cuda", {}).get("median_ms")
    onnx_cuda = row.get("onnx_cuda", {}).get("median_ms")
    print(f"\n{'=' * 60}")
    print(f"imgsz {row['imgsz']}   mAP50 {row['mAP50']}   mAP50-95 {row['mAP50_95']}")
    print(f"PyTorch CUDA : {'n/a' if torch_cuda is None else f'{torch_cuda:.2f} ms'}")
    print(f"ONNX  CUDA   : {'n/a' if onnx_cuda is None else f'{onnx_cuda:.2f} ms'}")
    print(f"ONNX  CPU    : {row['onnx_cpu']['median_ms']:.2f} ms")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
