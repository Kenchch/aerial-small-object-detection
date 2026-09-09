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
import math
import os
import shutil
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

ONNX_OPSET = 13
# Export flags. Constants because they are BOTH arguments to the export and
# part of the cache key: the manifest repeated `"simplify": True` as its own
# literal, so toggling the export flag would have left the manifest still
# claiming the old value and every cached graph looking current. One name,
# read by both.
ONNX_SIMPLIFY = True  # onnxslim folds constants; smaller, faster to load
ONNX_DYNAMIC = False  # static shapes let ORT pick better kernels


def _register_cudnn() -> bool:
    """Put cuDNN on the DLL search path on Windows, and say whether it worked.

    onnxruntime-gpu does not bundle cuDNN. On Linux the wheel's RPATH usually
    finds a system copy; on Windows nothing does, so
    `onnxruntime_providers_cuda.dll` fails to load with error 126 and ORT falls
    back to CPU with only a log line to say so. Measured here: a session asked
    for CUDAExecutionProvider and reported `['CPUExecutionProvider']`.

    verify_cuda_placement below refuses to report that as a GPU number, which is
    the important half. This is the other half: torch already ships cuDNN 9 in
    its own lib directory, so pointing the loader at it makes the provider
    available rather than making the benchmark unrunnable.

    Returns True if a directory was added, False if there was nothing to do.
    The path itself is deliberately not recorded: it is wherever this machine
    happens to have installed torch, which is noise in a committed report.
    """
    if not hasattr(os, "add_dll_directory"):
        return False  # not Windows
    try:
        import torch
    except ImportError:
        return False
    lib = Path(torch.__file__).resolve().parent / "lib"
    if not any(lib.glob("cudnn*.dll")):
        return False
    os.add_dll_directory(str(lib))
    return True


# One number, three places used to disagree: the CLI defaulted to 0.002, the
# function it calls defaulted to 0.01, and DESIGN.md quoted 0.01. Whichever a
# reader checked, one of the others was wrong. The measured delta is 0.0007, so
# 0.002 is the tolerance that would actually catch a regression; 0.01 is loose
# enough to pass an export that changed the model.
MAP_TOLERANCE = 0.002

WARMUP_ITERS = 20
TIMED_ITERS = 100


def _map_tolerance(value) -> float:
    """Parse the accuracy-loss ceiling without allowing it to disable itself.

    ``float`` accepts NaN and infinity, but neither is a usable threshold: every
    comparison with NaN is false, and a tolerance above 1 can never be exceeded
    by mAP.  This function is used both by argparse and by ``run_one`` so callers
    using the Python API get the same guard as the CLI.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"map tolerance must be a finite fraction in [0, 1], got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(
            f"map tolerance must be a finite fraction in [0, 1], got {value!r}"
        )
    return parsed


def _validated_map(name: str, value) -> float:
    """Return one mAP value, refusing a broken evaluation result.

    A NaN metric otherwise passes the deployment gate because ``NaN > limit``
    is false, then reaches benchmark.json as JavaScript's non-standard ``NaN``
    token.  mAP is a fraction, so values outside the unit interval are equally
    invalid evidence.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(
            f"{name} is not a numeric mAP value: {value!r}. Nothing was written."
        ) from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise SystemExit(
            f"{name} must be a finite fraction in [0, 1], got {value!r}. Nothing was written."
        )
    return parsed


def _summarise(times_ms: list[float]) -> dict:
    """Median / p95 / throughput from a list of per-iteration latencies.

    p95 is nearest-rank: ceil(0.95*n), not int(0.95*n). The two agree whenever
    0.95*n happens to be a whole number -- which it is at the TIMED_ITERS=100
    this script actually runs, so the committed benchmark.json is unaffected --
    and disagree by one rank everywhere else, always downward. Since the whole
    point of reporting p95 alongside the median is to expose the tail, a
    formula that quietly understates the tail on any other sample size is the
    wrong one to leave in place for the next person who changes TIMED_ITERS.
    """
    times_ms = sorted(times_ms)
    median = statistics.median(times_ms)
    return {
        "median_ms": round(median, 2),
        "p95_ms": round(times_ms[math.ceil(0.95 * len(times_ms)) - 1], 2),
        "min_ms": round(times_ms[0], 2),
        "fps": round(1000.0 / median, 1),
    }


def bench_pytorch(weights: Path, imgsz: int, device: str = "cuda") -> dict:
    """Latency of the raw PyTorch model, in two regimes.

    `core` feeds a tensor already resident in VRAM and leaves the output there.
    `transfer_inclusive` starts from a CPU tensor and brings the output back,
    so it pays the host-to-device and device-to-host copies.

    Both are reported because comparing one against the other is the easiest
    way to draw a wrong conclusion here. ONNX Runtime's `sess.run` takes a
    numpy array and returns numpy arrays, so it is transfer-inclusive by
    construction; timing that against a GPU-resident PyTorch forward charges
    ONNX for ~2 ms of copying that PyTorch never does, and understates the
    export's real advantage roughly threefold.
    """
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights)).model.fuse().eval().to(device)
    cuda = device == "cuda"

    def run(fn) -> dict:
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                fn()
            if cuda:
                torch.cuda.synchronize()
            times = []
            for _ in range(TIMED_ITERS):
                start = time.perf_counter()
                fn()
                if cuda:
                    torch.cuda.synchronize()  # see module docstring
                times.append((time.perf_counter() - start) * 1000)
        return _summarise(times)

    resident = torch.randn(1, 3, imgsz, imgsz, device=device)
    on_host = torch.randn(1, 3, imgsz, imgsz)

    def core():
        model(resident)

    def transfer_inclusive():
        out = model(on_host.to(device))
        # .cpu() on the first output is the device-to-host half; without it
        # only the upload would be counted.
        out[0].cpu() if isinstance(out, (list, tuple)) else out.cpu()

    return {"core": run(core), "transfer_inclusive": run(transfer_inclusive)}


# ORT reports an input's type as a string. Mapping it back is what lets the
# same code feed an FP32 and an FP16 graph: hardcoding float32 worked only for
# as long as there was one precision, and it fails loudly rather than quietly
# -- ORT rejects the mismatch -- which is the reason this is a lookup and not a
# cast.
_ORT_DTYPES = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
}


def _dummy_input(sess, imgsz: int):
    """A random NCHW batch of 1, in whatever dtype this graph declares."""
    import numpy as np

    spec = sess.get_inputs()[0]
    dtype = _ORT_DTYPES.get(spec.type)
    if dtype is None:
        raise RuntimeError(
            f"{spec.name} has type {spec.type}, which this benchmark does not "
            f"know how to synthesise an input for"
        )
    return spec.name, np.random.randn(1, 3, imgsz, imgsz).astype(dtype)


def bench_onnx(onnx_path: Path, imgsz: int, provider: str) -> dict:
    """Latency under ONNX Runtime for one execution provider.

    Asking for a provider is not the same as getting it. If the CUDA EP's
    native library fails to load -- a version mismatch between the
    onnxruntime-gpu build and the installed CUDA runtime is the usual cause --
    ORT logs the failure and silently falls back to CPU. The session then
    reports honest numbers under a dishonest label, which is worse than an
    outright error. So verify what actually got bound and record it.
    """

    _register_cudnn()
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=[provider])
    actual = sess.get_providers()
    if provider not in actual:
        raise RuntimeError(
            f"requested {provider} but ORT bound {actual} -- "
            f"refusing to report a CPU measurement as GPU"
        )

    name, dummy = _dummy_input(sess, imgsz)

    def run(fn) -> dict:
        for _ in range(WARMUP_ITERS):
            fn()
        times = []
        for _ in range(TIMED_ITERS):
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000)
        return _summarise(times)

    transfer_inclusive = run(lambda: sess.run(None, {name: dummy}))
    if provider != "CUDAExecutionProvider":
        # On CPU there is no copy to separate out; the two regimes coincide.
        return {"core": transfer_inclusive, "transfer_inclusive": transfer_inclusive}

    # IOBinding keeps input and output on the device, which is the like-for-like
    # counterpart to PyTorch's GPU-resident measurement.
    binding = sess.io_binding()
    gpu_in = ort.OrtValue.ortvalue_from_numpy(dummy, "cuda", 0)
    binding.bind_input(name, "cuda", 0, dummy.dtype, gpu_in.shape(), gpu_in.data_ptr())
    for out in sess.get_outputs():
        binding.bind_output(out.name, "cuda", 0)
    core = run(lambda: sess.run_with_iobinding(binding))
    return {"core": core, "transfer_inclusive": transfer_inclusive}


def verify_cuda_placement(onnx_path: Path, imgsz: int) -> dict:
    """Count how many graph nodes actually ran on CUDA.

    `sess.get_providers()` only says which providers the session registered. It
    catches a CUDA EP that failed to load entirely - that path returns
    ['CPUExecutionProvider'] and bench_onnx already refuses it - but it cannot
    see *partial* fallback, where CUDA loads and individual unsupported ops run
    on CPU anyway. Claiming "genuine CUDA execution" from the provider list was
    therefore claiming more than the evidence supported.

    ORT's profiler records the provider per node, which is the direct
    measurement. Run once, outside the timed loops, and recorded in
    benchmark.json so the claim in the README has something behind it.
    """
    import collections
    import json as _json

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.enable_profiling = True
    _register_cudnn()
    sess = ort.InferenceSession(
        str(onnx_path), opts, providers=["CUDAExecutionProvider"]
    )
    name, dummy = _dummy_input(sess, imgsz)
    sess.run(None, {name: dummy})
    trace = Path(sess.end_profiling())
    try:
        events = _json.loads(trace.read_text(encoding="utf-8"))
    finally:
        trace.unlink(missing_ok=True)

    counts = collections.Counter(
        e["args"]["provider"]
        for e in events
        if e.get("cat") == "Node" and "provider" in e.get("args", {})
    )
    total = sum(counts.values())
    return {
        "nodes_total": total,
        "by_provider": dict(counts),
        "cpu_fallback_nodes": counts.get("CPUExecutionProvider", 0),
        "all_on_cuda": total > 0 and counts.get("CPUExecutionProvider", 0) == 0,
    }


def check_placement(placement: dict, allow_cpu_fallback: bool = False) -> None:
    """Act on the placement result instead of only recording it.

    `all_on_cuda` was written into benchmark.json and nothing ever read it, so a
    run with half its graph on the CPU still produced a row labelled "ONNX
    CUDA" and published it. The measurement would be real; the label on it
    would not. Registering CUDAExecutionProvider does not mean the nodes went
    there - ORT silently places whatever that provider cannot run on the CPU,
    and an opset or an operator it does not support is the ordinary way that
    happens.

    Separate from run_one so it is testable without a GPU, a checkpoint or
    ultralytics, which is the environment CI runs in.
    """
    if placement.get("all_on_cuda") or allow_cpu_fallback:
        return
    # The message says exactly where the placement can be read, because this
    # exception escapes main() before --out is written. It previously claimed
    # "the placement is recorded in the output either way", which sent a
    # reader to reports/benchmark.json - a file still holding the PREVIOUS
    # run, asserting all_on_cuda: true. The report would have contradicted the
    # run that had just failed, and the message was what pointed there.
    raise SystemExit(
        f"{placement.get('cpu_fallback_nodes')} of {placement.get('nodes_total')} "
        f"nodes ran on the CPU: {placement.get('by_provider')}. Those numbers "
        f"would be published under a CUDA label, so nothing was written: the "
        f"existing report still describes the previous run. The placement is "
        f"above and in this message. Pass --allow-cpu-fallback to measure and "
        f"publish it anyway."
    )


def _environment(imgsz: int) -> dict:
    """What a latency figure needs alongside it to mean anything.

    Milliseconds without the GPU, the driver stack and the iteration counts are
    a number nobody can reproduce or compare against their own hardware.
    """
    import os
    import platform

    import onnxruntime as ort
    import torch

    env = {
        "cpu": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "ort_intra_op_num_threads": ort.SessionOptions().intra_op_num_threads,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "cudnn_from_torch": _register_cudnn(),
        "imgsz": imgsz,
        "batch_size": 1,
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "statistic": "median and p95 over timed_iters, warmup discarded",
    }
    return env


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(
    weights: Path, onnx_path: Path, imgsz: int, exporter=None, half: bool = False
) -> None:
    """Export `weights` to `onnx_path`, writing nothing into the weights dir.

    Ultralytics writes the .onnx next to the .pt it loaded - verified: exporting
    a checkpoint from a temp directory put probe.onnx in that same directory. So
    exporting straight from a read-only mount fails even when the final
    destination is writable, which is what `docker run -v ...:/weights:ro` does
    on the very first run, when no cached graph exists yet. The checkpoint is
    copied into the destination directory and exported from there instead.

    `exporter` is injected so the path can be tested without a GPU, a
    checkpoint, or ultralytics installed at all.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    staged_weights = onnx_path.parent / weights.name
    copied = not staged_weights.exists() or not staged_weights.samefile(weights)
    if copied and staged_weights.exists():
        # Something else already lives at the staging path - same basename,
        # different file. Do NOT copy over it: the `finally` below deletes this
        # path, so the sequence would be "overwrite a stranger's file, then
        # remove it". `best.pt` is ultralytics' default output name and is
        # gitignored here, so the stranger is very often another checkpoint
        # that only exists on disk.
        raise FileExistsError(
            f"{staged_weights} already exists and is not {weights}. The export "
            f"stages the checkpoint there and deletes it afterwards, which "
            f"would destroy that file. Point --cache-dir somewhere else."
        )
    if copied:
        shutil.copy2(weights, staged_weights)
    try:
        if exporter is None:  # pragma: no cover - exercised by the real run
            from ultralytics import YOLO

            def exporter(src: Path) -> str:
                return YOLO(str(src)).export(
                    format="onnx",
                    imgsz=imgsz,
                    opset=ONNX_OPSET,
                    simplify=ONNX_SIMPLIFY,
                    dynamic=ONNX_DYNAMIC,
                    half=half,
                )

        produced = Path(exporter(staged_weights))
        # .replace, not .rename: rename refuses an existing target on Windows,
        # and a digest mismatch forcing a re-export is exactly the case where
        # the stale graph has to be overwritten.
        produced.replace(onnx_path)
    finally:
        if copied:
            staged_weights.unlink(missing_ok=True)


def _MANIFEST_STAMP(onnx_path: Path) -> Path:
    """The one place the stamp's filename is decided.

    Reader and writer disagreeing about it is what made the cache unhittable,
    so neither spells it out any more.
    """
    return onnx_path.with_suffix(".onnx.manifest.json")


def _export_manifest(
    onnx_path: Path, weights: Path, imgsz: int, half: bool = False
) -> dict:
    """Everything that determines whether a cached .onnx is the right one.

    A weights digest alone was not enough. It catches a changed checkpoint, but
    not a re-run at a different --imgsz, a different opset, simplify toggled, or
    an ultralytics/onnx/onnxslim upgrade that emits a different graph from the
    same inputs. Any of those produce a stale cache hit that benchmarks one
    model and reports another's accuracy.
    """
    # importlib.metadata rather than importing the packages: this function is
    # the cache key, so it has to be computable wherever the cache is consulted,
    # and CI runs the suite with torch/ultralytics deliberately absent. A
    # package that is not installed records None - you cannot export without it
    # anyway, so a real manifest is only ever written where all three exist.
    from importlib.metadata import PackageNotFoundError, version

    def installed(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "weights_sha256": _sha256(weights),
        "onnx_sha256": _sha256(onnx_path) if onnx_path.exists() else None,
        "imgsz": imgsz,
        # In the key, not merely recorded: an FP16 and an FP32 export of the
        # same checkpoint at the same size are different graphs, and without
        # this the second run would hit the first one's cache and benchmark
        # the wrong precision under the right label.
        "half": half,
        "opset": ONNX_OPSET,
        "simplify": ONNX_SIMPLIFY,
        "dynamic": ONNX_DYNAMIC,
        "ultralytics": installed("ultralytics"),
        "onnx": installed("onnx"),
        "onnxslim": installed("onnxslim"),
    }


def _export_is_current(
    onnx_path: Path, weights: Path, imgsz: int, half: bool = False
) -> bool:
    """Does this .onnx correspond to these weights, at this size, from this
    toolchain?

    The export used to be skipped whenever a file of the right name existed, so
    retraining and re-running benchmarked the OLD graph against the NEW
    checkpoint's accuracy - two different models in one row, with nothing on
    screen to say so.
    """
    stamp = _MANIFEST_STAMP(onnx_path)
    if not (onnx_path.exists() and stamp.exists()):
        return False
    try:
        recorded = json.loads(stamp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return recorded == _export_manifest(onnx_path, weights, imgsz, half)


def run_one(
    weights: Path,
    imgsz: int,
    data: str,
    device: str = "0",
    map_tolerance: float = MAP_TOLERANCE,
    cache_dir: Path | None = None,
    allow_cpu_fallback: bool = False,
    half: bool = False,
) -> dict:
    """Accuracy + latency across every available backend.

    `half` exports and measures the ONNX side in FP16. The PyTorch row stays
    FP32 deliberately: it is the baseline the accuracy gate compares against,
    and an FP16-vs-FP16 comparison would answer a different question -- "did
    the export preserve the half-precision model" rather than "what did half
    precision cost".
    """
    map_tolerance = _map_tolerance(map_tolerance)

    import torch
    from ultralytics import YOLO

    # --- accuracy -------------------------------------------------------
    metrics = YOLO(str(weights)).val(
        data=data, imgsz=imgsz, device=device, verbose=False
    )
    pytorch_map50 = _validated_map("PyTorch mAP50", metrics.box.map50)
    pytorch_map95 = _validated_map("PyTorch mAP50-95", metrics.box.map)
    row = {
        "imgsz": imgsz,
        "precision": "FP16" if half else "FP32",
        "mAP50": round(pytorch_map50, 4),
        "mAP50_95": round(pytorch_map95, 4),
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
    # cache_dir, not weights.parent: the export and its manifest are build
    # artefacts, and writing them beside the checkpoint means the weights mount
    # has to be writable. Defaults to the weights directory so a local run is
    # unchanged.
    # The precision is in the filename as well as in the manifest. The manifest
    # would catch a mismatch on its own, by forcing a re-export -- but into the
    # same path, so the two precisions would take turns overwriting each other
    # and every run would export.
    suffix = "_fp16" if half else ""
    onnx_path = (cache_dir or weights.parent) / f"{weights.stem}_{imgsz}{suffix}.onnx"
    if not _export_is_current(onnx_path, weights, imgsz, half):
        export_onnx(weights, onnx_path, imgsz, half=half)
        # Write the SAME stamp _export_is_current() reads. It wrote
        # .onnx.sha256 while the check looked for .onnx.manifest.json, so the
        # check never found a stamp, always returned False, and every run
        # re-exported - the cache existed but could not be hit.
        _MANIFEST_STAMP(onnx_path).write_text(
            json.dumps(_export_manifest(onnx_path, weights, imgsz, half), indent=2),
            encoding="utf-8",
        )
        # A stamp left by the previous scheme would otherwise sit there forever.
        onnx_path.with_suffix(".onnx.sha256").unlink(missing_ok=True)

    row["onnx_size_mb"] = round(onnx_path.stat().st_size / 1024**2, 2)
    row["export"] = _export_manifest(onnx_path, weights, imgsz, half)

    # Validate the ONNX graph itself. Reporting the PyTorch mAP beside ONNX
    # latency invites the reader to assume the export preserved accuracy, which
    # is an assumption and not a measurement -- opset choice, constant folding
    # and fp precision can all move it.
    onnx_metrics = YOLO(str(onnx_path), task="detect").val(
        data=data, imgsz=imgsz, device=device, half=half, verbose=False
    )
    onnx_map50 = _validated_map("ONNX mAP50", onnx_metrics.box.map50)
    onnx_map95 = _validated_map("ONNX mAP50-95", onnx_metrics.box.map)
    row["accuracy"] = {
        "pytorch": {
            "mAP50": round(pytorch_map50, 4),
            "mAP50_95": round(pytorch_map95, 4),
        },
        "onnx": {
            "mAP50": round(onnx_map50, 4),
            "mAP50_95": round(onnx_map95, 4),
        },
    }
    d50 = row["accuracy"]["onnx"]["mAP50"] - row["accuracy"]["pytorch"]["mAP50"]
    d95 = row["accuracy"]["onnx"]["mAP50_95"] - row["accuracy"]["pytorch"]["mAP50_95"]
    row["accuracy"]["delta"] = {
        "mAP50": round(d50, 4),
        "mAP50_95": round(d95, 4),
        "tolerance": map_tolerance,
    }
    print(
        f"  ONNX accuracy : mAP50 {row['accuracy']['onnx']['mAP50']} "
        f"mAP50-95 {row['accuracy']['onnx']['mAP50_95']}  "
        f"(delta {d50:+.4f} / {d95:+.4f})"
    )
    if max(abs(d50), abs(d95)) > map_tolerance:
        raise SystemExit(
            f"ONNX accuracy differs from PyTorch by more than {map_tolerance}: "
            f"mAP50 {d50:+.4f}, mAP50-95 {d95:+.4f}. An export that changes the "
            f"model is not a deployment artefact -- investigate before publishing "
            f"latency for it."
        )

    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        row["onnx_cuda"] = bench_onnx(onnx_path, imgsz, "CUDAExecutionProvider")
        placement = verify_cuda_placement(onnx_path, imgsz)
        row["onnx_cuda_placement"] = placement
        print(f"  ONNX node placement: {placement}")
        print(f"  ONNXRuntime GPU: {row['onnx_cuda']}")
        check_placement(placement, allow_cpu_fallback)
    row["onnx_cpu"] = bench_onnx(onnx_path, imgsz, "CPUExecutionProvider")
    print(f"  ONNXRuntime CPU: {row['onnx_cpu']}")

    # The FP16 number is only useful as a ratio, and this repository's own
    # README says latency is reproducible within a session and not across them
    # -- it measured an 11% spread between sessions on this machine. Comparing
    # a fresh FP16 figure against a committed FP32 one would therefore be
    # reporting session noise as a speedup, which is the exact mistake that
    # paragraph exists to warn about.
    #
    # So the FP32 graph is measured again here, in this process, immediately
    # after. Both numbers then come from one session and the ratio between them
    # means something. The export is cached, so this costs a benchmark loop and
    # not a re-export.
    if half and "CUDAExecutionProvider" in available:
        fp32_path = (cache_dir or weights.parent) / f"{weights.stem}_{imgsz}.onnx"
        if not _export_is_current(fp32_path, weights, imgsz, half=False):
            export_onnx(weights, fp32_path, imgsz, half=False)
            _MANIFEST_STAMP(fp32_path).write_text(
                json.dumps(
                    _export_manifest(fp32_path, weights, imgsz, half=False), indent=2
                ),
                encoding="utf-8",
            )
        reference = bench_onnx(fp32_path, imgsz, "CUDAExecutionProvider")
        row["fp32_reference"] = {
            "onnx_cuda": reference,
            "onnx_size_mb": round(fp32_path.stat().st_size / 1024**2, 2),
            "note": (
                "The FP32 graph, benchmarked in this same process so the "
                "speedup is a within-session ratio. Its accuracy is not "
                "re-validated here; reports/benchmark.json is the FP32 record."
            ),
        }
        speedup = (
            1 - row["onnx_cuda"]["core"]["median_ms"] / reference["core"]["median_ms"]
        )
        row["fp32_reference"]["core_speedup_pct"] = round(100 * speedup, 1)
        print(f"  FP32 reference (same session): {reference['core']}")
        print(f"  FP16 core speedup: {100 * speedup:.1f}%")

    row["environment"] = _environment(imgsz)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--data", default="VisDrone.yaml")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument(
        "--device",
        default="0",
        help="Device for the accuracy .val() pass, e.g. '0' or "
        "'cpu'. The PyTorch-CUDA and ONNX-CUDA latency rows "
        "always need an actual CUDA device regardless of "
        "this setting -- that comparison is CUDA vs CPU by "
        "definition -- and are skipped, not forced, when "
        "one isn't available.",
    )
    p.add_argument(
        "--map-tolerance",
        type=_map_tolerance,
        default=MAP_TOLERANCE,
        help="Fail if ONNX mAP differs from PyTorch by more than "
        "this. An export is meant to change speed, not the model.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where the exported .onnx and its manifest are written. "
        "Defaults to the weights directory, which is fine locally "
        "but fails when the weights are mounted read-only - the "
        "container mounts /weights:ro and points this at /out.",
    )
    p.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Publish the ONNX-CUDA row even when some nodes were "
        "placed on the CPU. Off by default: a partly-CPU graph "
        "measured under a CUDA label is a wrong number, not a "
        "slow one.",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="Export and measure the ONNX side in FP16. The PyTorch row "
        "stays FP32: it is the baseline the accuracy gate compares "
        "against, so the delta reports what half precision cost rather "
        "than whether an FP16 export matches an FP16 model.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to reports/benchmark.json, or "
        "reports/benchmark_fp16.json with --half, so a half-precision run "
        "cannot silently replace the full-precision report.",
    )
    args = p.parse_args()

    if args.out is None:
        args.out = REPORTS_DIR / (
            "benchmark_fp16.json" if args.half else "benchmark.json"
        )

    row = run_one(
        args.weights,
        args.imgsz,
        args.data,
        args.device,
        args.map_tolerance,
        args.cache_dir,
        args.allow_cpu_fallback,
        args.half,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Standard JSON has no NaN or Infinity. The accuracy values are checked
    # above; this final guard also catches a broken latency measurement before
    # it can replace a previously valid report with a file other tools reject.
    args.out.write_text(json.dumps(row, indent=2, allow_nan=False))

    def med(key: str, regime: str) -> str:
        v = row.get(key, {}).get(regime, {}).get("median_ms")
        return "n/a" if v is None else f"{v:.2f} ms"

    print(f"\n{'=' * 60}")
    print(
        f"imgsz {row['imgsz']}   {row['precision']} ONNX   "
        f"mAP50 {row['mAP50']}   mAP50-95 {row['mAP50_95']}  (PyTorch FP32)"
    )
    print(f"{'':16}{'core':>12}{'+transfer':>12}")
    for label, key in (
        ("PyTorch CUDA", "pytorch_cuda"),
        ("ONNX  CUDA", "onnx_cuda"),
        ("ONNX  CPU", "onnx_cpu"),
    ):
        print(f"{label:16}{med(key, 'core'):>12}{med(key, 'transfer_inclusive'):>12}")
    print("core = in/out resident on device; +transfer = CPU in, CPU out")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
