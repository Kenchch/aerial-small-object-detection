"""Export and validate FP16 ONNX, then measure transfer-inclusive CUDA latency."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=Path("reports/benchmark_fp16.json"))
    args = parser.parse_args()
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    from ultralytics import __version__ as ultra_version

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    model = YOLO(str(args.weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            half=True,
            device=0,
            dynamic=False,
            simplify=True,
            opset=17,
        )
    )
    metrics = YOLO(str(exported), task="detect").val(
        data=args.data, imgsz=args.imgsz, device=0, half=True, verbose=False
    )
    options = ort.SessionOptions()
    session = ort.InferenceSession(
        str(exported), sess_options=options, providers=["CUDAExecutionProvider"]
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA provider was not activated")
    spec = session.get_inputs()[0]
    if spec.type != "tensor(float16)":
        raise RuntimeError(f"Export is not FP16: {spec.type}")
    sample = (
        np.random.default_rng(0)
        .random((1, 3, args.imgsz, args.imgsz))
        .astype(np.float16)
    )
    for _ in range(10):
        session.run(None, {spec.name: sample})
    elapsed = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {spec.name: sample})
        elapsed.append((time.perf_counter() - start) * 1000)
    evidence = {
        "precision": "FP16",
        "split": "val",
        "imgsz": args.imgsz,
        "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "latency": {
            "scope": "ORT session.run, host-to-device + forward + device-to-host; no preprocessing/NMS",
            "input": "seed-0 random float16 tensor; batch=1",
            "warmup": 10,
            "iterations": 100,
            "median_ms": float(np.median(elapsed)),
            "p95_ms": float(np.percentile(elapsed, 95)),
        },
        "environment": {
            "torch": torch.__version__,
            "ultralytics": ultra_version,
            "onnxruntime": ort.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cpu": platform.processor(),
            "providers": session.get_providers(),
            "intra_op_num_threads": options.intra_op_num_threads,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
