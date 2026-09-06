# Small-Object Detection on Drone Imagery

[![CI](https://github.com/Kenchch/aerial-small-object-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Kenchch/aerial-small-object-detection/actions/workflows/ci.yml)

YOLO11n detection on VisDrone2019, with ONNX deployment and a tracking pipeline
for studying accuracy, GPU placement and inference latency on a laptop GPU.

## Results at a glance

| Evidence | Result |
|---|---|
| Training | YOLO11n, 1024 px, 50 epochs, RTX 2070 Max-Q |
| Standalone validation | mAP50 0.375 / mAP50-95 0.222 |
| ONNX CUDA core latency | 9.3 ms; 21% faster than eager |
| CUDA placement and parity | 238/238 nodes; mAP50-95 delta +0.0007 |
| ONNX CPU | Approximately 11× slower than ONNX CUDA in this benchmark |
| Object scale / tracking | 92.4% of validation boxes small at 640 px; derived steady-state 25.8 FPS |

The tracking FPS is derived from stage medians, not measured end-to-end throughput.
See [benchmark](reports/benchmark.json), [tracking](reports/tracking.json) and
[full numerical evidence](docs/DESIGN.md). Training-time AMP validation differs
slightly from standalone fp32 evaluation of the same checkpoint.

## What I built

- Label-size analysis to motivate the 1024 px training resolution.
- Detection evaluation and per-class accuracy reports.
- ONNX export, accuracy parity checks and CUDA node-placement verification.
- Separate core and transfer-inclusive latency measurements.
- A profiled tracking demo with video digests and source provenance.

## Run it

Follow [environment setup](docs/DESIGN.md#setup) for the GPU runtime and dataset.
Download the recorded checkpoint before evaluation:

```bash
mkdir -p runs/n_1024/weights
curl -L -o runs/n_1024/weights/best.pt \
  https://github.com/Kenchch/aerial-small-object-detection/releases/download/v1.0/best.pt
```

[Training, evaluation and tracking commands](docs/DESIGN.md#usage).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use stubs and committed reports; they do not establish GPU performance.

## Limits

- Validation was used for checkpoint selection; no new test-dev result is claimed.
- There is no matched 640 px training ablation yet.
- Benchmark results depend on hardware, precision and transfer boundaries.
- The demo pans over one real image; it does not measure tracking accuracy on moving objects.
- Fifty epochs and one training run do not establish an optimal detector.
- Ultralytics and published weights use AGPL-3.0; see [licence and attribution](docs/DESIGN.md#licence-and-attribution).

[Design notes](docs/DESIGN.md)
