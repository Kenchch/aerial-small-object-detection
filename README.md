# Small-Object Detection on Drone Imagery

[![CI](https://github.com/Kenchch/aerial-small-object-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Kenchch/aerial-small-object-detection/actions/workflows/ci.yml)

YOLO11n detection on VisDrone2019, with ONNX deployment and a tracking pipeline
for studying accuracy, GPU placement and inference latency on a laptop GPU.

## Results at a glance

| Evidence | Result |
|---|---|
| Training | YOLO11n, 1024 px, 50 epochs, RTX 2070 Max-Q |
| Standalone validation | mAP50 0.375 / mAP50-95 0.222 |
| Held-out test-dev | mAP50 0.318 / mAP50-95 0.183 on 1,610 images, scored once after selection |
| ONNX CUDA core latency | 10.4 ms; 17% faster than eager |
| CUDA placement and parity | 238/238 nodes; mAP50-95 delta +0.0007 |
| ONNX CPU | Approximately 12× slower than ONNX CUDA in this benchmark |
| Object scale / tracking | 92.4% of validation boxes small at 640 px; 22.9 FPS steady-state, median of five repeats (17.8-24.9) |

Latency is reproducible within a session and not across them. Three
consecutive runs of `src/benchmark.py` on the same machine and checkpoint gave
ONNX CUDA core medians of 10.35, 10.37 and 10.43 ms — a spread under 1% — while
an earlier session on the same GPU recorded 9.3 ms. The 100 timed iterations
behind each median control the noise inside a run; they say nothing about
driver version, thermal state or what else the machine was doing. Read these as
one machine's numbers on one day, not as a device specification.

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

- The checkpoint was selected on validation, so the val figures are optimistic by construction. The test-dev row above is the held-out number: it was scored once, after selection, and is 0.0385 mAP50-95 below val.
- There is no matched 640 px training ablation yet.
- Benchmark results depend on hardware, precision and transfer boundaries.
- The demo pans over one real image; it does not measure tracking accuracy on moving objects.
- Fifty epochs and one training run do not establish an optimal detector.
- Ultralytics and published weights use AGPL-3.0; see [licence and attribution](docs/DESIGN.md#licence-and-attribution).

[Design notes](docs/DESIGN.md)

## Additional measurements

**Test-dev.** The v1.0 checkpoint was selected on validation. A separate
labelled test-dev evaluation at 1024 px gave mAP50 **0.3183** / mAP50-95
**0.1831** across 1,610 images — [evidence](reports/evaluation_test.json).
Reproduce with `src/evaluate.py --split test`.

**FP16.** [FP16 ONNX](reports/benchmark_fp16.json) reached mAP50-95 **0.2211**
on val against the FP32 PyTorch baseline's 0.2216 — a delta of **-0.0005**,
inside the same 0.002 gate the FP32 export has to pass.
The graph is 5.24 MB against 10.37, and **240/240 nodes ran on CUDA**.

Core latency was **7.69 ms** against **9.24 ms** for the FP32 graph:
**16.8% faster**. Both figures come from the same process, which is the only
way this ratio means anything — the FP32 graph is re-benchmarked during a
`--half` run for exactly that reason. Measured against the 10.43 ms in the
table above, from a different session, the same FP16 result would read 26%
faster; the difference between 16.8 and 26 is between-session variance, not
precision. Transfer-inclusive it was 9.92 ms.

Reproduce with `python src/benchmark.py --weights <best.pt> --data
<dataset.yaml> --half`.

**Tracking repeats.** [Five repeats](reports/tracking_repeats/summary.json),
each decoding and encoding the same 90-frame synthetic pan, gave
**8.4 FPS median**, range **7.2-8.9 FPS**. Machine-specific, not general
edge-device performance. Rebuild with `python scripts/summarize_tracking.py`.

**GPU smoke test.** Opt-in: `RUN_GPU=1 AERIAL_TEST_IMAGE=<local image> pytest
-m gpu`. It downloads and SHA-verifies the v1.0 checkpoint.

## How this was built

I set the problem, the data contracts and the quality rules, ran the benchmarks
and reviewed every diff; Claude Code and OpenAI Codex drafted code, refactored
and scaffolded tests. The full note — including the `Co-Authored-By` trailers
removed from this repository's history on 6 September 2026 — is on my profile:
[How I use AI tools](https://github.com/Kenchch/Kenchch#how-i-use-ai-tools).

Runtime upgrade decisions are recorded in [dependency review](docs/DEPENDENCIES.md).
