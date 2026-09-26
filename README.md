# Small-Object Detection on Drone Imagery

[![CI](https://github.com/Kenchch/aerial-small-object-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Kenchch/aerial-small-object-detection/actions/workflows/ci.yml)

How accurately, and how fast, a small model finds people and vehicles in drone
footage, where most objects are only a few pixels across. YOLO11n on
VisDrone2019, with ONNX deployment and a tracking pipeline for studying
accuracy, GPU placement and inference latency on a laptop GPU.

## Results at a glance

| Evidence | Result |
|---|---|
| Training | YOLO11n, 1024 px, 50 epochs, RTX 2070 Max-Q |
| Validation (the checkpoint was chosen on it) | mAP50 0.375 / mAP50-95 0.222 |
| Held-out test-dev | mAP50 0.318 / mAP50-95 0.183 on 1,610 images, only ever evaluated with the fixed v1.0 checkpoint; the gap to validation is the honest number |
| ONNX CUDA core latency | 10.0 ms against 12.8 ms eager PyTorch: 22% faster in this run, 7-26% across 4 sessions (median 23%) |
| CUDA placement and parity | 238/238 nodes; mAP50-95 delta +0.0000 against PyTorch at the same letterbox |
| ONNX CPU | Approximately 10× slower than ONNX CUDA, both host-in/host-out, in this benchmark |
| Object scale / tracking | 92.4% of validation boxes small at 640 px; 25.8 FPS steady-state, median of five repeats (25.0-27.0) |

Latency is reproducible within a session and not across them.
[Four separate runs](reports/benchmark_repeats/summary.json) of `src/benchmark.py` on the same
machine and checkpoint gave ONNX CUDA core medians from 9.54 to 11.70 ms,
and the eager-vs-ONNX ratio moved with them: 7% to 26% faster, median
23%. Each ratio is within one session; none compares two. The 100 timed iterations
behind each median control the noise inside a run; they say nothing about
driver version, thermal state or what else the machine was doing. Read these as
one machine's numbers on one day, not as a device specification.

The steady-state tracking FPS is measured over frames 2-90, after the cold
first frame; end-to-end throughput including it is lower, below.
See [benchmark](reports/benchmark.json), [tracking](reports/tracking.json) and
[full numerical evidence](docs/DESIGN.md). Training-time AMP validation differs
slightly from standalone fp32 evaluation of the same checkpoint.

## What I built

- Label-size analysis to motivate the 1024 px training resolution.
- Detection evaluation and per-class accuracy reports.
- ONNX export, accuracy parity checks and CUDA node-placement verification.
- Separate core and transfer-inclusive latency measurements.
- A throughput profile of the tracking pipeline, run over a synthetic pan of
  one real image: it measures speed, not tracking accuracy.

## Run it

Follow [environment setup](docs/DESIGN.md#setup) for the GPU runtime and dataset.
Download the recorded checkpoint before evaluation:

```bash
mkdir -p runs/n_1024/weights
curl -fL --retry 3 -o runs/n_1024/weights/best.pt \
  https://github.com/Kenchch/aerial-small-object-detection/releases/download/v1.0/best.pt
echo "8786213fc488fc8b94bdb1c8c576e377eb8f2befaa258e0338b3c5efbc26382e  runs/n_1024/weights/best.pt" | sha256sum -c -
```

On macOS use `shasum -a 256 -c -`; on Windows compare `Get-FileHash` output
with the digest.

[Training, evaluation and tracking commands](docs/DESIGN.md#usage).

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use stubs and committed reports; they do not establish GPU performance.

## Limits

- The checkpoint was selected on validation, so the val figures are optimistic by construction. The test-dev row above is the held-out number: it was never used for model or threshold selection (re-runs used the same fixed checkpoint), and is 0.0385 mAP50-95 below val.
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
Reproduce with `python src/evaluate.py --weights runs/n_1024/weights/best.pt
--data docker/VisDrone.yaml --data-root "$DATASETS/VisDrone" --split test`.

**FP16.** [FP16 ONNX](reports/benchmark_fp16.json) reached mAP50-95 **0.2211**
on val against the FP32 PyTorch baseline's 0.2223, both square-letterboxed
— a delta of **-0.0012**, inside the same 0.002 gate the FP32 export has to pass.
(The FP32 export itself measures +0.0000 on that protocol, so this is what
half precision cost.)
The graph is 5.24 MiB against 10.37, and **240/240 nodes ran on CUDA**.

Core latency was **7.06 ms** against **9.40 ms** for the FP32 graph in the
same process. Timed in 10 alternating blocks, so neither graph always ran on a
cooler or hotter card, FP16 was **24.6% faster**, with the per-block
ratio ranging 18.8-37.5%: the effect is real and its size is not
precise. Measured against the 10.03 ms in the table above, from a different
session, the same FP16 result would read 30% faster - a comparison
across sessions, which is why `--half` re-benchmarks the FP32 graph.
Transfer-inclusive it was 9.29 ms.

Reproduce with `python src/benchmark.py --weights <best.pt> --data
<dataset.yaml> --half`.

**Tracking repeats.** [Five repeats](reports/tracking_repeats/summary.json),
each decoding and encoding the same 90-frame synthetic pan, gave
**11.9 FPS median** end-to-end, range **11.4-12.1 FPS**: wall time over all 90
frames, including a 4.0-4.3 s first-frame warm-up. The 25.8 FPS steady-state
figure above comes from the same runs with the warm-up excluded (frames
2-90 over the time they took). Machine-specific, not general
edge-device performance. Rebuild with `python scripts/summarize_tracking.py`.

**GPU smoke test.** Opt-in: `RUN_GPU=1 AERIAL_TEST_IMAGE=<local image> pytest
-m gpu`. It downloads and SHA-verifies the v1.0 checkpoint.

## How this was built

I used Claude Code and OpenAI Codex as drafting tools. The problem, the data
contracts and the quality rules are mine, and so are the benchmark runs and the
review: every generated change was read and run before it was committed. The
tools drafted code, refactored and scaffolded tests.

Commits made before 6 September 2026 carried `Co-Authored-By` trailers naming
these tools. They were removed when I rewrote that history; most commits since
then carry them, and each pull request states its own AI involvement.

Runtime upgrade decisions are recorded in [dependency review](docs/DEPENDENCIES.md).
