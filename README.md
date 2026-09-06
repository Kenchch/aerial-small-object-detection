# Small-Object Detection on Drone Imagery

[![CI](https://github.com/Kenchch/aerial-small-object-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Kenchch/aerial-small-object-detection/actions/workflows/ci.yml)

Object detection, tracking and deployment benchmarking with YOLO11 on
VisDrone2019 — trained, evaluated, exported to ONNX, and run through a
tracking pipeline with a measured deployment profile.

Aerial frames are large and the things worth detecting in them are tiny. The
label statistics are measured before training to justify the input resolution
chosen, rather than assumed.

## Demo at a glance

![Object detection and tracking pipeline demonstration](reports/tracking_demo.gif)

This clip pans across a single real image to demonstrate the processing
pipeline. The camera motion is synthetic; it does not demonstrate tracking
accuracy on independently moving objects.

The study compares detection accuracy with inference speed on a laptop GPU.
Start with [Results](#results) for accuracy, [export and latency](#deployment-export-and-latency)
for speed, or [Usage](#usage) to reproduce the experiment.

> Uses Ultralytics (AGPL-3.0), and the published weights are AGPL-3.0 rather
> than MIT. If you intend to reuse any of this, read
> [Licence and attribution](#licence-and-attribution) first.

## Contents

- [Why this dataset](#why-this-dataset)
- [The problem, quantified](#the-problem-quantified)
- [Results](#results)
- [Deployment: export and latency](#deployment-export-and-latency)
- [Deployment: tracking](#deployment-tracking)
- [Engineering notes](#engineering-notes) — where the measured numbers came from, including four things that were wrong at first
- [What this does not establish](#what-this-does-not-establish)
- [Setup](#setup) · [Usage](#usage) · [Layout](#layout) · [Tests](#tests)
- [Licence and attribution](#licence-and-attribution)

---

## Why this dataset

[VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset) is drone-captured
imagery across 10 classes (pedestrian, car, van, truck, bus, motor, …).

- **6,471** training / **548** validation frames. Resolutions are mixed, not
  uniform: the val split is entirely 16:9 (1360×765, 960×540 or 1920×1080);
  train mixes those with 4:3 frames as large as 2000×1500.
- **38,759** annotated boxes in the validation split alone — a mean of **~71
  objects per frame**, against roughly 7 for COCO.

That object density is not incidental — it drives most of the engineering
decisions below.

![Training-split label distribution: class counts, spatial density and box width/height](runs/n_1024/labels.jpg)

Training split (6,471 frames), not val. The width/height scatter bottom-right
is the same "boxes are tiny" fact as the analysis below, from a different
angle: normalised box width and height both cluster near 0, with essentially
nothing past ~0.1 on either axis.

The bar labels are Ultralytics' own count and differ slightly from a direct
read of the label files: `car` 144,866 vs 144,867, `pedestrian` 79,335 vs
79,337, `motor` 29,646 vs 29,647. The other seven classes match exactly, so
the bars sum to 343,201 against a direct total of 343,205. The gap is four
boxes confined to the three largest classes, not a per-class offset. Counts
quoted elsewhere in this README are the direct ones (`class_balance` in
`reports/evaluation_train.json`), so adding up the bars will not quite
reproduce them.

---

## The problem, quantified

Before choosing an input resolution, measure what the labels actually look
like. Box areas are normalised in YOLO format, so `w*h` is directly the
fraction of frame covered — no need to open the images.

| ground-truth box scale (val, 38,759 boxes) | share |
| --- | --- |
| small — under 32×32 px | **92.4 %** |
| medium — 32×32 to 96×96 px | 7.5 % |
| large — over 96×96 px | 0.1 % |

COCO's small/medium/large thresholds, applied to the pixel area each box
actually occupies at a 640 px input *after letterboxing*. Applying them as a
fixed share of the frame instead (the usual `<0.25 %` shortcut) assumes the
image is squashed to 640×640, which is the very thing the paragraph below
says YOLO does not do.

From `reports/evaluation.json`. The train split is in
`reports/evaluation_train.json` and is slightly less extreme — 90.5 % small
across 343,205 boxes — because it mixes 4:3 frames with the val split's
uniform 16:9.

The median box covers 0.055 % of the frame. Converting that to a pixel size
needs the source image's actual dimensions, not just the target resolution —
YOLO letterboxes (scales both axes by the same factor, padding the short
side) rather than stretching to a square, so a naive `sqrt(area) * imgsz`
overstates the box's true rendered size whenever the source isn't already
square. Computed per-image from the val split's real dimensions (100% 16:9
here): the median box is a **11 px** object at the YOLO default of 640 px
input, or **18 px** at 1024 px. Better than nine in ten objects in this
dataset are in the regime where downscaling destroys them, which is why this
trains at 1024px rather than the default.

---

## Results

YOLO11n, 1024px, 50 epochs, 197.6 min on an RTX 2070 Max-Q.

**mAP50 0.373, mAP50-95 0.221.** This is the training run's own final
validation (`runs/n_1024/results.csv`, the pass used to select the checkpoint).
`src/benchmark.py` re-validates independently to pair accuracy with its own
latency numbers below and gets a very slightly different figure — mAP50
0.3748, mAP50-95 0.2216, in `reports/benchmark.json`. The in-training pass
runs under AMP (`amp: true` in `runs/n_1024/args.yaml`, so fp16) and the
standalone one in fp32, so the two are not numerically identical. Same
weights, same split.

![Training curves: box/cls/dfl losses and val metrics over 50 epochs](runs/n_1024/results.png)

Both mAP curves are still rising at epoch 50 — see "What this does not
establish" below.

Per-class breakdown, sorted by difficulty, shows why the aggregate number is
not the interesting one. These rows come from `src/evaluate.py`'s standalone
fp32 pass (`reports/evaluation.json`), so their arithmetic mean is exactly the
0.3748 / 0.2216 above and *not* results.csv's 0.3733 / 0.2209 — worth checking
against the table yourself. `share of labels` is the val split, matching the
metrics beside it.

| class | P | R | mAP50 | mAP50-95 | share of labels |
| --- | --- | --- | --- | --- | --- |
| bicycle | 0.298 | 0.152 | 0.111 | **0.049** | 3.3 % |
| awning-tricycle | 0.267 | 0.177 | 0.125 | 0.081 | 1.4 % |
| people | 0.524 | 0.300 | 0.304 | 0.113 | 13.2 % |
| tricycle | 0.407 | 0.296 | 0.254 | 0.142 | 2.7 % |
| motor | 0.507 | 0.434 | 0.423 | 0.182 | 12.6 % |
| pedestrian | 0.532 | 0.445 | 0.440 | 0.198 | 22.8 % |
| truck | 0.479 | 0.373 | 0.357 | 0.239 | 1.9 % |
| van | 0.516 | 0.428 | 0.423 | 0.290 | 5.1 % |
| bus | 0.725 | 0.466 | 0.515 | 0.374 | 0.6 % |
| car | 0.697 | 0.797 | 0.796 | **0.548** | 36.3 % |

**The spread between best and worst class (11×) is not explained by label
frequency.** Training signal has to be counted on the train split, not the val
shares above: there `pedestrian` has 79,337 boxes to `bus`'s 5,926, so 13×
more — and `bus` is not even the rarest training class, `awning-tricycle`
(3,246) is. Those counts are in `reports/evaluation_train.json`
(`python src/evaluate.py --split train`), so they can be checked without
downloading the dataset. `bus` still scores nearly double `pedestrian` on
mAP50-95 (0.374 vs 0.198). What separates them is apparent size: a bus occupies
tens of pixels from altitude, a pedestrian occupies a handful, and the
label-scale table above is exactly where that prediction came from.

The confusion matrix below isolates this. At the deployed threshold
`pedestrian` and `bus` are recognised about equally well (0.42 of true
instances correct against 0.44) and `pedestrian` still scores half the
mAP50-95. Since mAP50-95 averages IoU thresholds from 0.5 to 0.95, a two-pixel
boundary error on an 11 px box costs what it would not cost on a bus. The gap
is localisation precision under scale, not recognition.

![Normalised confusion matrix on the val split](runs/n_1024/confusion_matrix_normalized.png)

Columns are the true class and sum to 1, so each column splits three ways:
correct, missed (the `background` row), and misclassified (everything else).
Splitting them apart is the part the aggregate mAP hides, and the two failure
modes are not distributed evenly.

`src/evaluate.py` emits this split per class as `error_split` in
`reports/evaluation.json`, so the numbers below are citable rather than read
off the image:

Built at **conf = 0.25** — the threshold `src/track.py` deploys at — and
recorded as `error_split_conf` beside the split itself. That is not the
operating point of the P/R figures above, which need `conf → 0` or the PR
curve is truncated, and the two must not be read as one measurement.

| true class | correct | missed | misclassified | mostly as |
| --- | --- | --- | --- | --- |
| van | 0.32 | 0.26 | **0.42** | car |
| awning-tricycle | 0.11 | **0.55** | 0.34 | car |
| truck | 0.32 | **0.42** | 0.25 | car |
| bus | 0.44 | **0.36** | 0.20 | truck |
| tricycle | 0.23 | **0.57** | 0.20 | motor |
| bicycle | 0.11 | **0.75** | 0.14 | motor |
| people | 0.24 | **0.66** | 0.10 | pedestrian |
| motor | 0.36 | **0.57** | 0.07 | people |
| pedestrian | 0.42 | **0.55** | 0.03 | people |
| car | 0.77 | **0.21** | 0.02 | van |

**Misclassification outweighs missing for one of the ten classes.** Missing
dominates for the other nine — the model does not emit a box at all, at the
threshold it would be deployed at.

This reverses what this section used to claim, and the reason is worth stating.
The split was previously computed from the validator's own confusion matrix,
which ultralytics ≥ 8.4 builds at `args.conf = 0.001` (8.3.x clamped it to 0.25
internally; 8.4 dropped the clamp). At 0.001 the matrix is assembled from up to
`max_det = 300` boxes per image against ~71 real objects, and its matching is
class-agnostic, IoU-only and greedy on IoU — confidence plays no part in it. So
the sub-threshold junk tail wins ground-truth boxes that no deployment would
ever see, moving them out of "missed" and into "misclassified". On that
arithmetic seven of ten classes looked misclassification-limited; at the
deployed threshold, one does.

What survives is `van`, which is genuinely confused rather than missed
(0.42 against 0.26), and it goes to `car` — as do
`truck` and `awning-tricycle` for the share of their error that is confusion.
The four-wheeled classes collapsing into `car` (42 % of training boxes) is
real; it is just not the dominant error mode.

The label-scale analysis therefore stands rather than being narrowed. High
recall loss on the smallest classes is the largest single component of the
error budget, and it is consistent with training at 1024px rather than 640.

---

## Deployment: export and latency

A detector benchmarked on mAP alone says nothing about whether it fits a
latency budget, so the trained model is exported to ONNX and timed against the
raw PyTorch model across backends.

| backend | core | + host transfer |
| --- | --- | --- |
| PyTorch (CUDA, eager) | 11.74 ms · 85.2 FPS | 14.48 ms · 69.1 FPS |
| ONNX Runtime (CUDA) | **9.27 ms · 107.9 FPS** | **11.49 ms · 87.0 FPS** |
| ONNX Runtime (CPU) | 102.94 ms · 9.7 FPS | 102.94 ms · 9.7 FPS |

**Two regimes, because comparing across them is how this gets read wrong.**
`core` feeds a tensor already resident in VRAM and leaves the output there.
`+ host transfer` starts from a CPU array and brings the output back.
ONNX Runtime's `sess.run` takes and returns numpy, so it is transfer-inclusive
by construction; timing that against a GPU-resident PyTorch forward — which is
what this table used to do — charges ONNX ~2.2 ms of copying PyTorch never
paid, and reported the export as **2 % faster when like-for-like it is 21 %**.
On CPU there is no copy to separate, so the two columns coincide.

ONNX is 21 % faster core-to-core and 21 % transfer-to-transfer. The more
decisive number is still CPU: **11.1× slower** than ONNX on GPU — the case for
keeping inference on a GPU-equipped edge device rather than falling back to CPU.

Median of 100 timed iterations after 20 warmup, with `torch.cuda.synchronize()`
before each stop — GPU work is asynchronous, so timing without it measures
kernel *launch*, not the kernel. The full environment is recorded in
`reports/benchmark.json` alongside the numbers:

```
RTX 2070 Max-Q · CUDA 12.4 · cuDNN 9.1.0 · torch 2.6.0+cu124
onnxruntime-gpu 1.20.2 · imgsz 1024 · batch 1 · 20 warmup / 100 timed
```

**The export is validated, not assumed.** Latency beside a PyTorch mAP
invites the reader to take it that ONNX kept the accuracy, which is an
assumption: opset choice, constant folding and precision can all move it.
Both backends are validated on the same split — PyTorch mAP50 0.3748 / mAP50-95 0.2216, ONNX 0.3752 / 0.2223, a delta of +0.0004 / +0.0007 — and the run fails if it exceeds 0.01. The `.onnx` also carries the
sha256 of the checkpoint it came from, so retraining forces a re-export
rather than benchmarking yesterday's graph against today's weights.

The GPU path is genuine CUDA execution, and that is measured rather than
inferred. `sess.get_providers()` only reports which providers the session
registered — it catches a CUDA library that failed to load entirely, but not
*partial* fallback, where CUDA loads and unsupported ops quietly run on CPU
anyway. ORT's profiler gives per-node placement: **238 of 238 nodes on CUDA, 0 on CPU**, recorded as
`onnx_cuda_placement` in `reports/benchmark.json`.

---

## Deployment: tracking

`src/track.py` runs the trained detector through ByteTrack on a video source
and reports a staged latency profile — decode, detect+track, annotate, encode,
plus the once-per-run capture/writer open and flush — reconciled against
wall-clock time. What the reconciliation gives is a *published* remainder, not
a guarantee of none: `coverage_pct` and `unaccounted_ms` say how much of the
frame budget the stages account for, and the gap is Python-level overhead
between them.

**Footage.** VisDrone2019-DET is sampled from video at a 200-frame interval —
readable straight off the filenames, whose frame-index field steps
`1, 201, 401, 601 …` — so it cannot support a tracking demo on its own.
`src/make_demo_clip.py` instead pans a crop window across one real, densely
labelled frame, producing synthetic camera motion over real imagery. That
exercises the full decode → detect → track → encode pipeline and ByteTrack's
association logic under realistic latency; it does not exercise independently
moving objects or occlusion, so track-continuity numbers below describe the
*pipeline*, not tracking accuracy against ground truth.

The [demo preview above](#demo-at-a-glance) shows this pipeline in action.

90 frames, 1024px:

| stage | median | mean |
| --- | --- | --- |
| decode | 0.94 ms | 1.1 ms |
| detect + track | 32.21 ms | **97.93 ms** |
| ├ preprocess | 5.19 ms | |
| ├ forward | 13.16 ms | |
| ├ postprocess (NMS) | 1.77 ms | |
| └ association + overhead | 11.81 ms | |
| annotate | 3.11 ms | 3.55 ms |
| encode | 2.46 ms | 3.01 ms |
| open + flush (once per run) | | 0.105 ms |
| **accounted** | | **105.7 ms** |
| **wall per frame** | | **105.77 ms** |

The four sub-rows are each a median over frames, so they do not sum to the
parent median — 31.93 against 32.21 here. That is the honest form: the frame
with the median total is not the frame with the median preprocess time.

Full run in `reports/tracking.json`. Coverage 99.9 % — the profile accounts
for nearly all of wall-clock time.

**One run, on a thermally-limited GPU.** Re-running this on the same machine
over one session produced forward-pass medians of 12.4, 12.5, 12.5, 13.1, 14.1,
14.5 and 16.3 ms — a ±14 % spread driven by how hot the card already was, with
the slowest immediately after a benchmark run. The cold-start cost moves far
more: the first frame ranged 5.6 s to 11.2 s across the same runs, so the
`warmup_penalty_x` below is the least stable number in the report and is worth
reading as an order of magnitude rather than a measurement. The
committed profile is a cold-start run, which is the protocol the benchmark
section states, but the figures here should be read as one sample from that
range rather than a stable measurement. The *ratios* between stages are far
steadier than the absolute milliseconds, and they are what the argument below
rests on.

**`detect + track` (32.21 ms) is not the same measurement as the 11.74 ms
PyTorch core row in the backend table above, and the 20.47 ms between them is the
interesting part.** That row times an `nn.Module` forward pass on a tensor
already resident in VRAM. This one times `model.track(frame)` on a decoded BGR
frame, which additionally does letterbox resize, BGR→RGB, `/255`, HWC→CHW, the
host-to-device copy, NMS, and ByteTrack's association — so the four sub-rows
above, taken from Ultralytics' `Results.speed`, are where that time goes.
(`association + overhead` is the remainder after the three phases Ultralytics
reports; the tracker is not separately instrumented, so it carries the
per-call Python overhead too.)

**The forward pass is 41 % of this stage. Preprocessing and association are
another 53 %,** and neither is affected by the export format. That is the
argument against reading the backend table as an optimisation roadmap: ONNX
buys 2.47 ms on the forward pass, while 19.05 ms per frame sits in the
surrounding work — batching the host-to-device copy, or moving letterboxing
onto the GPU, is worth more here than anything the export format can do.

The remaining 11.74 → 13.16 ms difference on the forward pass itself is
per-call dispatch: benchmark.py reuses one pre-allocated tensor with static
shapes, `model.track()` does not.

Median and mean disagree sharply on one stage because of one frame: the first
frame costs **5,805 ms — 180× the steady-state 32.21 ms**, which is cold-start
initialisation: the CUDA context and cuDNN's autotuning, plus Ultralytics
building its predictor and the ByteTrack instance, both of which are
constructed lazily on the first `model.track()` call. It is not attributed to
CUDA alone because it has not been broken down — what the profile shows is that
the first frame costs this, not which part of it costs what.

Amortised over 90 frames that is most of the gap between the mean (97.93 ms)
and the median (32.21 ms), which is why this clip's end-to-end throughput
(**9.5 FPS**) is well below its steady-state rate.
Steady state has to sum every stage's median, not just the largest one — decode,
annotate and encode still happen every frame once the cold start is behind you
— which gives **38.72 ms/frame → 25.8 FPS**, not the ~31 FPS a detect+track-only
figure would suggest, and nothing like the 87 FPS the ONNX row implies. That
distinction matters for short-clip batch processing versus a long-running
stream.

Association statistics — not association *quality*, which cannot be stated
without ground-truth track ids this clip does not have: 306 unique tracks,
mean length 10.3 frames, 26.1 % single-frame ("fragmented") tracks. 306 distinct track ids appear in the
output; the largest is 9,525. At least 9,219 id values were therefore
consumed without ever producing a drawn box — ByteTrack's internal counter
increments for every *tentative* track, including ones spawned by detections
that never get confirmed, so anything keying on track id downstream (a
counter, a database) inherits the larger number.

---

## Engineering notes

Findings from profiling and cross-checking real runs.

### AutoBatch under-sized the batch by 4×

Ultralytics' `batch=-1` profiles a forward/backward pass and picks a batch
size targeting ~60% of VRAM. On VisDrone it selected **batch=4**, and the GPU
sat at 37% utilisation drawing 42 W with 2.4 of 8 GB in use — observed at
640px during an earlier configuration of this project, before it was
simplified to the single 1024px run recorded in this repo. The profiler
over-weights peak memory during label assignment, which scales with box count
— and at ~53 boxes/image on the training split (343,205 boxes / 6,471 images)
VisDrone is close to a worst case for that heuristic. Fixing the batch to a
size chosen from measured
steady-state usage (1.16 GB, at that same 640px configuration) rather than
the profiler's estimate took utilisation to 85%. The same reasoning — trust
measured steady-state VRAM over AutoBatch's estimate — is why the 1024px run
here uses `--batch 6` explicitly instead of AutoBatch.

### JPEG decode was the real bottleneck

Even at the correct batch size, decoding full-resolution JPEGs (up to
2000×1500) every epoch — 6,471 of them — kept the dataloader, not the GPU,
as the limiting factor.
`cache='disk'` pre-decodes each image to `.npy` once and reads that back on
subsequent epochs. That was measured while diagnosing the dataloader stall at
640px, where it took epochs from ~5 minutes to ~60 seconds — it is not a claim
about the published run, which was already on `cache: disk` throughout
(`runs/n_1024/args.yaml`) and averaged 3.95 min/epoch at 1024px, where the GPU
is the limiting factor again.

### CUDA version pinning

The driver on this machine (532.09) supports CUDA up to 12.1, but PyTorch's
default PyPI wheels are CPU-only and its current CUDA wheels target 12.8,
which needs a 570+ driver. The cu124 build works via CUDA 12.x minor-version
compatibility:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verified with a real device-side matmul rather than trusting
`torch.cuda.is_available()`, which returns `True` for builds that later fail
at kernel launch.

### ONNX Runtime silently fell back to CPU under a GPU label

The first latency measurement produced an "ORT-GPU" number nearly identical to
the CPU number. Buried in the log:

```
Error loading onnxruntime_providers_cuda.dll
which depends on "cublasLt64_13.dll" which is missing
```

`onnxruntime-gpu` 1.28 is built against CUDA 13; this machine runs CUDA 12.4.
ORT logged the load failure, **silently fell back to the CPU provider, and
returned success.** Passing `providers=['CUDAExecutionProvider']` is a
request, not a guarantee. Fix: pin `onnxruntime-gpu==1.20.2` (the CUDA 12
line), and make the benchmark verify `sess.get_providers()` against what was
requested rather than trusting it — a benchmark that silently mislabels its
backend is worse than one that crashes.

### Augmentation choices

Checked Ultralytics' own `cfg/default.yaml` rather than assuming: `fliplr=0.5`,
`scale=0.5`, `mosaic=1.0` and `close_mosaic=10` are already the defaults, and
they suit this dataset as-is, so training does not override them. There is
exactly **one** real deviation: `flipud` defaults to `0.0` (a vertical flip
would ruin the labels on a normal, ground-level photo), but VisDrone is shot
nadir — straight down — so there is no canonical "up" and a vertical flip is
label-preserving here. `flipud=0.5` is set explicitly for that reason; nothing
else is.

---

## What this does not establish

**Only one input resolution was trained.** 1024px was chosen from the label
statistics, not compared against alternatives on this model — a resolution
sweep would make that a measured trade-off rather than a single data point.

**Tracking numbers are pipeline throughput, not tracking accuracy.** The
demo clip has no independently moving objects or occlusion, and no ground
truth track ids exist to score association against.

**Latency figures come from a power-limited laptop GPU** (RTX 2070 Max-Q),
and from that GPU only. The absolute milliseconds are specific to it. The
backend ordering is the more portable half of the result, but one machine is
one machine — nothing here has been measured on a second device, so treat the
ordering as a finding about this hardware rather than a general one.

**The dominant error mode is missed detection, and it is a recall problem.**
At the deployed threshold the model fails to emit a box at all for most of the
error budget on nine of ten classes; only `van` is confusion-limited, and
it goes to `car`. Resolution was the lever pulled here and it is the right
family of lever, but it was pulled once, at one value. Higher input resolution
still, a lower deployment threshold traded against precision, more epochs (see
below — training was budget-limited, not converged), and class re-weighting for
the minority classes would be the next experiments. None were run.

**Training was epoch-budget-limited, not converged.** `best.pt` is epoch 50 —
the last epoch — and mAP50-95 was still rising when the budget ran out
(+0.0078 over the final 10 epochs, +0.0030 over the final 5, in
`runs/n_1024/results.csv`). `--patience 15` never triggered. The headline
mAP50/mAP50-95 figures above are a lower bound on what this configuration can
reach, not a converged result.

---

## Setup

```bash
conda create -n yolo -c conda-forge --override-channels python=3.11 pip -y
conda activate yolo

# torch first, from the cu124 index -- PyPI's default wheels are CPU-only.
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt` is the single source of truth for versions; the first
command only exists because pip cannot express "this package from a different
index" inside a requirements file. It installs the same two pins the file
declares, so the second command finds them already satisfied and moves on.

Both pins are load-bearing: `--index-url .../cu124` avoids the CPU-only wheels
PyPI serves by default, and `onnxruntime-gpu==1.20.2` avoids a build that
expects CUDA 13 and silently degrades to CPU.

Point Ultralytics at a drive with room — VisDrone plus its disk cache needs
~35 GB — and make sure every configured directory actually exists first;
Ultralytics silently resets all settings to defaults if one is missing:

```python
from pathlib import Path
from ultralytics import settings

for d in ("datasets", "runs", "weights"):
    Path(f"<path>/{d}").mkdir(parents=True, exist_ok=True)

settings.update(
    {
        "datasets_dir": "<path>/datasets",
        "runs_dir": "<path>/runs",
        "weights_dir": "<path>/weights",
    }
)
```

The dataset downloads automatically on first run.

Alternatively, build the provided image (CUDA 12.4, matching the pins above).
Weights and data are **mounted, not baked in** — a model inside an image cannot
be updated without a rebuild:

```bash
docker build -t aerial-detection .

# Default command: benchmark the mounted checkpoint, write the report out.
docker run --gpus all \
  -v "$PWD/runs/n_1024/weights:/weights:ro" \
  -v "$PWD/datasets:/data:ro" \
  -v "$PWD/reports:/out" \
  aerial-detection

# Anything else is an override of the command:
docker run --gpus all \
  -v "$PWD/runs/n_1024/weights:/weights:ro" \
  -v "$PWD/reports:/data" \
  -v "$PWD/reports:/out" \
  aerial-detection src/track.py --weights /weights/best.pt --source /data/demo_pan.mp4

# Training needs the dataset mounted as well:
docker run --gpus all \
  -v "$PWD/datasets:/workspace/datasets" \
  -v "$PWD/runs:/workspace/runs" \
  aerial-detection src/train.py --model yolo11n.pt --imgsz 1024 --batch 6     --name n_1024_rerun
```

The dataset is mounted at `/data`, and the image carries
`docker/VisDrone.yaml` pointing there. The bundled ultralytics spec has a
`download:` key, so without it the default command would pull the dataset
into the container's ephemeral filesystem on every run and fail outright
offline.

The Dockerfile copies exactly what the image needs — `src/`, `requirements.txt`
and `docker/VisDrone.yaml` — and `.dockerignore` keeps local data, weights, git
history and build output out of the context uploaded to the daemon. (The
context still carries the README, tests and docs; they are small, and the
explicit `COPY` lines are what decide the image's contents.) Without the ignore
file, `COPY . .` shipped 60 MB from this checkout — 33 MB of `runs/`
(including 21.6 MB of weights that `.gitignore` excludes from git and Docker
sends anyway, since `.gitignore` does not apply to a build context), 21 MB of
`.git`, and the demo video. CI builds the image on every push and checks that
every script answers `--help` inside it.

## Usage

Every command below except training needs the trained checkpoint, which is too
large for git. Fetch it from the release rather than retraining for 3.3 hours:

```bash
mkdir -p runs/n_1024/weights
curl -L -o runs/n_1024/weights/best.pt \
  https://github.com/Kenchch/aerial-small-object-detection/releases/download/v1.0/best.pt
```

5.2 MB, sha256 `8786213fc488fc8b94bdb1c8c576e377eb8f2befaa258e0338b3c5efbc26382e`.

The accuracy and latency numbers are reproducible from this checkpoint plus the
VisDrone val split plus a comparable environment — `reports/benchmark.json`
records the GPU, the CUDA and cuDNN runtimes, the torch and ONNX Runtime
versions, the registered providers and the export manifest they were measured
with. (Not the display driver version - the 532.09 constraint is discussed in
the engineering notes below and is not captured in the report.) The *tracking* numbers additionally depend on the demo clip,
which is build output and is not in the repo: `src/make_demo_clip.py` picks its
source frame by label count against whatever dataset revision is installed, and
the crop, pan and fps are arguments. So the clip identifies itself instead —
`reports/demo_pan.provenance.json` records the source frame and its sha256, the
crop and the pan, and the clip's own digest, and `reports/tracking.json` embeds
all of it under `source`, with `matches_generator_record` saying whether the
file profiled is the file that record describes.

```bash
# Train (1024px, chosen from the label-size distribution above).
# --batch 6, not the default 16: at 1024px, activation memory per image is
# high enough that batch 16 does not fit in 8 GB VRAM.
# --name: runs/n_1024/ is COMMITTED (args.yaml, results.csv, the plots), and
# train.py refuses to write into an existing run directory, so retraining
# under that name needs --overwrite and will replace the published evidence.
# Pick a new name unless that is what you want.
python src/train.py --model yolo11n.pt --imgsz 1024 --epochs 50 --batch 6   --name n_1024_rerun

# Per-class metrics + ground-truth object-size distribution
# -> reports/evaluation.json
python src/evaluate.py --weights runs/n_1024/weights/best.pt --imgsz 1024

# ONNX export + latency across backends (run on an idle GPU)
python src/benchmark.py --weights runs/n_1024/weights/best.pt --imgsz 1024

# Video inference + ByteTrack, with a staged latency profile.
#
# The plain form picks the densest val frame, which depends on which dataset
# revision is installed. To rebuild the exact clip the committed tracking
# numbers were measured on - byte for byte, verified - name the frame and its
# digest; the run refuses rather than producing a different clip:
python src/make_demo_clip.py --frames 90 --fps 15 \
  --source-image "$DATASETS/VisDrone/images/val/0000295_02400_d_0000033.jpg" \
  --expected-source-sha256 f4e6fc5838b411648e0d10845540309874c23bd79a402d6be68bf57cbedf6771 \
  --expected-clip-sha256 ef545c205123b91c1bee517613381b3cd87ab66930186427742f6c4e73b8e87b

python src/track.py --weights runs/n_1024/weights/best.pt --source reports/demo_pan.mp4
```

The second digest is the one that matters. Without it the sidecar is written
from whatever clip was just produced, so it always agrees with itself and
"verified" means nothing. With it, a clip that comes out different is refused,
and the published `demo_pan.mp4` and its provenance are left untouched.

The record carries **two** frame digests, because they answer different
questions and neither answers both:

| digest | over | survives | flag |
| --- | --- | --- | --- |
| `decoded_frames_sha256` | frames read back **out of the finished file** | a remux — different container bytes, same pixels | `--expected-decoded-frames-sha256` |
| `pre_encode_frames_sha256` | frames handed **to the encoder** | a change of codec — the generator made the same pixels | `--expected-pre-encode-frames-sha256` |

The decoded one is the only digest a consumer can recompute from the published
clip alone, without the source frame or the generator. Reading the file back is
also what catches an encoder that accepts ninety frames and writes eighty-seven
— a `VideoWriter` reports nothing when it does — so the frame count and
dimensions in the record are measured from the file rather than assumed from
the arguments.

`src/track.py` compares the source against that record **before it loads the
model**, and stops if they disagree - profiling ninety frames to be told
afterwards that the footage was not the footage is a fact discovered after
paying for it. `--allow-source-mismatch` opts in, and the report records that
it was used.

## Layout

```
src/train.py            training at a chosen input resolution
src/evaluate.py         per-class metrics; label-size distribution
src/benchmark.py        ONNX export; latency on PyTorch / ONNX Runtime GPU / CPU
src/track.py            video inference + ByteTrack; staged latency profile
src/make_demo_clip.py   synthetic-motion clip for the tracking demo
src/make_demo_gif.py    README GIF from track_out.mp4, with its digest
tests/                  unit tests; run with `pytest`
resume_training.ps1     restart an interrupted run from last.pt (Windows)
runs/                   training artefacts (weights gitignored)
reports/                evaluation (val + train), benchmark, tracking output (JSON)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

They run on a bare clone — no torch, ultralytics or CUDA required. The count
is deliberately not written here: it was stated as 37 through two rounds of
additions and was wrong both times, and a number nobody re-counts is worse
than no number. `pytest -q` prints it. Two things the suite is actually
guarding:

- **`--help` must work uninstalled.** `tests/test_cli.py` runs every script's
  `--help` in a subprocess and asserts exit 0. That only holds while the heavy
  imports stay inside the functions that use them; a module-scope `import
  torch` anywhere in `src/` turns the Dockerfile's default command into a
  traceback. A subprocess specifically so an already-imported torch cannot
  mask the regression.
- **Every file under `src/` must parse.** `tests/test_syntax.py` walks the
  directory rather than naming files, so a new script is covered the moment it
  is added.

The rest drive the pure functions: the export cache's manifest, the CUDA
placement gate, frame ordering, the source/output collision guard, the clip's
provenance record, the metric helpers and the report-path formatting.
Deliberately not enumerated exhaustively — a list in a README goes stale the
moment a test is added, and this one had. `pytest -q --collect-only` is the
current answer.

**The output video is evidence too.** `reports/track_out.mp4` is written to a
temporary file, decoded back after the writer closes, and checked for frame
count and dimensions before it replaces the previous output — a `VideoWriter`
accepts every frame and reports nothing, so `frames` in the profile only counts
what was *processed*. Its sha256 (`fe9e21cccb5711d1…`) and a digest of its
decoded frames go into `reports/tracking.json` under `output`.

Both artefacts are staged and published video-first, so the failure that
matters publishes neither: if the mp4 cannot be moved into place, the report is
still a `.tmp` and the previous pair stands untouched. **This is not atomic and
is not claimed to be.** Two files cannot be replaced in one operation, so a
crash between the two replaces leaves a new video beside the previous report.
That window is one `os.replace` of a 3 KB file, and it is *detectable* — the
report's `output.sha256` will not match the video beside it, which is the check
described below. Closing it entirely needs a version directory and a single
pointer, the way `retail-ai-pipeline` publishes.

`reports/tracking_demo.gif` is built from it by `src/make_demo_gif.py`, which
records the source mp4's digest in `reports/tracking_demo.provenance.json`.
That digest matches `output.sha256` above, which is what makes the GIF part of
the same chain rather than an illustration. It used to be a hand-run ffmpeg
command that lived in somebody's shell history, with nothing saying which run
it showed. `src/track.py --source
reports/demo_pan.mp4` regenerates the underlying video.

## Licence and attribution

Code in `src/` and `tests/` is available under the repository's MIT licence.
The pipeline depends on Ultralytics, which is distributed under AGPL-3.0;
running or distributing the combined work must comply with that licence. The
published `best.pt` model and its ONNX export were fine-tuned from
Ultralytics' `yolo11n.pt` and are offered under AGPL-3.0, not MIT.

VisDrone2019 is copyright the AISKYEYE team at Tianjin University. Its official
repository does not publish an explicit dataset licence, so this repository
does not assume commercial reuse rights. Demo media derived from VisDrone
frames is provided for research demonstration only. See
[`NOTICE`](NOTICE) for links and the dataset citation.

