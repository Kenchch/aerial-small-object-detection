# Dependency review — 7–8 September 2026

The training/export/inference runtime is an explicitly measured configuration.
CPU unit tests and a successful Docker build cannot establish unchanged GPU
accuracy, provider selection, export behavior or latency.

| Proposal | Decision and evidence |
|---|---|
| pytest >=9.1.1,<10 (#9) | Accepted after updating the branch and passing Python 3.11–3.13 tests and the real Docker build. This changes offline test tooling. |
| numpy 2.5.2 (#10) | Rejected for the supported Python 3.11 environment: the [failed build](https://github.com/Kenchch/aerial-small-object-detection/actions/runs/34082336844) reports Requires-Python >=3.12 and no matching distribution. Keep measured 2.4.4. |
| torch 2.14.0 (#8) | Deferred; keep torch 2.6.0 / torchvision 0.21.0 and the CUDA 12.4 image as a pair. No GPU migration result has been produced for this proposal. |
| onnxruntime-gpu 1.29.0 (#6) | Deferred; retain the measured 1.20.2 provider stack. Green CPU CI does not prove GPU-provider compatibility. |
| ultralytics 8.4.138 (#7) | Deferred; retain measured 8.4.110. Training, preprocessing and validation are version-sensitive; the newer version has not been rebenchmarked here. |

## The version-by-version ignores did not hold

Each decision above was recorded in `dependabot.yml` as an ignore against the
*version* that had been reviewed — `torch 2.14.*`, `onnxruntime-gpu 1.29.*`,
`ultralytics 8.4.138`, `numpy >=2.5,<3`. Within a month Dependabot had opened
four more pull requests:

| PR | Proposal | Why the ignore missed it |
|---|---|---|
| [#16](https://github.com/Kenchch/aerial-small-object-detection/pull/16) | torch 2.6.0 → 2.13.0 | 2.13.0 is not 2.14.\* |
| [#14](https://github.com/Kenchch/aerial-small-object-detection/pull/14) | ultralytics 8.4.110 → 8.4.137 | 8.4.137 is not 8.4.138 |
| [#15](https://github.com/Kenchch/aerial-small-object-detection/pull/15) | numpy 2.4.4 → 2.4.6 | a patch bump, below the 2.5 floor |
| [#17](https://github.com/Kenchch/aerial-small-object-detection/pull/17) | torchvision 0.21.0 → 0.29.0 | never ignored at all |

None of them is a new question. The reason for keeping torch 2.6.0 is that the
GPU driver constrains this machine to the cu124 line and every published figure
was measured there; that reason does not change when the proposed version does.
Writing the ignore against a version number meant re-asking the same question
under each release, and the four were closed unmerged.

The ignores are now per dependency, covering the whole measured runtime —
torch, torchvision, ultralytics, onnx, onnxslim, onnxruntime-gpu, opencv-python
and numpy. Pillow and PyYAML stay upgradable: they read image headers and parse
dataset specs, and cannot move a published number. `tests/test_dependabot_config.py`
holds the two lists against `requirements.txt`, because a `dependency-name` that
matches nothing is not an error — it is a config that reviews as correct and has
no effect.

Ignoring an upgrade is not a security decision, and this file is not where
advisories are handled. CI runs `pip-audit -r requirements.txt --strict` on
every push against a listed set of accepted advisory IDs, so a new advisory
against any of these pins fails the build whether or not a bot proposes an
upgrade. These decisions are reproducibility constraints, not a claim that old
versions will always be safe.

A runtime migration needs a separate branch with matching base image and package
versions, real CUDA/provider checks, checkpoint evaluation using the same data
and settings, ONNX export/inference checks, and refreshed benchmark artifacts.
Preserve the existing release and evidence rather than silently reinterpreting
its measurements under newer libraries.
