# Dependency review — 7 September 2026

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

Dependabot ignores these specific reviewed proposals (and NumPy 2.5–2.x while
Python 3.11 remains supported), so they do not recur without new evidence.
Other dependency proposals remain enabled. These decisions are reproducibility
constraints, not a claim that old versions will always be safe; review any
security advisory affecting a pin separately.

A runtime migration needs a separate branch with matching base image and package
versions, real CUDA/provider checks, checkpoint evaluation using the same data
and settings, ONNX export/inference checks, and refreshed benchmark artifacts.
Preserve the existing release and evidence rather than silently reinterpreting
its measurements under newer libraries.
