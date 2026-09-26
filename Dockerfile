# CUDA 12.4 to match the pinned onnxruntime-gpu==1.20.2 build (see
# docs/DESIGN.md, 'CUDA version pinning'). Run with `docker run --gpus all ...`.
#
# The torch minor has to match requirements.txt's pin, not just the CUDA line.
# The sed below drops torch from the pip install so the base image's cu124 build
# survives -- which means, inside this container, the base image tag *is* the
# torch version. A 2.4.1 base under a `torch==2.6.0` requirements file gives a
# container that quietly disagrees with the file docs/DESIGN.md calls the single
# source of truth for versions.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# WORKDIR creates /workspace root-owned, and the image is documented to run
# with --user. Ultralytics writes its runs/ under the working directory when it
# finds no git root (and .git is not in the image), so the default command's
# first .val() died with PermissionError for exactly the user the comments
# below recommend. The scripts now pass .val() a writable project, but anything
# that still defaults to the working directory should not be what fails.
# The sticky bit keeps one user from replacing another's files.
RUN mkdir -p /workspace/runs && chmod 1777 /workspace /workspace/runs

# opencv-python links against libGL and libglib, which the PyTorch runtime
# images do not ship. Without these, `import cv2` raises
# "libGL.so.1: cannot open shared object file" -- and since ultralytics itself
# depends on opencv-python (not the headless build), that takes every command
# in this image down with it, not just the ones that draw.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision come from the base image's cu124 build; drop them from
# requirements.txt so pip doesn't install a PyPI wheel over the top of it and
# break the CUDA 12.4 premise above. The version specifier is matched
# optionally and for any operator, so re-pinning the file cannot quietly turn
# this into a no-op.
#
# The sed makes the base image the torch version, so check first that it is
# the version requirements.txt pins. Without this a drifted FROM or pin builds
# cleanly into an image that disagrees with the file; and dropping the sed
# would not surface it either - pip would install the PyPI wheel over the cu124
# build, which is the thing the sed is there to prevent.
#
# The base image's own requests/urllib3/idna/filelock/jinja2/pip satisfy
# ultralytics' loose floors, so pip leaves them at versions with published
# advisories. The floors below are those advisories' fixed versions; none of
# these packages takes part in computing a published number.
COPY requirements.txt .
RUN python -c "import re, torch, torchvision; \
pins = dict(re.findall(r'^(torch|torchvision)==(\S+)', open('requirements.txt').read(), re.M)); \
got = {'torch': torch.__version__.split('+')[0], 'torchvision': torchvision.__version__.split('+')[0]}; \
assert got == pins, f'base image {got} != requirements.txt {pins}'" \
    && sed -i -E '/^(torch|torchvision)([=<>!~].*)?$/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
        'requests>=2.33' 'urllib3>=2.7' 'idna>=3.15' 'filelock>=3.20.3' \
        'jinja2>=3.1.6' 'pip>=26.2'

# Named explicitly rather than `COPY . .`, so what lands in the image is a
# decision rather than whatever the working directory happens to contain.
# .dockerignore already trims the context; this makes the intent readable from
# the Dockerfile itself.
COPY src/ ./src/
# A dataset spec pointing at /data, instead of the bundled one whose
# `download:` key pulls 35 GB into the container's ephemeral filesystem.
COPY docker/VisDrone.yaml ./docker/VisDrone.yaml

# Weights and data are MOUNTED, not baked in. A model inside the image cannot
# be updated without a rebuild, and the checkpoint is 5.2 MiB of build cache
# nobody asked for.
VOLUME ["/weights", "/data", "/out"]

# Run as your own user, not root:
#
#   docker run --user "$(id -u):$(id -g)" --gpus all -v "$PWD/reports:/out" ...
#
# The image needs no root: the only thing it writes is /out, and matching the
# container UID to the host directory's owner is what makes that writable.
# There is deliberately no `USER` line. Hardcoding one guesses the host's UID:
# a mount owned by anyone else -- a named volume, which Docker creates
# root-owned, or a host account that is not 1000 -- fails with EACCES on the
# first write. Passing --user at run time is the only form that is correct for
# whoever is actually running it.
#
# HOME is / in this base image and is not writable by a non-root user, so
# Ultralytics falls back to /tmp with a warning on every run. Name the
# directory instead of relying on that fallback. Ultralytics appends
# "Ultralytics" to this, so /tmp gives /tmp/Ultralytics -- the same path the
# fallback picks, reached deliberately and without the warning.
ENV YOLO_CONFIG_DIR=/tmp
ENV MPLCONFIGDIR=/tmp/matplotlib

# onnxruntime-gpu's CUDA provider has no RPATH and links the cuDNN 9
# sub-libraries (libcudnn_adv, _cnn, _ops, ...), cuBLAS, cuFFT, cuRAND and
# nvrtc directly. In this base image those exist only under torch's pip
# `nvidia/*/lib` directories, which are not on the loader path: importing
# torch maps libcudnn.so.9 but not its sub-libraries, so the provider failed to
# load, ORT fell back to the CPU, and benchmark.py refused the result.
ENV NV=/opt/conda/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# No pip installs at run time. Ultralytics otherwise installs what it thinks is
# missing: an unpinned `onnxruntime` over the pinned onnxruntime-gpu build when
# an ONNX model runs on the CPU, or whatever lap is newest for the tracker.
# Everything the scripts need is in requirements.txt.
ENV YOLO_AUTOINSTALL=false
# Load checkpoints with torch's weights_only unpickler. Ultralytics defaults to
# full pickle, under which a tampered .pt executes code as it loads.
ENV ULTRALYTICS_SAFE_LOAD=1

# A real default. `--help` as the CMD made `docker run <image>` a no-op that
# proved only that Python starts - which is exactly the "Docker was added to
# tick a box" impression it gives.
#
#   docker run --gpus all \
#     -v "$PWD/runs/n_1024/weights:/weights:ro" \
#     -v "$PWD/reports:/out" \
#     aerial-detection
#
# Override the command for anything else:
#   docker run --gpus all -v ... aerial-detection src/track.py --weights ...
ENTRYPOINT ["python"]
# --cache-dir keeps the exported .onnx and its manifest out of /weights,
# which is mounted read-only. Without it the FIRST run fails: no cached
# graph exists yet, so it exports - and ultralytics writes the .onnx
# beside the checkpoint it loaded.
CMD ["src/benchmark.py", "--weights", "/weights/best.pt", "--cache-dir", "/out/onnx-cache", "--data", "docker/VisDrone.yaml", "--imgsz", "1024", "--out", "/out/benchmark.json"]
