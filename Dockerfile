# CUDA 12.4 to match the pinned onnxruntime-gpu==1.20.2 build (see README
# engineering note 4). Run with `docker run --gpus all ...`.
#
# The torch minor has to match requirements.txt's pin, not just the CUDA line.
# The sed below drops torch from the pip install so the base image's cu124 build
# survives -- which means, inside this container, the base image tag *is* the
# torch version. A 2.4.1 base under a `torch==2.6.0` requirements file gives a
# container that quietly disagrees with the file the README calls the single
# source of truth for versions.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

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
COPY requirements.txt .
RUN sed -i -E '/^(torch|torchvision)([=<>!~].*)?$/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# Named explicitly rather than `COPY . .`, so what lands in the image is a
# decision rather than whatever the working directory happens to contain.
# .dockerignore already trims the context; this makes the intent readable from
# the Dockerfile itself.
COPY src/ ./src/
# A dataset spec pointing at /data, instead of the bundled one whose
# `download:` key pulls 35 GB into the container's ephemeral filesystem.
COPY docker/VisDrone.yaml ./docker/VisDrone.yaml

# Weights and data are MOUNTED, not baked in. A model inside the image cannot
# be updated without a rebuild, and the checkpoint is 5.3 MB of build cache
# nobody asked for.
VOLUME ["/weights", "/data", "/out"]

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
CMD ["src/benchmark.py", "--weights", "/weights/best.pt", "--data", "docker/VisDrone.yaml", "--imgsz", "1024", "--out", "/out/benchmark.json"]
