# CUDA 12.4 to match the pinned onnxruntime-gpu==1.20.2 build (see README
# engineering note 4). Run with `docker run --gpus all ...`.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# torch/torchvision come from the base image's cu124 build; drop them from
# requirements.txt so pip doesn't install a PyPI wheel over the top of it and
# break the CUDA 12.4 premise above. The version specifier is matched
# optionally and for any operator, so re-pinning the file cannot quietly turn
# this into a no-op.
COPY requirements.txt .
RUN sed -i -E '/^(torch|torchvision)([=<>!~].*)?$/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/train.py", "--help"]
