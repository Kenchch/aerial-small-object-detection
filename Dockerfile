# CUDA 12.4 to match the pinned onnxruntime-gpu==1.20.2 build (see README
# engineering note 4). Run with `docker run --gpus all ...`.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# torch/torchvision come from the base image's cu124 build; drop the plain
# entries from requirements.txt before installing the rest.
COPY requirements.txt .
RUN sed -i '/^torch$/d;/^torchvision$/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/train.py", "--help"]
