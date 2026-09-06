"""Explicitly enabled published-checkpoint inference; excluded by default."""

import hashlib
import os
import urllib.request
from pathlib import Path

import pytest


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("RUN_GPU") != "1", reason="Set RUN_GPU=1 and AERIAL_TEST_IMAGE"
)
def test_published_checkpoint_on_real_frame(tmp_path):
    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "RUN_GPU requires CUDA"
    frame = Path(os.environ["AERIAL_TEST_IMAGE"])
    assert frame.is_file()
    weights = tmp_path / "best.pt"
    urllib.request.urlretrieve(
        "https://github.com/Kenchch/aerial-small-object-detection/releases/download/v1.0/best.pt",
        weights,
    )
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == (
        "8786213fc488fc8b94bdb1c8c576e377eb8f2befaa258e0338b3c5efbc26382e"
    )
    result = YOLO(str(weights)).predict(str(frame), device=0, imgsz=1024, verbose=False)
    assert len(result) == 1
    assert result[0].boxes is not None
    assert torch.isfinite(result[0].boxes.data).all()
