"""Explicitly enabled published-checkpoint inference; excluded by default."""

import hashlib
import os
import urllib.request
from pathlib import Path

import pytest

RELEASE = "https://github.com/Kenchch/aerial-small-object-detection/releases/download/v1.0/best.pt"
DIGEST = "8786213fc488fc8b94bdb1c8c576e377eb8f2befaa258e0338b3c5efbc26382e"


def _release_checkpoint(tmp_path) -> Path:
    weights = tmp_path / "best.pt"
    urllib.request.urlretrieve(RELEASE, weights)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == DIGEST
    return weights


@pytest.mark.gpu
@pytest.mark.skipif(
    # Both, not only RUN_GPU: with the image unset the test failed on a
    # KeyError instead of skipping with the reason that names it.
    os.getenv("RUN_GPU") != "1" or not os.getenv("AERIAL_TEST_IMAGE"),
    reason="Set RUN_GPU=1 and AERIAL_TEST_IMAGE",
)
def test_published_checkpoint_on_real_frame(tmp_path):
    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "RUN_GPU requires CUDA"
    frame = Path(os.environ["AERIAL_TEST_IMAGE"])
    assert frame.is_file()
    weights = _release_checkpoint(tmp_path)
    result = YOLO(str(weights)).predict(str(frame), device=0, imgsz=1024, verbose=False)
    assert len(result) == 1
    assert result[0].boxes is not None
    assert torch.isfinite(result[0].boxes.data).all()


@pytest.mark.skipif(
    os.getenv("RUN_CHECKPOINT") != "1",
    reason="Set RUN_CHECKPOINT=1 (downloads the release checkpoint; CPU is enough)",
)
def test_published_checkpoint_loads_and_predicts_on_cpu(tmp_path):
    """The real checkpoint runs on a CPU, so checking it needs no GPU - but the
    only test that loaded it was gated on one. This one downloads, verifies
    and runs it anywhere torch is installed."""
    import numpy as np
    import torch
    from ultralytics import YOLO

    weights = _release_checkpoint(tmp_path)
    model = YOLO(str(weights))
    assert len(model.names) == 10
    assert model.ckpt["train_args"]["imgsz"] == 1024

    frame = os.getenv("AERIAL_TEST_IMAGE")
    source = frame if frame else np.zeros((540, 960, 3), dtype=np.uint8)
    result = model.predict(source, device="cpu", imgsz=1024, verbose=False)
    assert len(result) == 1
    assert torch.isfinite(result[0].boxes.data).all()
