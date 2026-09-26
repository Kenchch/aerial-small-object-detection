"""train.py's resume path and result printing, without torch or ultralytics.

`resume()` takes its checkpoint loader and YOLO class as arguments, so these
drive it with stand-ins: what matters is what it refuses and what it passes to
train(), not the training itself.
"""

import argparse
import types
import typing

import pytest

import train


def _args(**overrides) -> argparse.Namespace:
    values = {k: None for k in (*train.FRESH_DEFAULTS, "seed")}
    values.update(name="run", resume=True, overwrite=False)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    weights = tmp_path / "runs" / "run" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"checkpoint")
    return tmp_path / "runs" / "run"


class _Recorder:
    """Stands in for ultralytics.YOLO and remembers what train() was given."""

    calls: typing.ClassVar[list] = []

    def __init__(self, path):
        self.path = path

    def train(self, **kwargs):
        _Recorder.calls.append(kwargs)
        return types.SimpleNamespace(box=types.SimpleNamespace(map=0.5, map50=0.7))


@pytest.fixture(autouse=True)
def _reset_recorder():
    _Recorder.calls = []


def _resumable(data):
    return {"epoch": 10, "optimizer": {"state": {}}, "train_args": {"data": data}}


def test_a_finished_run_is_refused_not_retrained_on_coco8(run_dir):
    """A finished last.pt has epoch -1 and no optimizer. Ultralytics turned
    that into a new run on coco8.yaml; the script then printed RESUMED."""
    finished = {"epoch": -1, "optimizer": None, "train_args": {"data": "x.yaml"}}

    with pytest.raises(SystemExit, match="finished run"):
        train.resume(_args(), lambda _: finished, _Recorder)

    assert _Recorder.calls == [], "training started anyway"


def test_a_dataset_missing_here_needs_data_named(run_dir):
    """The checkpoint's dataset path is from the machine that trained it.
    Missing here, Ultralytics fell back to coco8.yaml without stopping."""
    ckpt = _resumable("C:/somewhere/else/VisDrone.yaml")

    with pytest.raises(SystemExit, match="pass --data explicitly"):
        train.resume(_args(), lambda _: ckpt, _Recorder)

    assert _Recorder.calls == []


def test_explicit_flags_and_the_run_directory_reach_train(run_dir, tmp_path):
    """--device and --batch are the usual reasons to resume differently, and
    were dropped. save_dir keeps the output under runs/<name>/, rather than
    wherever the checkpoint's recorded project points."""
    data = tmp_path / "VisDrone.yaml"
    data.write_text("path: .\n")
    ckpt = _resumable(str(data))

    train.resume(_args(device="cpu", batch=2, cache="False"), lambda _: ckpt, _Recorder)

    (kwargs,) = _Recorder.calls
    assert kwargs["resume"] is True
    assert kwargs["device"] == "cpu"
    assert kwargs["batch"] == 2
    assert kwargs["cache"] is False
    assert kwargs["save_dir"] == str(run_dir)
    assert kwargs["data"] == str(data)
    assert "workers" not in kwargs, "an unset flag must not override the checkpoint"


def test_data_given_on_the_command_line_is_used_when_the_recorded_one_is_gone(
    run_dir,
):
    ckpt = _resumable("C:/somewhere/else/VisDrone.yaml")

    train.resume(_args(data="docker/VisDrone.yaml"), lambda _: ckpt, _Recorder)

    assert _Recorder.calls[0]["data"] == "docker/VisDrone.yaml"


def test_no_checkpoint_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path / "runs")
    with pytest.raises(SystemExit, match="not found"):
        train.resume(_args(), lambda _: {}, _Recorder)


def test_headline_reads_a_single_process_result():
    result = types.SimpleNamespace(box=types.SimpleNamespace(map=0.22161, map50=0.37))
    assert train._headline(result) == ("0.2216", "0.3700")


def test_headline_reads_the_dict_multi_gpu_training_returns():
    """DDP returns a results.csv-style dict, and `.box` on it raised after a
    successful run, so the process exited non-zero with its weights written."""
    result = {"metrics/mAP50-95(B)": 0.2216, "metrics/mAP50(B)": 0.3748}
    assert train._headline(result) == ("0.2216", "0.3748")


@pytest.mark.parametrize("result", [None, {}, object()])
def test_headline_says_na_instead_of_raising(result):
    assert train._headline(result) == ("n/a", "n/a")
