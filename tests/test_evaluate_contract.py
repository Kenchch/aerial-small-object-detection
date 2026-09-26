"""The code that writes reports/evaluation*.json, held to its contract.

These ran only as a whole, on a GPU, and were checked afterwards by reading the
committed JSON - so swapping precision and recall, taking the confusion matrix
from the wrong pass, or dropping the conf=0.25 of the second pass all passed
the suite. Stubbed here, no torch or GPU needed.
"""

import sys
import types
from pathlib import Path

import numpy as np

import evaluate


def _result(p, r, ap50, ap, matrix):
    box = types.SimpleNamespace(
        ap_class_index=[0],
        p=[p],
        r=[r],
        ap50=[ap50],
        ap=[ap],
        mp=p,
        mr=r,
        map50=ap50,
        map=ap,
    )
    return types.SimpleNamespace(
        names={0: "car"},
        box=box,
        confusion_matrix=types.SimpleNamespace(matrix=np.array(matrix)),
    )


def test_accuracy_comes_from_pass_one_and_the_matrix_from_pass_two(monkeypatch):
    """mAP/P/R need conf -> 0; the error split needs the deployed 0.25. Each
    field has to come from the pass that was run for it."""
    first = _result(0.9, 0.7, 0.5, 0.25, [[1, 0], [9, 0]])
    second = _result(0.1, 0.1, 0.1, 0.1, [[8, 0], [2, 0]])
    results = iter([first, second])
    calls = []

    class Model:
        def val(self, **kwargs):
            calls.append(kwargs)
            return next(results)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda _: Model())
    )

    got = evaluate.per_class_table(Path("weights.pt"), "data.yaml", 1024, "cpu", "val")

    assert "conf" not in calls[0], "the accuracy pass must use val's own conf"
    assert calls[1]["conf"] == evaluate.CONFUSION_CONF
    assert got["per_class"]["car"] == {
        "precision": 0.9,
        "recall": 0.7,
        "mAP50": 0.5,
        "mAP50_95": 0.25,
    }
    assert got["overall"]["mAP50_95"] == 0.25
    assert got["overall"]["mAP50"] == 0.5
    assert got["error_split"] == evaluate.error_split(
        second.confusion_matrix.matrix, first.names
    )
    assert got["error_split_conf"] == evaluate.CONFUSION_CONF


def test_paths_outside_the_repo_are_reduced_to_a_name():
    assert evaluate._portable("/somewhere/else/VisDrone.yaml") == "VisDrone.yaml"


def test_paths_inside_the_repo_are_relative_and_posix():
    inside = evaluate.PROJECT_ROOT / "docker" / "VisDrone.yaml"
    assert evaluate._portable(inside) == "docker/VisDrone.yaml"


def test_default_output_names_do_not_collide():
    assert evaluate.default_out("val").name == "evaluation.json"
    assert evaluate.default_out("test").name == "evaluation_test.json"
    assert evaluate.default_out("train").name == "evaluation_train.json"


def test_the_matrix_figure_is_the_conf_025_pass(monkeypatch, tmp_path):
    """The figure handed to main() is the one drawn by the second pass - the
    pass `error_split` comes from - not the first pass's conf=0.001 matrix."""
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    for d in (first_dir, second_dir):
        d.mkdir()
        (d / "confusion_matrix_normalized.png").write_bytes(d.name.encode())
    first = _result(0.9, 0.7, 0.5, 0.25, [[1, 0], [9, 0]])
    second = _result(0.1, 0.1, 0.1, 0.1, [[8, 0], [2, 0]])
    first.save_dir, second.save_dir = first_dir, second_dir
    results = iter([first, second])

    class Model:
        def val(self, **kwargs):
            return next(results)

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda _: Model())
    )

    got = evaluate.per_class_table(Path("weights.pt"), "data.yaml", 1024, "cpu", "val")

    assert got["_confusion_plot"].read_bytes() == b"second"


def test_design_embeds_the_figure_the_report_names():
    """The embedded matrix is the conf=0.25 one evaluate.py wrote beside
    reports/evaluation.json, not the training-time conf=0.001 figure."""
    import json

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "reports/evaluation.json").read_text(encoding="utf-8"))
    figure = report["accuracy"]["error_split_plot"]
    assert (root / "reports" / figure).is_file()
    design = (root / "docs/DESIGN.md").read_text(encoding="utf-8")
    assert f"](../reports/{figure})" in design
    assert "](../runs/n_1024/confusion_matrix_normalized.png)" not in design
