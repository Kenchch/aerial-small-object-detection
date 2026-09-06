"""Test-dev accuracy uses the selected split in both validation passes."""

import sys
import types
from pathlib import Path

import numpy as np

from src import evaluate


def test_both_validation_passes_use_test_split(monkeypatch):
    calls = []
    box = types.SimpleNamespace(
        ap_class_index=[0],
        p=[0.5],
        r=[0.4],
        ap50=[0.3],
        ap=[0.2],
        mp=0.5,
        mr=0.4,
        map50=0.3,
        map=0.2,
    )
    result = types.SimpleNamespace(
        names={0: "car"},
        box=box,
        confusion_matrix=types.SimpleNamespace(matrix=np.array([[3, 0], [2, 0]])),
    )

    class Model:
        def val(self, **kwargs):
            calls.append(kwargs)
            return result

    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda _: Model())
    )
    evaluate.per_class_table(Path("weights.pt"), "data.yaml", 1024, "cpu", "test")
    assert len(calls) == 2
    assert all(call["split"] == "test" for call in calls)
