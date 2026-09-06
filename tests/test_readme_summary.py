import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_headlines_match_benchmark():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    benchmark = json.loads((ROOT / "reports/benchmark.json").read_text())
    assert (
        f"mAP50 {benchmark['mAP50']:.3f} / mAP50-95 {benchmark['mAP50_95']:.3f}"
        in readme
    )
    cuda = benchmark["onnx_cuda"]["core"]["median_ms"]
    eager = benchmark["pytorch_cuda"]["core"]["median_ms"]
    assert f"{cuda:.1f} ms; {100 * (1 - cuda / eager):.0f}% faster" in readme
    assert f"+{benchmark['accuracy']['delta']['mAP50_95']:.4f}" in readme
