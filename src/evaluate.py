"""
Evaluate a trained checkpoint and characterise *why* it fails.

A single mAP number is not an engineering artefact -- it does not tell you
which class or which object scale is dragging the average down. This script
produces two things:

  1. Per-class precision / recall / mAP, so the weak classes are named.
  2. The object-size distribution of the dataset labels themselves.

(2) is why this project trains at 1024px instead of the YOLO default of 640.
VisDrone's boxes are overwhelmingly tiny relative to the frame; once an object
is a handful of pixels after downscaling to the network's input size, no
amount of training recovers it.

Usage
-----
    python src/evaluate.py --weights runs/n_1024/weights/best.pt --imgsz 1024
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def per_class_table(weights: Path, data: str, imgsz: int) -> None:
    model = YOLO(str(weights))
    m = model.val(data=data, imgsz=imgsz, device=0, verbose=False)
    names = m.names

    print(f"\n{'=' * 66}")
    print(f"  Per-class detection performance  (imgsz={imgsz})")
    print(f"{'=' * 66}")
    print(f"{'class':<22}{'P':>9}{'R':>9}{'mAP50':>10}{'mAP50-95':>12}")
    print("-" * 66)

    # m.box.ap_class_index maps row order back onto dataset class ids.
    rows = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.p[i], m.box.r[i], m.box.ap50[i], m.box.ap[i]
        rows.append((names[c], p, r, ap50, ap))

    for name, p, r, ap50, ap in sorted(rows, key=lambda x: x[4]):
        print(f"{name:<22}{p:>9.3f}{r:>9.3f}{ap50:>10.3f}{ap:>12.3f}")

    print("-" * 66)
    print(f"{'ALL':<22}{m.box.mp:>9.3f}{m.box.mr:>9.3f}"
          f"{m.box.map50:>10.3f}{m.box.map:>12.3f}")

    worst = min(rows, key=lambda x: x[4])
    best = max(rows, key=lambda x: x[4])
    print(f"\nbest  class: {best[0]}  (mAP50-95 {best[4]:.3f})")
    print(f"worst class: {worst[0]}  (mAP50-95 {worst[4]:.3f})")


def label_size_distribution(data: str, split: str = "val", imgsz: int = 640) -> None:
    """Box-area distribution of the ground-truth labels, as % of frame area.

    YOLO labels are normalised (w, h in [0,1]), so w*h is directly the
    fraction of the image the box covers -- no need to read the images.
    """
    data_yaml = Path(SETTINGS["datasets_dir"]) / data if not Path(data).exists() else Path(data)
    # Ultralytics resolves the yaml itself; locate it via its own config dir.
    if not data_yaml.exists():
        from ultralytics.utils import ROOT
        data_yaml = ROOT / "cfg" / "datasets" / data
    # encoding must be explicit: Python on Windows defaults to cp1252, which
    # chokes on the non-ASCII bytes in Ultralytics' bundled dataset yamls.
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

    root = Path(SETTINGS["datasets_dir"]) / cfg["path"]
    label_dir = root / cfg[split].replace("images", "labels")
    files = list(label_dir.glob("*.txt"))
    if not files:
        print(f"\n[warn] no label files under {label_dir}")
        return

    areas, cls_counter = [], Counter()
    for f in files:
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_counter[int(parts[0])] += 1
            areas.append(float(parts[3]) * float(parts[4]))

    areas = np.asarray(areas)
    # COCO's convention: "small" is <32x32 px. On a 640px frame that is
    # 1024/409600 = 0.25% of the image area.
    small = (areas < 0.0025).mean() * 100
    medium = ((areas >= 0.0025) & (areas < 0.01)).mean() * 100
    large = (areas >= 0.01).mean() * 100

    print(f"\n{'=' * 66}")
    print(f"  Ground-truth object scale  ({split} split, {len(areas):,} boxes)")
    print(f"{'=' * 66}")
    print(f"  small  (<0.25% of frame) : {small:5.1f} %")
    print(f"  medium (0.25-1%)         : {medium:5.1f} %")
    print(f"  large  (>1%)             : {large:5.1f} %")
    print(f"  median box area          : {np.median(areas) * 100:.4f} % of frame")
    side = np.sqrt(np.median(areas))
    print(f"\n  -> a box at the median covers ~{side * 640:.1f} px on a 640px "
          f"input (the YOLO default), and ~{side * imgsz:.1f} px at {imgsz}px "
          f"(this project's chosen resolution).")
    print("     This is the argument for training at higher resolution.")

    names = cfg.get("names", {})
    print(f"\n  class balance:")
    for c, n in cls_counter.most_common():
        print(f"    {names.get(c, c):<20} {n:>8,}  ({n / len(areas) * 100:5.1f} %)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--data", default="VisDrone.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", default="val")
    args = p.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    per_class_table(args.weights, args.data, args.imgsz)
    label_size_distribution(args.data, args.split, args.imgsz)


if __name__ == "__main__":
    main()
