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
import json
from collections import Counter
from pathlib import Path

# numpy/PIL/yaml/ultralytics are imported inside the functions that use them,
# after parse_args(). At module scope they pull in torch and pin `--help` to a
# fully provisioned environment, which makes the CLI undiscoverable exactly
# when someone is trying to find out what it needs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def per_class_table(weights: Path, data: str, imgsz: int, device: str = "0") -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=data, imgsz=imgsz, device=device, verbose=False)
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

    return {
        "imgsz": imgsz,
        "overall": {
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
        },
        "per_class": {
            name: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "mAP50": round(float(ap50), 4),
                "mAP50_95": round(float(ap), 4),
            }
            for name, p, r, ap50, ap in rows
        },
        "best_class": best[0],
        "worst_class": worst[0],
    }


def label_size_distribution(data: str, split: str = "val", imgsz: int = 640) -> dict:
    """Box-area distribution of the ground-truth labels, as % of frame area,
    plus the true pixel size those boxes end up at once YOLO resizes the image.

    The frame-area share (w*h, both normalised by the image's own width/height
    independently) needs nothing but the label file -- that part of the
    original version was already correct.

    Converting area share to a pixel size is not: it is tempting to multiply
    by imgsz directly, but Ultralytics *letterboxes* rather than stretching --
    it scales the image uniformly so its long side matches imgsz, then pads
    the short side, rather than resizing width and height independently to
    imgsz x imgsz. A box's true rendered size therefore depends on its
    source image's aspect ratio, which is not uniform here: VisDrone's val
    split is 100% 16:9 (checked directly), but train mixes 4:3 and 16:9. So
    this reads each label's actual source image to get its real (W, H) and
    applies the correct per-image letterbox scale, rather than assuming one
    aspect ratio for the whole split.
    """
    import numpy as np
    import yaml
    from PIL import Image
    from ultralytics.utils import SETTINGS

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
    image_dir = root / cfg[split]
    files = list(label_dir.glob("*.txt"))
    if not files:
        print(f"\n[warn] no label files under {label_dir}")
        return {}

    areas, cls_counter = [], Counter()
    px_area_640, px_area_imgsz = [], []
    dims_seen = Counter()
    for f in files:
        img_path = next(
            (image_dir / f"{f.stem}{ext}" for ext in (".jpg", ".jpeg", ".png")
             if (image_dir / f"{f.stem}{ext}").exists()),
            None,
        )
        W = H = None
        if img_path is not None:
            # PIL.Image.open is lazy: .size comes from the header, so this
            # reads a few hundred bytes per file instead of decoding whole
            # frames. Only the dimensions are needed here, and at 6,471
            # train-split images the difference is minutes.
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
                dims_seen[(W, H)] += 1
            except OSError:
                W = H = None

        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_counter[int(parts[0])] += 1
            w_n, h_n = float(parts[3]), float(parts[4])
            areas.append(w_n * h_n)
            if W and H:
                # Letterbox scales both axes by the SAME factor (long side ->
                # target), unlike stretching to a imgsz x imgsz square.
                s640 = 640 / max(W, H)
                s_imgsz = imgsz / max(W, H)
                px_area_640.append((w_n * W * s640) * (h_n * H * s640))
                px_area_imgsz.append((w_n * W * s_imgsz) * (h_n * H * s_imgsz))

    if not areas:
        # np.mean/np.median on an empty array return nan without raising, so
        # without this guard every stat below would silently print "nan %"
        # instead of saying plainly that no boxes were found -- label files
        # existed (the earlier guard passed) but every one was empty (e.g.
        # an all-background split, or all lines shorter than 5 fields).
        print(f"\n[warn] {len(files)} label file(s) found under {label_dir}, "
              f"but none contained a valid box -- nothing to summarise.")
        return {}

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

    if px_area_640 and px_area_imgsz:
        side640 = np.sqrt(np.median(px_area_640))
        side_imgsz = np.sqrt(np.median(px_area_imgsz))
        print("\n  source image dimensions seen (w x h : count):")
        for (w, h), n in dims_seen.most_common():
            print(f"    {w}x{h}  ({n})")
        print(f"\n  -> under YOLO's letterbox resize (aspect ratio preserved, "
              f"not stretched), a box\n     at the median covers ~{side640:.1f} px "
              f"on a 640px input (the YOLO default),\n     and ~{side_imgsz:.1f} px "
              f"at {imgsz}px (this project's chosen resolution).")
    else:
        print("\n  [warn] could not resolve source images -- skipping pixel "
              "conversion (frame-area stats above are unaffected).")
    print("     This is the argument for training at higher resolution.")

    names = cfg.get("names", {})
    print("\n  class balance:")
    for c, n in cls_counter.most_common():
        print(f"    {names.get(c, c):<20} {n:>8,}  ({n / len(areas) * 100:5.1f} %)")

    return {
        "split": split,
        "boxes": len(areas),
        "images": len(files),
        "share_pct": {
            "small_lt_0.25pct": round(float(small), 1),
            "medium_0.25_to_1pct": round(float(medium), 1),
            "large_gt_1pct": round(float(large), 1),
        },
        "median_box_area_pct_of_frame": round(float(np.median(areas)) * 100, 4),
        "median_box_side_px": {
            "at_640": round(float(np.sqrt(np.median(px_area_640))), 1) if px_area_640 else None,
            f"at_{imgsz}": round(float(np.sqrt(np.median(px_area_imgsz))), 1) if px_area_imgsz else None,
        },
        "source_dims_seen": {f"{w}x{h}": n for (w, h), n in dims_seen.most_common()},
        "class_balance": {str(names.get(c, c)): n for c, n in cls_counter.most_common()},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--data", default="VisDrone.yaml")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="Should match the resolution the weights were "
                        "trained at (this project's default); evaluating a "
                        "1024px-trained model at a different size is valid "
                        "but not what the committed numbers reflect.")
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="0", help="'0' for GPU, or 'cpu'.")
    p.add_argument("--out", type=Path, default=REPORTS_DIR / "evaluation.json")
    args = p.parse_args()

    report = {
        "accuracy": per_class_table(args.weights, args.data, args.imgsz, args.device),
        "label_scale": label_size_distribution(args.data, args.split, args.imgsz),
        "config": {
            "weights": Path(args.weights).resolve().relative_to(PROJECT_ROOT).as_posix()
                       if Path(args.weights).resolve().is_relative_to(PROJECT_ROOT)
                       else str(args.weights),
            "data": args.data,
            "imgsz": args.imgsz,
            "split": args.split,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n  metrics -> {args.out}")


if __name__ == "__main__":
    main()
