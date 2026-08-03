"""
Train a YOLO detector on VisDrone2019 (drone-captured aerial imagery).

Aerial frames are large (~2000x1500) and the objects in them are tiny -- most
VisDrone boxes are under 20px on a side (see evaluate.py's label-size report).
That is why this trains at 1024px rather than the YOLO default of 640: at 640,
downscaling throws away most of the signal a small object has left.

Usage
-----
    python src/train.py --model yolo11n.pt --imgsz 1024 --epochs 50 --name n_1024
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO on VisDrone2019")
    p.add_argument("--model", default="yolo11n.pt",
                   help="Pretrained checkpoint. 'n' is the smallest variant -- "
                        "chosen as the baseline because the target is edge "
                        "deployment, and because it fits comfortably in 8 GB "
                        "VRAM at 1024px.")
    p.add_argument("--data", default="VisDrone.yaml",
                   help="Ultralytics dataset spec. VisDrone.yaml auto-downloads "
                        "the dataset and converts its annotation format to YOLO.")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="Training/inference resolution. VisDrone objects are "
                        "frequently <20px, so the YOLO default of 640 discards "
                        "most of the small-object signal.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=6,
                   help="Set explicitly rather than using AutoBatch (-1). "
                        "AutoBatch profiles a forward/backward pass and picks a "
                        "size targeting ~60%% VRAM, but VisDrone's training split "
                        "carries ~53 boxes/image (343,205 boxes / 6,471 images) "
                        "and the profiler over-weights the label assignment cost "
                        "-- on this 8 GB card it chose batch=4, "
                        "which left the GPU at 37%% utilisation and starved. "
                        "6 is what the recorded run actually used at 1024px; "
                        "activation memory scales with pixel count, so a batch "
                        "that fits at a lower --imgsz will not fit here.")
    p.add_argument("--cache", default="disk", choices=["disk", "ram", "False"],
                   help="VisDrone frames are ~2000x1500 JPEGs; decoding them "
                        "every epoch makes the dataloader the bottleneck, not "
                        "the GPU. 'disk' pre-decodes to .npy once. 'ram' is "
                        "faster still but needs ~8 GB free.")
    p.add_argument("--workers", type=int, default=8,
                   help="Dataloader processes. 8 of the 12 logical cores.")
    p.add_argument("--device", default="0")
    p.add_argument("--name", required=True, help="Run name under runs/")
    p.add_argument("--patience", type=int, default=15,
                   help="Early-stop patience on fitness. Untested at 1024px: "
                        "the recorded run never triggered this and was still "
                        "improving at epoch 50 (see README, 'What this does "
                        "not establish'), so 50 epochs is a budget here, not "
                        "a verified plateau.")
    p.add_argument("--seed", type=int, default=0, help="Fixed for reproducibility.")
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run from runs/<name>/weights/last.pt. "
                        "Ultralytics restores optimizer state, EMA and epoch "
                        "counter from the checkpoint, so this is not the same "
                        "as fine-tuning from last.pt with a fresh optimizer.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        last = RUNS_DIR / args.name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"cannot resume: {last} not found")
        print(f"resuming from {last}")
        # On resume Ultralytics reloads every hyperparameter from the
        # checkpoint, so passing them again here would be ignored.
        results = YOLO(str(last)).train(resume=True)
        print(f"\n=== RESUMED RUN COMPLETE ===\nmAP50-95 : {results.box.map:.4f}")
        return

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        cache=(False if args.cache == "False" else args.cache),
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        # --- Augmentation -------------------------------------------------
        # Ultralytics' defaults (fliplr=0.5, scale=0.5, mosaic=1.0,
        # close_mosaic=10) are already sensible for this dataset and are left
        # alone -- checked against cfg/default.yaml rather than assumed. The
        # one real change: flipud defaults to 0.0 (label-ruining for a normal,
        # ground-level photo) but VisDrone is shot nadir, straight down, so
        # there is no canonical "up" and a vertical flip is label-preserving
        # here.
        flipud=0.5,
        plots=True,
        val=True,
    )

    print("\n=== TRAINING COMPLETE ===")
    print(f"run       : {args.name}  (imgsz={args.imgsz})")
    print(f"mAP50-95  : {results.box.map:.4f}")
    print(f"mAP50     : {results.box.map50:.4f}")
    print(f"weights   : {RUNS_DIR / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
