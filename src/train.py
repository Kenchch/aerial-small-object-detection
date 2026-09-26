"""
Train a YOLO detector on VisDrone2019 (drone-captured aerial imagery).

VisDrone frames run from 960x540 to 2000x1500 (val is mostly 1360x765) and the
objects in them are tiny: at the network input the median box is ~11 px on a
side at 640 and ~18 px at 1024 (reports/evaluation.json, label_scale). That is
why this trains at 1024px rather than the YOLO default of 640: at 640,
downscaling throws away most of the signal a small object has left.

Usage
-----
    python src/train.py --model yolo11n.pt --imgsz 1024 --epochs 50 --name n_1024_rerun

(runs/n_1024 is the published run, and the script refuses to write into it.)
"""

import argparse
import os
from pathlib import Path

# Read when ultralytics is imported, so set before anything can import it: no
# pip installs at run time. (ULTRALYTICS_SAFE_LOAD is not defaulted here, as it
# is in the evaluation scripts: resuming loads optimizer state, and that path
# has not been checked under the weights_only unpickler.)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# ultralytics is imported inside main(), after parse_args(). Importing it at
# module scope pulls in torch and pins `--help` to a fully provisioned
# environment, which makes the CLI undiscoverable exactly when someone is
# trying to find out what it needs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

# The fresh-run defaults. The flags below default to None instead, so a resume
# can tell "given on the command line" from "left at the default" and forward
# only the former.
FRESH_DEFAULTS = {
    "data": "VisDrone.yaml",
    "batch": 6,
    "cache": "disk",
    "workers": 8,
    "device": "0",
    "patience": 15,
}
# What Ultralytics lets a resume override. Anything else comes from the
# checkpoint.
RESUME_OVERRIDES = ("device", "batch", "workers", "cache", "patience")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO on VisDrone2019")
    p.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Pretrained checkpoint. 'n' is the smallest variant -- "
        "chosen as the baseline because the target is edge "
        "deployment, and because it fits comfortably in 8 GB "
        "VRAM at 1024px.",
    )
    p.add_argument(
        "--data",
        default=None,
        help="Ultralytics dataset spec (default: VisDrone.yaml, which "
        "auto-downloads the dataset and converts its annotation format "
        "to YOLO). On --resume the checkpoint's own dataset wins, and "
        "this is used only when that path does not exist here.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Training/inference resolution. VisDrone objects are "
        "frequently <20px, so the YOLO default of 640 discards "
        "most of the small-object signal.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Default 6. Set explicitly rather than using AutoBatch (-1). "
        "AutoBatch profiles a forward/backward pass and picks a "
        "size targeting ~60%% VRAM, but VisDrone's training split "
        "carries ~53 boxes/image (343,205 boxes / 6,471 images) "
        "and the profiler over-weights the label assignment cost "
        "-- on this 8 GB card it chose batch=4, "
        "which left the GPU at 37%% utilisation and starved. "
        "6 is what the recorded run actually used at 1024px; "
        "activation memory scales with pixel count, so a batch "
        "that fits at a lower --imgsz will not fit here.",
    )
    p.add_argument(
        "--cache",
        default=None,
        choices=["disk", "ram", "False"],
        help="Default disk. VisDrone frames are ~2000x1500 JPEGs; decoding them "
        "every epoch makes the dataloader the bottleneck, not "
        "the GPU. 'disk' pre-decodes to .npy once. 'ram' is "
        "faster still but needs ~8 GB free.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dataloader processes. Default 8, of the 12 logical cores.",
    )
    p.add_argument("--device", default=None, help="Default '0'.")
    p.add_argument("--name", required=True, help="Run name under runs/")
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Default 15. Early-stop patience on fitness. Untested at 1024px: "
        "the recorded run never triggered this and was still "
        "improving at epoch 50 (see docs/DESIGN.md, 'What this does "
        "not establish'), so 50 epochs is a budget here, not "
        "a verified plateau.",
    )
    p.add_argument("--seed", type=int, default=0, help="Fixed for reproducibility.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from runs/<name>/weights/last.pt. "
        "Ultralytics restores optimizer state, EMA and epoch "
        "counter from the checkpoint, so this is not the same "
        "as fine-tuning from last.pt with a fresh optimizer. "
        "--device, --batch, --workers, --cache and --patience "
        "given here override the checkpoint's; nothing else does.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing runs/<name>/ to be written into. Off by "
        "default: runs/n_1024 is the run whose results.csv, plots "
        "and metrics this repo publishes, so running the training "
        "command under that name a second time silently overwrote "
        "the evidence behind every published number. Use a new "
        "--name, or pass this "
        "flag deliberately.",
    )
    return p.parse_args()


def _cache(value):
    return False if value == "False" else value


def _headline(results) -> tuple[str, str]:
    """mAP50-95 and mAP50 from whatever train() returned.

    A single-process run returns a metrics object with `.box`. Multi-GPU (DDP)
    training returns a plain dict keyed like results.csv, and reading `.box`
    off that raised AttributeError after training had succeeded, so the run
    exited non-zero with its weights already written.
    """
    box = getattr(results, "box", None)
    if box is not None:
        return f"{box.map:.4f}", f"{box.map50:.4f}"
    found = results if isinstance(results, dict) else {}

    def fmt(key: str) -> str:
        value = found.get(key)
        return "n/a" if value is None else f"{float(value):.4f}"

    return fmt("metrics/mAP50-95(B)"), fmt("metrics/mAP50(B)")


def resume(args, load_checkpoint=None, yolo=None):
    """Continue runs/<name>/ from last.pt, or refuse when that would not.

    `train(resume=True)` alone did three things nobody asked for, all without
    stopping. A FINISHED run's last.pt has its optimizer stripped, so
    Ultralytics logged a warning and started a new 100-epoch run on its
    default dataset, coco8, under runs/detect/train - and this script then
    printed "RESUMED RUN COMPLETE". A checkpoint whose recorded dataset path
    does not exist on this machine fell back to coco8 the same way, and its
    output went wherever the checkpoint's project said. And --device/--batch
    were never passed on, so the usual reasons to resume - a different card,
    a smaller batch after running out of memory - were silently ignored.

    The loaders are injected so this is testable without torch or ultralytics.
    """
    last = RUNS_DIR / args.name / "weights" / "last.pt"
    if not last.exists():
        raise SystemExit(f"cannot resume: {last} not found")

    if load_checkpoint is None:  # pragma: no cover - exercised by a real run
        import torch

        def load_checkpoint(path):
            return torch.load(path, map_location="cpu", weights_only=False)

    ckpt = load_checkpoint(last)
    if ckpt.get("epoch", -1) < 0 or ckpt.get("optimizer") is None:
        raise SystemExit(
            f"{last} is from a finished run (its optimizer state was stripped "
            f"when training ended), so there is nothing to resume. Train under "
            f"a new --name instead."
        )
    recorded = (ckpt.get("train_args") or {}).get("data")
    if args.data is None and not (recorded and Path(recorded).exists()):
        raise SystemExit(
            f"the checkpoint's dataset {recorded!r} does not exist here; pass "
            f"--data explicitly"
        )
    extra = {
        k: getattr(args, k) for k in RESUME_OVERRIDES if getattr(args, k) is not None
    }
    if "cache" in extra:
        extra["cache"] = _cache(extra["cache"])

    if yolo is None:  # pragma: no cover - exercised by a real run
        from ultralytics import YOLO as yolo

    print(f"resuming from {last}")
    return yolo(str(last)).train(
        resume=True,
        data=args.data or recorded,
        save_dir=str(RUNS_DIR / args.name),
        **extra,
    )


def main() -> None:
    args = parse_args()

    if args.resume:
        results = resume(args)
        mAP, _ = _headline(results)
        print(f"\n=== RESUMED RUN COMPLETE ===\nmAP50-95 : {mAP}")
        return

    for key, value in FRESH_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    run_dir = RUNS_DIR / args.name
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"{run_dir} already exists. The published metrics, plots and results.csv "
            f"for this project live under runs/n_1024/, and training into an existing "
            f"directory overwrites them in place. Pass a different --name, or "
            f"--overwrite if replacing that run is what you intend."
        )

    from ultralytics import YOLO

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        cache=_cache(args.cache),
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        # exist_ok stays True so Ultralytics writes into runs/<name>/ rather
        # than silently inventing runs/<name>2/. The guard above is what makes
        # that safe: reaching this line means the directory is new, or the
        # operator asked for it with --overwrite.
        exist_ok=True,
        # --- Augmentation -------------------------------------------------
        # Ultralytics' defaults (fliplr=0.5, scale=0.5, mosaic=1.0,
        # close_mosaic=10) are already sensible for this dataset and are left
        # alone -- checked against cfg/default.yaml rather than assumed. The
        # one change: flipud, whose default is 0.0. It was set on the premise
        # that VisDrone is shot nadir with no canonical "up"; most frames are
        # in fact oblique, some with sky in them, and no 0.0-vs-0.5 ablation
        # has been run. It stays because the published run used it.
        flipud=0.5,
        plots=True,
        val=True,
    )

    print("\n=== TRAINING COMPLETE ===")
    print(f"run       : {args.name}  (imgsz={args.imgsz})")
    mAP, mAP50 = _headline(results)
    print(f"mAP50-95  : {mAP}")
    print(f"mAP50     : {mAP50}")
    print(f"weights   : {RUNS_DIR / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
