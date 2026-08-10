"""AFSS training entrypoint.

Example:
    python train.py --data ./yaml/yolov11_flooddgate_detect.yaml --model ./ultralytics/cfg/models/11/yolo11s.yaml
    python train.py --data ./yaml/yolov11_swimmer_detect.yaml --weights yolo11s.pt --name SWIMMER_AFSS
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for AFSS training."""
    parser = argparse.ArgumentParser(description="Train YOLO with AFSS enabled.")
    parser.add_argument("--model", default="./ultralytics/cfg/models/11/yolo11s.yaml", help="model cfg or weights path")
    parser.add_argument(
        "--weights", default="yolo11s.pt", help="optional pretrained weights to load; empty disables load"
    )
    parser.add_argument("--data", default="./yaml/yolov11_fisher_detect.yaml", help="dataset yaml path")
    parser.add_argument("--imgsz", type=int, default=1280, help="train image size")
    parser.add_argument("--epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--patience", type=int, default=30, help="early stopping patience")
    parser.add_argument("--device", default="0", help="device, e.g. 0 or cpu")
    parser.add_argument("--optimizer", default="SGD", help="optimizer name")
    parser.add_argument("--close-mosaic", type=int, default=30, help="disable mosaic for final N epochs")
    parser.add_argument("--project", default="runs/train", help="save project directory")
    parser.add_argument("--name", default="FISHER_DETECT_v11s_1280_AFSS_0807", help="experiment name")
    parser.add_argument("--resume", action="store_true", help="resume training")
    parser.add_argument("--cache", action="store_true", help="cache dataset")

    parser.add_argument("--no-afss", action="store_true", help="disable AFSS and run normal training")
    parser.add_argument("--afss-auto-tune", action="store_true", default=True, help="auto-tune AFSS params")
    parser.add_argument(
        "--no-afss-auto-tune", dest="afss_auto_tune", action="store_false", help="use manual AFSS params"
    )
    parser.add_argument("--afss-easy-thresh", type=float, default=0.8, help="Easy threshold for sufficiency")
    parser.add_argument("--afss-hard-thresh", type=float, default=0.3, help="Hard threshold for sufficiency")
    parser.add_argument("--afss-easy-ratio", type=float, default=0.02, help="Easy image sampling ratio")
    parser.add_argument("--afss-moderate-ratio", type=float, default=0.4, help="Moderate image sampling ratio")
    parser.add_argument("--afss-update-interval", type=int, default=5, help="AFSS metric update interval in epochs")
    parser.add_argument(
        "--afss-warmup-epochs", type=int, default=10, help="full-data warmup epochs before AFSS sampling"
    )
    return parser.parse_args()


def main() -> None:
    """Run YOLO training with AFSS options passed into the trainer."""
    warnings.filterwarnings("ignore")
    args = parse_args()

    model = YOLO(args.model)
    if args.weights:
        weights = Path(args.weights)
        if weights.exists():
            model.load(str(weights))
        else:
            print(f"Warning: weights '{args.weights}' not found, training from model definition.")

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        optimizer=args.optimizer,
        close_mosaic=args.close_mosaic,
        resume=args.resume,
        project=args.project,
        name=args.name,
        cache=args.cache,
        afss=not args.no_afss,
        afss_auto_tune=args.afss_auto_tune,
        afss_easy_thresh=args.afss_easy_thresh,
        afss_hard_thresh=args.afss_hard_thresh,
        afss_easy_ratio=args.afss_easy_ratio,
        afss_moderate_ratio=args.afss_moderate_ratio,
        afss_update_interval=args.afss_update_interval,
        afss_warmup_epochs=args.afss_warmup_epochs,
    )


if __name__ == "__main__":
    main()
