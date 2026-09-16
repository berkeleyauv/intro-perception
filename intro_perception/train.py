"""Train the intro three-class YOLO gate detector."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def automatic_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/generated/dataset.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/yolo/best.pt"))
    args = parser.parse_args()

    from ultralytics import YOLO

    run = YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device or automatic_device(),
        project="artifacts/training",
        name="gate",
        exist_ok=True,
    )
    best = Path(run.save_dir) / "weights" / "best.pt"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    print(f"copied best weights to {args.output}")


if __name__ == "__main__":
    main()
