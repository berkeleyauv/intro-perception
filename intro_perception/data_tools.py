"""Extract, annotate, validate, and build a local YOLO gate dataset.

Workflow:
 
    intro-perception-data annotate --images data/raw
        draws boxes by hand and writes data/raw/<image>.txt next to each image
    intro-perception-data split --data data/raw --output data/generated
        copies labeled images into train/val/test and writes dataset.yaml
 
Label files use the YOLO format, one line per image:
``class center_x center_y width height`` with coordinates normalized to 0-1 and
class 0/1/2 = gate_left / gate_head_on / gate_right. The box covers the whole
gate. An *empty* label file means "no gate in this image".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import random

import cv2 as cv

from intro_perception.images import list_images, read_image

ORIENTATIONS = ("gate_left", "gate_head_on", "gate_right")
SPLITS = ("train", "val", "test")

def parse_label(path):
    """ Read a single YOLO label file

    Returns 'None' for an empty file, otherwise '(class_id, center_x, center_y, width, height)'
    """
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return None
    if len(lines) != 1:
        raise ValueError(f"{path}: expected one gate box per image, found {len(lines)}")
    parts = lines[0].split()
    if len(parts) != 5:
        raise ValueError(f"{path}: expected 'class cx cy w h', got {lines[0]!r}")
    try:
        class_id = int(parts[0])
        center_x, center_y, width, height = (float(value) for value in parts[1:])
    except ValueError:
        raise ValueError(f"{path}: non-numeric label {lines[0]!r}") from None
    if not 0 <= class_id < len(ORIENTATIONS):
        raise ValueError(
            f"{path}: class id must be 0-{len(ORIENTATIONS) - 1}, got {class_id}"
        )
    if width <= 0 or height <= 0 or not all(
        0.0 <= value <= 1.0 for value in (center_x, center_y, width, height)
    ):
        raise ValueError(f"{path}: box values must be in 0-1 with a positive size")
    return class_id, center_x, center_y, width, height

def write_label(path, class_id, box_xywh_pixels, image_shape):
    """Write a one-box YOLO label:
    
    - '(x, y, width, height)'
    """
    x, y, width, height = box_xywh_pixels
    image_height, image_width = image_shape[:2]
    Path(path).write_text(
        f"{class_id} {(x + width / 2) / image_width:.8f} "
        f"{(y + height / 2) / image_height:.8f} "
        f"{width / image_width:.8f} {height / image_height:.8f}\n",
        encoding="utf-8",
    )

def annotate_directory(images):
    """Label every image that has no corresponding .txt yet
    
    Draw one box around the entire gate, then enter its orientation. Cancel the
    box for a true no-gate image (an empty label file is written). Enter ``s``
    to move an ambiguous image into ``<images>/excluded/`` so it is never
    trained or scored on.
    """
    choices = {"l": 0, "h": 1, "r": 2}
    for image_path in list_images(images):
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            continue
        frame = read_image(image_path)
        print(f"\n{image_path.name}: draw the full gate, or cancel for no gate")
        x, y, width, height = cv.selectROI("Gate annotation", frame, showCrosshair=True)
        cv.destroyWindow("Gate annotation")
        if width == 0 or height == 0:
            label_path.write_text("", encoding="utf-8")
            continue
        choice = input("orientation [l]eft/[h]ead-on/[r]ight, [s]kip: ").strip().lower()
        if choice == "s":
            excluded = image_path.parent / "excluded"
            excluded.mkdir(exist_ok=True)
            shutil.move(str(image_path), str(excluded / image_path.name))
            continue
        if choice not in choices:
            print("invalid choice; leaving image unlabeled")
            continue
        write_label(
            label_path,
            choices[choice],
            (int(x), int(y), int(width), int(height)),
            frame.shape,
        )

def split_dataset(
    data_dir: Path, 
    output_dir: Path, 
    train_ratio=0.8, 
    val_ratio=0.1, 
    seed=42,
    overwrite=False
):
    """Split labeled images into train/val/test and write ``dataset.yaml``.
 
    Every image needs a ``.txt`` label next to it (empty = no gate). Each split
    gets at least one image, so at least three labeled images are required.
    Images are shuffled with ``seed`` after sorting, so the split is
    reproducible. Returns a per-split summary of image and class counts.
 
    The split is random per image. If several images come from the same scene or
    burst, near-duplicates can land in different splits and inflate your scores;
    keep such groups out of the data or split them by hand.
    """
    
        data_dir, output_dir = Path(data_dir), Path(output_dir)
    images = list_images(data_dir)
 
    missing = [image.name for image in images if not image.with_suffix(".txt").exists()]
    if missing:
        raise ValueError(
            f"{len(missing)} image(s) have no .txt label (e.g. {', '.join(missing[:3])}); "
            "run `intro-perception-data annotate`, or add an empty .txt for no-gate images"
        )
    labels = {image: parse_label(image.with_suffix(".txt")) for image in images}
 
    total = len(images)
    if total < 3:
        raise ValueError("need at least 3 labeled images (one each for train, val, test)")
    test_ratio = 1.0 - train_ratio - val_ratio
    val_count = max(1, round(total * val_ratio))
    test_count = max(1, round(total * test_ratio))
    train_count = total - val_count - test_count
    if train_count < 1:
        raise ValueError("ratios leave no images for training")
 
    existing = [
        output_dir / split
        for split in SPLITS
        if (output_dir / split).exists() and any((output_dir / split).iterdir())
    ]
    if existing and not overwrite:
        raise ValueError(
            f"{output_dir} already has data splits; pass --overwrite to replace them "
            "(stale files from an earlier split would leak between splits)"
        )
    for split in SPLITS:
        shutil.rmtree(output_dir / split, ignore_errors=True)
 
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    splits = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }
 
    summary = {}
    for split, files in splits.items():
        image_dir = output_dir / split / "images"
        label_dir = output_dir / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        counts = {"images": len(files), "no_gate": 0, **{name: 0 for name in ORIENTATIONS}}
        for image in files:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(image.with_suffix(".txt"), label_dir / f"{image.stem}.txt")
            parsed = labels[image]
            if parsed is None:
                counts["no_gate"] += 1
            else:
                counts[ORIENTATIONS[parsed[0]]] += 1
        summary[split] = counts
 
    # `path` is left out on purpose: Ultralytics then resolves the splits
    # relative to this file instead of the current working directory.
    (output_dir / "dataset.yaml").write_text(
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: gate_left\n"
        "  1: gate_head_on\n"
        "  2: gate_right\n",
        encoding="utf-8",
    )
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
 
    annotate = commands.add_parser("annotate", help="label images by drawing boxes")
    annotate.add_argument("--images", type=Path, default=Path("data/raw"))
 
    split = commands.add_parser("split", help="split labeled images into train/val/test")
    split.add_argument("--data", type=Path, default=Path("data/raw"))
    split.add_argument("--output", type=Path, default=Path("data/generated"))
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--overwrite", action="store_true")
 
    args = parser.parse_args()
 
    if args.command == "annotate":
        annotate_directory(args.images)
    else:
        summary = split_dataset(
            args.data, args.output, seed=args.seed, overwrite=args.overwrite
        )
        print(f"wrote dataset to {args.output}")
        for split_name, counts in summary.items():
            detail = ", ".join(f"{name}={value}" for name, value in counts.items())
            print(f"  {split_name}: {detail}")
            for name in ORIENTATIONS:
                if counts[name] == 0:
                    print(f"    warning: {split_name} has no {name} images")

if __name__ == "__main__":
    main()
