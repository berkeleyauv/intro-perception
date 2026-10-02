
"""Label, split, and package a local YOLO gate dataset from still images.
 
Workflow (images only):
 
    intro-perception-data annotate --images data/raw
        draws boxes by hand and writes data/raw/<image>.txt next to each image
    intro-perception-data split --data data/raw --output data/generated
        copies labeled images into train/val/test and writes dataset.yaml
 
If your labels are in a separate folder (for example the auto-labeling
pipeline's ``labels_detect/``), point ``split`` at both:
 
    intro-perception-data split --data data/images --labels data/labels_detect
 
Label files use the YOLO format, one line per image:
``class center_x center_y width height`` with coordinates normalized to 0-1.
There is a single class, 0 = gate. The box covers the whole visible gate (a gate
cut off by the image edge is boxed up to that edge). An *empty* label file means
"no gate in this image".
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import random

import cv2 as cv

from intro_perception.images import list_images, read_image

CLASS_NAMES = ("gate",)
SPLITS = ("train", "val", "test")

def parse_label(path):
    """ Read a single YOLO label file

    Returns ``None`` for an empty file (a true no-gate image), otherwise
    ``(center_x, center_y, width, height)``, all normalized. Raises
    ``ValueError`` on anything malformed so a bad label can never be trained or
    scored silently.
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
    if class_id != 0:
        raise ValueError(
            f"{path}: class id must be 0 (the only class is 'gate'), got {class_id}; "
            "labels from the old 3-orientation-class dataset need converting to class 0"
        )
    if width <= 0 or height <= 0 or not all(
        0.0 <= value <= 1.0 for value in (center_x, center_y, width, height)
    ):
        raise ValueError(f"{path}: box values must be in 0-1 with a positive size")
    return center_x, center_y, width, height

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

    for image_path in list_images(images):
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            continue
        frame = read_image(image_path)
        print(f"\n{image_path.name}: draw the gate, or cancel for no gate")
        x, y, width, height = cv.selectROI("Gate annotation", frame, showCrosshair=True)
        cv.destroyWindow("Gate annotation")
        if width == 0 or height == 0:
            label_path.write_text("", encoding="utf-8")
            continue
        choice = input("[Enter] save box, [s]kip this image: ").strip().lower()
        if choice == "s":
            excluded = image_path.parent / "excluded"
            excluded.mkdir(exist_ok=True)
            shutil.move(str(image_path), str(excluded / image_path.name))
            continue
        write_label(label_path, (int(x), int(y), int(width), int(height)), frame.shape)

def read_manifest(labels_dir):
    """Read the labeling pipeline's ``manifest.csv`` (if there is one), keyed by image file stem."""
    path = Path(labels_dir) / "manifest.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError(f"{path}: expected a 'filename' column")
        return {Path(row["filename"]).stem: row for row in reader}

def pair_images_and_labels(data_dir, labels_dir=None):
    """Return ``([(image, label), ...], unlabeled_image_names)``.
 
    Without ``labels_dir`` every image needs a ``.txt`` next to it. With it, the
    labels decide what is in the dataset: images without a label are left out
    (and reported), but a label with no matching image is an error.
    """
    images = list_images(data_dir)
 
    if labels_dir is None:
        missing = [image.name for image in images if not image.with_suffix(".txt").exists()]
        if missing:
            raise ValueError(
                f"{len(missing)} image(s) have no .txt label (e.g. {', '.join(missing[:3])}); "
                "run `intro-perception-data annotate`, add an empty .txt for no-gate images, "
                "or pass --labels if your labels are in a separate folder"
            )
        return [(image, image.with_suffix(".txt")) for image in images], []
 
    labels_dir = Path(labels_dir)
    if not labels_dir.is_dir():
        raise ValueError(f"{labels_dir} is not a folder")
    label_files = {
        path.stem: path
        for path in sorted(labels_dir.iterdir())
        if path.is_file() and path.suffix == ".txt"
    }
    if not label_files:
        raise ValueError(f"no .txt labels found in {labels_dir}")
 
    by_stem = {image.stem: image for image in images}
    orphans = sorted(stem for stem in label_files if stem not in by_stem)
    if orphans:
        raise ValueError(
            f"{len(orphans)} label(s) have no matching image in {data_dir} "
            f"(e.g. {', '.join(orphans[:3])}); are --data and --labels from the same dataset?"
        )
    unlabeled = [image.name for stem, image in by_stem.items() if stem not in label_files]
    return [(by_stem[stem], label_files[stem]) for stem in sorted(label_files)], unlabeled

def split_dataset(
    data_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    overwrite=False,
    labels_dir=None,
):
    """Split labeled images into train/val/test and write ``dataset.yaml``.
 
    By default every image needs a ``.txt`` label next to it (empty = no gate).
    With ``labels_dir`` the labels live in a separate folder and decide which
    images are used; images without a label are left out. Each split gets at
    least one image, so at least three labeled images are required. Images are
    shuffled with ``seed`` after sorting, so the split is reproducible.
 
    If ``labels_dir`` holds a ``manifest.csv`` from the labeling pipeline, a
    ``manifest.csv`` (filename, split, status, truncated) is written next to
    ``dataset.yaml`` so cut-off gates can be found later.
 
    Returns a per-split summary of image counts.
 
    The split is random per image. If several images come from the same scene or
    burst, near-duplicates can land in different splits and inflate your scores;
    keep such groups out of the data or split them by hand.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    pairs, unlabeled = pair_images_and_labels(data_dir, labels_dir)
    source_manifest = read_manifest(labels_dir) if labels_dir is not None else {}
 
    labels = {image: parse_label(label) for image, label in pairs}
 
    total = len(pairs)
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
 
    if unlabeled:
        print(f"note: {len(unlabeled)} image(s) have no label and were left out "
              f"(e.g. {', '.join(unlabeled[:3])})")
 
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    splits = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }
 
    summary = {}
    manifest_rows = []
    for split, files in splits.items():
        image_dir = output_dir / split / "images"
        label_dir = output_dir / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        counts = {"images": len(files), "gate": 0, "no_gate": 0}
        if source_manifest:
            counts["truncated"] = 0
        for image, label in files:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / f"{image.stem}.txt")
            status = "no_gate" if labels[image] is None else "gate"
            counts[status] += 1
            truncated = str(source_manifest.get(image.stem, {}).get("truncated", "")).lower() == "true"
            if source_manifest:
                counts["truncated"] += int(truncated)
                manifest_rows.append(
                    {"filename": image.name, "split": split, "status": status, "truncated": truncated}
                )
        summary[split] = counts
 
    if source_manifest:
        with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["filename", "split", "status", "truncated"], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(sorted(manifest_rows, key=lambda row: row["filename"]))
 
    # `path` is left out on purpose: Ultralytics then resolves the splits
    # relative to this file instead of the current working directory.
    (output_dir / "dataset.yaml").write_text(
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: gate\n",
        encoding="utf-8",
    )
    return summary

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
 
    annotate = commands.add_parser("annotate", help="label images by drawing boxes")
    annotate.add_argument("--images", type=Path, default=Path("data/raw"))
 
    split = commands.add_parser("split", help="split labeled images into train/val/test")
    split.add_argument("--data", type=Path, default=Path("data/raw"),
                       help="folder of images")
    split.add_argument("--labels", type=Path, default=None,
                       help="folder of YOLO .txt labels, if they are not next to the images; "
                            "images without a label are left out")
    split.add_argument("--output", type=Path, default=Path("data/generated"))
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--overwrite", action="store_true")
 
    args = parser.parse_args()
 
    if args.command == "annotate":
        annotate_directory(args.images)
    else:
        summary = split_dataset(
            args.data, args.output, seed=args.seed, overwrite=args.overwrite,
            labels_dir=args.labels,
        )
        print(f"wrote dataset to {args.output}")
        for split_name, counts in summary.items():
            detail = ", ".join(f"{name}={value}" for name, value in counts.items())
            print(f"  {split_name}: {detail}")
            if counts["gate"] == 0:
                print(f"    warning: {split_name} has no gate images")

if __name__ == "__main__":
    main()
