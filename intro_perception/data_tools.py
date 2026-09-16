"""Extract, annotate, validate, and build a local YOLO gate dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2 as cv


ORIENTATIONS = ("gate_left", "gate_head_on", "gate_right")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def evenly_spaced_indices(frame_count, count):
    if frame_count <= 0 or count <= 0:
        return []
    count = min(frame_count, count)
    if count == 1:
        return [frame_count // 2]
    return sorted({round(index * (frame_count - 1) / (count - 1)) for index in range(count)})


def extract_videos(videos, output, frames_per_video=60):
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for video in videos:
        capture = cv.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video}")
        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        for frame_index in evenly_spaced_indices(frame_count, frames_per_video):
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            image_path = output / f"{video.stem}__{frame_index:06d}.jpg"
            cv.imwrite(str(image_path), frame)
            annotation = {
                "source_video": video.name,
                "frame_index": frame_index,
                "visible": None,
                "excluded": False,
                "orientation": None,
                "bbox_xywh_pixels": None,
            }
            image_path.with_suffix(".json").write_text(
                json.dumps(annotation, indent=2) + "\n", encoding="utf-8"
            )
            written.append(image_path)
        capture.release()
    return written


def validate_annotation(annotation, image_shape):
    required = {"source_video", "frame_index", "visible"}
    missing = required - annotation.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if annotation.get("excluded", False):
        return
    if annotation["visible"] is None:
        raise ValueError("frame has not been labeled")
    if not annotation["visible"]:
        return
    if annotation.get("orientation") not in ORIENTATIONS:
        raise ValueError(f"invalid orientation: {annotation.get('orientation')}")
    bbox = annotation.get("bbox_xywh_pixels")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("visible frame needs bbox_xywh_pixels [x, y, w, h]")
    x, y, width, height = bbox
    image_height, image_width = image_shape[:2]
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        raise ValueError("bounding box must have positive size and origin")
    if x + width > image_width or y + height > image_height:
        raise ValueError("bounding box extends outside image")


def annotate_directory(images):
    for image_path in sorted(images.glob("*.jpg")):
        sidecar = image_path.with_suffix(".json")
        annotation = json.loads(sidecar.read_text(encoding="utf-8"))
        if annotation["visible"] is not None or annotation.get("excluded", False):
            continue
        frame = cv.imread(str(image_path))
        print(f"\n{image_path.name}: draw the full gate, or cancel for no gate")
        bbox = cv.selectROI("Gate annotation", frame, showCrosshair=True)
        cv.destroyWindow("Gate annotation")
        if bbox[2] == 0 or bbox[3] == 0:
            annotation.update(visible=False, orientation=None, bbox_xywh_pixels=None)
        else:
            label = input("orientation [l]eft/[h]ead-on/[r]ight, [s]kip: ").strip().lower()
            if label == "s":
                annotation.update(
                    excluded=True,
                    visible=None,
                    orientation=None,
                    bbox_xywh_pixels=None,
                )
                sidecar.write_text(
                    json.dumps(annotation, indent=2) + "\n", encoding="utf-8"
                )
                continue
            mapping = {"l": "gate_left", "h": "gate_head_on", "r": "gate_right"}
            if label not in mapping:
                print("invalid label; leaving frame unfinished")
                continue
            annotation.update(
                visible=True,
                orientation=mapping[label],
                bbox_xywh_pixels=[int(value) for value in bbox],
            )
        validate_annotation(annotation, frame.shape)
        sidecar.write_text(json.dumps(annotation, indent=2) + "\n", encoding="utf-8")


def split_sources(sources):
    sources = sorted(set(sources))
    if len(sources) < 3:
        raise ValueError("at least three source videos are required for leakage-safe splits")
    train_count = min(max(1, round(len(sources) * 0.6)), len(sources) - 2)
    validation_count = max(1, round(len(sources) * 0.2))
    validation_count = min(validation_count, len(sources) - train_count - 1)
    train_end = train_count
    val_end = train_count + validation_count
    return {
        "train": set(sources[:train_end]),
        "val": set(sources[train_end:val_end]),
        "test": set(sources[val_end:]),
    }


def build_dataset(images, output):
    records = []
    for image_path in sorted(images.glob("*.jpg")):
        annotation_path = image_path.with_suffix(".json")
        if not annotation_path.exists():
            raise ValueError(f"missing annotation: {annotation_path}")
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        frame = cv.imread(str(image_path))
        validate_annotation(annotation, frame.shape)
        if annotation.get("excluded", False):
            continue
        records.append((image_path, frame.shape, annotation))

    splits = split_sources(record[2]["source_video"] for record in records)
    source_to_split = {
        source: split for split, sources in splits.items() for source in sources
    }
    manifest = []
    truth = []
    class_ids = {name: index for index, name in enumerate(ORIENTATIONS)}
    for image_path, shape, annotation in records:
        split = source_to_split[annotation["source_video"]]
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        destination = image_dir / image_path.name
        shutil.copy2(image_path, destination)
        label_path = label_dir / f"{image_path.stem}.txt"
        label = ""
        record = {
            "frame_id": image_path.stem,
            "source_video": annotation["source_video"],
            "frame_index": annotation["frame_index"],
            "split": split,
            "visible": bool(annotation["visible"]),
        }
        if annotation["visible"]:
            x, y, width, height = annotation["bbox_xywh_pixels"]
            image_height, image_width = shape[:2]
            center_x = (x + width / 2) / image_width
            center_y = (y + height / 2) / image_height
            box_width = width / image_width
            box_height = height / image_height
            class_id = class_ids[annotation["orientation"]]
            label = f"{class_id} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}\n"
            record.update(
                orientation=annotation["orientation"].removeprefix("gate_"),
                confidence=1.0,
                center_x=center_x,
                center_y=center_y,
                box_x=x / image_width,
                box_y=y / image_height,
                box_width=box_width,
                box_height=box_height,
            )
        label_path.write_text(label, encoding="utf-8")
        manifest.append(record)
        if split == "test":
            truth.append(record)

    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: gate_left\n  1: gate_head_on\n  2: gate_right\n",
        encoding="utf-8",
    )
    for name, payload in (("manifest.jsonl", manifest), ("truth_test.jsonl", truth)):
        with (output / name).open("w", encoding="utf-8") as stream:
            for record in payload:
                stream.write(json.dumps(record) + "\n")
    return manifest


def video_paths(values):
    paths = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(
                item
                for item in sorted(path.iterdir())
                if item.suffix.lower() in VIDEO_SUFFIXES
            )
        else:
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("videos", nargs="+")
    extract.add_argument("--output", type=Path, default=Path("data/working"))
    extract.add_argument("--frames-per-video", type=int, default=60)
    annotate = commands.add_parser("annotate")
    annotate.add_argument("--images", type=Path, default=Path("data/working"))
    build = commands.add_parser("build")
    build.add_argument("--images", type=Path, default=Path("data/working"))
    build.add_argument("--output", type=Path, default=Path("data/generated"))
    args = parser.parse_args()

    if args.command == "extract":
        written = extract_videos(video_paths(args.videos), args.output, args.frames_per_video)
        print(f"extracted {len(written)} frames to {args.output}")
    elif args.command == "annotate":
        annotate_directory(args.images)
    else:
        manifest = build_dataset(args.images, args.output)
        print(f"built {len(manifest)} examples in {args.output}")


if __name__ == "__main__":
    main()
