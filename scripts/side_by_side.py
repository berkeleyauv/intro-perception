"""Draw several methods' predictions next to each other, from prediction files.

    python scripts/side_by_side.py --test-dir data/generated/test \
        --predictions output/predictions_classical.jsonl output/predictions_yolo.jsonl \
        --output output/side_by_side

Each output image shows the same frame once per method (left to right in the
order given) with the label box in green and the method's box in orange. It
needs only the ``.jsonl`` files, so YOLO predictions made in Colab can be drawn
on a machine without PyTorch. ``--only`` limits it to a list of frame ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv

from intro_perception.evaluate import box_iou, load_records, load_truth
from intro_perception.images import list_images, read_image


def draw_box(frame, box, color, thickness=2):
    height, width = frame.shape[:2]
    x1, y1 = int(box["box_x"] * width), int(box["box_y"] * height)
    x2 = int((box["box_x"] + box["box_width"]) * width)
    y2 = int((box["box_y"] + box["box_height"]) * height)
    cv.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def panel(frame, expected, record, name):
    canvas = frame.copy()
    if expected["visible"]:
        draw_box(canvas, expected, (0, 255, 0))
    if record.get("visible"):
        draw_box(canvas, record, (0, 165, 255), 3)
        iou = box_iou(expected, record) if expected["visible"] else 0.0
        text = f"{name}: gate {record['confidence']:.2f}" + (f" IoU {iou:.2f}" if expected["visible"] else "")
    else:
        text = f"{name}: no gate"
    cv.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv.putText(canvas, text, (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("output/side_by_side"))
    parser.add_argument("--only", nargs="*", help="frame ids (file names without extension)")
    args = parser.parse_args()

    truth = load_truth(args.test_dir)
    methods = [(path.stem.replace("predictions_", ""), load_records(path)) for path in args.predictions]
    args.output.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in list_images(args.test_dir / "images"):
        if args.only and image_path.stem not in args.only:
            continue
        frame = read_image(image_path)
        panels = [
            panel(frame, truth[image_path.stem], records.get(image_path.stem, {"visible": False}), name)
            for name, records in methods
        ]
        cv.imwrite(str(args.output / f"{image_path.stem}.jpg"), cv.hconcat(panels))
        count += 1
    print(f"wrote {count} images to {args.output}")


if __name__ == "__main__":
    main()
