"""List each method's mistakes on a labeled split and gather their images.

    python scripts/failure_gallery.py --test-dir data/generated/test \
        --predictions output/predictions_classical.jsonl output/predictions_yolo.jsonl \
        --output output/failures

A mistake is a missed gate, a box with IoU < 0.5, or a gate reported on a
no-gate image. For each one the script prints the frame, kind, IoU and
confidence, and writes ``<output>/<method>/<frame>.jpg`` showing the
prediction (orange) and label (green) on the image. Use it to pick and
describe the report's failure modes; it never changes any prediction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2 as cv

from intro_perception.evaluate import box_iou, load_records, load_truth, load_truncated
from intro_perception.images import list_images, read_image


def draw(frame, box, color):
    height, width = frame.shape[:2]
    x1, y1 = int(box["box_x"] * width), int(box["box_y"] * height)
    x2 = int((box["box_x"] + box["box_width"]) * width)
    y2 = int((box["box_y"] + box["box_height"]) * height)
    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def classify(expected, actual):
    if expected["visible"] and not actual.get("visible"):
        return "missed", 0.0
    if not expected["visible"] and actual.get("visible"):
        return "false_alarm", 0.0
    if expected["visible"]:
        iou = box_iou(expected, actual)
        return ("bad_box" if iou < 0.5 else None), iou
    return None, 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("output/failures"))
    args = parser.parse_args()

    truth = load_truth(args.test_dir)
    manifest = args.test_dir.parent / "manifest.csv"
    truncated = load_truncated(manifest) if manifest.is_file() else set()
    images = {path.stem: path for path in list_images(args.test_dir / "images")}

    for prediction_path in args.predictions:
        method = prediction_path.stem.replace("predictions_", "")
        records = load_records(prediction_path)
        out_dir = args.output / method
        out_dir.mkdir(parents=True, exist_ok=True)
        kinds = Counter()
        print(f"\n== {method} ==")
        for frame_id, expected in sorted(truth.items()):
            actual = records.get(frame_id, {"visible": False})
            kind, iou = classify(expected, actual)
            if kind is None:
                continue
            kinds[kind] += 1
            cut = " (cut off)" if frame_id in truncated else ""
            print(f"{kind:12s} {frame_id}{cut}  IoU {iou:.2f}  conf {actual.get('confidence', 0):.2f}")
            frame = read_image(images[frame_id])
            if expected["visible"]:
                draw(frame, expected, (0, 255, 0))
            if actual.get("visible"):
                draw(frame, actual, (0, 165, 255))
            cv.putText(frame, f"{method}: {kind}", (10, 26), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2, cv.LINE_AA)
            cv.imwrite(str(out_dir / f"{frame_id}.jpg"), frame)
        print(f"{method}: {dict(kinds) or 'no mistakes'} out of {len(truth)} frames")


if __name__ == "__main__":
    main()
