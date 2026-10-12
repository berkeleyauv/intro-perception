"""Optional extension: yaw of the gate from the classical post geometry.

    python scripts/orientation_demo.py --data data/generated/test/images \
        --output output/orientation

Writes ``orientation.csv`` (frame, detection kind, yaw in degrees, nearer post,
rear view, spacing ratio, height ratio) and annotated images for frames where
both posts were paired. There are no orientation labels in this dataset, so
this is a qualitative tool: check the yaw sign against the annotated images.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2 as cv

from intro_perception import classical
from intro_perception.images import list_images, read_image
from intro_perception.orientation import estimate_orientation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/orientation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    perceiver = classical.ClassicalGatePerceiver()
    rows = []
    for image_path in list_images(args.data):
        frame = read_image(image_path)
        detection = classical.detect(frame)
        result = estimate_orientation(detection)
        kind = detection.best.kind if detection.best is not None else "none"
        if result is None:
            rows.append([image_path.stem, kind, "", "", "", "", ""])
            continue
        rows.append([image_path.stem, kind, result.yaw_deg, result.nearer_post, result.from_behind,
                     result.spacing_ratio, result.height_ratio])
        canvas = classical.debug_frames(frame, perceiver.estimate(detection), detection)[0]
        text = f"yaw {result.yaw_deg:+.0f} deg, nearer post: {result.nearer_post}"
        cv.putText(canvas, text, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)
        cv.imwrite(str(args.output / f"{image_path.stem}.jpg"), canvas)
    with (args.output / "orientation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_id", "kind", "yaw_deg", "nearer_post", "from_behind",
                         "spacing_ratio", "height_ratio"])
        writer.writerows(rows)
    paired = [r for r in rows if r[2] != ""]
    print(f"{len(paired)} of {len(rows)} frames had a post pair; results in {args.output}")


if __name__ == "__main__":
    main()
