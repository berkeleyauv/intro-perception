"""Run one or both gate methods and write predictions plus annotated video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2 as cv

from intro_perception.classical import ClassicalGatePerceiver
from intro_perception.images import list_images, read_image
from intro_perception.yolo import YoloGatePerceiver

def create_methods(method, model):
    methods = {}
    if method in ("classical", "both"):
        methods["classical"] = ClassicalGatePerceiver()
    if method in ("yolo", "both"):
        methods["yolo"] = YoloGatePerceiver(model)
    return methods

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="an image or a folder of images"
    )
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--method", choices=("classical", "yolo", "both"), default="both"
    )
    parser.add_argument("--model", default="artifacts/yolo/best.pt")
    args = parser.parse_args()
 
    images = list_images(args.data)  # fails early on videos or empty folders
    annotated_dir = args.output / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    methods = create_methods(args.method, args.model)
    streams = {
        name: (args.output / f"predictions_{name}.jsonl").open("w", encoding="utf-8")
        for name in methods
    }
    try:
        for image_path in images:
            frame = read_image(image_path)
            annotated = []
            for name, perceiver in methods.items():
                started = time.perf_counter()
                estimate, debug_frames = perceiver.analyze(frame, debug=True)
                elapsed = time.perf_counter() - started
                record = {
                    "frame_id": image_path.stem,
                    "image": image_path.name,
                    "inference_sec": elapsed,
                    **estimate.to_dict(),
                }
                streams[name].write(json.dumps(record) + "\n")
                annotated.append(debug_frames[0])
            canvas = cv.vconcat(annotated) if len(annotated) > 1 else annotated[0]
            out_path = annotated_dir / f"{image_path.stem}.jpg"
            if not cv.imwrite(str(out_path), canvas):
                raise OSError(f"could not write {out_path}")
    finally:
        for stream in streams.values():
            stream.close()
    print(f"wrote predictions and annotated images for {len(images)} images to {args.output}")

if __name__ == "__main__":
    main()
