"""Run one or both gate methods and write predictions plus annotated video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2 as cv

from intro_perception.classical import ClassicalGatePerceiver
from intro_perception.yolo import YoloGatePerceiver


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def frames_from(path):
    if path.is_dir():
        for item in sorted(path.iterdir()):
            supported = IMAGE_SUFFIXES | {".mp4", ".mov", ".avi", ".mkv"}
            if item.is_file() and item.suffix.lower() in supported:
                yield from frames_from(item)
        return
    if path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv.imread(str(path))
        if frame is None:
            raise ValueError(f"could not read image: {path}")
        yield path.name, 0, frame
        return
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield path.name, int(capture.get(cv.CAP_PROP_POS_FRAMES)) - 1, frame
    finally:
        capture.release()


def create_methods(method, model):
    methods = {}
    if method in ("classical", "both"):
        methods["classical"] = ClassicalGatePerceiver()
    if method in ("yolo", "both"):
        methods["yolo"] = YoloGatePerceiver(model)
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--method", choices=("classical", "yolo", "both"), default="both"
    )
    parser.add_argument("--model", default="artifacts/yolo/best.pt")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    methods = create_methods(args.method, args.model)
    streams = {
        name: (args.output / f"predictions_{name}.jsonl").open("w", encoding="utf-8")
        for name in methods
    }
    writer = None
    try:
        for source_video, frame_index, frame in frames_from(args.data):
            annotated = []
            for name, perceiver in methods.items():
                started = time.perf_counter()
                estimate, debug_frames = perceiver.analyze(frame, debug=True)
                elapsed = time.perf_counter() - started
                record = {
                    "source_video": source_video,
                    "frame_index": frame_index,
                    "inference_sec": elapsed,
                    **estimate.to_dict(),
                }
                streams[name].write(json.dumps(record) + "\n")
                annotated.append(debug_frames[0])
            canvas = cv.vconcat(annotated) if len(annotated) > 1 else annotated[0]
            if writer is None:
                height, width = canvas.shape[:2]
                writer = cv.VideoWriter(
                    str(args.output / "comparison.mp4"),
                    cv.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
                )
            writer.write(canvas)
    finally:
        for stream in streams.values():
            stream.close()
        if writer is not None:
            writer.release()
    print(f"wrote comparison artifacts to {args.output}")


if __name__ == "__main__":
    main()
