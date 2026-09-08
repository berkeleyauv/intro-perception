"""Run the starter perceiver on an image or video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv

from intro_perception.perceiver import IntroGatePerceiver


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def frames_from(path: Path):
    if path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv.imread(str(path))
        if frame is None:
            raise ValueError(f"could not read image: {path}")
        yield frame
        return

    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    perceiver = IntroGatePerceiver()
    writer = None
    predictions_path = args.output / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as predictions:
        for frame_index, frame in enumerate(frames_from(args.data)):
            estimate, debug_frames = perceiver.analyze(frame, debug=True)
            annotated = debug_frames[0]
            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv.VideoWriter(
                    str(args.output / "annotated.mp4"),
                    cv.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (width, height),
                )
            writer.write(annotated)
            record = {"frame_index": frame_index, **estimate.to_dict()}
            predictions.write(json.dumps(record) + "\n")
    if writer is not None:
        writer.release()
    print(f"wrote {predictions_path}")


if __name__ == "__main__":
    main()
