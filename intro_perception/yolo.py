"""Ultralytics YOLO adapter for the common gate-estimate contract."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from intro_perception.production import prepare_production_namespace

prepare_production_namespace()

from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.registry import register_perceiver

from intro_perception.classical import annotate
from intro_perception.types import GateEstimate, GateOrientation


CLASS_ORIENTATIONS = {
    "gate_left": GateOrientation.LEFT,
    "gate_head_on": GateOrientation.HEAD_ON,
    "gate_right": GateOrientation.RIGHT,
}


def estimate_from_result(result, image_shape):
    """Convert the highest-confidence Ultralytics box into a GateEstimate."""
    boxes = result.boxes
    if boxes is None or len(boxes.conf) == 0:
        return GateEstimate.invisible()
    confidences = boxes.conf.cpu().tolist()
    best = max(range(len(confidences)), key=confidences.__getitem__)
    class_id = int(boxes.cls[best].cpu().item())
    class_name = result.names[class_id]
    if class_name not in CLASS_ORIENTATIONS:
        raise ValueError(f"unexpected YOLO class: {class_name}")
    x1, y1, x2, y2 = boxes.xyxy[best].cpu().tolist()
    height, width = image_shape[:2]
    return GateEstimate(
        True,
        float(confidences[best]),
        (x1 + x2) / (2.0 * width),
        (y1 + y2) / (2.0 * height),
        CLASS_ORIENTATIONS[class_name],
        x1 / width,
        y1 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ).normalized()


@register_perceiver(task="gate", algo="intro_yolo")
class YoloGatePerceiver(TaskPerceiver):
    def __init__(self, model_path=None):
        super().__init__()
        path = model_path or os.environ.get(
            "INTRO_YOLO_MODEL", "artifacts/yolo/best.pt"
        )
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"YOLO weights not found at {path}; train first or set INTRO_YOLO_MODEL"
            )
        from ultralytics import YOLO

        self.model = YOLO(path)

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals=None):
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")
        result = self.model.predict(frame, verbose=False)[0]
        estimate = estimate_from_result(result, frame.shape)
        if not debug:
            return estimate
        return estimate, [annotate(frame, estimate, "yolo")]
