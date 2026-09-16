"""Intentionally weak classical gate detector baseline."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from intro_perception.production import prepare_production_namespace

prepare_production_namespace()

from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.registry import register_perceiver

from intro_perception.types import GateEstimate, GateOrientation


@register_perceiver(task="gate", algo="intro_classical")
class ClassicalGatePerceiver(TaskPerceiver):
    """Find two orange vertical posts with fixed HSV thresholds."""

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals=None):
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array((5, 80, 80)), np.array((35, 255, 255)))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        candidates = []
        for contour in contours:
            area = float(cv.contourArea(contour))
            x, y, width, height = cv.boundingRect(contour)
            if area >= 30.0 and height >= max(10, int(width * 1.5)):
                candidates.append((x, y, width, height, area))
        candidates.sort(key=lambda candidate: candidate[-1], reverse=True)
        selected = sorted(candidates[:2], key=lambda candidate: candidate[0])

        estimate = GateEstimate.invisible()
        if len(selected) == 2:
            left, right = selected
            left_center = left[0] + left[2] / 2.0
            right_center = right[0] + right[2] / 2.0
            x1 = min(left[0], right[0])
            y1 = min(left[1], right[1])
            x2 = max(left[0] + left[2], right[0] + right[2])
            y2 = max(left[1] + left[3], right[1] + right[3])

            ratio = right[2] / max(left[2], 1)
            if ratio > 1.12:
                orientation = GateOrientation.LEFT
            elif ratio < 1.0 / 1.12:
                orientation = GateOrientation.RIGHT
            else:
                orientation = GateOrientation.HEAD_ON

            separation = abs(right_center - left_center) / frame.shape[1]
            area_fraction = (left[-1] + right[-1]) / (
                frame.shape[0] * frame.shape[1]
            )
            confidence = min(1.0, separation * 1.5 + area_fraction * 8.0)
            estimate = GateEstimate(
                True,
                confidence,
                (left_center + right_center) / (2.0 * frame.shape[1]),
                (y1 + y2) / (2.0 * frame.shape[0]),
                orientation,
                x1 / frame.shape[1],
                y1 / frame.shape[0],
                (x2 - x1) / frame.shape[1],
                (y2 - y1) / frame.shape[0],
            ).normalized()

        if not debug:
            return estimate
        annotated = annotate(frame, estimate)
        for x, y, width, height, _ in selected:
            cv.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
        return estimate, [annotated, cv.cvtColor(mask, cv.COLOR_GRAY2BGR)]


def annotate(frame, estimate, label_prefix="classical"):
    canvas = frame.copy()
    if estimate.visible:
        height, width = frame.shape[:2]
        x1 = int(estimate.box_x * width)
        y1 = int(estimate.box_y * height)
        x2 = int((estimate.box_x + estimate.box_width) * width)
        y2 = int((estimate.box_y + estimate.box_height) * height)
        cv.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{label_prefix}: {estimate.orientation.value} {estimate.confidence:.2f}"
        cv.putText(canvas, label, (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                   (255, 255, 255), 2, cv.LINE_AA)
    else:
        cv.putText(canvas, f"{label_prefix}: no gate", (10, 26),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
    return canvas
