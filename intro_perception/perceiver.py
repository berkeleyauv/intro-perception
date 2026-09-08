"""Weak but runnable gate-perception baseline."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.registry import register_perceiver

from intro_perception.tracker import GateTracker
from intro_perception.types import GateEstimate


@register_perceiver(task="gate", algo="intro")
class IntroGatePerceiver(TaskPerceiver):
    """Detect two orange vertical posts and track their midpoint."""

    def __init__(self):
        super().__init__()
        self.tracker = GateTracker()

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals=None):
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array((5, 80, 80)), np.array((35, 255, 255)))
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
        candidates: list[tuple[int, int, int, int, float]] = []
        for contour in contours:
            area = float(cv.contourArea(contour))
            x, y, width, height = cv.boundingRect(contour)
            if area >= 30.0 and height >= max(10, int(width * 1.5)):
                candidates.append((x, y, width, height, area))
        candidates.sort(key=lambda item: item[-1], reverse=True)

        raw = GateEstimate.invisible()
        selected = candidates[:2]
        if len(selected) == 2:
            centers_x = [x + width / 2.0 for x, _, width, _, _ in selected]
            centers_y = [y + height / 2.0 for _, y, _, height, _ in selected]
            separation = abs(centers_x[0] - centers_x[1]) / frame.shape[1]
            area_fraction = sum(item[-1] for item in selected) / (
                frame.shape[0] * frame.shape[1]
            )
            confidence = min(1.0, separation * 1.5 + area_fraction * 8.0)
            raw = GateEstimate(
                visible=True,
                confidence=confidence,
                center_x=sum(centers_x) / (2.0 * frame.shape[1]),
                center_y=sum(centers_y) / (2.0 * frame.shape[0]),
                orientation_rad=None,
            ).normalized()

        estimate = self.tracker.update(raw)
        if not debug:
            return estimate

        annotated = frame.copy()
        for x, y, width, height, _ in selected:
            cv.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
        if estimate.visible:
            point = (
                int(estimate.center_x * frame.shape[1]),
                int(estimate.center_y * frame.shape[0]),
            )
            cv.circle(annotated, point, 6, (0, 0, 255), -1)
            cv.putText(
                annotated,
                f"confidence={estimate.confidence:.2f}",
                (10, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
        mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        return estimate, [annotated, mask_bgr]
