"""Intentionally weak classical gate detector baseline.

Gate appearance (as seen facing the gate):
 
* left post:  black on top, red on the bottom
* right post: red on top, black on the bottom
"""

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
    """Find two vertical posts with fixed HSV thresholds."""

    # ADDED: Default Slider Values
    def __init__(self):
        super().__init__(
            red_h_min=((0, 180), 0),
            red_h_max=((0, 180), 10),
            red_s_min=((0, 255), 100),
            red_v_min=((0, 255), 100),
            black_v_max=((0, 255), 50),
        )

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals=None):
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")

        if slider_vals is None:
            slider_vals = {}

        # Slider controls for HSV tuning
        r_h_min = slider_vals.get("red_h_min", 0)
        r_h_max = slider_vals.get("red_h_max", 10)
        r_s_min = slider_vals.get("red_s_min", 100)
        r_v_min = slider_vals.get("red_v_min", 100)
        
        b_v_max = slider_vals.get("black_v_max", 50)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        red_mask = cv.inRange(
            hsv, 
            np.array((r_h_min, r_s_min, r_v_min)), 
            np.array((r_h_max, 255, 255))
        )
        
        # Broad black mask
        black_mask = cv.inRange(
            hsv, 
            np.array((0, 0, 0)), 
            np.array((180, 255, b_v_max))
        )

        combined_mask = cv.bitwise_or(red_mask, black_mask)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        contours = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        candidates = []
        for contour in contours:
            area = float(cv.contourArea(contour))
            x, y, width, height = cv.boundingRect(contour)
            # Basic vertical aspect ratio check
            if area >= 30.0 and height >= max(10, int(width * 1.2)):
                candidates.append((x, y, width, height, area))
        
        # Sort by area and pick top 2
        candidates.sort(key=lambda c: c[-1], reverse=True)
        selected = sorted(candidates[:2], key=lambda c: c[0])

        estimate = GateEstimate.invisible()
        
        if len(selected) == 2:
            left_post, right_post = selected
            
            x1 = min(left_post[0], right_post[0])
            y1 = min(left_post[1], right_post[1])
            x2 = max(left_post[0] + left_post[2], right_post[0] + right_post[2])
            y2 = max(left_post[1] + left_post[3], right_post[1] + right_post[3])
 
            left_center = left_post[0] + left_post[2] / 2.0
            right_center = right_post[0] + right_post[2] / 2.0
 
            separation = abs(right_center - left_center) / frame.shape[1]
            area_fraction = (left_post[-1] + right_post[-1]) / (frame.shape[0] * frame.shape[1])
            confidence = min(1.0, separation * 1.5 + area_fraction * 8.0)
            
            estimate = GateEstimate(
                True,
                confidence,
                (left_center + right_center) / (2.0 * frame.shape[1]),
                (y1 + y2) / (2.0 * frame.shape[0]),
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
            
        return estimate, [annotated, cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)]

def annotate(frame, estimate, label_prefix="classical"):
    canvas = frame.copy()
    if estimate.visible:
        height, width = frame.shape[:2]
        x1 = int(estimate.box_x * width)
        y1 = int(estimate.box_y * height)
        x2 = int((estimate.box_x + estimate.box_width) * width)
        y2 = int((estimate.box_y + estimate.box_height) * height)
        cv.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{label_prefix}: gate {estimate.confidence:.2f}"
        cv.putText(canvas, label, (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                   (255, 255, 255), 2, cv.LINE_AA)
    else:
        cv.putText(canvas, f"{label_prefix}: no gate", (10, 26),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
    return canvas
