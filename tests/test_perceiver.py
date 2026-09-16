import cv2 as cv
import numpy as np

from intro_perception.classical import ClassicalGatePerceiver
from intro_perception.types import GateOrientation
from perception.tasks.registry import get_perceiver


def test_intro_classical_uses_production_registry():
    assert get_perceiver("gate", "intro_classical") is ClassicalGatePerceiver


def test_baseline_finds_two_orange_posts_and_box():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    cv.rectangle(frame, (80, 35), (92, 150), (0, 140, 255), -1)
    cv.rectangle(frame, (225, 35), (237, 150), (0, 140, 255), -1)
    estimate, debug_frames = ClassicalGatePerceiver().analyze(frame, debug=True)
    assert estimate.visible
    assert abs(estimate.center_x - 0.5) < 0.02
    assert estimate.orientation is GateOrientation.HEAD_ON
    assert estimate.box_width > 0.4
    assert len(debug_frames) == 2


def test_baseline_requires_two_posts():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    cv.rectangle(frame, (80, 35), (92, 150), (0, 140, 255), -1)
    assert not ClassicalGatePerceiver().analyze(frame, debug=False).visible
