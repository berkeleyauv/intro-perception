import cv2 as cv
import numpy as np

from intro_perception.perceiver import IntroGatePerceiver
from perception.tasks.registry import get_perceiver


def test_intro_perceiver_uses_production_registry():
    assert get_perceiver("gate", "intro") is IntroGatePerceiver


def test_baseline_finds_two_orange_posts():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    cv.rectangle(frame, (80, 35), (92, 150), (0, 140, 255), -1)
    cv.rectangle(frame, (225, 35), (237, 150), (0, 140, 255), -1)

    estimate, debug_frames = IntroGatePerceiver().analyze(frame, debug=True)

    assert estimate.visible
    assert abs(estimate.center_x - 0.5) < 0.02
    assert len(debug_frames) == 2
