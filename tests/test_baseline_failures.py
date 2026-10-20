from pathlib import Path

import cv2
import pytest

from intro_perception.classical import ClassicalGatePerceiver

FIXTURES = Path(__file__).parent / "fixtures"


#@pytest.mark.xfail(strict=True, reason="starter detector reports a gate in a no-gate photo")
def test_no_gate_photo_reports_no_gate():
    image = cv2.imread(str(FIXTURES / "false_alarm.jpg"))
    assert image is not None

    detector = ClassicalGatePerceiver()
    estimate = detector.analyze(image, debug=False)

    assert not estimate.visible
