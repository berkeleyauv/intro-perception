from gate_helpers import make_gate_frame

from intro_perception.classical import ClassicalGatePerceiver
from intro_perception.types import GateOrientation
from perception.tasks.registry import get_perceiver


def test_intro_classical_uses_production_registry():
    assert get_perceiver("gate", "intro_classical") is ClassicalGatePerceiver


def test_baseline_finds_spec_gate_and_box():
    estimate, debug_frames = ClassicalGatePerceiver().analyze(make_gate_frame(), debug=True)
    assert estimate.visible
    assert abs(estimate.center_x - 0.5) < 0.02
    assert abs(estimate.center_y - 93 / 180) < 0.01
    assert estimate.orientation is GateOrientation.HEAD_ON
    assert estimate.box_width > 0.4
    # Regression for the y2 typo (right_post[0] used instead of right_post[1]),
    # which stretched the box to the bottom of the image.
    assert abs(estimate.box_y - 35 / 180) < 0.01
    assert abs(estimate.box_height - 116 / 180) < 0.01
    assert len(debug_frames) == 2

def test_baseline_requires_two_posts():
    for frame in (make_gate_frame(right=False), make_gate_frame(left=False)):
        assert not ClassicalGatePerceiver().analyze(frame, debug=False).visible

def test_declared_slider_defaults_match_code_defaults():
    perceiver = ClassicalGatePerceiver()
    frame = make_gate_frame()
    defaults = {name: default for name, (_range, default) in perceiver.kwargs.items()}
    assert set(defaults) == {
        "red_h_min", "red_h_max", "red_s_min", "red_v_min", "black_v_max"
    }
    assert perceiver.analyze(frame, debug=False, slider_vals=defaults) == perceiver.analyze(
        frame, debug=False
    )


