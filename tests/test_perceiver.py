from gate_helpers import make_gate_frame

from intro_perception.classical import ClassicalGatePerceiver
from perception.tasks.registry import get_perceiver


def test_intro_classical_uses_production_registry():
    assert get_perceiver("gate", "intro_classical") is ClassicalGatePerceiver


def test_finds_spec_gate_and_box():
    estimate, debug_frames = ClassicalGatePerceiver().analyze(make_gate_frame(), debug=True)
    assert estimate.visible
    assert abs(estimate.center_x - 0.5) < 0.02
    assert abs(estimate.center_y - 93 / 180) < 0.02
    assert estimate.box_width > 0.4
    # Regression for the y2 typo (right_post[0] used instead of right_post[1]),
    # which stretched the box to the bottom of the image.
    assert abs(estimate.box_y - 35 / 180) < 0.02
    assert abs(estimate.box_height - 116 / 180) < 0.02
    # annotated frame and the red/black paint-mask overlay, both input-sized
    assert len(debug_frames) == 2
    assert all(d.shape == make_gate_frame().shape for d in debug_frames)


def test_lone_post_means_a_gate_cut_off_by_the_frame():
    # The baseline required two posts. In the supplied data a lone complete
    # post is a gate whose other post is outside the frame (769 of 1480 gates),
    # so the box runs from the post to the edge on the missing post's side.
    left_only = ClassicalGatePerceiver().analyze(make_gate_frame(right=False), debug=False)
    assert left_only.visible and left_only.box_x + left_only.box_width > 0.99
    assert abs(left_only.box_x - 80 / 320) < 0.02
    right_only = ClassicalGatePerceiver().analyze(make_gate_frame(left=False), debug=False)
    assert right_only.visible and right_only.box_x < 0.01


def test_no_posts_means_no_gate():
    frame = make_gate_frame(left=False, right=False)
    assert not ClassicalGatePerceiver().analyze(frame, debug=False).visible


def test_declared_slider_defaults_match_code_defaults():
    perceiver = ClassicalGatePerceiver()
    frame = make_gate_frame()
    defaults = {name: default for name, (_range, default) in perceiver.kwargs.items()}
    assert set(defaults) == {"black_min", "red_min"}
    assert perceiver.analyze(frame, debug=False, slider_vals=defaults) == perceiver.analyze(
        frame, debug=False
    )
