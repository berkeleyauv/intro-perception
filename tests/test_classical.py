"""Regression tests for the classical detector on pool-like synthetic scenes.

The scenes (``gate_helpers.make_scene``) use paint and water colors measured
on the supplied images and reproduce what the real frames contain: support
pipes above the crossbar painted like the other post, diagonal lane lines that
touch the posts, gates cut off by the frame, rotation padding. Each test pins
one behavior; ``scripts/hard_cases.py`` prints the same cases as a report.
"""

import numpy as np
import pytest
from gate_helpers import box_iou_pixels, make_gate_frame, make_scene

from intro_perception import classical
from intro_perception.classical import ClassicalGatePerceiver
from intro_perception.color_model import padding_mask


@pytest.fixture(scope="module")
def perceiver():
    return ClassicalGatePerceiver()


GATE_SCENES = {
    "pool_gate": {},
    "angled_far_post_shorter": {"gate_width": 220, "right_height": 160},
    "small_far_gate": {"gate_x": 380, "gate_width": 120, "top": 150,
                       "post_height": 80, "post_width": 6},
    "no_supports_or_crossbar": {"supports": False, "crossbar": False},
    "dark_frame": {"brightness": -35},
    "rotated_with_padding": {"rotate": 6},
    "jpeg": {"jpeg": 50},
}


@pytest.mark.parametrize("name", sorted(GATE_SCENES))
def test_finds_gate(perceiver, name):
    frame, truth = make_scene(**GATE_SCENES[name])
    estimate = perceiver.analyze(frame, debug=False)
    assert estimate.visible, name
    assert box_iou_pixels(estimate, truth, frame.shape) >= 0.7, name


def test_box_top_is_the_crossbar_not_the_support_pipe(perceiver):
    # The support pipe continues the post's top color up to the surface; the
    # box must stop at the crossbar (one bottom-half length above the color
    # change), not run to the top of the frame.
    frame, truth = make_scene()
    estimate = perceiver.analyze(frame, debug=False)
    assert abs(estimate.box_y * frame.shape[0] - truth[1]) < 12
    assert box_iou_pixels(estimate, truth, frame.shape) >= 0.9


@pytest.mark.parametrize("gate_x", [420, -230])
def test_gate_cut_off_at_side_is_boxed_to_the_frame_edge(perceiver, gate_x):
    # Half of the supplied gates are cut off like this; the baseline said "no gate".
    frame, truth = make_scene(gate_x=gate_x, gate_width=420)
    estimate = perceiver.analyze(frame, debug=False)
    assert estimate.visible
    assert box_iou_pixels(estimate, truth, frame.shape) >= 0.8
    touches_edge = estimate.box_x < 0.01 or estimate.box_x + estimate.box_width > 0.99
    assert touches_edge


def test_one_post_gate_is_less_confident_than_a_full_gate():
    full = classical.detect(make_scene()[0]).best
    cut = classical.detect(make_scene(gate_x=420, gate_width=420)[0]).best
    assert (full.kind, cut.kind) == ("pair", "one_post")
    assert cut.score < full.score


@pytest.mark.parametrize("kwargs", [
    {},
    {"lane_lines": False, "noise": 10.0},
    # Dark vertical poles and a pool-wall stripe, but no red/black posts.
    {"distractors": [(100, 40, 106, 300, (60, 40, 10)), (400, 20, 404, 330, (50, 35, 10)),
                     (0, 150, 640, 160, (40, 25, 5))]},
    {"rotate": 8},
])
def test_no_gate_scenes_stay_invisible(perceiver, kwargs):
    frame, _truth = make_scene(left=False, right=False, **kwargs)
    assert not perceiver.analyze(frame, debug=False).visible


def test_gate_seen_from_behind_is_found_with_lower_score(perceiver):
    frame, truth = make_scene(red=(32, 17, 3), black=(109, 84, 84))
    estimate = perceiver.analyze(frame, debug=False)
    assert estimate.visible
    assert box_iou_pixels(estimate, truth, frame.shape) >= 0.7
    assert classical.detect(frame).best.score < classical.detect(make_scene()[0]).best.score


def test_rotation_padding_is_found_and_ignored():
    frame, _truth = make_scene(rotate=10)
    padding = padding_mask(frame)
    assert padding[0, 0] and padding[-1, -1]
    assert not padding[frame.shape[0] // 2, frame.shape[1] // 2]
    # Real gray-ish pool pixels are textured, so they are not padding.
    assert not padding_mask(make_scene(rotate=0)[0]).any()


def test_large_frames_are_downscaled_but_boxes_stay_normalized(perceiver):
    frame, truth = make_scene()
    big = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
    estimate = perceiver.analyze(big, debug=False)
    assert box_iou_pixels(estimate, truth, frame.shape) >= 0.9


def test_debug_frames_match_input_size(perceiver):
    frame, _truth = make_scene()
    estimate, debug_frames = perceiver.analyze(frame, debug=True)
    assert estimate.visible
    assert [d.shape for d in debug_frames] == [frame.shape] * 2


def test_confidence_is_monotonic_in_score():
    calibration = classical.load_calibration()
    scores = [0.3, 0.5, 0.7, 0.9]
    confidences = [classical.calibrated_confidence(s, calibration) for s in scores]
    assert confidences == sorted(confidences)
    assert all(0.0 < c < 1.0 for c in confidences)


def test_calibration_file_overrides_default(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    path.write_text('{"a": 2.0, "b": 0.1}')
    monkeypatch.setenv("INTRO_CLASSICAL_CALIBRATION", str(path))
    assert classical.load_calibration() == {"a": 2.0, "b": 0.1}


def test_spec_gate_still_found(perceiver):
    assert perceiver.analyze(make_gate_frame(), debug=False).visible
