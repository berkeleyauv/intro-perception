from gate_helpers import make_scene

from intro_perception import classical
from intro_perception.orientation import estimate_orientation

FRONT_RATIO = (300 - 18) / 220  # center spacing / height of make_scene()'s gate


def orientation(typical=FRONT_RATIO, **kwargs):
    frame, _truth = make_scene(**kwargs)
    return estimate_orientation(classical.detect(frame), typical_ratio=typical)


def test_frontal_gate_has_near_zero_yaw():
    result = orientation()
    assert result.nearer_post == "even"
    assert abs(result.yaw_deg) < 10
    assert not result.from_behind


def test_shorter_right_post_means_right_side_is_farther():
    # Turned gate: narrower spacing and a shorter right post.
    result = orientation(gate_width=200, right_height=170)
    assert result.nearer_post == "left"
    assert result.yaw_deg > 20


def test_shorter_left_post_gives_negative_yaw():
    result = orientation(gate_width=200, post_height=170, right_height=220, top=90)
    assert result.nearer_post == "right"
    assert result.yaw_deg < -20


def test_rear_view_is_flagged():
    result = orientation(red=(25, 30, 20), black=(40, 30, 150))
    assert result.from_behind


def test_no_angle_without_a_pair():
    assert orientation(left=False) is None
    assert orientation(left=False, right=False) is None
