from intro_perception.types import GateEstimate, GateOrientation


def test_normalized_bounds_values():
    estimate = GateEstimate(
        True, 2.0, -0.5, 1.5, "left", -0.2, 0.1, 1.4, 0.8
    ).normalized()
    assert estimate.confidence == 1.0
    assert estimate.center_x == 0.0
    assert estimate.center_y == 1.0
    assert estimate.orientation is GateOrientation.LEFT
    assert estimate.box_x == 0.0
    assert estimate.box_width == 1.0


def test_box_is_clipped_to_image_boundary():
    estimate = GateEstimate(
        True, 1.0, 0.9, 0.9, "head_on", 0.8, 0.7, 0.5, 0.5
    ).normalized()
    assert estimate.box_x + estimate.box_width == 1.0
    assert estimate.box_y + estimate.box_height == 1.0


def test_invisible_discards_detection_fields():
    estimate = GateEstimate(False, 0.8, 0.1, 0.2, "right", 0.1, 0.1, 0.5, 0.5).normalized()
    assert estimate == GateEstimate.invisible()
    assert estimate.to_dict()["orientation"] is None
