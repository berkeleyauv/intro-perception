from intro_perception.types import GateEstimate


def test_normalized_bounds_values():
    estimate = GateEstimate(True, 2.0, -0.5, 1.5).normalized()
    assert estimate.confidence == 1.0
    assert estimate.center_x == 0.0
    assert estimate.center_y == 1.0


def test_invisible_uses_neutral_center():
    estimate = GateEstimate.invisible()
    assert not estimate.visible
    assert estimate.confidence == 0.0
    assert (estimate.center_x, estimate.center_y) == (0.5, 0.5)
