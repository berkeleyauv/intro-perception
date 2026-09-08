from intro_perception.tracker import GateTracker
from intro_perception.types import GateEstimate


def detection(x=0.4):
    return GateEstimate(True, 0.8, x, 0.5)


def test_tracker_smooths_new_observation():
    tracker = GateTracker(alpha=0.5)
    tracker.update(detection(0.2))
    result = tracker.update(detection(0.6))
    assert result.center_x == 0.4


def test_tracker_resets_after_miss_budget():
    tracker = GateTracker(max_missed_frames=1)
    tracker.update(detection())
    assert tracker.update(GateEstimate.invisible()).visible
    assert not tracker.update(GateEstimate.invisible()).visible
