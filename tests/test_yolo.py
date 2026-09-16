from types import SimpleNamespace

import pytest

from intro_perception.types import GateOrientation
from intro_perception.yolo import YoloGatePerceiver, estimate_from_result


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value

    def item(self):
        return self.value

    def __getitem__(self, index):
        return FakeTensor(self.value[index])

    def __len__(self):
        return len(self.value)


def test_yolo_result_conversion():
    boxes = SimpleNamespace(
        conf=FakeTensor([0.4, 0.9]),
        cls=FakeTensor([0, 2]),
        xyxy=FakeTensor([[0, 0, 10, 10], [20, 10, 100, 90]]),
    )
    result = SimpleNamespace(boxes=boxes, names={0: "gate_left", 2: "gate_right"})
    estimate = estimate_from_result(result, (100, 200, 3))
    assert estimate.confidence == 0.9
    assert estimate.orientation is GateOrientation.RIGHT
    assert estimate.center_x == 0.3
    assert estimate.box_height == 0.8


def test_yolo_no_boxes_is_invisible():
    result = SimpleNamespace(
        boxes=SimpleNamespace(conf=FakeTensor([])), names={}
    )
    assert not estimate_from_result(result, (100, 200, 3)).visible


def test_yolo_reports_missing_weights_before_importing_optional_dependency(tmp_path):
    with pytest.raises(FileNotFoundError, match="train first"):
        YoloGatePerceiver(tmp_path / "missing.pt")
