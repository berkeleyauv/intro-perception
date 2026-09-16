"""Stable output contract shared by both intro gate detectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class GateOrientation(str, Enum):
    LEFT = "left"
    HEAD_ON = "head_on"
    RIGHT = "right"


@dataclass(frozen=True)
class GateEstimate:
    """One frame's gate estimate in normalized image coordinates."""

    visible: bool
    confidence: float
    center_x: float
    center_y: float
    orientation: GateOrientation | None = None
    box_x: float = 0.0
    box_y: float = 0.0
    box_width: float = 0.0
    box_height: float = 0.0

    def normalized(self) -> "GateEstimate":
        if not self.visible:
            return self.invisible()
        orientation = self.orientation
        if isinstance(orientation, str):
            orientation = GateOrientation(orientation)
        box_x = _clamp01(self.box_x)
        box_y = _clamp01(self.box_y)
        return GateEstimate(
            True,
            _clamp01(self.confidence),
            _clamp01(self.center_x),
            _clamp01(self.center_y),
            orientation,
            box_x,
            box_y,
            min(_clamp01(self.box_width), 1.0 - box_x),
            min(_clamp01(self.box_height), 1.0 - box_y),
        )

    @classmethod
    def invisible(cls) -> "GateEstimate":
        return cls(False, 0.0, 0.5, 0.5)

    def to_dict(self):
        payload = asdict(self)
        payload["orientation"] = self.orientation.value if self.orientation else None
        return payload
