"""Stable assignment data contracts."""

from __future__ import annotations

from dataclasses import dataclass


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class GateEstimate:
    """One frame's gate estimate in normalized image coordinates."""

    visible: bool
    confidence: float
    center_x: float
    center_y: float
    orientation_rad: float | None = None

    def normalized(self) -> "GateEstimate":
        """Return a copy with bounded confidence and image coordinates."""
        return GateEstimate(
            visible=bool(self.visible),
            confidence=_clamp01(self.confidence),
            center_x=_clamp01(self.center_x),
            center_y=_clamp01(self.center_y),
            orientation_rad=self.orientation_rad,
        )

    @classmethod
    def invisible(cls) -> "GateEstimate":
        return cls(False, 0.0, 0.5, 0.5, None)

    def to_dict(self) -> dict[str, bool | float | None]:
        return {
            "visible": self.visible,
            "confidence": self.confidence,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "orientation_rad": self.orientation_rad,
        }
