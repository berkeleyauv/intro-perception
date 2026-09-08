"""Intentionally simple temporal filter for students to improve."""

from __future__ import annotations

from intro_perception.types import GateEstimate


class GateTracker:
    """Smooth detections and bridge a few missing frames.

    This baseline has no motion model or distractor rejection. Those omissions
    are deliberate project opportunities, not recommended production behavior.
    """

    def __init__(self, alpha: float = 0.35, max_missed_frames: int = 3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")
        self.alpha = alpha
        self.max_missed_frames = max_missed_frames
        self._estimate: GateEstimate | None = None
        self._missed_frames = 0

    def reset(self) -> None:
        self._estimate = None
        self._missed_frames = 0

    def update(self, observation: GateEstimate) -> GateEstimate:
        observation = observation.normalized()
        if observation.visible:
            self._missed_frames = 0
            if self._estimate is None:
                self._estimate = observation
            else:
                previous = self._estimate
                alpha = self.alpha
                orientation = observation.orientation_rad
                if orientation is None:
                    orientation = previous.orientation_rad
                self._estimate = GateEstimate(
                    visible=True,
                    confidence=(
                        alpha * observation.confidence
                        + (1.0 - alpha) * previous.confidence
                    ),
                    center_x=(
                        alpha * observation.center_x
                        + (1.0 - alpha) * previous.center_x
                    ),
                    center_y=(
                        alpha * observation.center_y
                        + (1.0 - alpha) * previous.center_y
                    ),
                    orientation_rad=orientation,
                )
            return self._estimate

        if self._estimate is None:
            return GateEstimate.invisible()

        self._missed_frames += 1
        if self._missed_frames > self.max_missed_frames:
            self.reset()
            return GateEstimate.invisible()

        confidence_scale = 1.0 - (
            self._missed_frames / (self.max_missed_frames + 1.0)
        )
        previous = self._estimate
        self._estimate = GateEstimate(
            visible=True,
            confidence=previous.confidence * confidence_scale,
            center_x=previous.center_x,
            center_y=previous.center_y,
            orientation_rad=previous.orientation_rad,
        )
        return self._estimate
