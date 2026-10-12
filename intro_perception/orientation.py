"""Optional extension: rough gate yaw from the classical detector's post geometry.

This does not change the ``GateEstimate`` contract; it is an extra estimate
that can be computed from the same :class:`intro_perception.classical.Detection`.

Seen head-on, both posts have the same apparent height and their spacing over
height is the gate's true ratio (``GATE_RATIO_TYPICAL``). Turned by a yaw angle
``theta`` (pinhole camera, gate far compared with its width):

* the spacing shrinks roughly by ``cos(theta)``, and
* the far post looks shorter, so the shorter post tells the sign.

``cos(theta)`` is taken from the spacing ratio; the height ratio is used for
the sign and as a consistency check. Only a full left/right pair (front or rear
view) gives an angle; single posts and diagonals return ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from intro_perception import classical


@dataclass(frozen=True)
class GateOrientation:
    yaw_deg: float          # 0 = facing the gate; positive = right post is farther
    nearer_post: str        # "left", "right" or "even"
    from_behind: bool       # posts appear mirrored
    spacing_ratio: float    # observed center spacing / post height
    height_ratio: float     # right post height / left post height (image left/right)


def estimate_orientation(detection, typical_ratio=None, even_tolerance=0.06):
    """Return a :class:`GateOrientation` for a paired detection, else ``None``."""
    best = detection.best
    if best is None or best.kind not in ("pair", "mirrored") or len(best.posts) != 2:
        return None
    typical = typical_ratio or classical.GATE_RATIO_TYPICAL
    first, second = sorted(best.posts, key=lambda post: post.cx)
    height = max(first.height, second.height)
    spacing = (second.cx - first.cx) / max(1.0, height)
    cos_yaw = max(0.0, min(1.0, spacing / typical))
    yaw = math.degrees(math.acos(cos_yaw))
    height_ratio = second.height / max(1.0, first.height)
    if abs(height_ratio - 1.0) <= even_tolerance:
        nearer = "even"
        yaw = 0.0 if spacing >= typical * (1 - even_tolerance) else yaw
    else:
        nearer = "left" if height_ratio < 1.0 else "right"
    signed = yaw if nearer == "left" else -yaw if nearer == "right" else yaw
    return GateOrientation(
        round(signed, 1), nearer, best.kind == "mirrored", round(spacing, 3), round(height_ratio, 3)
    )
