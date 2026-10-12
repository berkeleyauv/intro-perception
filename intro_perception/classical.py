"""Classical (OpenCV) gate detector.

Gate appearance (as seen facing the gate). The two posts hang from a black
crossbar; the label box spans from the crossbar to the bottom of the posts.

* left post:  black on top, red on the bottom
* right post: red on top, black on the bottom

Above the crossbar the support pipes continue upward in the same color as the
post's top half, so the top half often looks much longer than the bottom half.

Pipeline (each stage is a small function so it can be tested and tuned alone):

1. **Paint maps** - a color look-up table (``color_model.py``, fitted on the
   training split by ``scripts/fit_color_model.py``) scores every pixel as
   black paint / red paint / background from its Lab color *relative to the
   image's median color*, which cancels the water tint and the dataset's hue
   and brightness augmentation. Rotation padding (pure black borders) is
   ignored.
2. **Masks** - thresholded log-odds, cleaned with morphology.
3. **Segments** - connected components that are long, thin and roughly
   vertical (tilt allowed). Collinear pieces of one color split by ripples or
   the hanging panels are merged.
4. **Posts from the color pattern** - a black segment directly above a red one
   is a *left* post; red above black is a *right* post. The two halves have the
   same length, so the post's top (the crossbar) is placed one bottom-half
   length above the color change; this cuts off the support pipe above the
   crossbar.
5. **Pairing** - a left post to the left of a right post with similar length,
   aligned tops and plausible spacing. A mirrored pair (gate seen from behind)
   is accepted at a lower score.
6. **One visible post** - about half of the gates in this data are cut off by
   the left or right frame edge. A complete post with no partner gives a box
   from the post to the frame edge on the side where the other post must be.
7. **Confidence** - the geometric score is mapped through a logistic curve
   fitted on the validation split (``scripts/calibrate_classical.py``). Below
   ``MIN_SCORE`` the detector returns invisible rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import cv2 as cv
import numpy as np

from intro_perception.production import prepare_production_namespace

prepare_production_namespace()

from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.registry import register_perceiver

from intro_perception.color_model import ColorModel, padding_mask
from intro_perception.types import GateEstimate


# --- tunables --------------------------------------------------------------------

# Slider name -> ((min, max), default). Thresholds are paint log-odds x 10.
SLIDERS = {
    "black_min": ((0, 60), 12),
    "red_min": ((0, 60), 8),
}
DEFAULTS = {name: default for name, (_range, default) in SLIDERS.items()}

WORK_WIDTH = 640           # frames are processed at most this wide
MIN_SEGMENT_LENGTH = 0.05  # of frame height
MAX_SEGMENT_WIDTH = 0.08   # of frame width
MIN_ELONGATION = 2.5       # length / width
MAX_TILT_DEG = 30.0        # from vertical
VERTICAL_BLUR = 3          # work pixels, height of the score averaging
VERTICAL_OPEN = 15         # work pixels
MERGE_GAP = 0.06           # of frame height, between collinear pieces

# Gate spacing prior: distance between post centers divided by post height.
# Measured on the training labels (scripts/gate_stats.py): whole-gate box
# width/height has median 1.35 (p10 1.08, p90 1.53); the post-center spacing
# is a post width less. The pairing range is wide to allow angled views.
GATE_RATIO_MIN = 0.6
GATE_RATIO_MAX = 2.6
GATE_RATIO_TYPICAL = 1.3   # head-on spacing / height, used by orientation.py

MIRRORED_FACTOR = 0.8      # gate seen from behind: red-top post on the left
HALF_POST_FACTOR = 0.6     # a post seen only as one half reaching the frame bottom
ONE_POST_FACTOR = 0.75     # a gate inferred from one post (cut off by the frame)
MIN_SCORE = 0.45           # below this, report "no gate"

# Logistic confidence calibration: confidence = 1 / (1 + exp(-a * (score - b))).
# Fitted values live in classical_calibration.json next to this file, or the
# file named by INTRO_CLASSICAL_CALIBRATION.
DEFAULT_CALIBRATION = {"a": 9.0, "b": 0.45}


def load_calibration():
    path = Path(
        os.environ.get(
            "INTRO_CLASSICAL_CALIBRATION",
            Path(__file__).with_name("classical_calibration.json"),
        )
    )
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"a": float(data["a"]), "b": float(data["b"])}
    return dict(DEFAULT_CALIBRATION)


_COLOR_MODEL = None


def color_model():
    global _COLOR_MODEL
    if _COLOR_MODEL is None:
        _COLOR_MODEL = ColorModel.load()
    return _COLOR_MODEL


# --- data types --------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One elongated blob of a single paint color, in work pixels.

    The blob's axis runs from ``(top_x, top_y)`` to ``(bottom_x, bottom_y)``.
    """

    color: str  # "red" or "black"
    top_x: float
    top_y: float
    bottom_x: float
    bottom_y: float
    width: float
    area: int
    touches_top: bool
    touches_bottom: bool

    @property
    def length(self):
        return self.bottom_y - self.top_y

    @property
    def cx(self):
        return (self.top_x + self.bottom_x) / 2.0

    def x_at(self, y):
        if self.bottom_y == self.top_y:
            return self.cx
        t = (y - self.top_y) / (self.bottom_y - self.top_y)
        return self.top_x + t * (self.bottom_x - self.top_x)

    @property
    def fill(self):
        return min(1.0, self.area / max(1.0, self.length * self.width))


@dataclass(frozen=True)
class Post:
    side: str  # "left" or "right"
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    halves: int  # 2 = both colors seen, 1 = only one half
    junction: float = 0.0

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2.0


@dataclass(frozen=True)
class Candidate:
    box: tuple  # (x1, y1, x2, y2) in work pixels
    score: float
    kind: str  # "pair", "mirrored", "one_post"
    posts: tuple


# --- stages 1-2: paint maps and masks -------------------------------------------


def paint_masks(frame, values):
    """Return ``(red_mask, black_mask, padding)`` for a BGR work frame."""
    smooth = cv.GaussianBlur(frame, (5, 5), 0)
    padding = padding_mask(frame)
    black, red = color_model().paint_scores(smooth, padding)
    # Average along the vertical: posts are long and near-vertical, so this
    # gathers faint evidence along them and washes out blobs and diagonals.
    kernel = (3, VERTICAL_BLUR)
    black = cv.blur(black, kernel)
    red = cv.blur(red, kernel)
    black_mask = (black > values["black_min"] / 10.0) & (black > red)
    red_mask = (red > values["red_min"] / 10.0) & (red >= black)
    return clean_mask(red_mask), clean_mask(black_mask), padding


def clean_mask(mask):
    """Keep vertical structure: a tall opening removes speckle, the crossbar,
    and the diagonal lane lines that touch the posts; closing fills ripples."""
    mask = mask.astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((VERTICAL_OPEN, 3), np.uint8))
    return cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((9, 3), np.uint8))


# --- stage 3: segments --------------------------------------------------------------


def find_segments(mask, color, padding=None):
    height, width = mask.shape[:2]
    min_length = MIN_SEGMENT_LENGTH * height
    max_width = MAX_SEGMENT_WIDTH * width
    # Padding borders count as frame edges for "touches" checks.
    if padding is not None and padding.any():
        rows = np.flatnonzero((~padding).any(axis=1))
        top_edge, bottom_edge = rows[0], rows[-1]
    else:
        top_edge, bottom_edge = 0, height - 1
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    segments = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if h < min_length or area < 20:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == index)
        points = np.column_stack((xs + x, ys + y)).astype(np.float32)
        (cx, cy), (rw, rh), angle = cv.minAreaRect(points)
        length, thickness = max(rw, rh), max(1.0, min(rw, rh))
        if length < min_length or thickness > max_width or length < MIN_ELONGATION * thickness:
            continue
        vx, vy, x0, y0 = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01).ravel()
        if abs(vy) < 1e-6:
            continue
        tilt = math.degrees(math.atan2(abs(vx), abs(vy)))
        if tilt > MAX_TILT_DEG:
            continue
        top_y, bottom_y = float(y), float(y + h)
        top_x = float(x0 + (top_y - y0) * vx / vy)
        bottom_x = float(x0 + (bottom_y - y0) * vx / vy)
        segments.append(Segment(
            color, top_x, top_y, bottom_x, bottom_y, float(thickness), area,
            y <= top_edge + 2, y + h >= bottom_edge - 1,
        ))
    return merge_collinear(segments, height)


def merge_collinear(segments, frame_height):
    """Join pieces of one post split by ripples, panels or the crossbar."""
    segments = sorted(segments, key=lambda s: s.top_y)
    merged = True
    while merged:
        merged = False
        for i, upper in enumerate(segments):
            for j in range(i + 1, len(segments)):
                lower = segments[j]
                gap = lower.top_y - upper.bottom_y
                if gap < -0.02 * frame_height or gap > MERGE_GAP * frame_height:
                    continue
                offset = abs(upper.x_at(lower.top_y) - lower.top_x)
                if offset > max(upper.width, lower.width) * 1.2 + 3:
                    continue
                if ratio(upper.width, lower.width) < 0.4:
                    continue
                joined = Segment(
                    upper.color, upper.top_x, upper.top_y, lower.bottom_x, lower.bottom_y,
                    max(upper.width, lower.width), upper.area + lower.area,
                    upper.touches_top, lower.touches_bottom,
                )
                segments = [s for k, s in enumerate(segments) if k not in (i, j)] + [joined]
                segments.sort(key=lambda s: s.top_y)
                merged = True
                break
            if merged:
                break
    return segments


# --- stage 4: posts from the color pattern ------------------------------------------


def ratio(small, large):
    small, large = sorted((float(small), float(large)))
    return small / large if large > 0 else 0.0


def stack_post(upper, lower, frame_height):
    """Score ``upper`` directly above ``lower`` as one post, or return None."""
    gap = lower.top_y - upper.bottom_y
    tallest = max(upper.length, lower.length)
    if gap < -0.25 * min(upper.length, lower.length) or gap > 0.05 * frame_height + 4:
        return None
    junction = (upper.bottom_y + lower.top_y) / 2.0
    offset = abs(upper.x_at(junction) - lower.x_at(junction))
    widest = max(upper.width, lower.width)
    if offset > 1.5 * widest + 4:
        return None
    width_ratio = ratio(upper.width, lower.width)
    if width_ratio < 0.35:
        return None

    # Both halves have the same length; the upper color usually continues
    # above the crossbar, so the lower half sets the post's height.
    if lower.touches_bottom:
        half = None
        top = upper.top_y
    else:
        half = lower.length
        top = max(upper.top_y, junction - half)
    bottom = lower.bottom_y
    upper_visible = junction - top
    balance = 1.0 if half is None else min(1.0, upper_visible / max(1.0, half))
    if half is not None and balance < 0.3:
        return None

    xs = [upper.x_at(top), upper.x_at(junction), lower.x_at(junction), lower.x_at(bottom)]
    x1 = min(xs) - widest / 2.0
    x2 = max(xs) + widest / 2.0
    side = "left" if upper.color == "black" else "right"
    adjacency = 1.0 - min(1.0, max(0.0, abs(gap)) / (0.05 * frame_height + 4))
    alignment = 1.0 - min(1.0, offset / (1.5 * widest + 4))
    size = min(1.0, (bottom - top) / (0.25 * frame_height))
    score = (
        0.25 * adjacency
        + 0.2 * alignment
        + 0.15 * width_ratio
        + 0.2 * balance
        + 0.2 * min(upper.fill, lower.fill)
    ) * (0.7 + 0.3 * size)
    return Post(side, x1, top, x2, bottom, score, 2, junction)


def half_post(segment, frame_height):
    """A lone segment running into the bottom of the frame: a post's top half.

    Black running off the bottom is the left post's top half; red is the
    right post's.
    """
    if not segment.touches_bottom or segment.touches_top:
        return None
    if segment.length < 0.25 * frame_height:
        return None
    side = "left" if segment.color == "black" else "right"
    x1 = min(segment.top_x, segment.bottom_x) - segment.width / 2.0
    x2 = max(segment.top_x, segment.bottom_x) + segment.width / 2.0
    score = (0.5 + 0.5 * segment.fill) * HALF_POST_FACTOR
    return Post(side, x1, segment.top_y, x2, segment.bottom_y, score, 1, segment.bottom_y)


def find_posts(red_segments, black_segments, frame_height):
    posts = []
    used = set()
    for black in black_segments:
        for red in red_segments:
            for upper, lower in ((black, red), (red, black)):
                if upper.top_y >= lower.top_y:
                    continue
                post = stack_post(upper, lower, frame_height)
                if post is not None:
                    posts.append(post)
                    used.update((id(black), id(red)))
    for segment in (*red_segments, *black_segments):
        if id(segment) in used:
            continue
        post = half_post(segment, frame_height)
        if post is not None:
            posts.append(post)
    return suppress_posts(posts)


def suppress_posts(posts):
    """Drop support pipes, then keep the best post among overlapping ones.

    Above the crossbar each post's support pipe is painted like the *other*
    post (the left post's support is red over black), so a post-like pattern
    sitting directly above another post on the same vertical line is the
    support, not a post: the posts hang below the crossbar.
    """
    posts = [
        post for post in posts
        if not any(is_support_above(post, other) for other in posts if other is not post)
    ]
    kept = []
    for post in sorted(posts, key=lambda p: p.score, reverse=True):
        if all(box_overlap(other, post) < 0.5 for other in kept):
            kept.append(post)
    return kept


def is_support_above(upper, lower):
    if abs(upper.cx - lower.cx) > max(upper.width, lower.width) + 6:
        return False
    return upper.junction < lower.y1 + 0.25 * lower.height and upper.y1 < lower.y1


def box_overlap(a, b):
    inter_w = min(a.x2, b.x2) - max(a.x1, b.x1)
    inter_h = min(a.y2, b.y2) - max(a.y1, b.y1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    smaller = min(a.width * a.height, b.width * b.height)
    return inter_w * inter_h / float(smaller) if smaller else 0.0


# --- stages 5-6: pairing and one-post gates --------------------------------------------


def spacing_score(left_cx, right_cx, height):
    spacing = (right_cx - left_cx) / max(1.0, height)
    if spacing < 0.5 * GATE_RATIO_MIN or spacing > 1.5 * GATE_RATIO_MAX:
        return 0.0
    if spacing < GATE_RATIO_MIN:
        return (spacing - 0.5 * GATE_RATIO_MIN) / (0.5 * GATE_RATIO_MIN)
    if spacing > GATE_RATIO_MAX:
        return (1.5 * GATE_RATIO_MAX - spacing) / (0.5 * GATE_RATIO_MAX)
    return 1.0


def pair_posts(left, right):
    if right.cx <= left.cx:
        return None
    height_similarity = ratio(left.height, right.height)
    full = left.halves == 2 and right.halves == 2
    if full and height_similarity < 0.4:
        return None
    tallest = max(left.height, right.height)
    top_alignment = 1.0 - min(1.0, abs(left.y1 - right.y1) / (0.5 * tallest))
    vertical_overlap = (min(left.y2, right.y2) - max(left.y1, right.y1)) / max(
        1.0, min(left.height, right.height)
    )
    if vertical_overlap < 0.4:
        return None
    width_similarity = ratio(left.width, right.width)
    if width_similarity < 0.25:
        return None
    spacing = spacing_score(left.cx, right.cx, tallest)
    if spacing <= 0:
        return None
    geometry = (
        0.25 * (height_similarity if full else 0.7)
        + 0.25 * top_alignment
        + 0.15 * min(1.0, vertical_overlap)
        + 0.1 * width_similarity
        + 0.25 * spacing
    )
    score = math.sqrt(left.score * right.score) * (0.4 + 0.6 * geometry)
    box = (
        min(left.x1, right.x1),
        min(left.y1, right.y1),
        max(left.x2, right.x2),
        max(left.y2, right.y2),
    )
    return Candidate(box, score, "pair", (left, right))


def one_post_gate(post, frame_width):
    """A gate cut off by the left or right frame edge, from its one visible post.

    The left post's partner is to its right, so the box runs to the right edge;
    the right post's runs to the left edge.
    """
    if post.side == "left":
        box = (post.x1, post.y1, frame_width, post.y2)
    else:
        box = (0, post.y1, post.x2, post.y2)
    return Candidate(box, post.score * ONE_POST_FACTOR, "one_post", (post,))


def choose_candidate(posts, frame_width):
    candidates = []
    lefts = [p for p in posts if p.side == "left"]
    rights = [p for p in posts if p.side == "right"]
    for left in lefts:
        for right in rights:
            candidate = pair_posts(left, right)
            if candidate is not None:
                candidates.append(candidate)
            # From behind, the gate looks mirrored. It is still the gate (the
            # only class), but a front view is preferred.
            mirrored = pair_posts(right, left)
            if mirrored is not None:
                candidates.append(Candidate(
                    mirrored.box, mirrored.score * MIRRORED_FACTOR, "mirrored", mirrored.posts
                ))
    for post in posts:
        candidates.append(one_post_gate(post, frame_width))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


def calibrated_confidence(score, calibration):
    return 1.0 / (1.0 + math.exp(-calibration["a"] * (score - calibration["b"])))


# --- whole pipeline --------------------------------------------------------------


@dataclass
class Detection:
    """Everything one frame produced, in work (downscaled) pixels."""

    best: Candidate | None
    posts: list
    red_mask: np.ndarray
    black_mask: np.ndarray
    work_shape: tuple
    scale: float  # work pixels per original pixel


def detect(frame, slider_vals=None):
    """Run stages 1-6 and return the best candidate with intermediates."""
    if frame is None or frame.size == 0:
        raise ValueError("frame must be a non-empty image")
    values = dict(DEFAULTS)
    values.update(slider_vals or {})
    scale = min(1.0, WORK_WIDTH / float(frame.shape[1]))
    work = (
        cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        if scale < 1.0
        else frame
    )
    work_height, work_width = work.shape[:2]
    red_mask, black_mask, padding = paint_masks(work, values)
    red_segments = find_segments(red_mask, "red", padding)
    black_segments = find_segments(black_mask, "black", padding)
    posts = find_posts(red_segments, black_segments, work_height)
    best = choose_candidate(posts, work_width)
    return Detection(best, posts, red_mask, black_mask, work.shape[:2], scale)


# --- perceiver ----------------------------------------------------------------------


@register_perceiver(task="gate", algo="intro_classical")
class ClassicalGatePerceiver(TaskPerceiver):
    """Detect the gate from its two red/black posts."""

    def __init__(self):
        super().__init__(**SLIDERS)
        self.calibration = load_calibration()
        color_model()  # fail early if the color table is missing

    def analyze(self, frame: np.ndarray, debug: bool, slider_vals=None):
        detection = detect(frame, slider_vals)
        estimate = self.estimate(detection)
        if not debug:
            return estimate
        return estimate, debug_frames(frame, estimate, detection)

    def estimate(self, detection):
        best = detection.best
        if best is None or best.score < MIN_SCORE:
            return GateEstimate.invisible()
        work_height, work_width = detection.work_shape
        x1, y1, x2, y2 = best.box
        return GateEstimate(
            True,
            calibrated_confidence(best.score, self.calibration),
            (x1 + x2) / (2.0 * work_width),
            (y1 + y2) / (2.0 * work_height),
            x1 / work_width,
            y1 / work_height,
            (x2 - x1) / work_width,
            (y2 - y1) / work_height,
        ).normalized()


def debug_frames(frame, estimate, detection):
    """Annotated frame and the red (red) / black (green) paint-mask overlay.

    Both are the size of the input frame, as the visualizer requires.
    """
    size = (frame.shape[1], frame.shape[0])
    annotated = annotate(frame, estimate)
    to_original = 1.0 / detection.scale
    for post in detection.posts:
        color = (0, 255, 0) if post.side == "left" else (255, 0, 255)
        cv.rectangle(
            annotated,
            (int(post.x1 * to_original), int(post.y1 * to_original)),
            (int(post.x2 * to_original), int(post.y2 * to_original)),
            color,
            1,
        )
    if detection.best is not None:
        cv.putText(annotated, f"{detection.best.kind} score {detection.best.score:.2f}",
                   (10, 52), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    masks = np.zeros((*detection.work_shape, 3), np.uint8)
    masks[:, :, 2] = detection.red_mask
    masks[:, :, 1] = detection.black_mask
    return [annotated, cv.resize(masks, size, interpolation=cv.INTER_NEAREST)]


def annotate(frame, estimate, label_prefix="classical"):
    canvas = frame.copy()
    if estimate.visible:
        height, width = frame.shape[:2]
        x1 = int(estimate.box_x * width)
        y1 = int(estimate.box_y * height)
        x2 = int((estimate.box_x + estimate.box_width) * width)
        y2 = int((estimate.box_y + estimate.box_height) * height)
        cv.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{label_prefix}: gate {estimate.confidence:.2f}"
        cv.putText(canvas, label, (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.65,
                   (255, 255, 255), 2, cv.LINE_AA)
    else:
        cv.putText(canvas, f"{label_prefix}: no gate", (10, 26),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
    return canvas
