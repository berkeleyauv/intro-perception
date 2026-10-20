from __future__ import annotations
 
import cv2 as cv
import numpy as np
 
from intro_perception.production import prepare_production_namespace
 
prepare_production_namespace()
 
from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.registry import register_perceiver
 
from intro_perception.types import GateEstimate


# Red also appears at the top of OpenCV's hue range (170-180), because hue is a
# circle: 179 and 0 are both "red". Your old range (0-10) only caught one side.
RED_WRAP_H_MIN = 170
 
# How different the red amount must be between the top half and the bottom half
# of a blob before we trust that it has the post pattern. 0.0 = no difference,
# 1.0 = one half completely red and the other half has no red at all.
MIN_PATTERN_CONTRAST = 0.3
 
# If the gate confidence is below this, we say "no gate" instead of guessing.
MIN_CONFIDENCE = 0.5
 
 
def blob_check(a, b):
    # (unchanged from your version)
    top_tol = 0.3
    min_height_ratio = 0.5
    min_overlap = 0.5
 
    # we're checking if 2 blobs look like the two posts of a gate
    # (only indexes 0-3 are used, so extra fields on the tuple don't matter)
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
 
    # this prevents a post and its reflection from being registered as a gate
    if min(ax + aw, bx + bw) > max(ax, bx):
        return False
    taller, shorter = max(ah, bh), min(ah, bh)
 
    # we want our posts to have similar height
    if shorter / taller < min_height_ratio:
        return False
 
    # the tops should be at about the same height
    if abs(ay - by) > top_tol * taller:
        return False
 
    # the vertical ranges should overlap
    overlap = min(ay + ah, by + bh) - max(ay, by)
    return overlap >= min_overlap * shorter
 
 

# NEW: the color-pattern check

def post_pattern(red_mask, box):
    """Decide whether a blob looks like a left post, a right post, or neither.
 
    red_mask: black/white image, white where a pixel is red.
    box:      (x, y, width, height) of the blob.
 
    Returns (side, score):
      side  = "left"  -> black on top, red on bottom
              "right" -> red on top, black on bottom
              None    -> no clear pattern (not a post)
      score = 0.0 to 1.0, how strong the pattern is
    """
    x, y, w, h = box
    crop = red_mask[y:y + h, x:x + w]  # only the pixels inside this blob's box
 
    mid = h // 2
    if mid < 1:
        return None, 0.0  # too short to split into two halves
 
    top_half = crop[:mid]
    bottom_half = crop[mid:]
 
    # fraction of each half that is red (0.0 = no red, 1.0 = all red)
    top_red = np.count_nonzero(top_half) / top_half.size
    bottom_red = np.count_nonzero(bottom_half) / bottom_half.size
 
    # positive -> more red on the bottom, negative -> more red on the top
    contrast = bottom_red - top_red
 
    if contrast >= MIN_PATTERN_CONTRAST:
        return "left", contrast        # black top, red bottom = LEFT post
    if contrast <= -MIN_PATTERN_CONTRAST:
        return "right", -contrast      # red top, black bottom = RIGHT post
    # A blob with NO red (like the dark blobs in your false-positive image)
    # has contrast near 0, so it ends up here and gets rejected.
    return None, 0.0
 
 
def best_pair(candidates, max_candidates=6):
    """Best valid pair among the biggest candidates, ordered left to right.
 
    Candidate tuple layout (NEW):  (x, y, w, h, side, pattern_score, area)
    `area` stays LAST so sorting by c[-1] still means "sort by area".
    """
    top = sorted(candidates, key=lambda c: c[-1], reverse=True)[:max_candidates]
    best, best_score = [], 0.0
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            # same geometric checks as before
            if not blob_check(top[i], top[j]):
                continue
 
            # put them in left-to-right order, then check the colors.
            # The left blob must look like a LEFT post, the right blob like a
            # RIGHT post. Two left-type posts, or two right-type posts, are
            # not a gate.
            left, right = sorted((top[i], top[j]), key=lambda c: c[0])
            if left[4] != "left" or right[4] != "right":
                continue
 
            # prefer the pair with the strongest color pattern
            # (before, it preferred the pair with the biggest area)
            score = left[5] + right[5]
            if score > best_score:
                best, best_score = [left, right], score
    return best
 
 
# confidence from "does this look like a gate", not "is it big"
def gate_confidence(left, right):
    """0.0 to 1.0. The weights below are guesses; check them against your data."""
    # 1) how strong the red/black pattern is on both posts (0 to 1)
    pattern = (left[5] + right[5]) / 2.0
 
    # 2) posts of a real gate are about the same height (1.0 = identical)
    taller = max(left[3], right[3])
    shorter = min(left[3], right[3])
    height_similarity = shorter / taller
 
    # 3) post tops should line up (1.0 = perfectly aligned)
    top_alignment = 1.0 - min(1.0, abs(left[1] - right[1]) / taller)
 
    return 0.6 * pattern + 0.2 * height_similarity + 0.2 * top_alignment
 

@register_perceiver(task="gate", algo="intro_classical")
class ClassicalGatePerceiver(TaskPerceiver):
    # Find two vertical posts with fixed HSV thresholds.
 
    def __init__(self):
        super().__init__(
            red_h_min=((0, 180), 0),
            red_h_max=((0, 180), 10),  # hue must be between 0 and 10
            red_s_min=((0, 255), 100),  # saturation must be at least 100 out of 255
            red_v_min=((0, 255), 100),  # value (brightness) must be at least 100 out of 255
            black_v_max=((0, 255), 110),  # black's brightness must be no more than 110 out of 255
        )
 
    def analyze(self, frame: np.ndarray, debug: bool, slider_vals=None):
 
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")
 
        if slider_vals is None:
            slider_vals = {}
 
        r_h_min = slider_vals.get("red_h_min", 0)
        r_h_max = slider_vals.get("red_h_max", 10)
        r_s_min = slider_vals.get("red_s_min", 100)
        r_v_min = slider_vals.get("red_v_min", 100)
        b_v_max = slider_vals.get("black_v_max", 110)
 
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 
        # NEW: red mask in TWO pieces, because red wraps around the hue circle.
        #   piece 1: low hues (your original range, ~0-10)
        #   piece 2: high hues (~170-180)
        # A pixel is "red" if it falls in either piece.
        red_low = cv.inRange(
            hsv,
            np.array((r_h_min, r_s_min, r_v_min)),
            np.array((r_h_max, 255, 255)),
        )
        red_high = cv.inRange(
            hsv,
            np.array((RED_WRAP_H_MIN, r_s_min, r_v_min)),
            np.array((180, 255, 255)),
        )
        red_mask = cv.bitwise_or(red_low, red_high)
 
        # Broad black mask (unchanged)
        black_mask = cv.inRange(
            hsv,
            np.array((0, 0, 0)),
            np.array((180, 255, b_v_max)),
        )
 
        # We still use the COMBINED mask to find blob shapes, because a real
        # post is black + red stacked, and combined they form one tall blob.
        # But red_mask stays separate so we can check the colors afterward.
        combined_mask = cv.bitwise_or(red_mask, black_mask)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
 
        contours = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
 
        candidates = []
        rejected = []  # NEW: tall-ish blobs that failed the color check (for debug)
        for contour in contours:
            area = float(cv.contourArea(contour))
            x, y, width, height = cv.boundingRect(contour)
            # Basic vertical aspect ratio check (unchanged)
            if area >= 30.0 and height >= max(10, int(width * 1.2)):
                # NEW: does this blob have the red/black post pattern?
                side, pattern_score = post_pattern(red_mask, (x, y, width, height))
                if side is None:
                    rejected.append((x, y, width, height))
                    continue  # no pattern -> not a post, don't even consider it
                candidates.append((x, y, width, height, side, pattern_score, area))
 
        selected = best_pair(candidates)
 
        # default answer is no gate unless 2 posts were selected
        estimate = GateEstimate.invisible()
 
        if len(selected) == 2:
            left_post, right_post = selected
 
            # NEW: confidence from how gate-like the pair is
            confidence = gate_confidence(left_post, right_post)
 
            # NEW: only accept the pair if we're confident enough.
            # Below the floor we keep the invisible estimate.
            if confidence >= MIN_CONFIDENCE:
                x1 = min(left_post[0], right_post[0])
                y1 = min(left_post[1], right_post[1])
                x2 = max(left_post[0] + left_post[2], right_post[0] + right_post[2])
                y2 = max(left_post[1] + left_post[3], right_post[1] + right_post[3])
 
                left_center = left_post[0] + left_post[2] / 2.0
                right_center = right_post[0] + right_post[2] / 2.0
 
                estimate = GateEstimate(
                    True,
                    float(confidence),
                    (left_center + right_center) / (2.0 * frame.shape[1]),
                    (y1 + y2) / (2.0 * frame.shape[0]),
                    x1 / frame.shape[1],
                    y1 / frame.shape[0],
                    (x2 - x1) / frame.shape[1],
                    (y2 - y1) / frame.shape[0],
                ).normalized()
 
        if not debug:
            return estimate
 
        annotated = annotate(frame, estimate)
        # green = posts that were paired
        for x, y, width, height, *_ in selected:
            cv.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
        # NEW: yellow = blobs thrown out by the color check
        for x, y, width, height in rejected:
            cv.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 255), 1)
 
        def to_bgr(mask):
            return cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
 
        # NEW: one panel per stage, so you can see where a gate gets lost
        return estimate, [
            annotated,
            to_bgr(red_mask),
            to_bgr(black_mask),
            to_bgr(combined_mask),
        ]
 
 
def annotate(frame, estimate, label_prefix="classical"):
    # (unchanged from your version)
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
 