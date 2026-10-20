"""Intentionally weak classical gate detector baseline.

Gate appearance (as seen facing the gate):
 
* left post:  black on top, red on the bottom
* right post: red on top, black on the bottom
"""

from __future__ import annotations

import cv2 as cv
import numpy as np

from intro_perception.production import prepare_production_namespace

prepare_production_namespace()

from perception.tasks.TaskPerceiver import TaskPerceiver
from perception.tasks.registry import register_perceiver

from intro_perception.types import GateEstimate

def blob_check(a, b):
    
    top_tol=0.3
    min_height_ratio = 0.5
    min_overlap=0.5

    # we're checking if 2 blobs look like the two posts of a gate
    ax = a[0]   # x; the left edge
    ay = a[1]   # y; the top edge
    aw = a[2]   # width
    ah = a[3]   # height
    ax_bottom = ay + ah

    bx = b[0]
    by = b[1]
    bw = b[2]
    bh = b[3]
    bx_bottom = by + bh
    
    # this prevents a post and its reflection from being registered as a gate
    # by checking what horizontal area is being overlapped

    
    if min(ax + aw, bx + bw) > max(ax, bx):
        return False
    
    taller = max(ah, bh)
    shorter = min(ah, bh)
    
    
    # we want our posts to have similar height
    # sometimes the algo registers two "posts" but they appear in totally different locations in the picture
    # because they aren't actually a part of the gate
    if shorter / taller < min_height_ratio:
        return False
    
    # checking for the same issue here
    # sometimes 2 "posts" are found but they don't exist at the same height
    # therefore not a gate..
    if abs(ay - by) > top_tol * taller:
        return False

    # checking if the vertical ranges overlap
    overlap = min(ay + ah, by + bh) - max(ay, by)
    return overlap >= min_overlap * shorter


MIN_PATTERN_CONTRAST = 0.3   # red amount must differ this much between top and bottom half
MIN_RED_IN_RED_HALF = 0.3    # the "red half" must be at least half red (rejects stray pixels)
MIN_CONFIDENCE = 0.2         # below this, return invisible. Starting guess; tune on your data.


def post_pattern(red_mask, box):
    """Return ("left" | "right" | None, score 0-1) from the red/black layout of a blob."""
    x, y, w, h = box
    crop = red_mask[y:y + h, x:x + w]
    mid = h // 2
    if mid < 1:
        return None, 0.0
    top, bottom = crop[:mid], crop[mid:]
    top_red = np.count_nonzero(top) / top.size        # fraction of top half that is red
    bottom_red = np.count_nonzero(bottom) / bottom.size

    # the red half has to be mostly red, not just a few stray pixels
    if max(top_red, bottom_red) < MIN_RED_IN_RED_HALF:
        return None, 0.0

    contrast = bottom_red - top_red
    if contrast >= MIN_PATTERN_CONTRAST:
        return "left", contrast       # black top, red bottom
    if contrast <= -MIN_PATTERN_CONTRAST:
        return "right", -contrast     # red top, black bottom
    return None, 0.0


def gate_confidence(left, right):
    """Color score scaled by geometry. Geometry can lower it, never replace it."""
    pattern = (left[5] + right[5]) / 2.0
    taller, shorter = max(left[3], right[3]), min(left[3], right[3])
    height_similarity = shorter / taller
    top_alignment = 1.0 - min(1.0, abs(left[1] - right[1]) / taller)
    geometry = (height_similarity + top_alignment) / 2.0
    return pattern * (0.5 + 0.5 * geometry)

def best_pair(candidates, max_candidates=6):
    """Best valid pair: left blob must be a left post, right blob a right post."""
    top = sorted(candidates, key=lambda c: c[-1], reverse=True)[:max_candidates]
    best, best_score = [], 0.0
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if not blob_check(top[i], top[j]):
                continue
            left, right = sorted((top[i], top[j]), key=lambda c: c[0])
            if left[4] != "left" or right[4] != "right":
                continue
            score = left[5] + right[5]
            if score > best_score:
                best, best_score = [left, right], score
    return best

@register_perceiver(task="gate", algo="intro_classical")
class ClassicalGatePerceiver(TaskPerceiver):
    # Find two vertical posts with fixed HSV thresholds.

    # ADDED: Default Slider Values
    def __init__(self):
        # so we're finding posts using color alone, but we're not taking lighting or brightness into consideration
        # Issues I am noticing:
        # - red looks different underwater, but the algo is only taking into consideration
        #    how red looks normally
        # - only checks brightness of black, doesn't take into consideration hues and saturation
        # - must be why the algo is detecting shadows, reflections, or other objects as gates
        super().__init__(
            red_h_min=((0, 180), 0),
            red_h_max=((0, 180), 25), # hue must be between 0 and 10 #changed to 25
            red_s_min=((0, 255), 35), # saturation must be at least 100 out of 225
            red_v_min=((0, 255), 35), # value (how bright) must be at least 100 out of 225
            black_v_max=((0, 255), 110), # OG: black's brightness must be no more than 50 out of 225
                                            # I am changing it to 110 out of 225

        )

    
    # Understanding this:
    #    - frame is the photo, stored in an array as a grid of pixels
    #    - slider_vals - threshold settings, optional since the None
    
    def analyze(self, frame: np.ndarray, debug: bool, slider_vals=None):

        # if the picture doesn't exist, or it isn't loading properly, raises value error
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")


        # if no slider values were passed in, use 0
        if slider_vals is None:
            slider_vals = {}

        # Slider controls for HSV tuning
        r_h_min = slider_vals.get("red_h_min", 0)
        r_h_max = slider_vals.get("red_h_max", 25)
        r_s_min = slider_vals.get("red_s_min", 35)
        r_v_min = slider_vals.get("red_v_min", 35)
        # fix the ranges on here because they're still not capturing red correctly
        # need to take size into consideration too
        #
        
        b_v_max = slider_vals.get("black_v_max", 110) # initially 50, I am changing to 110
        
        # OpenCV --> BGR 
        # Now converting to HSV: hue (which color), saturation (how vivid), and value (how bright). 
        # HSV is easier for picking out “red” because the color is in one number.
        
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # if you're inside all 3 ranges you become white, outside you're black
        # so now we get a black and white picture called a mask
        #red_mask = cv.inRange(
        #    hsv, 
        #    np.array((r_h_min, r_s_min, r_v_min)), 
        #    np.array((r_h_max, 255, 255))
        #)
        
        red_low = cv.inRange(hsv, np.array((r_h_min, r_s_min, r_v_min)), np.array((r_h_max, 255, 255)))
        red_high = cv.inRange(hsv, np.array((120, r_s_min, r_v_min)), np.array((180, 255, 255)))
        red_mask = cv.bitwise_or(red_low, red_high)
        
        # Broad black mask
        black_mask = cv.inRange(
            hsv, 
            np.array((0, 0, 0)), 
            np.array((180, 255, b_v_max))
        )

        combined_mask = cv.bitwise_or(red_mask, black_mask)
        k = max(15, frame.shape[0] // 25)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, np.ones((k, 3), np.uint8))
        
        contours = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

        #candidates = []
        #for contour in contours:
           #area = float(cv.contourArea(contour))
            #x, y, width, height = cv.boundingRect(contour)
            # Basic vertical aspect ratio check
            #if area >= 30.0 and height >= max(10, int(width * 1.2)):
                #candidates.append((x, y, width, height, area))
        
        

        candidates = []
        rejected = []   # blobs that failed the color check (shown in yellow in debug)
        for contour in contours:
            area = float(cv.contourArea(contour))
            x, y, width, height = cv.boundingRect(contour)
            if area >= 30.0 and height >= max(10, int(width * 1.2)):
                side, score = post_pattern(red_mask, (x, y, width, height))
                if side is None:
                    rejected.append((x, y, width, height))
                    continue
                candidates.append((x, y, width, height, side, score, area))
        
        # Sort by area and pick top 2
        candidates.sort(key=lambda c: c[-1], reverse=True)
        selected = best_pair(candidates)

        # default answer is no gate unless 2 posts were selected
        # IMPORTANT: if only 1 post is selected the algo still concludes no gate.
        estimate = GateEstimate.invisible()
        
        confidence = gate_confidence(*selected) if len(selected) == 2 else 0.0
        if len(selected) == 2 and confidence >= MIN_CONFIDENCE:
            left_post, right_post = selected
            
            x1 = min(left_post[0], right_post[0])
            y1 = min(left_post[1], right_post[1])
            x2 = max(left_post[0] + left_post[2], right_post[0] + right_post[2])
            y2 = max(left_post[1] + left_post[3], right_post[1] + right_post[3])
 
            left_center = left_post[0] + left_post[2] / 2.0
            right_center = right_post[0] + right_post[2] / 2.0

            # This is the confidence score..
            #separation = abs(right_center - left_center) / frame.shape[1]
            #area_fraction = (left_post[-1] + right_post[-1]) / (frame.shape[0] * frame.shape[1])
            #confidence = min(1.0, separation * 1.5 + area_fraction * 8.0)
            
            estimate = GateEstimate(
                True,
                confidence,
                (left_center + right_center) / (2.0 * frame.shape[1]),
                (y1 + y2) / (2.0 * frame.shape[0]),
                x1 / frame.shape[1],
                y1 / frame.shape[0],
                (x2 - x1) / frame.shape[1],
                (y2 - y1) / frame.shape[0],
            ).normalized()

        #print("conf:", round(confidence, 2), "pairs:", [(s[4], round(s[5], 2)) for s in selected])
        
        if not debug:
            return estimate
            
        #annotated = annotate(frame, estimate)
        #for x, y, width, height, _ in selected:
           # cv.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
        #return estimate, [annotated, cv.cvtColor(combined_mask, cv.COLOR_GRAY2BGR)]

        annotated = annotate(frame, estimate)
        for x, y, width, height, *_ in selected:          # note *_ now, not _
            cv.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
        for x, y, width, height in rejected:              # yellow = failed color check
            cv.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 255), 1)

        def to_bgr(m):
            return cv.cvtColor(m, cv.COLOR_GRAY2BGR)

        return estimate, [annotated, to_bgr(red_mask), to_bgr(black_mask), to_bgr(combined_mask)]

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
