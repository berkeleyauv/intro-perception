"""Synthetic gates that follow the real gate spec, shared by the tests.
 
Gate spec: the left post is black on top and red on the bottom; the right post
is red on top and black on the bottom. The background is mid-gray so only the
painted black halves (not the whole frame) fall under the black threshold.
"""
 
import cv2 as cv
import numpy as np
 
HEIGHT, WIDTH = 180, 320
BLACK = (0, 0, 0)
RED = (0, 0, 255)  # OpenCV uses BGR
GRAY = (128, 128, 128)
 
POST_WIDTH = 13
TOP, MIDDLE, BOTTOM = 35, 93, 151  # post spans rows [TOP, BOTTOM); halves split at MIDDLE
LEFT_X, RIGHT_X = 80, 225
 
 
def draw_post(frame, x, top_color, bottom_color):
    cv.rectangle(frame, (x, TOP), (x + POST_WIDTH - 1, MIDDLE - 1), top_color, -1)
    cv.rectangle(frame, (x, MIDDLE), (x + POST_WIDTH - 1, BOTTOM - 1), bottom_color, -1)
 
 
def make_gate_frame(shift=0, left=True, right=True):
    """A BGR frame holding a spec gate; pass ``left``/``right`` False to drop a post."""
    frame = np.full((HEIGHT, WIDTH, 3), GRAY, dtype=np.uint8)
    if left:
        draw_post(frame, LEFT_X + shift, BLACK, RED)
    if right:
        draw_post(frame, RIGHT_X + shift, RED, BLACK)
    return frame
 
 
def gate_label(class_id=1, shift=0):
    """YOLO label line for the whole gate drawn by ``make_gate_frame``."""
    x1, x2 = LEFT_X + shift, RIGHT_X + shift + POST_WIDTH
    center_x, center_y = (x1 + x2) / 2 / WIDTH, (TOP + BOTTOM) / 2 / HEIGHT
    width, height = (x2 - x1) / WIDTH, (BOTTOM - TOP) / HEIGHT
    return f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n"
 
 
def write_labeled_image(directory, name, *, shift=0, class_id=1, gate=True, ext=".png"):
    """Write ``<name><ext>`` plus its YOLO ``.txt`` label; ``gate=False`` makes a no-gate image.
 
    PNG is the default so colors survive exactly (JPEG blurs red toward the
    hue wrap-around that the baseline deliberately misses).
    """
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / f"{name}{ext}"
    if gate:
        frame = make_gate_frame(shift=shift)
    else:
        frame = np.full((HEIGHT, WIDTH, 3), GRAY, dtype=np.uint8)
    assert cv.imwrite(str(image_path), frame)
    image_path.with_suffix(".txt").write_text(gate_label(class_id, shift) if gate else "")
    return image_path
