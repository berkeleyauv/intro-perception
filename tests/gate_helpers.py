"""Synthetic gates that follow the real gate spec, shared by the tests.

Gate spec: the left post is black on top and red on the bottom; the right post
is red on top and black on the bottom. Colors are the median paint and water
colors measured on the supplied pool images (BGR), so the classical detector's
fitted color model sees them as it sees real frames.
"""

import pathlib

import cv2 as cv
import numpy as np

HEIGHT, WIDTH = 180, 320
BLACK = (32, 17, 3)      # black paint under pool water
RED = (109, 84, 84)      # red paint under pool water (muted mauve)
WATER = (206, 170, 59)   # median pool background
GRAY = WATER             # old name: the plain background of a no-gate frame

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
 
 
def gate_label(shift=0):
    """YOLO label line for the whole gate drawn by ``make_gate_frame``."""
    x1, x2 = LEFT_X + shift, RIGHT_X + shift + POST_WIDTH
    center_x, center_y = (x1 + x2) / 2 / WIDTH, (TOP + BOTTOM) / 2 / HEIGHT
    width, height = (x2 - x1) / WIDTH, (BOTTOM - TOP) / HEIGHT
    return f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n"
 
 
def write_labeled_image(directory, name, *, shift=0, gate=True, ext=".png", labels_dir=None):
    """Write ``<name><ext>`` plus its YOLO ``.txt`` label; ``gate=False`` makes a no-gate image.

    The label goes next to the image unless ``labels_dir`` is given.
 
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
    label_dir = pathlib.Path(labels_dir) if labels_dir else directory
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"{name}.txt").write_text(gate_label(shift) if gate else "")
    return image_path

# --- pool-like synthetic scenes ---------------------------------------------------
#
# Not a substitute for real images: each option reproduces one thing seen in
# the supplied frames (support pipes above the crossbar, diagonal lane lines,
# washed-out or dark frames, rotation padding, gates cut off by the frame), so
# each behavior can be pinned down by a regression test without committing
# the dataset.

SCENE_HEIGHT, SCENE_WIDTH = 360, 640


def make_scene(
    *,
    gate_x=170,
    gate_width=300,
    top=110,
    post_height=200,
    post_width=12,
    right_height=None,
    left=True,
    right=True,
    red=RED,
    black=BLACK,
    water=WATER,
    crossbar=True,
    supports=True,
    lane_lines=True,
    contrast=1.0,
    brightness=0.0,
    rotate=0.0,
    blur=3,
    noise=4.0,
    jpeg=None,
    distractors=(),
    seed=0,
):
    """Draw a pool-like scene and return ``(frame, truth_box)``.

    ``truth_box`` is the labeled gate box ``(x1, y1, x2, y2)`` in pixels: from
    the crossbar (post tops) to the post bottoms and from post to post, clipped
    to the frame (a gate cut off by the frame is boxed to the edge), or
    ``None`` when no post is drawn. ``supports`` adds the pipes above the
    crossbar, painted like the *other* post as on the real gate. ``rotate``
    (degrees) adds the flat gray borders of rotation augmentation; the truth box
    is then only approximate, so keep it small in IoU tests.
    """
    rng = np.random.default_rng(seed)
    frame = np.empty((SCENE_HEIGHT, SCENE_WIDTH, 3), np.float32)
    # Darker deep water above, bright floor below, like the pool frames.
    rows = np.linspace(0.75, 1.1, SCENE_HEIGHT)[:, None, None]
    frame[:] = np.array(water, np.float32)[None, None, :] * rows
    if lane_lines:
        for start in range(-200, SCENE_WIDTH + 200, 160):
            cv.line(frame, (start, SCENE_HEIGHT), (start + 120, int(SCENE_HEIGHT * 0.55)),
                    (120, 90, 30), 6)
    for x1, y1, x2, y2, color in distractors:
        frame[max(y1, 0):y2, max(x1, 0):x2] = color

    right_height = post_height if right_height is None else right_height
    left_x, right_x = gate_x, gate_x + gate_width - post_width
    drawn = []

    def post(x, height, top_color, bottom_color):
        middle = top + height // 2
        frame[max(top, 0):max(middle, 0), max(x, 0):max(x + post_width, 0)] = top_color
        frame[max(middle, 0):max(top + height, 0), max(x, 0):max(x + post_width, 0)] = bottom_color
        if supports:  # support pipe above the crossbar, painted like the other post
            frame[0:max(top - height // 3, 0), max(x, 0):max(x + post_width, 0)] = bottom_color
            frame[max(top - height // 3, 0):max(top, 0), max(x, 0):max(x + post_width, 0)] = top_color
        drawn.append((x, top, x + post_width, top + height))

    if crossbar and (left or right):
        y = max(top - 3, 0)
        frame[y:top + 3, max(left_x - 30, 0):max(right_x + post_width + 30, 0)] = black
    if left:
        post(left_x, post_height, black, red)
    if right:
        post(right_x, right_height, red, black)

    frame = frame * contrast + brightness + (1 - contrast) * 128
    if noise:
        frame += rng.normal(0.0, noise, frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    if blur:
        frame = cv.GaussianBlur(frame, (blur | 1, blur | 1), 0)
    if rotate:
        matrix = cv.getRotationMatrix2D((SCENE_WIDTH / 2, SCENE_HEIGHT / 2), rotate, 1.0)
        frame = cv.warpAffine(frame, matrix, (SCENE_WIDTH, SCENE_HEIGHT),
                              borderMode=cv.BORDER_CONSTANT, borderValue=(56, 56, 56))
    if jpeg:
        ok, encoded = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, jpeg])
        assert ok
        frame = cv.imdecode(encoded, cv.IMREAD_COLOR)

    if not drawn:
        return frame, None
    x1 = max(0, left_x if left else 0)
    x2 = min(SCENE_WIDTH, right_x + post_width if right else SCENE_WIDTH)
    y1 = max(0, min(b[1] for b in drawn))
    y2 = min(SCENE_HEIGHT, max(b[3] for b in drawn))
    if x2 <= x1 or y2 <= y1:
        return frame, None
    return frame, (x1, y1, x2, y2)


def box_iou_pixels(estimate, truth_box, shape):
    """IoU between a GateEstimate and a pixel ``(x1, y1, x2, y2)`` box."""
    height, width = shape[:2]
    ax1, ay1 = estimate.box_x * width, estimate.box_y * height
    ax2, ay2 = ax1 + estimate.box_width * width, ay1 + estimate.box_height * height
    bx1, by1, bx2, by2 = truth_box
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0
