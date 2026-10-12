"""Per-pixel paint-color model for the classical detector.

The gate's paint does not have a fixed color in these images: the water tints
everything, and the dataset's augmentation shifts hue and brightness. What is
stable is the paint's color *relative to the rest of the image*. Each pixel is
described by its Lab color minus the image's median Lab color, quantized into a
3-D histogram, and a look-up table holds, per bin, the log-likelihood of
"black paint", "red paint" and "background". This is classic histogram
back-projection, fitted from the labeled training split by
``scripts/fit_color_model.py``; no neural network is involved.
"""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

CLASSES = ("background", "black", "red")
BINS = 16
# Ranges of (L, a, b) minus the image median, in OpenCV's 8-bit Lab units.
RANGES = ((-200.0, 80.0), (-40.0, 60.0), (-50.0, 50.0))
TYPICAL_L_IQR = 74.0  # lightness spread of an ordinary frame (8-bit Lab units)
GAIN_MIN, GAIN_MAX = 1.0, 1.0
LOCAL_WINDOW = 41  # 0 = compare with the whole image only
MODEL_PATH = Path(__file__).with_name("gate_colors.npz")


def padding_mask(frame):
    """Flat gray borders added by rotation augmentation.

    The fill is a single neutral gray (any brightness, after the dataset's
    brightness augmentation) touching the image edge; real water and pool
    pixels are colored and textured.
    """
    blue, green, red = cv.split(frame)
    spread = cv.subtract(cv.max(cv.max(blue, green), red), cv.min(cv.min(blue, green), red))
    neutral = (spread <= 8).astype(np.uint8)
    count, labels, stats, _ = cv.connectedComponentsWithStats(neutral, connectivity=4)
    height, width = neutral.shape
    gray = green
    padding = np.zeros(count, bool)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 50 or not (x == 0 or y == 0 or x + w == width or y + h == height):
            continue
        values = gray[y:y + h, x:x + w][labels[y:y + h, x:x + w] == index]
        padding[index] = values.std() < 6.0
    mask = padding[labels]
    if mask.any():  # also drop the blended seam between padding and image
        mask = cv.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return mask


def relative_lab(frame, padding=None):
    """Lab of ``frame`` relative to the image: ``(L - median) * contrast`` etc.

    Subtracting the median color cancels the water tint and hue shifts;
    dividing lightness by its spread (interquartile range, relative to a
    typical image) cancels brightness/contrast augmentation, so a washed-out
    frame's faint posts look like ordinary ones.
    """
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB).astype(np.float32)
    if padding is None:
        padding = padding_mask(frame)
    # Statistics from every 4th pixel in each direction: same answer, 16x cheaper.
    sample_lab, sample_padding = lab[::4, ::4], padding[::4, ::4]
    valid = sample_lab[~sample_padding] if (~sample_padding).any() else sample_lab.reshape(-1, 3)
    median = np.median(valid, axis=0)
    q1, q3 = np.percentile(valid[:, 0], (25, 75))
    gain = np.clip(TYPICAL_L_IQR / max(q3 - q1, 4.0), GAIN_MIN, GAIN_MAX)
    if LOCAL_WINDOW:
        # Compare with the surroundings (padding replaced by the median so it
        # does not darken the local average near the borders).
        filled = lab.copy()
        filled[padding] = median
        local = cv.blur(filled, (LOCAL_WINDOW, LOCAL_WINDOW))
        relative = lab - (local + median) / 2.0
    else:
        relative = lab - median
    relative[..., 0] *= gain
    return relative


def bin_indices(relative):
    """Flat histogram bin index of every pixel."""
    index = np.zeros(relative.shape[:2], np.int32)
    for channel, (low, high) in enumerate(RANGES):
        scaled = (relative[..., channel] - low) * (BINS / (high - low))
        index = index * BINS + np.clip(scaled, 0, BINS - 1).astype(np.int32)
    return index


class ColorModel:
    def __init__(self, log_likelihood):
        # (n_classes, BINS**3) log p(bin | class)
        self.log_likelihood = np.asarray(log_likelihood, np.float32)

    @classmethod
    def load(cls, path=MODEL_PATH):
        with np.load(path) as data:
            return cls(data["log_likelihood"])

    def save(self, path=MODEL_PATH):
        np.savez_compressed(path, log_likelihood=self.log_likelihood.astype(np.float16))

    @classmethod
    def fit(cls, samples, smoothing=1.0):
        """``samples`` maps class name -> 1-D array of bin indices."""
        rows = []
        for name in CLASSES:
            counts = np.bincount(samples[name], minlength=BINS ** 3).astype(np.float64)
            counts += smoothing
            rows.append(np.log(counts / counts.sum()))
        return cls(np.array(rows))

    def paint_scores(self, frame, padding=None):
        """Log-odds maps ``(black, red)`` against background, padding set very low."""
        if padding is None:
            padding = padding_mask(frame)
        bins = bin_indices(relative_lab(frame, padding))
        background = self.log_likelihood[0][bins]
        black = self.log_likelihood[1][bins] - background
        red = self.log_likelihood[2][bins] - background
        black[padding] = -10.0
        red[padding] = -10.0
        return black, red
