"""Fit the classical detector's paint-color model from the training split.

    python scripts/fit_color_model.py --split data/generated/train

Only *training* images are used. For every whole (not cut-off) gate, the posts
lie along the left and right edges of the label box: the left post is black in
the top half and red in the bottom half, the right post the other way round.
In each row near a box edge the pixel most unlike the image's median color is
taken as the post, which gives "black" and "red" samples without any manual
pixel labeling. Pixels away from the posts are "background". The fitted look-up
table is written to ``intro_perception/gate_colors.npz``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2 as cv
import numpy as np

from intro_perception.color_model import (
    BINS, ColorModel, bin_indices, padding_mask, relative_lab,
)
from intro_perception.data_tools import parse_label
from intro_perception.images import list_images, read_image


def post_pixels(relative, rows, x_low, x_high, half_width):
    """Pixel positions of a post running through ``rows`` between ``x_low``/``x_high``.

    In each row the pixel most unlike the image median is a post candidate; a
    robust line fit through those candidates follows the (possibly tilted)
    post and ignores rows where a lane line or panel won instead.
    """
    height, width = relative.shape[:2]
    x_low, x_high = max(0, x_low), min(width, x_high)
    if x_high - x_low < 3 or len(rows) < 10:
        return [], []
    distance = np.linalg.norm(relative[rows, x_low:x_high], axis=2)
    centers = (x_low + distance.argmax(axis=1)).astype(np.float32)
    points = np.column_stack((centers, rows.astype(np.float32)))
    vx, vy, x0, y0 = cv.fitLine(points, cv.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if abs(vy) < 0.85:  # not a near-vertical post
        return [], []
    line_x = x0 + (rows - y0) * vx / vy
    ys, xs = [], []
    for y, center, on_line in zip(rows, centers, line_x):
        if abs(center - on_line) > half_width + 2:  # this row's pick is not the post
            continue
        for x in range(int(round(on_line)) - half_width, int(round(on_line)) + half_width + 1):
            if 0 <= x < width:
                ys.append(int(y))
                xs.append(x)
    return ys, xs


def collect(split_dir, rng, background_per_image=1500):
    split_dir = Path(split_dir)
    manifest = split_dir.parent / "manifest.csv"
    truncated = set()
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            truncated = {Path(r["filename"]).stem for r in csv.DictReader(stream)
                         if r["truncated"].lower() == "true"}
    samples = {"background": [], "black": [], "red": []}
    used = 0
    for image_path in list_images(split_dir / "images"):
        label = parse_label(split_dir / "labels" / f"{image_path.stem}.txt")
        frame = read_image(image_path)
        padding = padding_mask(frame)
        relative = relative_lab(frame, padding)
        bins = bin_indices(relative)
        height, width = frame.shape[:2]
        keep_out = padding.copy()
        if label is not None and image_path.stem not in truncated:
            cx, cy, w, h = label
            x1, x2 = int((cx - w / 2) * width), int((cx + w / 2) * width)
            y1, y2 = int((cy - h / 2) * height), int((cy + h / 2) * height)
            reach = int(0.12 * (x2 - x1)) + 12
            top = np.arange(y1 + int(0.08 * (y2 - y1)), y1 + int(0.42 * (y2 - y1)))
            bottom = np.arange(y1 + int(0.58 * (y2 - y1)), y1 + int(0.92 * (y2 - y1)))
            top, bottom = top[(top >= 0) & (top < height)], bottom[(bottom >= 0) & (bottom < height)]
            for rows, x_low, x_high, name in (
                (top, x1 - 6, x1 + reach, "black"),
                (bottom, x1 - 6, x1 + reach, "red"),
                (top, x2 - reach, x2 + 6, "red"),
                (bottom, x2 - reach, x2 + 6, "black"),
            ):
                ys, xs = post_pixels(relative, rows, x_low, x_high, 3)
                samples[name].append(bins[ys, xs])
            keep_out[max(0, y1 - 20):y2 + 20, max(0, x1 - 30):x1 + reach + 30] = True
            keep_out[max(0, y1 - 20):y2 + 20, max(0, x2 - reach - 30):x2 + 30] = True
            used += 1
        candidates = np.flatnonzero(~keep_out.ravel())
        if candidates.size:
            pick = rng.choice(candidates, size=min(background_per_image, candidates.size), replace=False)
            samples["background"].append(bins.ravel()[pick])
    return {k: np.concatenate(v) for k, v in samples.items()}, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/generated/train"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    samples, used = collect(args.split, np.random.default_rng(0))
    print(f"{used} whole gates; samples: " + ", ".join(f"{k}={v.size}" for k, v in samples.items()))
    model = ColorModel.fit(samples)
    if args.output:
        model.save(args.output)
        print(f"wrote {args.output}")
    else:
        model.save()
        print("wrote intro_perception/gate_colors.npz")
    print(f"look-up table: {BINS}^3 bins x 3 classes")


if __name__ == "__main__":
    main()
