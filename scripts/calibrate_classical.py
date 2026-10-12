"""Fit the classical detector's confidence calibration on a labeled split.

    python scripts/calibrate_classical.py --split data/generated/val

For every image the detector's raw geometric score is paired with whether its
box is correct (IoU >= 0.5 with the label). A logistic curve
``confidence = 1 / (1 + exp(-a * (score - b)))`` is fitted to those outcomes
and written to ``intro_perception/classical_calibration.json`` (use ``--output``
to write elsewhere). Fit on the validation split, never on test, so the test
metrics stay honest. The script also reports precision/recall at a few
``MIN_SCORE`` values so the cut-off can be chosen from data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from intro_perception import classical
from intro_perception.data_tools import parse_label
from intro_perception.evaluate import box_iou
from intro_perception.images import list_images, read_image


def outcomes(split_dir):
    """``[(score, correct, has_gate)]`` for every image in ``split_dir``."""
    perceiver = classical.ClassicalGatePerceiver()
    rows = []
    for image_path in list_images(Path(split_dir) / "images"):
        label = parse_label(Path(split_dir) / "labels" / f"{image_path.stem}.txt")
        detection = classical.detect(read_image(image_path))
        score = detection.best.score if detection.best is not None else 0.0
        correct = False
        if label is not None and detection.best is not None:
            estimate = perceiver.estimate(detection) if score >= classical.MIN_SCORE else None
            if estimate is None:  # score the box even below MIN_SCORE
                work_h, work_w = detection.work_shape
                x1, y1, x2, y2 = detection.best.box
                guess = {"box_x": x1 / work_w, "box_y": y1 / work_h,
                         "box_width": (x2 - x1) / work_w, "box_height": (y2 - y1) / work_h}
            else:
                guess = estimate.to_dict()
            cx, cy, w, h = label
            truth = {"box_x": cx - w / 2, "box_y": cy - h / 2, "box_width": w, "box_height": h}
            correct = box_iou(guess, truth) >= 0.5
        rows.append((score, correct, label is not None))
    return rows


def fit_logistic(scores, labels, steps=5000, rate=0.5):
    """Plain gradient descent on log loss; returns ``(a, b)``."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    w, c = 5.0, -2.5  # logit = w * x + c
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(w * x + c)))
        w -= rate * float(np.mean((p - y) * x))
        c -= rate * float(np.mean(p - y))
    a = w
    b = -c / w if w else 0.5
    return a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/generated/val"))
    parser.add_argument(
        "--output", type=Path,
        default=Path(classical.__file__).with_name("classical_calibration.json"),
    )
    args = parser.parse_args()

    rows = outcomes(args.split)
    detected = [(s, ok) for s, ok, _gate in rows if s > 0]
    positives = sum(gate for _s, _ok, gate in rows)
    print(f"{len(rows)} images, {positives} with a gate, {len(detected)} with a candidate")
    for threshold in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        kept = [ok for s, ok in detected if s >= threshold]
        tp = sum(kept)
        precision = tp / len(kept) if kept else 0.0
        recall = tp / positives if positives else 0.0
        print(f"MIN_SCORE {threshold:.2f}: precision {precision:.2f} recall {recall:.2f}")
    if len({ok for _s, ok in detected}) < 2:
        raise SystemExit("need both correct and wrong candidates to fit a calibration")
    a, b = fit_logistic([s for s, _ in detected], [ok for _, ok in detected])
    args.output.write_text(json.dumps({"a": round(a, 4), "b": round(b, 4),
                                       "fitted_on": str(args.split),
                                       "samples": len(detected)}, indent=2) + "\n")
    print(f"wrote a={a:.3f} b={b:.3f} to {args.output}")


if __name__ == "__main__":
    main()
