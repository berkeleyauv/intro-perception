"""Score gate JSONL predictions against frame-indexed ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

from intro_perception.data_tools import parse_label
from intro_perception.images import list_images, read_image

ORIENTATIONS = ("left", "head_on", "right")

def load_truth(test_dir):
    """Ground truth for every image in ``<test_dir>/images``, keyed by file stem."""
    test_dir = Path(test_dir)
    truth = {}
    for image_path in list_images(test_dir / "images"):
        label_path = test_dir / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"missing label for {image_path.name}: expected {label_path}")
        parsed = parse_label(label_path)
        if parsed is None:
            truth[image_path.stem] = {"visible": False}
            continue
        class_id, center_x, center_y, width, height = parsed
        truth[image_path.stem] = {
            "visible": True,
            "orientation": ORIENTATIONS[class_id],
            "center_x": center_x,
            "center_y": center_y,
            "box_x": center_x - width / 2,
            "box_y": center_y - height / 2,
            "box_width": width,
            "box_height": height,
        }
    return truth

def warn_about_truth(truth, stream=None):
    """Say so when the test set cannot support some metrics, instead of printing quiet zeros."""
    stream = stream or sys.stderr
    counts = {name: 0 for name in ORIENTATIONS}
    for expected in truth.values():
        if expected.get("visible"):
            counts[expected["orientation"]] += 1
    if not any(counts.values()):
        print("warning: the test set has no gate images, so precision/recall/mAP "
              "are meaningless (0.0)", file=stream)
        return
    for name, count in counts.items():
        if count == 0:
            print(f"warning: the test set has no '{name}' gates; "
                  "that orientation's metrics are not meaningful", file=stream)

def load_records(path):
    """Read a predictions JSONL written by ``intro-perception-run``, keyed by frame_id."""
    records = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "frame_id" not in record:
                raise ValueError(f"{path}:{line_number}: missing frame_id")
            records[record["frame_id"]] = record
    return records

def predict_test_images(test_dir, perceivers):
    """Run each perceiver on every test image, timing each call.
 
    Returns ``{method_name: {frame_id: record}}``; every record includes
    ``inference_sec`` so the FPS metric is populated.
    """
    predictions = {name: {} for name in perceivers}
    for image_path in list_images(Path(test_dir) / "images"):
        frame = read_image(image_path)
        for name, perceiver in perceivers.items():
            started = time.perf_counter()
            estimate = perceiver.analyze(frame, debug=False)
            elapsed = time.perf_counter() - started
            predictions[name][image_path.stem] = {
                "inference_sec": elapsed,
                **estimate.to_dict(),
            }
    return predictions

def evaluate_test_set(test_dir, method="both", model_path="artifacts/yolo/best.pt"):
    """Run the chosen detector(s) on ``test_dir`` and return ``{name: scores}``."""
    from intro_perception.run import create_methods  # imported lazily: it pulls in YOLO
 
    truth = load_truth(test_dir)
    warn_about_truth(truth)
    predictions = predict_test_images(test_dir, create_methods(method, model_path))
    return {name: score_records(truth, records) for name, records in predictions.items()}

def box_iou(first, second):
    ax1, ay1 = float(first["box_x"]), float(first["box_y"])
    ax2 = ax1 + float(first["box_width"])
    ay2 = ay1 + float(first["box_height"])
    bx1, by1 = float(second["box_x"]), float(second["box_y"])
    bx2 = bx1 + float(second["box_width"])
    by2 = by1 + float(second["box_height"])
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0

def average_precision(events, positives):
    if positives == 0:
        return None
    events.sort(key=lambda event: event[0], reverse=True)
    tp = fp = 0
    points = []
    for _, is_true in events:
        tp += int(is_true)
        fp += int(not is_true)
        points.append((tp / positives, tp / (tp + fp)))
    return sum(max((precision for recall, precision in points if recall >= threshold), default=0.0)
               for threshold in (index / 100 for index in range(101))) / 101

def score_records(truth, predictions):
    true_positive = false_positive = false_negative = 0
    center_errors = []
    confusion = {expected: {actual: 0 for actual in ORIENTATIONS} for expected in ORIENTATIONS}
    orientation_misses = {orientation: 0 for orientation in ORIENTATIONS}
    ap_events = {orientation: [] for orientation in ORIENTATIONS}
    positive_counts = {orientation: 0 for orientation in ORIENTATIONS}
    inference_times = []
 
    for frame_id, expected in truth.items():
        actual = predictions.get(frame_id, {"visible": False, "confidence": 0.0})
        expected_visible = bool(expected.get("visible", False))
        actual_visible = bool(actual.get("visible", False))
        expected_orientation = expected.get("orientation")
        if expected_visible:
            positive_counts[expected_orientation] += 1
        matched = expected_visible and actual_visible and box_iou(expected, actual) >= 0.5
        if matched:
            true_positive += 1
            center_errors.append(math.hypot(
                float(actual["center_x"]) - float(expected["center_x"]),
                float(actual["center_y"]) - float(expected["center_y"]),
            ))
            if actual.get("orientation") in ORIENTATIONS:
                confusion[expected_orientation][actual["orientation"]] += 1
        elif expected_visible:
            false_negative += 1
            if actual_visible:
                false_positive += 1
        elif actual_visible:
            false_positive += 1
 
        if expected_visible and (
            not matched or actual.get("orientation") not in ORIENTATIONS
        ):
            orientation_misses[expected_orientation] += 1
 
        if actual_visible and actual.get("orientation") in ORIENTATIONS:
            predicted_orientation = actual["orientation"]
            ap_events[predicted_orientation].append((
                float(actual.get("confidence", 0.0)),
                matched and predicted_orientation == expected_orientation,
            ))
        if "inference_sec" in actual:
            inference_times.append(float(actual["inference_sec"]))
 
    aps = [average_precision(ap_events[name], positive_counts[name]) for name in ORIENTATIONS]
    valid_aps = [value for value in aps if value is not None]
    f1_values = []
    for orientation in ORIENTATIONS:
        if positive_counts[orientation] == 0:
            continue
        tp = confusion[orientation][orientation]
        fp = sum(
            confusion[other][orientation]
            for other in ORIENTATIONS
            if other != orientation
        )
        fn = orientation_misses[orientation] + sum(
            confusion[orientation][other]
            for other in ORIENTATIONS
            if other != orientation
        )
        f1_values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "frames": len(truth),
        "precision_iou50": true_positive / precision_denominator if precision_denominator else 0.0,
        "recall_iou50": true_positive / recall_denominator if recall_denominator else 0.0,
        "map50": sum(valid_aps) / len(valid_aps) if valid_aps else 0.0,
        "mean_center_error": sum(center_errors) / len(center_errors) if center_errors else None,
        "orientation_macro_f1": (
            sum(f1_values) / len(f1_values) if f1_values else 0.0
        ),
        "orientation_confusion": confusion,
        "fps": (
            len(inference_times) / sum(inference_times)
            if inference_times and sum(inference_times)
            else None
        ),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir", type=Path, required=True,
        help="folder containing images/ and labels/, e.g. data/generated/test",
    )
    parser.add_argument(
        "--predictions", type=Path,
        help="predictions_*.jsonl from intro-perception-run (omit to run live)",
    )
    parser.add_argument("--method", choices=("classical", "yolo", "both"), default="both")
    parser.add_argument("--model", default="artifacts/yolo/best.pt")
    args = parser.parse_args()
 
    if args.predictions is None:
        scores = evaluate_test_set(args.test_dir, args.method, args.model)
    else:
        truth = load_truth(args.test_dir)
        warn_about_truth(truth)
        predictions = load_records(args.predictions)
        missing = sorted(set(truth) - set(predictions))
        if missing:
            raise SystemExit(
                f"{args.predictions} has no prediction for {len(missing)} test image(s), "
                f"e.g. {', '.join(missing[:3])}; was it run on this test set?"
            )
        scores = score_records(truth, predictions)
    print(json.dumps(scores, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
