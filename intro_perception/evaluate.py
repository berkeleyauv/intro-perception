"""Evaluate JSONL gate predictions against frame-indexed ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_records(path: Path) -> dict[int, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "frame_index" not in record:
                raise ValueError(f"{path}:{line_number}: missing frame_index")
            records[int(record["frame_index"])] = record
    return records


def score_records(truth: dict[int, dict], predictions: dict[int, dict]) -> dict:
    true_positive = false_positive = false_negative = true_negative = 0
    center_errors = []
    visible_centers = []

    for frame_index, expected in truth.items():
        actual = predictions.get(frame_index, {"visible": False})
        expected_visible = bool(expected.get("visible", False))
        actual_visible = bool(actual.get("visible", False))
        if expected_visible and actual_visible:
            true_positive += 1
            dx = float(actual["center_x"]) - float(expected["center_x"])
            dy = float(actual["center_y"]) - float(expected["center_y"])
            center_errors.append(math.hypot(dx, dy))
            visible_centers.append((float(actual["center_x"]), float(actual["center_y"])))
        elif expected_visible:
            false_negative += 1
        elif actual_visible:
            false_positive += 1
        else:
            true_negative += 1

    total = max(len(truth), 1)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    jitter = []
    for previous, current in zip(visible_centers, visible_centers[1:]):
        jitter.append(math.hypot(current[0] - previous[0], current[1] - previous[1]))

    return {
        "frames": len(truth),
        "visibility_accuracy": (true_positive + true_negative) / total,
        "precision": true_positive / precision_denominator if precision_denominator else 0.0,
        "recall": true_positive / recall_denominator if recall_denominator else 0.0,
        "mean_center_error": (
            sum(center_errors) / len(center_errors) if center_errors else None
        ),
        "mean_visible_jitter": sum(jitter) / len(jitter) if jitter else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    scores = score_records(load_records(args.truth), load_records(args.predictions))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
