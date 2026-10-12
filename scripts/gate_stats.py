"""Dataset counts by split and the gate shape prior used by the classical detector.

    python scripts/gate_stats.py --data data/generated

Prints, per split, the number of gate, no-gate and cut-off (truncated) images
(the cut-off flag comes from ``data/generated/manifest.csv`` when present,
otherwise a box touching the image border counts as cut off), and the
distribution of gate width / height in pixels for whole gates. The median of
that ratio is a good value for ``GATE_RATIO_TYPICAL`` in
``intro_perception/classical.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from intro_perception.data_tools import parse_label
from intro_perception.images import list_images

SPLITS = ("train", "val", "test")
EDGE = 0.005


def touches_border(box):
    cx, cy, w, h = box
    return cx - w / 2 <= EDGE or cy - h / 2 <= EDGE or cx + w / 2 >= 1 - EDGE or cy + h / 2 >= 1 - EDGE


def read_truncated(data_dir):
    manifest = Path(data_dir) / "manifest.csv"
    if not manifest.is_file():
        return None
    with manifest.open(newline="", encoding="utf-8") as stream:
        return {
            Path(row["filename"]).stem
            for row in csv.DictReader(stream)
            if str(row.get("truncated", "")).lower() == "true"
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/generated"))
    parser.add_argument("--json", type=Path, help="also write the summary here")
    args = parser.parse_args()

    truncated_ids = read_truncated(args.data)
    source = "manifest.csv" if truncated_ids is not None else "box touches the border"
    summary = {"truncated_source": source, "splits": {}}
    ratios = []
    sizes = {}
    for split in SPLITS:
        images = list_images(args.data / split / "images")
        counts = {"images": len(images), "gate": 0, "no_gate": 0, "cut_off": 0}
        for image_path in images:
            box = parse_label(args.data / split / "labels" / f"{image_path.stem}.txt")
            if box is None:
                counts["no_gate"] += 1
                continue
            counts["gate"] += 1
            cut = (
                image_path.stem in truncated_ids
                if truncated_ids is not None
                else touches_border(box)
            )
            if cut:
                counts["cut_off"] += 1
                continue
            height, width = cv.imread(str(image_path)).shape[:2]
            sizes[(width, height)] = sizes.get((width, height), 0) + 1
            ratios.append(box[2] * width / (box[3] * height))
        summary["splits"][split] = counts

    if ratios:
        values = np.array(ratios)
        summary["whole_gate_width_over_height"] = {
            "count": int(values.size),
            "p10": round(float(np.percentile(values, 10)), 3),
            "median": round(float(np.median(values)), 3),
            "p90": round(float(np.percentile(values, 90)), 3),
        }
    summary["image_sizes"] = {f"{w}x{h}": n for (w, h), n in sorted(sizes.items())}

    print(f"cut-off flag from: {source}")
    print(f"{'split':6s} {'images':>6s} {'gate':>5s} {'no-gate':>7s} {'cut-off':>7s}")
    for split, counts in summary["splits"].items():
        print(f"{split:6s} {counts['images']:6d} {counts['gate']:5d} {counts['no_gate']:7d} {counts['cut_off']:7d}")
    if ratios:
        print("whole-gate width/height (pixels):", summary["whole_gate_width_over_height"])
    print("image sizes:", summary["image_sizes"])
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
