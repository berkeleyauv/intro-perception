"""Score the classical detector on named synthetic hard cases.

Each case reproduces one failure mode of the baseline (see REPORT.md). Run:

    python scripts/hard_cases.py [--save output/hard_cases]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2 as cv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from gate_helpers import SCENE_HEIGHT, SCENE_WIDTH, box_iou_pixels, make_scene  # noqa: E402

from intro_perception.classical import ClassicalGatePerceiver, annotate  # noqa: E402

CASES = {
    "pool_gate": {},
    "angled_far_post_shorter": {"gate_width": 220, "right_height": 160},
    "small_far_gate": {"gate_x": 380, "gate_width": 120, "top": 150, "post_height": 80, "post_width": 6},
    "dark_frame": {"brightness": -35},
    "washed_out": {"contrast": 0.5, "brightness": 40},
    "rotated_with_padding": {"rotate": 6},
    "jpeg": {"jpeg": 50},
    "cut_off_right": {"gate_x": 420, "gate_width": 420},
    "cut_off_left": {"gate_x": -230, "gate_width": 420},
    "seen_from_behind": {"red": (32, 17, 3), "black": (109, 84, 84)},
    "no_gate": {"left": False, "right": False},
    "no_gate_dark_poles": {"left": False, "right": False,
                           "distractors": [(100, 40, 106, 300, (60, 40, 10)),
                                           (400, 20, 404, 330, (50, 35, 10)),
                                           (0, 150, 640, 160, (40, 25, 5))]},
    "no_gate_rotated": {"left": False, "right": False, "rotate": 8},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="write annotated images here")
    args = parser.parse_args()
    perceiver = ClassicalGatePerceiver()
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
    passed = 0
    for name, params in CASES.items():
        frame, truth = make_scene(**params)
        estimate, debug = perceiver.analyze(frame, debug=True)
        if truth is None:
            ok = not estimate.visible
            detail = "visible" if estimate.visible else "invisible"
        else:
            iou = box_iou_pixels(estimate, truth, frame.shape) if estimate.visible else 0.0
            ok = iou >= 0.5
            detail = f"IoU {iou:.2f}"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:22s} {detail:12s} conf {estimate.confidence:.2f}")
        if args.save:
            panel = debug[0].copy()
            if truth is not None:
                cv.rectangle(panel, truth[:2], (truth[2] - 1, truth[3] - 1), (0, 255, 0), 1)
            tiles = [panel] + [cv.resize(d, (SCENE_WIDTH, SCENE_HEIGHT)) for d in debug[1:]]
            cv.imwrite(str(args.save / f"{name}.jpg"), cv.hconcat(tiles[:2]) if len(tiles) > 1 else panel)
    print(f"{passed}/{len(CASES)} hard cases pass")


if __name__ == "__main__":
    main()
