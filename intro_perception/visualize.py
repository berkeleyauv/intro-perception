"""Run either intro algorithm through the production visualizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from intro_perception.production import prepare_production_namespace

prepare_production_namespace()

from intro_perception.perceiver import ClassicalGatePerceiver, YoloGatePerceiver  # noqa: F401
from perception.tasks import registry
from perception.vis.vis import run


def data_sources(path):
    if path == "webcam":
        return [path]
    candidate = Path(path)
    if candidate.is_dir():
        return [str(item) for item in sorted(candidate.iterdir()) if item.is_file()]
    return [str(candidate)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="webcam")
    parser.add_argument(
        "--algo",
        choices=("intro_classical", "intro_yolo"),
        default="intro_classical",
    )
    parser.add_argument("--compare", help="second registered gate algorithm")
    parser.add_argument("--resize", default=1.0, type=float)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    registry.discover_all()
    algorithm = registry.get_perceiver("gate", args.algo)()
    comparison = registry.get_perceiver("gate", args.compare)() if args.compare else None
    run(data_sources(args.data), algorithm, save_video=args.save_video,
        resize=args.resize, compare_algorithm=comparison, algo_label=args.algo,
        compare_label=args.compare)


if __name__ == "__main__":
    main()
