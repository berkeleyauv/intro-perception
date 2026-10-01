"""Run either intro algorithm through the production visualizer."""

from __future__ import annotations

import argparse

from intro_perception.images import list_images
from intro_perception.production import prepare_production_namespace

prepare_production_namespace()

from intro_perception.classical import ClassicalGatePerceiver  # noqa: F401
from intro_perception.yolo import YoloGatePerceiver  # noqa: F401
from perception.tasks import registry
from perception.vis.vis import run

VISUAlIZER_SUFFIXES = (".jpg", ".png")

def data_sources(path):
    images = list_images(path)
    unsupported = [
        image.name for image in images if image.suffix.lower() not in VISUALIZER_SUFFIXES
    ]
    if unsupported:
        raise ValueError(
            f"the visualizer only reads {', '.join(VISUALIZER_SUFFIXES)} images; "
            f"convert or rename: {', '.join(unsupported[:5])}"
        )
    return [str(image) for image in images]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/generated/val/images", help="an image or a folder of images"
    )
    parser.add_argument(
        "--algo",
        choices=("intro_classical", "intro_yolo"),
        default="intro_classical",
    )
    parser.add_argument("--compare", help="second registered gate algorithm")
    parser.add_argument("--resize", default=1.0, type=float)
    args = parser.parse_args()
    registry.discover_all()
    algorithm = registry.get_perceiver("gate", args.algo)()
    comparison = registry.get_perceiver("gate", args.compare)() if args.compare else None
    run(data_sources(args.data), algorithm, save_video=False,
        resize=args.resize, compare_algorithm=comparison, algo_label=args.algo,
        compare_label=args.compare)

if __name__ == "__main__":
    main()
