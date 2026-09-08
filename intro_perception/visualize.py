"""Run the intro algorithm through the production perception visualizer."""

from __future__ import annotations

import argparse
from pathlib import Path

# Importing the student implementation executes its production registry decorator.
from intro_perception.perceiver import IntroGatePerceiver  # noqa: F401
from perception.tasks import registry
from perception.vis.vis import run


def data_sources(path: str) -> list[str]:
    if path == "webcam":
        return [path]

    candidate = Path(path)
    if candidate.is_dir():
        return [str(item) for item in sorted(candidate.iterdir()) if item.is_file()]
    return [str(candidate)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the intro gate perceiver using production tooling."
    )
    parser.add_argument("--data", default="webcam")
    parser.add_argument("--resize", default=1.0, type=float)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--compare",
        help="Production gate algorithm to show alongside intro, e.g. segmentation_a.",
    )
    args = parser.parse_args()

    # Load production algorithms too, allowing direct comparisons in the same UI.
    registry.discover_all()
    algorithm = registry.get_perceiver("gate", "intro")()
    comparison = (
        registry.get_perceiver("gate", args.compare)() if args.compare else None
    )
    run(
        data_sources(args.data),
        algorithm,
        save_video=args.save_video,
        resize=args.resize,
        compare_algorithm=comparison,
        algo_label="intro",
        compare_label=args.compare,
    )


if __name__ == "__main__":
    main()
