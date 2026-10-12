"""Run the starting (baseline) classical detector from ``main`` on a folder.

    python scripts/run_baseline.py --data data/generated/test/images \
        --output output/predictions_baseline.jsonl

The baseline source is read from git (``main:intro_perception/classical.py``)
and loaded under another registry name, so the report's "before" numbers can
be reproduced from this branch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time

from intro_perception.images import list_images, read_image


def load_baseline(ref):
    source = subprocess.run(
        ["git", "show", f"{ref}:intro_perception/classical.py"],
        check=True, capture_output=True, text=True,
    ).stdout.replace('algo="intro_classical"', 'algo="intro_classical_baseline"')
    path = Path(tempfile.mkdtemp()) / "baseline_classical.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("baseline_classical", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ClassicalGatePerceiver()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/predictions_baseline.jsonl"))
    parser.add_argument("--ref", default="main", help="git ref holding the baseline")
    args = parser.parse_args()
    perceiver = load_baseline(args.ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images = list_images(args.data)
    with args.output.open("w", encoding="utf-8") as stream:
        for image_path in images:
            frame = read_image(image_path)
            started = time.perf_counter()
            estimate = perceiver.analyze(frame, debug=False)
            elapsed = time.perf_counter() - started
            stream.write(json.dumps({"frame_id": image_path.stem, "image": image_path.name,
                                     "inference_sec": elapsed, **estimate.to_dict()}) + "\n")
    print(f"wrote {len(images)} baseline predictions to {args.output}")


if __name__ == "__main__":
    main()
