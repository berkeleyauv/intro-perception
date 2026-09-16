# Berkeley AUV Perception Intro Project

Build two qualification-gate detectors from supplied underwater videos: a
classical OpenCV pipeline and a fine-tuned YOLO model. Both methods return the
same gate box, center, confidence, and view orientation (`left`, `head_on`, or
`right`) so their accuracy, speed, and failure modes can be compared fairly.

## Start here

```bash
git clone --recurse-submodules https://github.com/berkeleyauv/intro-perception.git
cd intro-perception
git switch -c <your-name>/perception-intro
./scripts/setup.sh
source .venv/bin/activate
./scripts/test.sh
```

The production `perception/` repository is a pinned, read-only submodule.
Student code belongs in `intro_perception/`. Follow [GUIDE.md](GUIDE.md) for the
2–3 week milestone sequence.

## Core commands

```bash
# Extract and label roughly 180 frames from at least three videos.
intro-perception-data extract path/to/videos --output data/working
intro-perception-data annotate --images data/working
intro-perception-data build --images data/working --output data/generated

# Install the ML dependencies and train the nano model.
./scripts/setup.sh --yolo
source .venv/bin/activate
intro-perception-train

# Compare both methods on held-out videos.
intro-perception-run --data path/to/held-out-videos --method both --output output
intro-perception-evaluate --truth data/generated/truth_test.jsonl \
  --predictions output/predictions_yolo.jsonl
```

Raw videos, extracted frames, generated datasets, model weights, and run
artifacts are intentionally ignored by Git.
