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
# Put your images in data/raw/, then draw one box per image and pick its
# orientation. This writes data/raw/<image>.txt next to each image (an empty
# .txt means "no gate"). Already-labeled images are skipped.
intro-perception-data annotate --images data/raw

# Split into train/val/test (80/10/10) and write data/generated/dataset.yaml.
intro-perception-data split --data data/raw --output data/generated

# Install the ML dependencies and train the nano model.
./scripts/setup.sh --yolo
source .venv/bin/activate
intro-perception-train

# Compare both methods on the held-out test images.
intro-perception-run --data data/generated/test/images --method both --output output
intro-perception-evaluate --test-dir data/generated/test \
  --predictions output/predictions_yolo.jsonl

# Or let evaluate run the detectors itself and score both:
intro-perception-evaluate --test-dir data/generated/test
```

`run` writes `predictions_<method>.jsonl` and an annotated copy of every image
under `output/annotated/`. Use `--method classical` until you have trained YOLO
weights.
 
Labels you create elsewhere work too: any YOLO-format `.txt` file next to its
image (`class cx cy w h`, normalized, class 0/1/2 = `gate_left` / `gate_head_on`
/ `gate_right`, one gate box per image) is accepted by `split`.
 
Raw images, generated datasets, model weights, and run artifacts are
intentionally ignored by Git.