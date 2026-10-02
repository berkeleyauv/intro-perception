# Project Guide

This project works on still images only (`.jpg`, `.jpeg`, `.png`).

## Output contract

Every image returns `GateEstimate`: visibility, confidence, normalized center,
and the normalized bounding box of the visible gate. Do not change this
contract.

There is a single class, `gate`. A gate that is partly cut off by the image
edge still counts: its box covers the visible part, up to the image border.
Skip genuinely ambiguous images rather than inventing labels.

## Gate appearance

Facing the gate:

- the **left post** is **black on top and red on the bottom**;
- the **right post** is **red on top and black on the bottom**.

This top/bottom color pattern is the most reliable cue for telling the posts
apart and for pairing them. The starting `ClassicalGatePerceiver` does not use
it.

## Milestone 0: Workflow and baseline

Clone recursively, work on a feature branch, create the environment, and run
the tests. Run `intro-perception-run --method classical` on every supplied image
folder. Save three different baseline failures and commit one regression test.

Use the production visualizer when developing the classical method (press a key
to move to the next image; the HSV sliders appear as trackbars):

```bash
intro-perception-vis --data data/generated/val/images --algo intro_classical \
  --compare segmentation_a
```

The visualizer reads `.jpg` and `.png` images only.

## Milestone 1: Dataset

Collect roughly 180 still images that cover different lighting, distances, and
viewing angles, including some where the gate is partly cut off by the frame. Put them in `data/raw/`, then label them:

```bash
intro-perception-data annotate --images data/raw
intro-perception-data split --data data/raw --output data/generated
```

The annotation tool opens each image: draw one box around the gate (up to the
image edge if it is cut off), then press Enter to save it. Enter `s` to exclude
an ambiguous view (it is moved to `data/raw/excluded/`), or cancel the box for a
true no-gate image.

`split` validates every label, shuffles the images with a fixed seed, and writes
an 80/10/10 train/val/test split plus `dataset.yaml`. It prints gate and no-gate
counts per split and warns when a split has no gates; review those counts before
training. Avoid including many near-duplicate images (bursts of the same
scene): because images are split individually, near-duplicates can land in
different splits and inflate your scores.

If you were given labels in a separate folder (for example `labels_detect/` from
the auto-labeling pipeline), pass it with `--labels`. Images without a label are
left out, and a label without an image is an error:

```bash
intro-perception-data split --data data/images --labels data/labels_detect \
  --output data/generated
```

If that folder has a `manifest.csv`, `split` writes `data/generated/manifest.csv`
listing each used image with its split and a `truncated` flag for gates that are
cut off by the frame, so you can look at those separately.

## Milestone 2: Classical CV

Improve `ClassicalGatePerceiver`. A strong solution normally includes lighting
normalization, color or brightness segmentation, morphology, geometric
candidate filtering, post pairing, a calibrated confidence score, and a plan for
gates where only one post is visible (see "Gate appearance" above).

Keep useful intermediate masks in debug output. Return invisible rather than a
confident guess when two plausible posts cannot be established.

## Milestone 3: YOLO

Re-run setup with `--yolo`, then train the default pretrained YOLO26 nano model
for 30 epochs at 640 px. There is one detection class, `gate`; each box covers
the visible gate.

Training automatically selects CUDA, Apple MPS, or CPU and copies the best
weights to `artifacts/yolo/best.pt`. Machines that cannot train locally may use
`notebooks/train_colab.ipynb` with the same data, model, image size, and epochs
(upload the whole `data/generated/` folder). Do not commit weights.

## Milestone 4: Comparison

Run both methods on the held-out test images (`data/generated/test`). Report
IoU@0.50 precision and recall, mAP50, normalized center error, and FPS.
`intro-perception-evaluate` prints all of them. If `data/generated/manifest.csv`
exists, it also prints a `by_truncation` section that scores cut-off gates
separately from whole ones. Include side-by-side annotated images from `output/annotated/` and
discuss:

- three failure modes;
- the accuracy/speed tradeoff;
- how training-data coverage affects YOLO;
- where the classical assumptions fail, including on cut-off gates;
- which method you would deploy and what you would improve next.

## Submission

Open one draft PR against `main` with at least three meaningful commits. Include
clean setup commands, dataset counts by split (gate, no-gate, and cut-off gates), reproducible training
arguments, both metric reports, annotated images, known limitations, and each
member's contribution.

Temporal tracking, or estimating the gate's orientation from its post geometry on
top of the detections, is an optional extension only after the required
comparison works.