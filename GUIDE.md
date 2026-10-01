# Project Guide

This project works on still images only (`.jpg`, `.jpeg`, `.png`).

## Output contract

Every image returns `GateEstimate`: visibility, confidence, normalized center,
normalized full-gate bounding box, and one of `left`, `head_on`, `right`, or
`null` when no gate is visible. Do not change this contract.

Orientation describes the viewing angle: `left` means the gate's right side
appears closer, `head_on` means both posts have similar perspective, and
`right` means the left side appears closer. Skip genuinely ambiguous images
rather than inventing labels.

## Gate appearance

Facing the gate:

- the **left post** is **black on top and red on the bottom**;
- the **right post** is **red on top and black on the bottom**.

This top/bottom color pattern is the most reliable cue for telling the posts
apart and for estimating orientation. The starting `ClassicalGatePerceiver` does
not use it.

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
viewing angles. Put them in `data/raw/`, then label them:

```bash
intro-perception-data annotate --images data/raw
intro-perception-data split --data data/raw --output data/generated
```

The annotation tool opens each image: draw one box around the entire gate, then
enter `l`, `h`, or `r`. Enter `s` to exclude an ambiguous view (it is moved to
`data/raw/excluded/`), or cancel the box for a true no-gate image.

`split` validates every label, shuffles the images with a fixed seed, and writes
an 80/10/10 train/val/test split plus `dataset.yaml`. It prints image and class
counts per split and warns when a split is missing a class; review those counts
before training. Avoid including many near-duplicate images (bursts of the same
scene): because images are split individually, near-duplicates can land in
different splits and inflate your scores.

## Milestone 2: Classical CV

Improve `ClassicalGatePerceiver`. A strong solution normally includes lighting
normalization, color or brightness segmentation, morphology, geometric
candidate filtering, post pairing, a calibrated confidence score, and an
orientation cue based on the relative appearance of the posts (see "Gate
appearance" above).

Keep useful intermediate masks in debug output. Return invisible rather than a
confident guess when two plausible posts cannot be established.

## Milestone 3: YOLO

Re-run setup with `--yolo`, then train the default pretrained YOLO26 nano model
for 30 epochs at 640 px. The three detection classes are `gate_left`,
`gate_head_on`, and `gate_right`; each box covers the complete gate.

Training automatically selects CUDA, Apple MPS, or CPU and copies the best
weights to `artifacts/yolo/best.pt`. Machines that cannot train locally may use
`notebooks/train_colab.ipynb` with the same data, model, image size, and epochs
(upload the whole `data/generated/` folder). Do not commit weights.

## Milestone 4: Comparison

Run both methods on the held-out test images (`data/generated/test`). Report
IoU@0.50 precision and recall, mAP50, normalized center error, orientation
macro-F1, the confusion matrix, and FPS. `intro-perception-evaluate` prints all
of them. Include side-by-side annotated images from `output/annotated/` and
discuss:

- three failure modes;
- the accuracy/speed tradeoff;
- how training-data coverage affects YOLO;
- where the classical assumptions fail;
- which method you would deploy and what you would improve next.

## Submission

Open one draft PR against `main` with at least three meaningful commits. Include
clean setup commands, dataset counts by split/class, reproducible training
arguments, both metric reports, annotated images, known limitations, and each
member's contribution.

Temporal tracking or a YOLO-box-plus-classical-orientation fusion is an optional
extension only after the required comparison works.