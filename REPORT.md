# Perception intro project: classical vs YOLO gate detection

Branch `satyajith/perception-intro`. Two gate detectors that return the same
`GateEstimate` (visible, confidence, center, box): a classical OpenCV pipeline
and a fine-tuned YOLO26-nano, scored on the same held-out test split.

> Step-by-step instructions for finishing and submitting: [docs/STEPS.md](docs/STEPS.md).
>
> **Status:** classical pipeline, dataset split, tests and tooling are done and
> scored on the test split. YOLO is trained in Colab
> (`notebooks/train_colab.ipynb`); `scripts/finish_yolo.sh` fills its column
> of the comparison table below. Sections marked _(after YOLO)_ are finished
> from those results.

## Setup

```bash
git clone --recurse-submodules https://github.com/berkeleyauv/intro-perception.git
cd intro-perception
git switch satyajith/perception-intro
./scripts/setup.sh            # add --yolo only if you will train/run YOLO locally
source .venv/bin/activate
./scripts/test.sh             # 103 tests
```

## Reproducing the results

The mentor data unpacks to `images/` (1,833 `.jpg`) and `intro_detect/`
(1,833 YOLO labels + `manifest.csv`).

```bash
# Step 1 (CPU, ~1 min): split, dataset counts, baseline + classical on test,
# and output/generated.zip for Colab.
./scripts/full_pipeline.sh /path/to/images /path/to/intro_detect

# Step 2: in Colab run notebooks/train_colab.ipynb with output/generated.zip,
# download yolo_results.zip, then:
./scripts/finish_yolo.sh /path/to/yolo_results.zip
```

Training arguments (local `intro-perception-train` and Colab are identical):
`yolo26n.pt` pretrained, `epochs=30`, `imgsz=640`, `seed=0`,
`deterministic=True`, one class `gate`, data `data/generated/dataset.yaml`.

## Dataset

Auto-labeled and reviewed labels (`manifest.csv`): **1,833 images, 1,480 with
a gate, 353 without, 769 gates cut off by the frame** (almost all at the left
or right edge; their box runs from the visible post to that edge). Images are
640×640.

**The supplied images contain near-duplicates.** They are a Roboflow export of
611 photos × 3 augmented copies (`<photo>_jpg.rf.<hash>`: rotation with gray
borders, hue/brightness/contrast changes), and the photos are video frames
~0.8 s apart from two sessions. With the default per-image split, **179 of the
183 test images had a copy of the same photo in train**, which would inflate
every test score. `intro-perception-data split` now keeps the copies of a
photo together by default, and `--burst-gap 5` (used by the pipeline) also
keeps consecutive frames less than 5 s apart together (59 bursts). Seed 42,
80/10/10 by image count:

| split | images | gate | no-gate | cut-off |
|---|---:|---:|---:|---:|
| train | 1479 | 1184 | 295 | 565 |
| val | 171 | 150 | 21 | 98 |
| test | 183 | 146 | 37 | 106 |

Whole-gate boxes: width/height median 1.35 (p10 1.08, p90 1.53).

Usage of the splits: the classical color table is fitted on **train**, its
thresholds and confidence calibration chosen on **val**, and **test** was run
once at the end. YOLO trains on train, selects its best epoch on val.

## Classical pipeline (`intro_perception/classical.py`, `color_model.py`)

What the real frames look like, and what each stage does about it:

1. **Paint color has no fixed value.** Water tint plus the dataset's hue and
   brightness augmentation move "red" from maroon to mauve to purple. A
   per-pixel **color look-up table** (16³ Lab bins, black/red/background
   log-likelihoods) is applied to each pixel's color *relative to the image
   median and its 41-px surroundings*. It is fitted from the training labels
   only (`scripts/fit_color_model.py`): along the left and right edges of
   each whole-gate box the post is found row by row and a robust line fit,
   giving black and red samples without hand-labeling pixels. This is
   histogram back-projection, not a neural network. Rotation padding (flat
   gray borders) is detected and ignored.
2. **Masks keep vertical structure.** Thresholded log-odds, then a tall
   opening that removes speckle, the crossbar and the diagonal lane lines that
   touch the posts.
3. **Segments** are long, thin, roughly vertical components (tilt ≤ 30° via
   line fits); collinear pieces split by ripples or panels are merged.
4. **Posts from the color pattern.** Black directly above red = left post; red
   above black = right post. Above the crossbar the support pipes continue,
   painted like the *other* post, so the post's top is set one bottom-half
   length above the color change (the halves are equal; on training data the
   color change sits at 0.49 of the box height). Post-like patterns stacked
   directly above a post (supports) are dropped.
5. **Pairing**: left post left of a right post, similar length, aligned tops,
   spacing/height in 0.6–2.6. A mirrored pair (gate seen from behind) scores
   ×0.8.
6. **Cut-off gates**: a lone complete post gives a box from the post to the
   frame edge on the missing post's side (left post → right edge), score ×0.75.
7. **Confidence**: logistic map of the geometric score fitted on val
   (`classical_calibration.json`); below 0.45 the answer is "no gate".

Debug output (`intro-perception-vis`): annotated frame with left (green) and
right (magenta) post boxes plus the candidate type, and the red/black mask
overlay. Sliders: `black_min`, `red_min`.

What changed the validation numbers most (precision / recall / mAP50):
first version tuned on synthetic scenes 0.26 / 0.04 / 0.03 → color table
0.50 / 0.39 / 0.28 → vertical-structure masks 0.77 / 0.63 / 0.60 → color
relative to the 41-px surroundings 0.78 / 0.79 / 0.75 → no-gate cut-off 0.45
chosen on val **0.84 / 0.79 / 0.75**.

### Baseline failures (Milestone 0)

On the test split the starting detector found 9 of 146 gates (precision
0.13, recall 0.06, mAP50 0.02). Its three main failures:

1. **Red is never found.** Its red band is HSV hue 0–10 with V ≥ 100; under
   the pool water the red paint is a muted, blue-dominant mauve (median BGR ≈
   109, 84, 84), nowhere near that band.
2. **Largest dark blobs win.** With red missing it pairs the two largest tall
   dark shapes: support pipes, lane lines, the far wall, rotation borders.
3. **Cut-off gates are impossible.** It requires two posts, but 52% of the
   gates have one post in view (recall on them: 0.02).

Regression tests pin these down on synthetic pool scenes drawn with the
measured paint and water colors (`tests/test_classical.py`,
`scripts/hard_cases.py`; 12/13 pass, the failing one is the washed-out
limitation below). Before/after on synthetic scenes:

![pool gate](docs/images/synthetic_pool_gate.jpg)
![cut off](docs/images/synthetic_cut_off_right.jpg)
![no gate](docs/images/synthetic_no_gate_dark_poles.jpg)

## Results on the held-out test split

<!-- comparison:start -->
| metric | baseline | classical | yolo |
|---|---:|---:|---:|
| precision @ IoU 0.5 | 0.125 | 0.835 | _(after YOLO)_ |
| recall @ IoU 0.5 | 0.062 | 0.726 | _(after YOLO)_ |
| mAP50 | 0.017 | 0.701 | _(after YOLO)_ |
| mean center error (normalized) | 0.1006 | 0.0163 | _(after YOLO)_ |
| FPS | 582.1 | 25.3 | _(after YOLO)_ |
| recall, whole gates | 0.175 | 0.750 | _(after YOLO)_ |
| recall, cut-off gates | 0.019 | 0.717 | _(after YOLO)_ |
| precision, cut-off gates | 0.047 | 0.844 | _(after YOLO)_ |
| test frames | 183 | 183 | 183 |
<!-- comparison:end -->

FPS is single-image latency on the machine that ran each method. Baseline and
classical were timed on a 2-core cloud CPU; YOLO's `yolo` column is timed on
the Colab GPU and `yolo_cpu` on the Colab CPU, which is the fairer comparison
with the classical numbers.

Annotated test images: `output/annotated/` (classical), `output/side_by_side/`
(classical | YOLO, label in green, prediction in orange), mistakes in
`output/failures/<method>/`.

### Classical failure analysis (41 mistakes in 183 test frames)

Every mistake was looked at (`output/failures/classical/`):

| failure mode | count | what happens |
|---|---:|---|
| Faint, washed-out or distant posts | 15 | low-contrast or bright augmented frames and far gates: the paint is not separated from the water, the post is missed |
| Support pipe read as a post | 11 | a left post's support (red over black, above the crossbar) looks exactly like a right post when the real post's red half is too faint; the box goes above the crossbar toward the wrong edge |
| Box extent wrong | 8 | a post runs off the frame bottom, or one post of a whole gate is missed and the box runs to an edge |
| Strong camera roll (>30°) | 6 | posts are too tilted for the near-vertical assumption |
| Label disagreement | 1 | a reviewer marked "no gate" while a post is visible at the edge |

Examples (left: baseline, right: classical; green = label, orange =
prediction) are in `output/report_images/`: `success_whole_gate.jpg`,
`success_cut_off_gate.jpg`, `failure_support_pipe.jpg`,
`failure_faint_post.jpg`, `failure_camera_roll.jpg`. Add them to the PR
description (they are dataset images, so they are not committed).

Where the classical assumptions break: it assumes both colors of a post are
visible and separable from the water, that posts are near-vertical, and that a
lone post means the gate continues past the frame edge. Cut-off gates work
well when the visible post is clear (recall 0.72, precision 0.84) because the
one-post rule matches how they are labeled; they fail when the visible post is
faint or is confused with its own support pipe.

## YOLO _(after YOLO)_

Training curves are in `artifacts/yolo/colab/results.png`; Ultralytics' own
test metrics in `artifacts/yolo/colab/ultralytics_test_metrics.json`.
`output/failures/yolo/` lists its mistakes.

## Discussion _(after YOLO)_

- **Accuracy vs speed** – classical: no training, CPU only, ~25 FPS at 640 px,
  every decision explainable; YOLO: compare the table.
- **Training-data coverage** – compare YOLO's recall on whole vs cut-off gates
  with how many of each are in train (619 whole, 565 cut-off), and check
  whether its mistakes cluster in the same faint/rolled frames.
- **Deploy** – decide from the table.

## Optional extension: orientation

`intro_perception/orientation.py` estimates yaw from a classical post pair:
`cos(yaw) ≈ spacing / head-on spacing` (head-on spacing/height 1.3 from the
labels), sign from the shorter, farther post; mirrored pairs are flagged as
rear views. `scripts/orientation_demo.py` writes a CSV and annotated images
(21 of 183 test frames have a pair; |yaw| median 16°). There are no orientation
labels, so this is checked by eye only; a YOLO box could seed the same post
search inside the box for more pairs.

## Known limitations

- Washed-out frames (contrast halved) and very dark frames lose the paint.
- Support pipes vs posts relies on the red half being visible.
- Camera roll beyond ~30° breaks the near-vertical post assumption.
- The color table and thresholds are fitted to this pool and this
  augmentation; a new pool needs `REFIT=1 ./scripts/full_pipeline.sh ...`.
- A cut-off gate seen from behind is boxed toward the wrong edge, because its
  color pattern points to the opposite side (seen once in the training data).

## Contributions

_TODO: each member's contribution._
