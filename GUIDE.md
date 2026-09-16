# Project Guide

## Output contract

Every frame returns `GateEstimate`: visibility, confidence, normalized center,
normalized full-gate bounding box, and one of `left`, `head_on`, `right`, or
`null` when no gate is visible. Do not change this contract.

Orientation describes the viewing angle: `left` means the gate's right side
appears closer, `head_on` means both posts have similar perspective, and
`right` means the left side appears closer. Skip genuinely ambiguous frames
rather than inventing labels.

## Milestone 0: Workflow and baseline

Clone recursively, work on a feature branch, create the environment, and run
the tests. Run `intro-perception-run --method classical` on every supplied clip.
Save three different baseline failures and commit one regression test.

Use the production visualizer when developing the classical method:

```bash
intro-perception-vis --data clip.mp4 --algo intro_classical \
  --compare segmentation_a
```

## Milestone 1: Dataset

Extract 60 evenly spaced frames from each of at least three different videos.
The local annotation tool opens each image: draw one box around the entire gate,
then enter `l`, `h`, or `r`. Enter `s` to exclude an ambiguous view, or cancel
the box for a true no-gate frame.

The build command validates every label and assigns whole source videos to
60/20/20 train/validation/test splits. Adjacent frames from the same video must
never cross splits. Review the generated manifest and class balance before
training.

## Milestone 2: Classical CV

Improve `ClassicalGatePerceiver`. A strong solution normally includes lighting
normalization, color or brightness segmentation, morphology, geometric
candidate filtering, post pairing, a calibrated confidence score, and an
orientation cue based on the relative appearance of the posts.

Keep useful intermediate masks in debug output. Return invisible rather than a
confident guess when two plausible posts cannot be established.

## Milestone 3: YOLO

Re-run setup with `--yolo`, then train the default pretrained YOLO26 nano model
for 30 epochs at 640 px. The three detection classes are `gate_left`,
`gate_head_on`, and `gate_right`; each box covers the complete gate.

Training automatically selects CUDA, Apple MPS, or CPU and copies the best
weights to `artifacts/yolo/best.pt`. Machines that cannot train locally may use
`notebooks/train_colab.ipynb` with the same data, model, image size, and epochs.
Do not commit weights.

## Milestone 4: Comparison

Run both methods on every held-out source video. Report IoU@0.50 precision and
recall, mAP50, normalized center error, orientation macro-F1, the confusion
matrix, and FPS. Include the side-by-side video and discuss:

- three failure modes;
- the accuracy/speed tradeoff;
- how training-data coverage affects YOLO;
- where the classical assumptions fail;
- which method you would deploy and what you would improve next.

## Submission

Open one draft PR against `main` with at least three meaningful commits. Include
clean setup commands, dataset counts by split/class, reproducible training
arguments, both metric reports, annotated video, known limitations, and each
member's contribution.

Temporal tracking or a YOLO-box-plus-classical-orientation fusion is an optional
extension only after the required comparison works.
