# Project Guide

## Goal

Given an image or video containing a qualification gate, publish one structured
estimate per frame and create an annotated debug video. The estimator should be
stable across lighting variation, false contours, motion blur, and short
occlusions.

The required output type is `GateEstimate` in `intro_perception/types.py`.
Coordinates are normalized so `(0, 0)` is the top-left of the image and
`(1, 1)` is the bottom-right.

## What is provided

- A color-threshold baseline detector.
- A simple exponential-moving-average tracker.
- Image/video loading and annotated output.
- JSONL prediction output.
- A small evaluator and public tests.
- The production `perception` package as a pinned, read-only submodule.
- Registration as the production-style `gate/intro` algorithm.
- The production visualizer for inspecting debug frames and comparing against
  an existing gate algorithm.

The provided system should run before you change any code. Record its metrics
and save one failure example; that is your baseline.

Use `intro-perception-vis` during development. This deliberately exercises the
same `TaskPerceiver`, registry, and visualization workflow used by production
algorithms. Use `intro-perception-run` when you need reproducible JSONL output
for the project evaluator.

## Milestones

### 1. Baseline and failure analysis

- Run the starter on every public clip.
- Identify at least three distinct failure modes.
- Add a regression test for one failure.

### 2. Single-frame detection

- Improve gate candidate generation and rejection.
- Return a meaningful confidence value rather than a fixed value.
- Avoid false detections when fewer than two plausible posts are visible.
- Preserve useful intermediate images in debug mode.

### 3. Temporal tracking

- Reduce frame-to-frame center jitter.
- Recover after short occlusions without locking onto a distractor.
- Reset after a sufficiently long loss.
- Document the state maintained between frames.

### 4. Evaluation and communication

- Run the public evaluator.
- Produce an annotated output video.
- Explain one design tradeoff using measured evidence.
- Add tests covering your tracker and confidence behavior.

## Constraints

- Use the `GateEstimate` output contract unchanged.
- Keep the `TaskPerceiver.analyze` interface and `gate/intro` registration.
- Treat the `perception/` submodule as read-only student infrastructure.
- Do not hard-code answers for individual frames or filenames.
- The estimator must run offline on a normal laptop.
- A learned detector is optional, not required.
- The program must handle an empty or unreadable input with a clear error.

## Submission

Push all work to your assigned branch and open a draft pull request against
`main`. In the PR description include:

- Setup and run commands.
- Baseline and final metrics.
- A link to or attachment of the annotated video.
- Known failure cases.
- A short description of each member's contribution.
