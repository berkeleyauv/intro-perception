# Berkeley AUV Perception Intro Project

Build a gate detector and temporal tracker that remains useful when underwater
video becomes noisy, blurred, or briefly occluded.

This repository contains the assignment scaffold. The production perception
package is pinned as a Git submodule under `perception/`. The student algorithm
uses its `TaskPerceiver` interface, algorithm registry, and visualizer, while
student-owned code stays in `intro_perception/`.

## Clone and set up

```bash
git clone --recurse-submodules https://github.com/berkeleyauv/intro-perception.git
cd intro_perception
./scripts/setup.sh
source .venv/bin/activate
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Run the public tests:

```bash
./scripts/test.sh
```

Run the baseline on an image or video:

```bash
python -m intro_perception.run --data path/to/video.mp4 --output output
```

Run it through the production perception visualizer:

```bash
intro-perception-vis --data path/to/video.mp4
```

The implementation is registered as `gate/intro`, just like a production
algorithm. To compare it against an existing production algorithm:

```bash
intro-perception-vis --data path/to/video.mp4 --compare segmentation_a
```

The baseline is intentionally weak but complete. Read [GUIDE.md](GUIDE.md)
for the milestones and [RUBRIC.md](RUBRIC.md) for evaluation criteria.

## Student-owned files

The main implementation lives in:

```text
intro_perception/perceiver.py
intro_perception/tracker.py
```

You may add supporting modules and tests. Treat `perception/` as a read-only
dependency: use its APIs and tools, but do not edit it unless a mentor
explicitly asks you to test a production-package change. After the project, a
strong solution can be migrated into the production repository in a separate
PR.
