# Berkeley AUV Perception Intro Project

Build a gate detector and temporal tracker that remains useful when underwater
video becomes noisy, blurred, or briefly occluded.

This repository contains the assignment scaffold. The production perception
package is pinned as a Git submodule under `perception/`; students should make
their changes in `intro_perception/`, not inside the submodule.

## Clone and set up

```bash
git clone --recurse-submodules <REPOSITORY_URL>
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

The baseline is intentionally weak but complete. Read [GUIDE.md](GUIDE.md)
for the milestones and [RUBRIC.md](RUBRIC.md) for evaluation criteria.

## Student-owned files

The main implementation lives in:

```text
intro_perception/perceiver.py
intro_perception/tracker.py
```

You may add supporting modules and tests. Do not edit `perception/` unless a
mentor explicitly asks you to test a production-package change.
