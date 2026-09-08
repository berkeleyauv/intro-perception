# Rubric

| Area | Weight | Evidence |
|---|---:|---|
| Single-frame detection | 25% | Center accuracy, precision/recall, difficult frames |
| Temporal behavior | 25% | Jitter, occlusion recovery, clean reset after loss |
| Confidence and failure handling | 15% | Confidence calibration and safe invisible outputs |
| Evaluation and debugging | 15% | Reproducible metrics and useful annotated frames |
| Code quality and tests | 15% | Readable decomposition, tests, no hard-coded cases |
| Communication | 5% | PR explanation, failure analysis, demo |

Correct behavior on unseen footage matters more than tuning exclusively for
the public examples.
