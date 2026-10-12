#!/usr/bin/env bash
# Step 2 of 2: bring the Colab YOLO results back and build the comparison.
#
#   ./scripts/finish_yolo.sh /path/to/yolo_results.zip
#
# Puts best.pt at artifacts/yolo/best.pt, scores the YOLO test predictions,
# lists YOLO's failures, draws classical and YOLO side by side, and fills the
# comparison table in REPORT.md. No PyTorch needed on this machine.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

results="${1:?usage: finish_yolo.sh /path/to/yolo_results.zip}"
mkdir -p artifacts/yolo output
rm -rf artifacts/yolo/colab && mkdir -p artifacts/yolo/colab
unzip -qo "$results" -d artifacts/yolo/colab
cp artifacts/yolo/colab/best.pt artifacts/yolo/best.pt
cp artifacts/yolo/colab/predictions_yolo.jsonl output/predictions_yolo.jsonl

intro-perception-evaluate --test-dir data/generated/test \
  --predictions output/predictions_yolo.jsonl | tee output/metrics_yolo.json
if [[ -f artifacts/yolo/colab/predictions_yolo_cpu.jsonl ]]; then
  cp artifacts/yolo/colab/predictions_yolo_cpu.jsonl output/predictions_yolo_cpu.jsonl
  intro-perception-evaluate --test-dir data/generated/test \
    --predictions output/predictions_yolo_cpu.jsonl > output/metrics_yolo_cpu.json
fi

python scripts/failure_gallery.py --test-dir data/generated/test \
  --predictions output/predictions_classical.jsonl output/predictions_yolo.jsonl \
  --output output/failures | tee output/failures_both.txt
python scripts/side_by_side.py --test-dir data/generated/test \
  --predictions output/predictions_classical.jsonl output/predictions_yolo.jsonl \
  --output output/side_by_side

metrics=(output/metrics_baseline.json output/metrics_classical.json output/metrics_yolo.json)
[[ -f output/metrics_yolo_cpu.json ]] && metrics+=(output/metrics_yolo_cpu.json)
python scripts/compare.py "${metrics[@]}" --report REPORT.md
echo
echo "Step 2 done: REPORT.md table updated; images in output/side_by_side/ and output/failures/."
