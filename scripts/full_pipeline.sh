#!/usr/bin/env bash
# Step 1 of 2: everything that runs without a GPU.
#
#   ./scripts/full_pipeline.sh <images-dir> <labels-dir>
#
# <images-dir> holds the .jpg files, <labels-dir> the YOLO .txt labels and
# manifest.csv (the mentor zips unpack to images/ and intro_detect/).
# It splits the data, prints dataset counts, scores the baseline and the new
# classical detector on the test split, and writes output/generated.zip to
# upload to notebooks/train_colab.ipynb. Step 2 is scripts/finish_yolo.sh.
#
# Set BURST_GAP=0 to keep only augmented copies of a photo together (default 5 s
# keeps consecutive frames of one pass together too). Set REFIT=1 to refit the
# classical color table and calibration from this split instead of using the
# committed ones.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

images="${1:?usage: full_pipeline.sh <images-dir> <labels-dir>}"
labels="${2:?usage: full_pipeline.sh <images-dir> <labels-dir>}"
burst_gap="${BURST_GAP:-5}"

if [[ -d data/generated ]]; then
  echo "data/generated already exists; reusing it (delete it to re-split)"
else
  intro-perception-data split --data "$images" --labels "$labels" \
    --output data/generated --burst-gap "$burst_gap"
fi

mkdir -p output
python scripts/gate_stats.py --data data/generated --json output/dataset_stats.json

if [[ "${REFIT:-0}" == 1 ]]; then
  python scripts/fit_color_model.py --split data/generated/train
  python scripts/calibrate_classical.py --split data/generated/val
fi

# Classical: predictions + annotated images, then the scores.
intro-perception-run --data data/generated/test/images --method classical --output output
intro-perception-evaluate --test-dir data/generated/test \
  --predictions output/predictions_classical.jsonl | tee output/metrics_classical.json

# Baseline (the starting detector on main), for the before/after comparison.
python scripts/run_baseline.py --data data/generated/test/images \
  --output output/predictions_baseline.jsonl
intro-perception-evaluate --test-dir data/generated/test \
  --predictions output/predictions_baseline.jsonl > output/metrics_baseline.json

python scripts/failure_gallery.py --test-dir data/generated/test \
  --predictions output/predictions_classical.jsonl --output output/failures \
  | tee output/failures_classical.txt
python scripts/compare.py output/metrics_baseline.json output/metrics_classical.json

rm -f output/generated.zip
(cd data && zip -qr ../output/generated.zip generated -x "*.DS_Store")
echo
echo "Step 1 done. Upload output/generated.zip to notebooks/train_colab.ipynb in Colab,"
echo "then run: ./scripts/finish_yolo.sh ~/Downloads/yolo_results.zip"
