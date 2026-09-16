#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

install_yolo=false
if [[ "${1:-}" == "--yolo" ]]; then
  install_yolo=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: ./scripts/setup.sh [--yolo]" >&2
  exit 2
fi

git submodule update --init --recursive
uv venv --python 3.11
if [[ "$install_yolo" == true ]]; then
  uv pip install -r perception/requirements-torch.txt
else
  uv pip install -r perception/requirements-classical.txt
fi
uv pip install -e perception
uv pip install -e '.[test]'

echo "Setup complete. Run: source .venv/bin/activate"
if [[ "$install_yolo" == false ]]; then
  echo "Re-run with --yolo before the training milestone."
fi
