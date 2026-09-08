#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"

git submodule update --init --recursive
uv venv --python 3.11
uv pip install -r perception/requirements-classical.txt
uv pip install -e perception
uv pip install -e '.[test]'

echo "Setup complete. Run: source .venv/bin/activate"
