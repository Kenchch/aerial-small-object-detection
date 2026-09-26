#!/usr/bin/env bash
# Usage: bash scripts/resume.sh [run_name] [train.py flags...]
# e.g.   bash scripts/resume.sh n_1024 --device 0 --batch 4
set -euo pipefail
cd "$(dirname "$0")/.."
name="${1:-n_1024}"
if [[ ! -f "runs/$name/weights/last.pt" ]]; then
  echo "No checkpoint for $name" >&2
  exit 1
fi
exec "${PYTHON:-python}" src/train.py --name "$name" --resume "${@:2}"
