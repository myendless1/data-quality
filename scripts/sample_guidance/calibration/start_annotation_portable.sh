#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/outputs/calibration_bbox_portable}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

python3 "$SCRIPT_DIR/portable_annotation_server.py" \
  --data-dir "$DATA_DIR" \
  --host "$HOST" \
  --port "$PORT"
