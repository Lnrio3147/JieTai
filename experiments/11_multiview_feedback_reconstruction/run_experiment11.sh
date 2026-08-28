#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 MANIFEST [OUTPUT_DIR]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$(realpath "$1")"
OUTPUT=""
if [ "$#" -eq 2 ]; then
  OUTPUT="$(realpath -m "$2")"
fi
cd "$SCRIPT_DIR"

if ! "$PYTHON_BIN" -c 'import numpy, open3d' >/dev/null 2>&1; then
  echo "PYTHON_BIN must provide NumPy and Open3D." >&2
  exit 1
fi

if [ "$#" -eq 2 ]; then
  PREFLIGHT="${OUTPUT%/}_preflight.json"
  "$PYTHON_BIN" preflight_sequence.py \
    --manifest "$MANIFEST" --output "$PREFLIGHT"
  exec "$PYTHON_BIN" reconstruct.py \
    --manifest "$MANIFEST" --preflight-report "$PREFLIGHT" --output "$OUTPUT"
fi
PREFLIGHT="$SCRIPT_DIR/results/preflight_$(basename "${MANIFEST%.json}").json"
"$PYTHON_BIN" preflight_sequence.py \
  --manifest "$MANIFEST" --output "$PREFLIGHT"
exec "$PYTHON_BIN" reconstruct.py \
  --manifest "$MANIFEST" --preflight-report "$PREFLIGHT"
