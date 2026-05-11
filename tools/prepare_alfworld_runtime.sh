#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -U \
  "gymnasium==0.29.1" \
  "termcolor" \
  "textworld==1.6.2" \
  "alfworld==0.4.2"

ALFWORLD_CACHE_DIR="${ALFWORLD_CACHE_DIR:-${HOME}/.cache/alfworld}"
if [ ! -d "${ALFWORLD_CACHE_DIR}/json_2.1.1" ]; then
  mkdir -p "${ALFWORLD_CACHE_DIR}"
  alfworld-download --dir "${ALFWORLD_CACHE_DIR}"
fi
echo "ALFWorld runtime is ready. Cache: ${ALFWORLD_CACHE_DIR}"
