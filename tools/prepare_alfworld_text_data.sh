#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_REPO_ROOT="${BASE_REPO_ROOT:-${PROJECT_ROOT}/extern/base_agent_rl}"
ALFWORLD_DATA_DIR="${ALFWORLD_DATA_DIR:-${PROJECT_ROOT}/data/alfworld_full}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-64}"
TEST_DATA_SIZE="${TEST_DATA_SIZE:-128}"

export PYTHONPATH="${PROJECT_ROOT}:${BASE_REPO_ROOT}:${BASE_REPO_ROOT}/agent_system/environments/env_package/alfworld:${PYTHONPATH:-}"
mkdir -p "${ALFWORLD_DATA_DIR}"

cd "${BASE_REPO_ROOT}"
python3 -m examples.data_preprocess.prepare \
  --mode text \
  --local_dir "${ALFWORLD_DATA_DIR}" \
  --train_data_size "${TRAIN_DATA_SIZE}" \
  --val_data_size "${VAL_DATA_SIZE}" \
  --test_data_size "${TEST_DATA_SIZE}"

echo "ALFWorld parquet data written to ${ALFWORLD_DATA_DIR}"
