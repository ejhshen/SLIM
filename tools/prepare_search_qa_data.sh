#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_REPO_ROOT="${BASE_REPO_ROOT:-${PROJECT_ROOT}/extern/base_agent_rl}"
SEARCH_QA_DATA_DIR="${SEARCH_QA_DATA_DIR:-${PROJECT_ROOT}/data/search_qa_processed}"
SEARCH_QA_HF_REPO="${SEARCH_QA_HF_REPO:-PeterJinGo/nq_hotpotqa_train}"

export PYTHONPATH="${PROJECT_ROOT}:${BASE_REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${SEARCH_QA_DATA_DIR}"
cd "${BASE_REPO_ROOT}"

python3 -m examples.data_preprocess.preprocess_search_r1_dataset \
  --hf_repo_id "${SEARCH_QA_HF_REPO}" \
  --local_dir "${SEARCH_QA_DATA_DIR}"

python3 -m examples.data_preprocess.generate_search_r1_val \
  --input "${SEARCH_QA_DATA_DIR}/test.parquet" \
  --output_dir "${SEARCH_QA_DATA_DIR}" \
  --max_sample "${SEARCH_QA_VAL_SIZE:-4000}"

echo "Search-QA train/test/val_4000 parquet data written to ${SEARCH_QA_DATA_DIR}"
