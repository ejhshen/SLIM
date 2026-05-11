#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_REPO_ROOT="${BASE_REPO_ROOT:-${PROJECT_ROOT}/extern/base_agent_rl}"
SEARCH_QA_RETRIEVER_DIR="${SEARCH_QA_RETRIEVER_DIR:-${PROJECT_ROOT}/data/search_qa_retriever}"

export PYTHONPATH="${PROJECT_ROOT}:${BASE_REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${SEARCH_QA_RETRIEVER_DIR}"
cd "${BASE_REPO_ROOT}"

python3 -m examples.search.searchr1_download --save_dir "${SEARCH_QA_RETRIEVER_DIR}"
echo "Search-QA retriever files written to ${SEARCH_QA_RETRIEVER_DIR}"
