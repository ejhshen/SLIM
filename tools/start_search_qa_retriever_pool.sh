#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_REPO_ROOT="${BASE_REPO_ROOT:-${PROJECT_ROOT}/extern/base_agent_rl}"
SEARCH_QA_RETRIEVER_DIR="${SEARCH_QA_RETRIEVER_DIR:-${PROJECT_ROOT}/data/search_qa_retriever}"
SEARCH_QA_RETRIEVER_PORTS="${SEARCH_QA_RETRIEVER_PORTS:-8000 8001 8002 8003}"
SEARCH_QA_RETRIEVER_TOPK="${SEARCH_QA_RETRIEVER_TOPK:-3}"
SEARCH_QA_RETRIEVER_MODEL_NAME="${SEARCH_QA_RETRIEVER_MODEL_NAME:-e5}"
SEARCH_QA_RETRIEVER_LOG_DIR="${SEARCH_QA_RETRIEVER_LOG_DIR:-${PROJECT_ROOT}/logs/search_qa_retriever}"
SEARCH_QA_RETRIEVER_WAIT_SECONDS="${SEARCH_QA_RETRIEVER_WAIT_SECONDS:-240}"

INDEX_PATH="${SEARCH_QA_RETRIEVER_DIR}/e5_Flat.index"
CORPUS_PATH="${SEARCH_QA_RETRIEVER_DIR}/wiki-18.jsonl"
test -f "${INDEX_PATH}" || { echo "Missing index: ${INDEX_PATH}" >&2; exit 1; }
test -f "${CORPUS_PATH}" || { echo "Missing corpus: ${CORPUS_PATH}" >&2; exit 1; }

mkdir -p "${SEARCH_QA_RETRIEVER_LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${BASE_REPO_ROOT}:${PYTHONPATH:-}"

for port in ${SEARCH_QA_RETRIEVER_PORTS}; do
  if ss -ltn "( sport = :${port} )" | grep -q ":${port}"; then
    echo "Retriever port ${port} already listening; skip start"
    continue
  fi
  (
    cd "${BASE_REPO_ROOT}"
    nohup python3 examples/search/retriever/retrieval_server.py \
      --index_path "${INDEX_PATH}" \
      --corpus_path "${CORPUS_PATH}" \
      --topk "${SEARCH_QA_RETRIEVER_TOPK}" \
      --retriever_name "${SEARCH_QA_RETRIEVER_MODEL_NAME}" \
      --port "${port}" \
      > "${SEARCH_QA_RETRIEVER_LOG_DIR}/retriever_${port}.log" 2>&1 &
    echo $! > "${SEARCH_QA_RETRIEVER_LOG_DIR}/retriever_${port}.pid"
  )
done

deadline=$((SECONDS + SEARCH_QA_RETRIEVER_WAIT_SECONDS))
for port in ${SEARCH_QA_RETRIEVER_PORTS}; do
  until curl -fsS "http://127.0.0.1:${port}/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["health check"],"topk":1}' >/dev/null 2>&1; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      echo "Retriever pool did not become ready on port ${port}" >&2
      exit 1
    fi
    sleep 2
  done
  echo "Retriever ready: ${port}"
done
