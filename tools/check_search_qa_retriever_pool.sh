#!/usr/bin/env bash
set -euo pipefail
SEARCH_QA_RETRIEVER_PORTS="${SEARCH_QA_RETRIEVER_PORTS:-8000 8001 8002 8003}"
for port in ${SEARCH_QA_RETRIEVER_PORTS}; do
  echo "=== port ${port} ==="
  ss -ltnp "( sport = :${port} )" || true
  curl -fsS "http://127.0.0.1:${port}/retrieve" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["health check"],"topk":1}' >/dev/null && echo "ready" || echo "not ready"
done
