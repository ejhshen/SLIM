#!/usr/bin/env bash
set -euo pipefail
SEARCH_QA_RETRIEVER_PORTS="${SEARCH_QA_RETRIEVER_PORTS:-8000 8001 8002 8003}"
for port in ${SEARCH_QA_RETRIEVER_PORTS}; do
  pids="$(ss -ltnp "( sport = :${port} )" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)"
  if [ -n "${pids}" ]; then
    echo "Stopping retriever on port ${port}: ${pids}"
    kill ${pids} 2>/dev/null || true
  fi
done
