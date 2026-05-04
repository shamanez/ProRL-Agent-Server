#!/usr/bin/env bash
# Start the PolicyRegistry gRPC service (slot 5.7 — S4).
# Step 3b — can start in parallel with start_live_store.sh.
#
# Env vars:
#   POLICY_REGISTRY_SOCKET   (default /tmp/prorl_policy_registry.sock)
#   POLICY_REGISTRY_DB       (default /tmp/prorl_policy_registry.db)
#   VLLM_POOL_ENDPOINTS      space-separated list, e.g.
#                            "http://vllm-instance:8100 http://vllm-instance:8101"
set -euo pipefail

SOCKET="${POLICY_REGISTRY_SOCKET:-/tmp/prorl_policy_registry.sock}"
DB="${POLICY_REGISTRY_DB:-/tmp/prorl_policy_registry.db}"
ENDPOINTS="${VLLM_POOL_ENDPOINTS:-http://vllm-instance:8100 http://vllm-instance:8101 http://vllm-instance:8102 http://vllm-instance:8103}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Convert space-separated endpoints to Python list literal
ENDPOINTS_PY="[$(echo "${ENDPOINTS}" | tr ' ' '\n' | sed "s/.*/'&'/" | paste -sd,)]"

echo "[policy_registry] starting on unix:${SOCKET} db=${DB}"
echo "[policy_registry] pool endpoints: ${ENDPOINTS}"

exec /home/ubuntu/.cache/pypoetry/virtualenvs/openhands-ai-342rfuwh-py3.12/bin/python -c "
import logging, os, signal, sys
logging.basicConfig(level=os.environ.get('LOG_LEVEL','INFO'),
    format='%(asctime)s %(levelname)s policy_registry: %(message)s')
sys.path.insert(0, '${REPO_ROOT}')
from policy_registry.server import serve
server = serve(
    socket_path='${SOCKET}',
    db_path='${DB}',
    pool_endpoints=${ENDPOINTS_PY},
)
print('[policy_registry] healthy unix:${SOCKET}', flush=True)
def _stop(s, f):
    server.stop(grace=5.0)
    sys.exit(0)
signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)
server.wait_for_termination()
"
