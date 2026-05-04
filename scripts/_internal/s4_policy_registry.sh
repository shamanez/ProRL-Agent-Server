#!/usr/bin/env bash
# Stage S4 — PolicyRegistry (slot 5.7).
#
# Single source of truth for the active policy version. Owns the §3.3
# abort-gate fanout to the vLLM pool and the gRPC streaming subscription
# the rollout worker consumes. Replaces the trainer-direct pool fanout
# and the S2 JSON-manifest scaffolding.
#
# Run order (post-S4):
#   1. scripts/_internal/s0_5_live_store.sh        (LiveStore sidecar)
#   2. scripts/_internal/s3_replay_archive.sh      (in-process; no separate proc)
#   3. scripts/_internal/s4_policy_registry.sh     (this — host venv)
#   4. scripts/_internal/s2_rollout_worker.sh      (worker — host venv)
#   5. scripts/_internal/s3_fullasync_docker.sh    (trainer container)

set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f /home/ubuntu/.prorl_creds.env ]; then
    # shellcheck disable=SC1091
    source /home/ubuntu/.prorl_creds.env
fi

export POLICY_REGISTRY_SOCKET="${POLICY_REGISTRY_SOCKET:-/tmp/prorl_policy_registry.sock}"
export POLICY_REGISTRY_DB="${POLICY_REGISTRY_DB:-/tmp/prorl_policy_registry.db}"
export POLICY_REGISTRY_LOG_LEVEL="${POLICY_REGISTRY_LOG_LEVEL:-INFO}"
# Comma-separated; one per pool child:
#   POOL_ENDPOINTS="http://vllm-instance:8100,http://vllm-instance:8101,..."
: "${POOL_ENDPOINTS:?POOL_ENDPOINTS required (comma-separated http URLs of vLLM children)}"

LOGFILE="${POLICY_REGISTRY_LOGFILE:-/tmp/policy_registry.log}"

echo "[s4_policy_registry] starting PolicyRegistry on unix:${POLICY_REGISTRY_SOCKET}"
echo "[s4_policy_registry]   db=${POLICY_REGISTRY_DB}"
echo "[s4_policy_registry]   pool=${POOL_ENDPOINTS}"
echo "[s4_policy_registry]   log=${LOGFILE}"

exec poetry run python -m policy_registry.main \
    --socket "${POLICY_REGISTRY_SOCKET}" \
    --db "${POLICY_REGISTRY_DB}" \
    --pool-endpoints "${POOL_ENDPOINTS}" \
    2>&1 | tee "${LOGFILE}"
