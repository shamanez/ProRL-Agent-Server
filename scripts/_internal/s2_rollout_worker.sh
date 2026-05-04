#!/usr/bin/env bash
# Stage S2 — RolloutWorker (slot 5.3).
#
# Owns the dataloader (§3.8 — data ownership), the agent loop (calls
# ProRL :8006), the DAPO eager-push seam (§3.7), and the producer-side
# filters. Subscribes to the trainer's policy manifest at 1 Hz.
#
# Run order (post-S2):
#   1. scripts/_internal/s0_5_live_store.sh        (LiveStore sidecar)
#   2. scripts/_internal/s2_rollout_worker.sh      (this — host venv)
#   3. scripts/_internal/s3_fullasync_docker.sh    (trainer container)
#
# The trainer never imports openhands / aiohttp / fastapi / uvicorn
# after S2.

set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f /home/ubuntu/.prorl_creds.env ]; then
    # shellcheck disable=SC1091
    source /home/ubuntu/.prorl_creds.env
fi

export LIVE_STORE_SOCKET="${LIVE_STORE_SOCKET:-/tmp/prorl_live_store.sock}"
export POLICY_MANIFEST_PATH="${POLICY_MANIFEST_PATH:-/tmp/prorl_policy_manifest.json}"
export POLICY_ID="${POLICY_ID:-qwen3-4b-skyrl}"
export ENVIRONMENT_ID="${ENVIRONMENT_ID:-prorl_default}"
export POLICY_POLL_INTERVAL_S="${POLICY_POLL_INTERVAL_S:-1.0}"
export ROLLOUT_WORKER_LOG_LEVEL="${ROLLOUT_WORKER_LOG_LEVEL:-INFO}"

LOGFILE="${ROLLOUT_WORKER_LOGFILE:-/tmp/rollout_worker.log}"

echo "[s2_rollout_worker] starting RolloutWorker"
echo "[s2_rollout_worker]   live_store=${LIVE_STORE_SOCKET}"
echo "[s2_rollout_worker]   manifest=${POLICY_MANIFEST_PATH}"
echo "[s2_rollout_worker]   policy=${POLICY_ID} env=${ENVIRONMENT_ID}"
echo "[s2_rollout_worker]   log=${LOGFILE}"

exec poetry run python -m rollout_worker.main \
    --live-store-socket "${LIVE_STORE_SOCKET}" \
    --policy-manifest-path "${POLICY_MANIFEST_PATH}" \
    --policy-id "${POLICY_ID}" \
    --environment-id "${ENVIRONMENT_ID}" \
    --policy-poll-interval-s "${POLICY_POLL_INTERVAL_S}" \
    2>&1 | tee "${LOGFILE}"
