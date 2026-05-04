#!/usr/bin/env bash
# Stage S1 — LiveStore sidecar (slot 5.4).
#
# Same-machine gRPC service over Unix domain socket. The trainer container
# mounts the socket via -v ${LIVE_STORE_SOCKET_DIR}:/tmp/live_store on the
# Docker run line; the producer (still in-process at S1) and the trainer
# (always in-process here) connect via LiveStoreClient.
#
# Run order: this launcher must be up BEFORE the trainer container starts.
# scripts/_internal/s3_fullasync_docker.sh sources LIVE_STORE_SOCKET so the
# container sees the same path.

set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f /home/ubuntu/.prorl_creds.env ]; then
    # shellcheck disable=SC1091
    source /home/ubuntu/.prorl_creds.env
fi

export LIVE_STORE_SOCKET="${LIVE_STORE_SOCKET:-/tmp/prorl_live_store.sock}"
export LIVE_STORE_MAX_SIZE="${LIVE_STORE_MAX_SIZE:-256}"
export LIVE_STORE_STALENESS_K="${LIVE_STORE_STALENESS_K:-4}"
export LIVE_STORE_NO_PROGRESS_S="${LIVE_STORE_NO_PROGRESS_S:-1800}"
export LIVE_STORE_MAX_WORKERS="${LIVE_STORE_MAX_WORKERS:-16}"
export LIVE_STORE_LOG_LEVEL="${LIVE_STORE_LOG_LEVEL:-INFO}"

LOGFILE="${LIVE_STORE_LOGFILE:-/tmp/live_store.log}"

echo "[s0_5_live_store] starting LiveStore on unix:${LIVE_STORE_SOCKET}"
echo "[s0_5_live_store]   max_size=${LIVE_STORE_MAX_SIZE} k=${LIVE_STORE_STALENESS_K}"
echo "[s0_5_live_store]   no_progress_s=${LIVE_STORE_NO_PROGRESS_S} log=${LOGFILE}"

exec poetry run python -m live_store.main \
    --socket "${LIVE_STORE_SOCKET}" \
    --max-size "${LIVE_STORE_MAX_SIZE}" \
    --staleness-cutoff-k "${LIVE_STORE_STALENESS_K}" \
    --no-progress-timeout-s "${LIVE_STORE_NO_PROGRESS_S}" \
    --max-workers "${LIVE_STORE_MAX_WORKERS}" \
    2>&1 | tee "${LOGFILE}"
