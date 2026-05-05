#!/usr/bin/env bash
# ============================================================
#  Rollout Fabric — full startup orchestrator
#  Starts all services in strict dependency order.
#  Each step health-probes before the next begins (Sec.0.1).
#
#  Startup sequence:
#    Step 1  InferenceBackend  (vLLM pool :8100-8103)
#    Step 2  EnvironmentProvider (ProRL :8006)
#    Step 3a LiveStore (gRPC UDS)            ─┐ parallel
#    Step 3b PolicyRegistry (gRPC UDS)       ─┘
#    Step 3c ReplayArchive (HTTP :8080, opt)
#    Step 4  RolloutManager
#    Step 5  TrainerAdapter (Docker container)
#
#  Stop in reverse. Trainer before worker; worker before store+registry.
#
#  Required env vars:
#    DATA_FILES          space-separated parquet paths for the worker
#    VLLM_POOL_ENDPOINTS space-separated pool URLs
#
#  Optional (have defaults):
#    PRORL_PORT, LIVE_STORE_SOCKET, POLICY_REGISTRY_SOCKET, etc.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---- health probe helpers -----------------------------------------------

probe_http() {
    local url="$1" retries="${2:-60}" interval="${3:-2}"
    echo "  probing ${url} ..."
    for ((i=1; i<=retries; i++)); do
        if curl -sf --max-time 3 "${url}" > /dev/null 2>&1; then
            echo "  ✓ ${url} healthy (attempt ${i})"
            return 0
        fi
        sleep "${interval}"
    done
    echo "  ✗ ${url} did not become healthy after $((retries * interval))s"
    return 1
}

probe_socket() {
    local path="$1" retries="${2:-30}" interval="${3:-1}"
    echo "  probing unix:${path} ..."
    for ((i=1; i<=retries; i++)); do
        if [[ -S "${path}" ]]; then
            echo "  ✓ unix:${path} exists (attempt ${i})"
            return 0
        fi
        sleep "${interval}"
    done
    echo "  ✗ unix:${path} did not appear after $((retries * interval))s"
    return 1
}

# ---- pid tracking --------------------------------------------------------

declare -a PIDS=()
declare -a NAMES=()

start_bg() {
    local name="$1"; shift
    "$@" &
    local pid=$!
    PIDS+=("$pid")
    NAMES+=("$name")
    echo "[start_all] ${name} started (pid=${pid})"
}

cleanup() {
    echo ""
    echo "[start_all] shutting down in reverse order ..."
    for ((i=${#PIDS[@]}-1; i>=0; i--)); do
        local pid="${PIDS[$i]}" name="${NAMES[$i]}"
        if kill -0 "${pid}" 2>/dev/null; then
            echo "  stopping ${name} (pid=${pid})"
            kill -TERM "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
    echo "[start_all] shutdown complete"
}
trap cleanup EXIT INT TERM

# =========================================================================
# Step 1 — InferenceBackend (vLLM pool)
# =========================================================================
echo ""
echo "=== Step 1: InferenceBackend (vLLM pool) ==="
bash "${REPO_ROOT}/inference/vllm/scripts/launch_remote_vllm_pool.sh" start

VLLM_BASE="${VLLM_BASE_URL:-http://${REMOTE_DNS:-vllm-instance}}"
for port in 8100 8101 8102 8103; do
    probe_http "${VLLM_BASE}:${port}/health"
done
echo "Step 1 complete."

# =========================================================================
# Step 2 — EnvironmentProvider (ProRL)
# =========================================================================
echo ""
echo "=== Step 2: EnvironmentProvider (ProRL) ==="
start_bg "env_provider" bash "${SCRIPT_DIR}/start_env_provider.sh"
probe_http "http://localhost:${PRORL_PORT:-8006}/health" 60 2
# Activate the agent server (CLAUDE.md Step 2 — POST /start must follow health check).
curl -sf -X POST "http://localhost:${PRORL_PORT:-8006}/start" \
     -H "Content-Type: application/json" -d '{}' \
  || { echo "ERROR: ProRL /start failed"; exit 1; }
probe_http "http://localhost:${PRORL_PORT:-8006}/status" 30 1
echo "Step 2 complete."

# =========================================================================
# Steps 3a + 3b — LiveStore and PolicyRegistry (parallel)
# =========================================================================
echo ""
echo "=== Steps 3a + 3b: LiveStore + PolicyRegistry (parallel) ==="

start_bg "live_store" bash "${SCRIPT_DIR}/start_live_store.sh"
start_bg "policy_registry" bash "${SCRIPT_DIR}/start_policy_registry.sh"

probe_socket "${LIVE_STORE_SOCKET:-/tmp/prorl_live_store.sock}" 30 1
probe_socket "${POLICY_REGISTRY_SOCKET:-/tmp/prorl_policy_registry.sock}" 30 1
echo "Steps 3a + 3b complete."

# =========================================================================
# Step 3c — ReplayArchive (optional)
# =========================================================================
REPLAY_ARCHIVE_DISABLED="${REPLAY_ARCHIVE_DISABLED:-0}"
if [[ "${REPLAY_ARCHIVE_DISABLED}" != "1" ]]; then
    echo ""
    echo "=== Step 3c: ReplayArchive ==="
    start_bg "replay_archive" bash "${SCRIPT_DIR}/start_replay_archive.sh"
    probe_http "http://localhost:${REPLAY_ARCHIVE_PORT:-8080}/health" 30 1
    echo "Step 3c complete."
else
    echo "Step 3c: ReplayArchive disabled (REPLAY_ARCHIVE_DISABLED=1)"
fi

# =========================================================================
# Step 4 — RolloutManager
# Depends on steps 1, 2, 3a, 3b.
# BC-14: DATA_FILES must be set — worker owns the dataset, NOT the trainer.
# BC-16: worker fills the live store before trainer starts.
#        Start worker first and wait for it to push at least 1 group.
# =========================================================================
echo ""
echo "=== Step 4: RolloutManager ==="

if [[ -z "${DATA_FILES:-}" ]]; then
    echo "ERROR: DATA_FILES must be set (BC-14 — worker owns the dataset, not the trainer)"
    exit 1
fi

start_bg "rollout_manager" bash "${SCRIPT_DIR}/start_rollout_manager.sh"

# The worker registers with the LiveStore; wait for it to push ≥ 1 group.
# This is the BC-16 warm-up gate: trainer must not start until buffer has data.
echo "  waiting for rollout manager to push ≥ 1 group (BC-16 warm-up) ..."
WARMUP_TIMEOUT="${WORKER_WARMUP_TIMEOUT_S:-300}"
# Use ROLLOUT_FABRIC_PYTHON if set; otherwise the default pre-populated env.
_DEFAULT_PYTHON="$(cd "${REPO_ROOT}/core" && poetry env info --path 2>/dev/null)/bin/python"
WARMUP_PYTHON="${ROLLOUT_FABRIC_PYTHON:-${POETRY_PYTHON:-${_DEFAULT_PYTHON}}}"
"${WARMUP_PYTHON}" - <<PYEOF
import sys, time
sys.path.insert(0, '${REPO_ROOT}/core')
from rollout_fabric.live_store.client import LiveStoreClient
cli = LiveStoreClient(
    '${LIVE_STORE_SOCKET:-/tmp/prorl_live_store.sock}',
    policy_id='${POLICY_ID:-qwen3-4b-skyrl}',
    environment_id='${ENVIRONMENT_ID:-prorl_default}',
)
deadline = time.monotonic() + ${WARMUP_TIMEOUT}
while time.monotonic() < deadline:
    n = cli.num_groups()
    if n >= 1:
        print(f'  ✓ live store has {n} group(s) — worker is producing')
        cli.close()
        sys.exit(0)
    time.sleep(5)
print(f'  ✗ live store still empty after ${WARMUP_TIMEOUT}s — worker may be wedged')
cli.close()
sys.exit(1)
PYEOF

echo "Step 4 complete."

# =========================================================================
# Step 5 — TrainerAdapter
# Depends on steps 3a (LiveStore) + 3b (PolicyRegistry) + 4 (worker pushing).
# BC-15: trainer connects ONLY to LiveStore + PolicyRegistry.
# =========================================================================
echo ""
echo "=== Step 5: TrainerAdapter ==="
start_bg "trainer" bash "${REPO_ROOT}/trainers/verl/scripts/start.sh"
echo "Step 5 started. Trainer will call get_batch and block until N groups available."
echo "Step 5 complete."

# =========================================================================
# All services running — wait for any to exit
# =========================================================================
echo ""
echo "[start_all] All services running. PIDs: ${PIDS[*]}"
echo "[start_all] Press Ctrl-C to shutdown in reverse order."
echo ""

wait -n "${PIDS[@]}" 2>/dev/null || true
echo "[start_all] A service exited. Running cleanup."
