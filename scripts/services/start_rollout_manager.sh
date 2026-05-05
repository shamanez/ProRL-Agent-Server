#!/usr/bin/env bash
# Start the RolloutManager service (slot 5.3 — S2).
# Step 4 — depends on InferenceBackend (step 1), EnvironmentProvider (step 2),
# LiveStore (step 3a), and PolicyRegistry (step 3b).
#
# BC-13: zero VERL/OpenHands imports inside the worker process.
# BC-14: worker owns the dataset via --data-files (NOT the trainer).
# BC-0:  one PolicyVersionSnapshot read per group dispatch — all N siblings
#        stamped with the same behavior_policy_version.
#
# Env vars:
#   LIVE_STORE_SOCKET      (default /tmp/prorl_live_store.sock)
#   PRORL_URL              (default http://localhost:8006)
#   POLICY_ID              (default qwen3-4b-skyrl)
#   ENVIRONMENT_ID         (default prorl_default)
#   DATA_FILES             space-separated parquet paths (REQUIRED)
#   GROUP_SIZE             GRPO/DAPO group size n (default 16)
#   POLICY_MANIFEST_PATH   (default /tmp/prorl_policy_manifest.json)
#   REPLAY_ARCHIVE_ROOT    (default /home/ubuntu/replay_archive)
#   REPLAY_ARCHIVE_DISABLED 1 to skip archive tee (default 0)
#   FILTER_ZERO_VARIANCE   1 to filter zero-variance groups (default 1)
set -euo pipefail

if [[ -z "${DATA_FILES:-}" ]]; then
    echo "[rollout_manager] ERROR: DATA_FILES env var is required (BC-14 — worker owns dataset)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[rollout_manager] starting"
echo "[rollout_manager] DATA_FILES=${DATA_FILES}"
echo "[rollout_manager] PRORL_URL=${PRORL_URL:-http://localhost:8006}"
echo "[rollout_manager] GROUP_SIZE=${GROUP_SIZE:-16} (n siblings per GRPO group)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ROLLOUT_FABRIC_PYTHON: override to use a fabric-only venv (see docs/service-envs.md).
_DEFAULT_PYTHON="/home/ubuntu/.cache/pypoetry/virtualenvs/openhands-ai-342rfuwh-py3.12/bin/python"
PYTHON="${ROLLOUT_FABRIC_PYTHON:-${POETRY_PYTHON:-${_DEFAULT_PYTHON}}}"

# Convert DATA_FILES to --data-files args
DATA_FILES_ARGS=()
for f in ${DATA_FILES}; do
    DATA_FILES_ARGS+=("$f")
done

exec "${PYTHON}" -m rollout_manager.main \
    --live-store-socket "${LIVE_STORE_SOCKET:-/tmp/prorl_live_store.sock}" \
    --prorl-url "${PRORL_URL:-http://localhost:8006}" \
    --policy-id "${POLICY_ID:-qwen3-4b-skyrl}" \
    --environment-id "${ENVIRONMENT_ID:-prorl_default}" \
    --data-files "${DATA_FILES_ARGS[@]}" \
    --group-size "${GROUP_SIZE:-16}" \
    --policy-manifest-path "${POLICY_MANIFEST_PATH:-/tmp/prorl_policy_manifest.json}" \
    --archive-root "${REPLAY_ARCHIVE_ROOT:-/home/ubuntu/replay_archive}" \
    --archive-dead-letter "${REPLAY_ARCHIVE_DEAD_LETTER:-/tmp/replay_archive_deadletter.jsonl}" \
    ${REPLAY_ARCHIVE_DISABLED:+--archive-disabled} \
    ${FILTER_ZERO_VARIANCE:+--filter-zero-variance}
