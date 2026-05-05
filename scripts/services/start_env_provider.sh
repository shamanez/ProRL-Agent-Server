#!/usr/bin/env bash
# Start the EnvironmentProvider (ProRL — slot 5.1, frozen through S4).
# Step 2 — no upstream service dependencies.
#
# This is a thin wrapper around the existing s0_prorl.sh launcher.
# ProRL is FROZEN (openhands/nvidia/async_server.py unchanged through S4).
# The rollout manager calls it via HTTP POST /process — no OpenHands imports
# in the worker (BC-13).
#
# Health: GET :${PRORL_PORT}/health → 200
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PRORL_PORT="${PRORL_PORT:-8006}"
PRORL_HOST="${PRORL_HOST:-0.0.0.0}"

echo "[env_provider] starting ProRL on ${PRORL_HOST}:${PRORL_PORT}"

exec bash "${REPO_ROOT}/scripts/_internal/s0_prorl.sh"
