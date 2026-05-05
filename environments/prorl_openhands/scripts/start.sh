#!/bin/bash
# Start the ProRL FastAPI server on :8006.
set -eo pipefail
source /home/ubuntu/.prorl_creds.env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

echo "[prorl] starting $(date -u +%FT%TZ)"

# Build --llm-server-address args from REMOTE_DNS so vLLM endpoints survive
# any future ProRL restart without needing POST /add_llm_server again.
VLLM_ADDR_ARGS=()
if [[ -n "${REMOTE_DNS:-}" ]]; then
  # --llm-server-address uses nargs='*' so pass ALL ports as ONE flag invocation.
  VLLM_ADDR_ARGS=(--llm-server-address \
    "http://${REMOTE_DNS}:8100" \
    "http://${REMOTE_DNS}:8101" \
    "http://${REMOTE_DNS}:8102" \
    "http://${REMOTE_DNS}:8103")
  echo "[prorl] baking in vLLM endpoints: http://${REMOTE_DNS}:8100-8103"
fi

# PRORL_OPENHANDS_PYTHON: override to use a specific full env (see docs/service-envs.md).
# Requires openhands + litellm + fastapi + Singularity — never use a fabric-core venv here.
_DEFAULT_PYTHON="$(cd "${REPO_ROOT}/environments/prorl_openhands" && poetry env info --path 2>/dev/null || true)/bin/python"
PYTHON="${PRORL_OPENHANDS_PYTHON:-${POETRY_PYTHON:-${_DEFAULT_PYTHON}}}"

VENDOR_ROOT="/home/ubuntu/unextractable-agentic-rl/vendor/ProRL-Agent-Server"
PYTHONPATH="${REPO_ROOT}/environments/prorl_openhands:${REPO_ROOT}/core:${VENDOR_ROOT}:${PYTHONPATH:-}" \
"${PYTHON}" "${REPO_ROOT}/environments/prorl_openhands/scripts/start_server.py" \
  --host 0.0.0.0 --port 8006 \
  --max-init-workers 64 --max-run-workers 64 --timeout 1200 \
  "${VLLM_ADDR_ARGS[@]}" \
  2>&1 | tee /tmp/s0-prorl.log
