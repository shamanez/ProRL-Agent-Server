#!/bin/bash
# Start the ProRL FastAPI server on :8006.
set -eo pipefail
source /home/ubuntu/.prorl_creds.env
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server

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

poetry run python scripts/start_server.py \
  --host 0.0.0.0 --port 8006 \
  --max-init-workers 64 --max-run-workers 64 --timeout 1200 \
  "${VLLM_ADDR_ARGS[@]}" \
  2>&1 | tee /tmp/s0-prorl.log
