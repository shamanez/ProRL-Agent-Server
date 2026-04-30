#!/bin/bash
# Start the ProRL FastAPI server on :8006.
set -eo pipefail
source /home/ubuntu/.prorl_creds.env
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server

echo "[prorl] starting $(date -u +%FT%TZ)"
poetry run python scripts/start_server.py \
  --host 0.0.0.0 --port 8006 \
  --max-init-workers 64 --max-run-workers 64 --timeout 1200 \
  2>&1 | tee /tmp/s0-prorl.log
