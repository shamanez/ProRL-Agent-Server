#!/bin/bash
# Stage 0 prerequisite: pull the SkyRL-v0-293 parquet dataset.
set -eo pipefail
source /home/ubuntu/.prorl_creds.env
echo "[pull-data] starting $(date -u +%FT%TZ)"
/opt/pytorch/bin/huggingface-cli download \
  NovaSky-AI/SkyRL-v0-293-data \
  --repo-type dataset \
  --local-dir /home/ubuntu/data/SkyRL-v0-293
echo "[pull-data] listing result:"
ls -la /home/ubuntu/data/SkyRL-v0-293/
echo "[pull-data] done $(date -u +%FT%TZ)"
