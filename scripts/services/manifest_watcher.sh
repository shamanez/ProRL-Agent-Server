#!/usr/bin/env bash
# Manifest watcher — host-side bridge for the manifest-write fix.
#
# The running Docker trainer has old code that does NOT write the policy
# manifest after LoRA publish. This script watches /tmp/trainer.log for
# "publish_lora_adapter" events and writes the manifest itself, so the
# RolloutWorker's FilePollingPolicySubscription can pick up the new version
# within 1 second.
#
# Run once on the HOST (not inside Docker):
#   bash scripts/services/manifest_watcher.sh &
#
# Safe to run alongside a trainer that already has the manifest-write fix —
# it will attempt a duplicate write, but write_manifest uses atomic rename
# so there's no corruption.
#
# Stop: kill $! or Ctrl-C.
set -euo pipefail

TRAINER_LOG="${1:-/tmp/trainer.log}"
MANIFEST_PATH="${POLICY_MANIFEST_PATH:-/tmp/prorl_policy_manifest.json}"
REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
PYTHON=/home/ubuntu/.cache/pypoetry/virtualenvs/openhands-ai-342rfuwh-py3.12/bin/python

source /home/ubuntu/.prorl_creds.env 2>/dev/null || true

echo "[manifest_watcher] watching ${TRAINER_LOG} → will write ${MANIFEST_PATH}"

LAST_VERSION=0

tail -f "${TRAINER_LOG}" 2>/dev/null | while IFS= read -r line; do
  # Trainer logs JSON: {"event":"publish_lora_adapter","policy_version":N,"adapter_bytes":...}
  if echo "$line" | grep -q '"event":"publish_lora_adapter"'; then
    VERSION=$(echo "$line" | python3 -c "import sys,json,re; m=re.search(r'\{.*\}', sys.stdin.read()); d=json.loads(m.group()) if m else {}; print(d.get('policy_version',0))" 2>/dev/null || echo "0")
    if [[ "$VERSION" -gt "$LAST_VERSION" ]] 2>/dev/null; then
      LAST_VERSION="$VERSION"
      # Find the adapter directory from the checkpoint path
      # Pattern: outputs/ProAgent/fullasync/global_step_N/actor/lora_adapter
      STEP=$(( VERSION ))
      ADAPTER_DIR="${REPO}/outputs/ProAgent/fullasync/global_step_${STEP}/actor/lora_adapter"
      # Fallback: scan for latest checkpoint
      if [[ ! -d "${ADAPTER_DIR}" ]]; then
        ADAPTER_DIR=$(find "${REPO}/outputs" -name "lora_adapter" -type d 2>/dev/null | sort | tail -1 || echo "")
      fi
      echo "[manifest_watcher] publish_lora_adapter detected: version=${VERSION} adapter=${ADAPTER_DIR}"
      PYTHONPATH=${REPO} ${PYTHON} -c "
import sys, time
sys.path.insert(0, '${REPO}')
from policy_registry.file_registry import PolicyManifest, write_manifest
write_manifest(
  PolicyManifest(
    policy_id='${POLICY_ID:-qwen3-4b-skyrl}',
    version=${VERSION},
    adapter_uri='file://${ADAPTER_DIR}',
    trainer_id='trainer-0',
    published_at=time.time(),
  ),
  path='${MANIFEST_PATH}',
)
print('[manifest_watcher] manifest written: version=${VERSION}')
" 2>&1 || echo "[manifest_watcher] manifest write failed for version=${VERSION}"
    fi
  fi
done

echo "[manifest_watcher] trainer log closed — exiting"
