#!/bin/bash
# TrainerAdapter launcher (slot 5.6) — contract-first rollout fabric.
#
# Starts the VERL FSDP trainer inside Docker. The trainer's only two
# external connections are:
#   - LiveStore      (get_batch via gRPC UDS — reads training groups)
#   - PolicyRegistry (publish_policy_version via gRPC UDS — pushes LoRA)
#
# All rollout generation, dataset ownership, ProRL, and vLLM routing
# belong to the other services. Start them first via start_all.sh.
#
# Usage:
#   bash trainers/verl/scripts/start.sh [extra hydra overrides]
#
# Key knobs (override via env vars):
#   BATCH_SIZE            4       groups drawn per training step  (32 for prod)
#   TOTAL_TRAINING_STEPS  500     stop after N steps
#   SAVE_FREQ             1       publish LoRA every N steps
#   USE_TEMPORAL_IS       False   IS correction for stale replay groups
#   SWAP_PROTOCOL         pinning vLLM LoRA swap: pinning or quiesce
set -eo pipefail
source /home/ubuntu/.prorl_creds.env

REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server

# ── VERL source tree ────────────────────────────────────────────────────────
# Pinned commit: a4351480871347092436d17573ad3ccf75b24122
# Branch: main @ github.com/verl-project/verl
# See trainers/verl/VERL_PIN.md for rationale.
VERL_COMMIT="a4351480871347092436d17573ad3ccf75b24122"
VERL_DIR="${VERL_DIR:-/tmp/verl}"
if [[ ! -d "${VERL_DIR}/.git" ]]; then
  echo "[trainer] /tmp/verl missing — cloning VERL @ ${VERL_COMMIT:0:8} ..."
  git clone --depth=1 https://github.com/verl-project/verl "${VERL_DIR}" 2>&1
  git -C "${VERL_DIR}" fetch --depth=1 origin "${VERL_COMMIT}" 2>&1
  git -C "${VERL_DIR}" checkout FETCH_HEAD 2>&1
  echo "[trainer] VERL cloned to ${VERL_DIR}"
else
  actual=$(git -C "${VERL_DIR}" rev-parse HEAD 2>/dev/null || echo "unknown")
  if [[ "${actual}" != "${VERL_COMMIT}" ]]; then
    echo "[trainer] WARNING: ${VERL_DIR} is at ${actual:0:8}, expected ${VERL_COMMIT:0:8}"
    echo "[trainer] Remove ${VERL_DIR} and re-run to pull the pinned commit."
  else
    echo "[trainer] VERL at pinned commit ${actual:0:8} ✓"
  fi
fi
# ───────────────────────────────────────────────────────────────────────────

# Build once: cd ProRL-Agent-Server && docker build -f trainers/verl/Dockerfile -t prorl/verl-trainer:vllm018 .
IMG="${TRAINER_IMG:-prorl/verl-trainer:vllm018}"
CNAME=prorl-trainer
REMOTE_DNS="${REMOTE_DNS:-ec2-3-87-168-160.compute-1.amazonaws.com}"

# Training knobs
BATCH_SIZE="${BATCH_SIZE:-4}"
STALENESS_CUTOFF_K="${STALENESS_CUTOFF_K:-4}"
REPLAY_ENABLE="${REPLAY_ENABLE:-True}"
BUFFER_SIZE="${BUFFER_SIZE:-64}"
USE_TEMPORAL_IS="${USE_TEMPORAL_IS:-False}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1000}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
TEST_FREQ="${TEST_FREQ:--1}"
SWAP_PROTOCOL="${SWAP_PROTOCOL:-pinning}"
LOG_PATH="${LOG_PATH:-/tmp/s3-fullasync.log}"

echo "[trainer] starting $(date -u +%FT%TZ)"
echo "[trainer] image=$IMG  steps=$TOTAL_TRAINING_STEPS  save_freq=$SAVE_FREQ  swap=$SWAP_PROTOCOL"

docker rm -f "$CNAME" >/dev/null 2>&1 || true

docker run --rm --name "$CNAME" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$REPO":/workspace \
  -v /tmp/verl:/opt/verl \
  -v /tmp:/tmp \
  -v /home/ubuntu/data:/data:ro \
  -v /home/ubuntu/.prorl_creds.env:/creds.env:ro \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  -e WANDB_API_KEY \
  -e HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN \
  -e PYTHONPATH=/workspace/core \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e REMOTE_DNS="$REMOTE_DNS" \
  -e TOTAL_EPOCHS="$TOTAL_EPOCHS" \
  -e TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
  -e SAVE_FREQ="$SAVE_FREQ" \
  -e TEST_FREQ="$TEST_FREQ" \
  -e VAL_BEFORE_TRAIN="$VAL_BEFORE_TRAIN" \
  -e BATCH_SIZE="$BATCH_SIZE" \
  -e STALENESS_CUTOFF_K="$STALENESS_CUTOFF_K" \
  -e REPLAY_ENABLE="$REPLAY_ENABLE" \
  -e BUFFER_SIZE="$BUFFER_SIZE" \
  -e USE_TEMPORAL_IS="$USE_TEMPORAL_IS" \
  -e SWAP_PROTOCOL="$SWAP_PROTOCOL" \
  -e REPO_HOST_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" \
  -e CKPT_PATH="/workspace/outputs" \
  -e LIVE_STORE_SOCKET="/tmp/prorl_live_store.sock" \
  -e POLICY_REGISTRY_SOCKET="/tmp/prorl_policy_registry.sock" \
  -e RAY_memory_usage_threshold=0.98 \
  -e RAY_memory_monitor_refresh_ms=250 \
  -e RAY_object_store_memory=21474836480 \
  "$IMG" \
  bash -c '
    set -eo pipefail
    source /creds.env

    # Install upstream VERL and our custom trainer package.
    # Extra pip deps (scipy, peft, etc.) are baked into the Docker image.
    pip install --no-deps -e /opt/verl >/dev/null 2>&1
    pip install --no-deps -e /workspace/trainers/verl >/dev/null 2>&1

    # Pre-flight: vLLM pool must be healthy before training starts (BC-15).
    HEALTH_TIMEOUT=300
    for port in 8100 8101 8102 8103; do
      url="http://${REMOTE_DNS}:${port}/health"
      deadline=$((SECONDS + HEALTH_TIMEOUT))
      until curl -sf --max-time 5 "${url}" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
          echo "[verl-trainer] ERROR: ${url} unreachable — start vLLM pool first" >&2; exit 1
        fi
        sleep 2
      done
      echo "[verl-trainer] vLLM :${port} healthy"
    done

    cd /workspace
    bash trainers/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh \
      trainer.total_epochs="$TOTAL_EPOCHS" \
      ++trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
      trainer.save_freq="$SAVE_FREQ" \
      trainer.resume_mode=auto \
      trainer.val_before_train="$VAL_BEFORE_TRAIN" \
      trainer.test_freq="$TEST_FREQ" \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=49152 \
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152 \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=49152 \
      +actor_rollout_ref.actor.calculate_entropy=false \
      actor_rollout_ref.actor.entropy_checkpointing=true \
      actor_rollout_ref.model.use_fused_kernels=True \
      +actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
      +actor_rollout_ref.actor.use_fused_kernels=True \
      +actor_rollout_ref.actor.use_remove_padding=True \
      trainer.default_local_dir=/workspace/outputs/ProAgent/fullasync \
      ++actor_rollout_ref.rollout.custom.rollout_save_dir=/workspace/outputs/rollout_data_fullasync \
      replay.staleness_cutoff_k="$STALENESS_CUTOFF_K" \
      replay.use_temporal_is="$USE_TEMPORAL_IS" \
      ++replay.live_store_socket="${LIVE_STORE_SOCKET:-/tmp/prorl_live_store.sock}" \
      ++replay.policy_registry_socket="${POLICY_REGISTRY_SOCKET:-/tmp/prorl_policy_registry.sock}" \
      ++replay.no_progress_timeout_s=5400 \
      ++replay.swap_protocol="$SWAP_PROTOCOL" \
      "$@"
  ' _ "$@" 2>&1 | tee "$LOG_PATH"
