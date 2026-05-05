#!/bin/bash
# TrainerAdapter launcher (slot 5.6) — contract-first rollout fabric.
#
# Starts the VERL FSDP trainer inside Docker. The trainer's only two
# external connections are:
#   - LiveStore  (get_batch via gRPC UDS — reads training groups)
#   - PolicyRegistry (publish_policy_version via gRPC UDS — pushes LoRA)
#
# Everything else — rollout generation, dataset, ProRL, vLLM routing —
# is owned by separate services started before this script.
#
# Usage:
#   bash scripts/_internal/s3_fullasync_docker.sh [extra hydra overrides]
#
# Key knobs (override via env vars):
#   BATCH_SIZE            4      groups per training step (32 for prod)
#   TOTAL_TRAINING_STEPS  500    stop after this many steps
#   SAVE_FREQ             1      publish LoRA every N steps (1 = every step)
#   FILTER_GROUPS         True   drop zero-variance groups; False for bootstrapping
#   CONTINUOUS_PRODUCER   False  external RolloutWorker is the producer
#   REPLAY_ENABLE         True   read from LiveStore (must be True)
set -eo pipefail
source /home/ubuntu/.prorl_creds.env

REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
IMG=verlai/verl:vllm018.dev1
CNAME=s3-fullasync
REMOTE_DNS="${REMOTE_DNS:-ec2-3-87-168-160.compute-1.amazonaws.com}"

# Training knobs
REPLAY_ENABLE="${REPLAY_ENABLE:-True}"
BUFFER_SIZE="${BUFFER_SIZE:-256}"
STALENESS_CUTOFF_K="${STALENESS_CUTOFF_K:-4}"
USE_TEMPORAL_IS="${USE_TEMPORAL_IS:-False}"
CONTINUOUS_PRODUCER="${CONTINUOUS_PRODUCER:-False}"   # external worker; never True
FILTER_GROUPS="${FILTER_GROUPS:-True}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-1000}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
TEST_FREQ="${TEST_FREQ:--1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SWAP_PROTOCOL="${SWAP_PROTOCOL:-pinning}"
LOG_PATH="${LOG_PATH:-/tmp/s3-fullasync.log}"

echo "[trainer] starting $(date -u +%FT%TZ)"
echo "[trainer] image=$IMG  steps=$TOTAL_TRAINING_STEPS  save_freq=$SAVE_FREQ"
echo "[trainer] batch=$BATCH_SIZE  filter_groups=$FILTER_GROUPS  swap=$SWAP_PROTOCOL"

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
  -e PYTHONPATH=/workspace \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e REMOTE_DNS="$REMOTE_DNS" \
  -e TOTAL_EPOCHS="$TOTAL_EPOCHS" \
  -e TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
  -e SAVE_FREQ="$SAVE_FREQ" \
  -e TEST_FREQ="$TEST_FREQ" \
  -e VAL_BEFORE_TRAIN="$VAL_BEFORE_TRAIN" \
  -e REPLAY_ENABLE="$REPLAY_ENABLE" \
  -e BUFFER_SIZE="$BUFFER_SIZE" \
  -e STALENESS_CUTOFF_K="$STALENESS_CUTOFF_K" \
  -e USE_TEMPORAL_IS="$USE_TEMPORAL_IS" \
  -e CONTINUOUS_PRODUCER="$CONTINUOUS_PRODUCER" \
  -e FILTER_GROUPS="$FILTER_GROUPS" \
  -e BATCH_SIZE="$BATCH_SIZE" \
  -e SWAP_PROTOCOL="$SWAP_PROTOCOL" \
  -e REPO_HOST_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" \
  -e RAY_memory_usage_threshold=0.98 \
  -e RAY_memory_monitor_refresh_ms=250 \
  -e RAY_object_store_memory=21474836480 \
  "$IMG" \
  bash -c '
    set -eo pipefail
    source /creds.env

    # Install verl + verl_custom (mounted at /opt/verl and /workspace).
    pip install --no-deps -e /opt/verl >/dev/null
    pip install --no-deps -e /workspace/trainer_integration/verl >/dev/null
    pip install scipy math_verify tabulate absl-py async_generator codetiming peft 2>/dev/null || true

    # Pre-flight: vLLM pool must be healthy before training starts.
    HEALTH_TIMEOUT=300
    for port in 8100 8101 8102 8103; do
      url="http://${REMOTE_DNS}:${port}/health"
      deadline=$((SECONDS + HEALTH_TIMEOUT))
      until curl -sf --max-time 5 "${url}" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
          echo "[trainer] ERROR: ${url} unreachable after ${HEALTH_TIMEOUT}s — start vLLM pool first" >&2
          exit 1
        fi
        sleep 2
      done
      echo "[trainer] vLLM :${port} healthy"
    done

    cd /workspace
    STAGE2_OUT=/workspace/outputs/ProAgent/fullasync

    bash trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh \
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
      data.train_files=[/data/SkyRL-v0-293/train.parquet] \
      data.val_files=[/data/SkyRL-v0-293/validation.parquet] \
      trainer.default_local_dir="$STAGE2_OUT" \
      ++actor_rollout_ref.rollout.custom.rollout_save_dir=/workspace/outputs/rollout_data_fullasync \
      replay.enable="$REPLAY_ENABLE" \
      replay.buffer_size="$BUFFER_SIZE" \
      replay.staleness_cutoff_k="$STALENESS_CUTOFF_K" \
      replay.use_temporal_is="$USE_TEMPORAL_IS" \
      replay.continuous_producer="$CONTINUOUS_PRODUCER" \
      ++replay.live_store_socket="${LIVE_STORE_SOCKET:-/tmp/prorl_live_store.sock}" \
      ++replay.policy_registry_socket="${POLICY_REGISTRY_SOCKET:-/tmp/prorl_policy_registry.sock}" \
      ++replay.no_progress_timeout_s=5400 \
      ++replay.swap_protocol="$SWAP_PROTOCOL" \
      ++algorithm.filter_groups.enable="$FILTER_GROUPS" \
      "$@"
  ' _ "$@" 2>&1 | tee "$LOG_PATH"
