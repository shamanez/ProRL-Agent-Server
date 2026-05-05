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
IMG=verlai/verl:vllm018.dev1
CNAME=s3-fullasync
REMOTE_DNS="${REMOTE_DNS:-ec2-3-87-168-160.compute-1.amazonaws.com}"

# Training knobs
BATCH_SIZE="${BATCH_SIZE:-4}"
# GEN_BATCH_SIZE: number of rows per dataloader batch. Must be <= number of
# rows in data.train_files parquet. Default 4× BATCH_SIZE for full dataset;
# set smaller (e.g. 3) when only a few SIF images are built.
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$((BATCH_SIZE * 4))}"
STALENESS_CUTOFF_K="${STALENESS_CUTOFF_K:-4}"
REPLAY_ENABLE="${REPLAY_ENABLE:-True}"
BUFFER_SIZE="${BUFFER_SIZE:-64}"
USE_TEMPORAL_IS="${USE_TEMPORAL_IS:-False}"
CONTINUOUS_PRODUCER="${CONTINUOUS_PRODUCER:-False}"
FILTER_GROUPS="${FILTER_GROUPS:-True}"
PRODUCER_BATCH_SIZE="${PRODUCER_BATCH_SIZE:-4}"
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
  -e GEN_BATCH_SIZE="$GEN_BATCH_SIZE" \
  -e STALENESS_CUTOFF_K="$STALENESS_CUTOFF_K" \
  -e REPLAY_ENABLE="$REPLAY_ENABLE" \
  -e BUFFER_SIZE="$BUFFER_SIZE" \
  -e USE_TEMPORAL_IS="$USE_TEMPORAL_IS" \
  -e CONTINUOUS_PRODUCER="$CONTINUOUS_PRODUCER" \
  -e FILTER_GROUPS="$FILTER_GROUPS" \
  -e PRODUCER_BATCH_SIZE="$PRODUCER_BATCH_SIZE" \
  -e SWAP_PROTOCOL="$SWAP_PROTOCOL" \
  -e REPO_HOST_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" \
  -e RAY_memory_usage_threshold=0.98 \
  -e RAY_memory_monitor_refresh_ms=250 \
  -e RAY_object_store_memory=21474836480 \
  "$IMG" \
  bash -c '
    set -eo pipefail
    source /creds.env

    pip install --no-deps -e /opt/verl >/dev/null
    pip install --no-deps -e /workspace/trainers/verl >/dev/null
    pip install scipy math_verify tabulate absl-py async_generator codetiming peft 2>/dev/null || true

    # Pre-flight: vLLM pool must be healthy before training starts.
    HEALTH_TIMEOUT=300
    for port in 8100 8101 8102 8103; do
      url="http://${REMOTE_DNS}:${port}/health"
      deadline=$((SECONDS + HEALTH_TIMEOUT))
      until curl -sf --max-time 5 "${url}" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
          echo "[trainer] ERROR: ${url} unreachable — start vLLM pool first" >&2; exit 1
        fi
        sleep 2
      done
      echo "[trainer] vLLM :${port} healthy"
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
      # BC-14 note: data.train_files is a legacy VERL Hydra config requirement for
      # dataset schema inference. The trainer does NOT use this for rollout generation.
      # Rollout data comes exclusively from LiveStoreClient.get_batch().
      # TODO: replace with a LiveStoreOnlyDataset dummy config to remove the parquet mount.
      data.train_files=[/data/SkyRL-v0-293/train.parquet] \
      data.val_files=[/data/SkyRL-v0-293/validation.parquet] \
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
