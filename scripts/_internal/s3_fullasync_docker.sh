#!/bin/bash
# Phase 2 trainer launcher: fully-async decoupled topology (ProRL on host,
# vLLM pool on EC2 `vllm-instance`, FSDP trainer inside this container)
# with bounded replay buffer, continuous rollout producer, and clipped
# temporal importance-sampling correction.
#
# Sibling of s2_weightsync_docker.sh. All Phase 1 invariants preserved:
# closed-loop rank-16 LoRA weight-sync via POST /reload_lora after each
# _save_checkpoint; remote pool is the single inference backend; abort on
# endpoints_failed > 0.
#
# What is new (Phase 2):
#   - Replay store (TrajectoryStore) buffers rollouts between trainer steps.
#   - Producer thread generates rollouts continuously; trainer samples on
#     its own cadence (clock separation). Both controlled by the
#     +replay.continuous_producer=True Hydra override.
#   - Temporal importance-sampling (IS) correction gated by
#     +replay.use_temporal_is=True feeds behavior-policy logprobs from
#     the buffer into the existing tis_imp_ratio code in core_algos.py.
#
# Testing-order reminder (plans-n-solutions/stages/full_async.md §5a):
#   ALWAYS run filter_groups=False FIRST. DAPO's generate_sequences_dapo
#   waits for train_batch_size surviving groups, not train_batch_size
#   prompts — ~2-3x wall-clock vs plain GRPO because SWE-Gym drops ~50%
#   of groups for sign-shared rewards. Promote to filter_groups=True only
#   after the plain run is clean.
#
# Environment knobs (all optional; defaults below):
#   REPLAY_ENABLE             True
#   BUFFER_SIZE               128       (= 4 × train_batch_size × n = 4×4×8)
#   STALENESS_CUTOFF_K        4         (handsoff §12.3)
#   PRODUCER_BATCH_SIZE       4
#   USE_TEMPORAL_IS           True
#   CONTINUOUS_PRODUCER       True
#   FILTER_GROUPS             False     (flip to True ONLY after E1 clean)
#   TOTAL_EPOCHS              10
#   TOTAL_TRAINING_STEPS      500
#   SAVE_FREQ                 1
#
# verl version: 0.8.0.dev (shamanez/verl main branch)
# Docker image: verlai/verl:vllm018.dev1 (vLLM 0.18, PyTorch 2.6+)
set -eo pipefail
source /home/ubuntu/.prorl_creds.env

REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
IMG=verlai/verl:vllm018.dev1
CNAME=s3-fullasync
REMOTE_DNS="${REMOTE_DNS:-ec2-54-145-77-207.compute-1.amazonaws.com}"

# Phase 2 replay knobs.
REPLAY_ENABLE="${REPLAY_ENABLE:-True}"
BUFFER_SIZE="${BUFFER_SIZE:-128}"
STALENESS_CUTOFF_K="${STALENESS_CUTOFF_K:-4}"
PRODUCER_BATCH_SIZE="${PRODUCER_BATCH_SIZE:-4}"
USE_TEMPORAL_IS="${USE_TEMPORAL_IS:-True}"
CONTINUOUS_PRODUCER="${CONTINUOUS_PRODUCER:-True}"
# DEFAULT False. Primary E2E smoke test runs plain GRPO. Flip to True
# only when running the DAPO gate (Phase E2) — see full_async.md §5a.
FILTER_GROUPS="${FILTER_GROUPS:-False}"

# Trainer scale knobs — overridable per run.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-1}"
LOG_PATH="${LOG_PATH:-/tmp/s3-fullasync.log}"

echo "[fullasync/docker] starting $(date -u +%FT%TZ)"
echo "[fullasync/docker] image: $IMG"
echo "[fullasync/docker] remote pool: $REMOTE_DNS:8100-8103"
echo "[fullasync/docker] replay: enable=$REPLAY_ENABLE buffer=$BUFFER_SIZE K=$STALENESS_CUTOFF_K producer_bs=$PRODUCER_BATCH_SIZE tis=$USE_TEMPORAL_IS continuous=$CONTINUOUS_PRODUCER"
echo "[fullasync/docker] filter_groups=$FILTER_GROUPS (False = plain GRPO; flip to True only after E1 clean — full_async.md §5a)"

# Clean up any stale container from a prior attempt.
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
  -e REPLAY_ENABLE="$REPLAY_ENABLE" \
  -e BUFFER_SIZE="$BUFFER_SIZE" \
  -e STALENESS_CUTOFF_K="$STALENESS_CUTOFF_K" \
  -e PRODUCER_BATCH_SIZE="$PRODUCER_BATCH_SIZE" \
  -e USE_TEMPORAL_IS="$USE_TEMPORAL_IS" \
  -e CONTINUOUS_PRODUCER="$CONTINUOUS_PRODUCER" \
  -e FILTER_GROUPS="$FILTER_GROUPS" \
  -e RAY_memory_usage_threshold=0.98 \
  -e RAY_memory_monitor_refresh_ms=250 \
  -e RAY_object_store_memory=21474836480 \
  "$IMG" \
  bash -c '
    set -eo pipefail
    source /creds.env
    echo "[in-container] python: $(which python3) $(python3 --version)"
    echo "[in-container] decoupled pkgs:"
    python3 -c "import vllm, torch, transformers, ray; print(\"  vllm=\"+vllm.__version__); print(\"  torch=\"+torch.__version__); print(\"  transformers=\"+transformers.__version__); print(\"  ray=\"+ray.__version__)"

    # Install verl (shamanez/verl main, mounted at /opt/verl) in editable
    # mode so future edits take effect without rebuild. --no-deps because the
    # image already ships the heavy runtime stack.
    pip install --no-deps -e /opt/verl >/dev/null
    # Install our verl_custom extension on top.
    pip install --no-deps -e /workspace/trainer_integration/verl >/dev/null

    # Any extra deps verl_custom needs that are not in the image.
    pip install scipy math_verify tabulate absl-py async_generator codetiming 2>/dev/null || true
    # peft is required for rank-16 LoRA training; image may not ship it.
    pip install peft 2>/dev/null || true

    python3 -c "import verl, verl_custom; print(\"verl=\"+verl.__version__); print(\"verl_custom ok\")"

    # Pre-flight: wait for the remote pool to report healthy before the
    # trainer starts. Fail loud here rather than deep inside the rollout
    # loop.
    HEALTH_TIMEOUT=300
    for port in 8100 8101 8102 8103; do
      url="http://${REMOTE_DNS}:${port}/health"
      echo "[fullasync/docker] waiting for ${url} (timeout ${HEALTH_TIMEOUT}s)..."
      deadline=$((SECONDS + HEALTH_TIMEOUT))
      until curl -sf --max-time 5 "${url}" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
          echo "[fullasync/docker] ERROR: ${url} still unreachable after ${HEALTH_TIMEOUT}s" >&2
          echo "[fullasync/docker] run scripts/serving/launch_remote_vllm_pool.sh start first" >&2
          echo "[fullasync/docker] confirm EC2 SG inbound 8100-8103 from this public IP" >&2
          exit 1
        fi
        sleep 2
      done
      echo "[fullasync/docker]   :${port} healthy"
    done
    echo "[fullasync/docker] remote pool ${REMOTE_DNS}:8100-8103 healthy"

    cd /workspace

    # Resume path — distinct output dir from Phase 1 so the two stacks can
    # coexist on disk and resume independently. Buffer starts empty on
    # resume per full_async.md §2.
    STAGE2_OUT=/workspace/outputs/ProAgent/fullasync

    echo "[fullasync/docker] scale: epochs=$TOTAL_EPOCHS steps=$TOTAL_TRAINING_STEPS save_freq=$SAVE_FREQ"
    if (( $# > 0 )); then
      echo "[fullasync/docker] extra hydra overrides: $*"
    fi
    bash trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh \
      trainer.total_epochs="$TOTAL_EPOCHS" \
      ++trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
      trainer.save_freq="$SAVE_FREQ" \
      trainer.resume_mode=auto \
      trainer.val_before_train=False \
      trainer.test_freq=-1 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=36864 \
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=36864 \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=36864 \
      +actor_rollout_ref.actor.calculate_entropy=false \
      actor_rollout_ref.actor.entropy_checkpointing=true \
      data.train_files=[/data/SkyRL-v0-293/train.parquet] \
      data.val_files=[/data/SkyRL-v0-293/validation.parquet] \
      trainer.default_local_dir="$STAGE2_OUT" \
      ++actor_rollout_ref.rollout.custom.rollout_save_dir=/workspace/outputs/rollout_data_fullasync \
      replay.enable="$REPLAY_ENABLE" \
      replay.buffer_size="$BUFFER_SIZE" \
      replay.staleness_cutoff_k="$STALENESS_CUTOFF_K" \
      replay.producer_batch_size="$PRODUCER_BATCH_SIZE" \
      replay.use_temporal_is="$USE_TEMPORAL_IS" \
      replay.continuous_producer="$CONTINUOUS_PRODUCER" \
      +algorithm.filter_groups.enable="$FILTER_GROUPS" \
      "$@"
  ' _ "$@" 2>&1 | tee "$LOG_PATH"
