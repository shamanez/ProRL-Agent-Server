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
# Topology reminder (Cut 5+ / handsoff.md §Topology):
#   filter_groups=True is the target. DAPO's generate_sequences_dapo is the
#   only path wired to eager-push each survivor into the replay store the
#   moment it clears filter_easy_hard_instance — plain GRPO has no such
#   seam. Set FILTER_GROUPS=False only for throwaway plain-GRPO debugging.
#
# Environment knobs (all optional; defaults below):
#   REPLAY_ENABLE             True
#   BUFFER_SIZE               256
#   STALENESS_CUTOFF_K        4         (handsoff §12.3)
#   PRODUCER_BATCH_SIZE       4
#   USE_TEMPORAL_IS           True
#   CONTINUOUS_PRODUCER       True
#   FILTER_GROUPS             True      (DAPO eager-push is the target path)
#   TOTAL_EPOCHS              10
#   TOTAL_TRAINING_STEPS      500
#   SAVE_FREQ                 1
#   VAL_BEFORE_TRAIN          False     (smoke-test default; True for prod)
#   TEST_FREQ                 -1        (-1 disables; 1/5 for prod)
#   BATCH_SIZE                4         (smoke-test; 32 for prod)
#   GEN_BATCH_SIZE            16        (smoke-test; 128 for prod)
#
# verl version: 0.8.0.dev (shamanez/verl main branch)
# Docker image: verlai/verl:vllm018.dev1 (vLLM 0.18, PyTorch 2.6+)
set -eo pipefail
source /home/ubuntu/.prorl_creds.env

REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
IMG=verlai/verl:vllm018.dev1
CNAME=s3-fullasync
REMOTE_DNS="${REMOTE_DNS:-ec2-3-87-168-160.compute-1.amazonaws.com}"

# Phase 2 replay knobs.
REPLAY_ENABLE="${REPLAY_ENABLE:-True}"
BUFFER_SIZE="${BUFFER_SIZE:-256}"
STALENESS_CUTOFF_K="${STALENESS_CUTOFF_K:-4}"
PRODUCER_BATCH_SIZE="${PRODUCER_BATCH_SIZE:-4}"
USE_TEMPORAL_IS="${USE_TEMPORAL_IS:-True}"
CONTINUOUS_PRODUCER="${CONTINUOUS_PRODUCER:-True}"
# DEFAULT True. Cut 5 onward the eager-push path is DAPO-specific, so the
# production topology always runs filter_groups=True. Override to False
# only for throwaway plain-GRPO debugging (full_async.md §5a).
FILTER_GROUPS="${FILTER_GROUPS:-True}"

# Trainer scale knobs — overridable per run.
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-1}"
# Validation knobs — smoke-test defaults are OFF. A full validation pass
# costs ~10–15 min; turning it off lets a quick functional test finish in
# minutes instead of hours.
#   VAL_BEFORE_TRAIN=False  — skip the once-up-front validation pass.
#   TEST_FREQ=-1            — disable in-training pass@k validation.
# Production runs: VAL_BEFORE_TRAIN=True, TEST_FREQ=1 every step while we
# verify the validation path stays inside total_len (now bounded by
# enable_history_truncation=False in openhands/nvidia/swe_agent/utils.py),
# then bump to 5 once stable.
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
TEST_FREQ="${TEST_FREQ:--1}"
LOG_PATH="${LOG_PATH:-/tmp/s3-fullasync.log}"

# Cut 6: producer / trainer batch decoupling. ``BATCH_SIZE`` is the
# trainer's per-step group draw from replay; ``GEN_BATCH_SIZE`` is the
# DAPO producer's per-call survivor target.
#
# Smoke-test defaults: BATCH_SIZE=4, GEN_BATCH_SIZE=16 — keeps a quick
# functional test from spending an hour on a single producer call.
# Production defaults (Cut 9 step-20 wedge stabilization):
# BATCH_SIZE=32 (stronger PPO gradient signal), GEN_BATCH_SIZE=128
# (decoupled from the prior 4× formula to keep the producer fed under
# DAPO filter pressure).
BATCH_SIZE="${BATCH_SIZE:-4}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"

# How the vLLM pool retires the prior LoRA adapter on /reload_lora.
#   pinning  — default. Multi-tenant, path-versioned. Pins every trajectory
#              and every GRPO sibling group to its dispatch-time policy
#              version via /v{N}/generate. No cross-call mixing under
#              save_freq=1. Requires the matching child startup flag in
#              scripts/serving/_remote_vllm_runner.sh (already wired).
#   quiesce  — fallback. Drains in-flight before remove_lora; HTTP 503 on
#              drain timeout. Throughput cost grows with session length.
#              Use only if a pinning-mode regression surfaces in E2E.
SWAP_PROTOCOL="${SWAP_PROTOCOL:-pinning}"

echo "[fullasync/docker] starting $(date -u +%FT%TZ)"
echo "[fullasync/docker] image: $IMG"
echo "[fullasync/docker] remote pool: $REMOTE_DNS:8100-8103"
echo "[fullasync/docker] replay: enable=$REPLAY_ENABLE buffer=$BUFFER_SIZE K=$STALENESS_CUTOFF_K producer_bs=$PRODUCER_BATCH_SIZE tis=$USE_TEMPORAL_IS continuous=$CONTINUOUS_PRODUCER"
echo "[fullasync/docker] batches: train=$BATCH_SIZE (groups/step) gen=$GEN_BATCH_SIZE (survivors/producer call)"
echo "[fullasync/docker] cadence: total_steps=$TOTAL_TRAINING_STEPS save_freq=$SAVE_FREQ test_freq=$TEST_FREQ val_before_train=$VAL_BEFORE_TRAIN"
echo "[fullasync/docker] filter_groups=$FILTER_GROUPS (False = plain GRPO; flip to True only after E1 clean — full_async.md §5a)"
echo "[fullasync/docker] swap_protocol=$SWAP_PROTOCOL (pinning = path-versioned multi-tenant; quiesce = drain-and-swap)"

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
  -e PRODUCER_BATCH_SIZE="$PRODUCER_BATCH_SIZE" \
  -e USE_TEMPORAL_IS="$USE_TEMPORAL_IS" \
  -e CONTINUOUS_PRODUCER="$CONTINUOUS_PRODUCER" \
  -e FILTER_GROUPS="$FILTER_GROUPS" \
  -e BATCH_SIZE="$BATCH_SIZE" \
  -e GEN_BATCH_SIZE="$GEN_BATCH_SIZE" \
  -e SWAP_PROTOCOL="$SWAP_PROTOCOL" \
  -e REPO_HOST_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" \
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
      replay.producer_batch_size="$PRODUCER_BATCH_SIZE" \
      replay.use_temporal_is="$USE_TEMPORAL_IS" \
      replay.continuous_producer="$CONTINUOUS_PRODUCER" \
      ++replay.live_store_socket="${LIVE_STORE_SOCKET:-/tmp/prorl_live_store.sock}" \
      ++replay.policy_registry_socket="${POLICY_REGISTRY_SOCKET:-/tmp/prorl_policy_registry.sock}" \
      ++replay.stop_timeout_s=300 \
      ++replay.no_progress_timeout_s=5400 \
      ++replay.swap_protocol="$SWAP_PROTOCOL" \
      ++algorithm.filter_groups.enable="$FILTER_GROUPS" \
      "$@"
  ' _ "$@" 2>&1 | tee "$LOG_PATH"
