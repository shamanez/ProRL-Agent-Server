#!/bin/bash
# Phase 1 sibling of s1_remote_docker.sh. Same decoupled topology
# (ProRL on host, vLLM pool on EC2 `vllm-instance`, FSDP trainer here)
# but enables rank-16 LoRA + POST /reload_lora weight-sync. Design sketch:
# plans-n-solutions/stages/weight_sync_lora.md.
#
# Deltas vs s1_remote_docker.sh:
#   - CNAME=s2-weightsync
#   - STAGE1_OUT points at the weight-sync experiment dir
#   - Inner Hydra script → run_proagent_qwn3_4B_instruct_weightsync.sh
#     (enables lora_rank=16, publish_on_save=True)
#   - trainer.save_freq=5 so /reload_lora fires on steps 5, 10, 15, 20
#   - tee /tmp/s2-weightsync.log
#
# verl version: 0.8.0.dev (shamanez/verl main branch)
# Docker image: verlai/verl:vllm018.dev1 (vLLM 0.18, PyTorch 2.6+)
set -eo pipefail
source /home/ubuntu/.prorl_creds.env

REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
IMG=verlai/verl:vllm018.dev1
CNAME=s2-weightsync
REMOTE_DNS="${REMOTE_DNS:-ec2-54-145-77-207.compute-1.amazonaws.com}"

echo "[weightsync/docker] starting $(date -u +%FT%TZ)"
echo "[weightsync/docker] image: $IMG"
echo "[weightsync/docker] remote pool: $REMOTE_DNS:8100-8103"

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
    # loop. 300 s budget accommodates cross-region RTT + Qwen3-4B cold
    # load on the remote.
    HEALTH_TIMEOUT=300
    for port in 8100 8101 8102 8103; do
      url="http://${REMOTE_DNS}:${port}/health"
      echo "[weightsync/docker] waiting for ${url} (timeout ${HEALTH_TIMEOUT}s)..."
      deadline=$((SECONDS + HEALTH_TIMEOUT))
      until curl -sf --max-time 5 "${url}" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
          echo "[weightsync/docker] ERROR: ${url} still unreachable after ${HEALTH_TIMEOUT}s" >&2
          echo "[weightsync/docker] run scripts/serving/launch_remote_vllm_pool.sh start first" >&2
          echo "[weightsync/docker] confirm EC2 SG inbound 8100-8103 from this public IP" >&2
          exit 1
        fi
        sleep 2
      done
      echo "[weightsync/docker]   :${port} healthy"
    done
    echo "[weightsync/docker] remote pool ${REMOTE_DNS}:8100-8103 healthy"

    cd /workspace

    # Fresh run — a resumed run would skip the initial steps we gate on and
    # silently reuse old weights/optimizer state + pre-existing LoRA on the
    # pool would desync from policy_version=0 on the trainer.
    STAGE1_OUT=/workspace/outputs/ProAgent/weight-sync-decup-prorl
    echo "[weightsync/docker] clearing $STAGE1_OUT for a fresh run"
    rm -rf "$STAGE1_OUT"

    bash trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh \
      trainer.total_epochs=3 \
      ++trainer.total_training_steps=20 \
      trainer.save_freq=5 \
      trainer.resume_mode=disable \
      data.max_prompt_length=8192 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
      +actor_rollout_ref.actor.calculate_entropy=false \
      actor_rollout_ref.actor.entropy_checkpointing=true \
      data.train_files=[/data/SkyRL-v0-293/train.filtered.parquet] \
      data.val_files=[/data/SkyRL-v0-293/validation.filtered.parquet] \
      trainer.default_local_dir="$STAGE1_OUT" \
      ++actor_rollout_ref.rollout.custom.rollout_save_dir=/workspace/outputs/rollout_data_weightsync \
      "$@"
  ' 2>&1 | tee /tmp/s2-weightsync.log
