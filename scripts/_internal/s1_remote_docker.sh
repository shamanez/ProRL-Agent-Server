#!/bin/bash
# Stage 1 Cut C: DECOUPLED GRPO run where rollouts live on a REMOTE EC2 host.
# Sibling of s2_decoupled_docker.sh — same verl image, same mounts, same
# FSDP contract; the only topology change is that the vLLM pool is on
# `vllm-instance` (public DNS ec2-54-145-77-207.compute-1.amazonaws.com)
# instead of local GPUs 4-7, so the trainer reclaims all 8 A100s.
#
# Architecture:
#   - Host (trainer box)
#     - ProRL FastAPI :8006 (Terminal 1 — scripts/_internal/s0_prorl.sh)
#     - This container (Terminal 3)
#       - FSDP trainer on GPUs 0-7
#   - Remote host `vllm-instance` (Terminal 2)
#     - 4 × _vllm_child.py on GPUs 0-3, ports 8100-8103
#     - Brought up by scripts/serving/launch_remote_vllm_pool.sh start
#
# Deltas vs s2_decoupled_docker.sh:
#   - CNAME=s1-remote-decoupled
#   - --gpus all (was "device=0,1,2,3"): trainer owns all 8 local A100s
#   - Health probe targets the public EC2 DNS of the remote pool (not
#     localhost). 300 s timeout because cross-region RTT + Qwen3-4B cold
#     load can exceed the 180 s local budget.
#   - Inner Hydra script → run_proagent_qwn3_4B_instruct_remote_decoupled.sh
#   - STAGE1_OUT path reflects the new experiment name
#
# verl version: 0.8.0.dev (shamanez/verl main branch)
# Docker image: verlai/verl:vllm018.dev1 (vLLM 0.18, PyTorch 2.6+)
set -eo pipefail
source /home/ubuntu/.prorl_creds.env

REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
IMG=verlai/verl:vllm018.dev1
CNAME=s1-remote-decoupled
REMOTE_DNS="${REMOTE_DNS:-ec2-54-145-77-207.compute-1.amazonaws.com}"

echo "[remote-decoupled/docker] starting $(date -u +%FT%TZ)"
echo "[remote-decoupled/docker] image: $IMG"
echo "[remote-decoupled/docker] remote pool: $REMOTE_DNS:8100-8103"

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

    python3 -c "import verl, verl_custom; print(\"verl=\"+verl.__version__); print(\"verl_custom ok\")"

    # Pre-flight: wait for the remote pool to report healthy before the
    # trainer starts. Fail loud here rather than deep inside the rollout
    # loop. 300 s budget accommodates cross-region RTT + Qwen3-4B cold
    # load on the remote.
    HEALTH_TIMEOUT=300
    for port in 8100 8101 8102 8103; do
      url="http://${REMOTE_DNS}:${port}/health"
      echo "[remote-decoupled/docker] waiting for ${url} (timeout ${HEALTH_TIMEOUT}s)..."
      deadline=$((SECONDS + HEALTH_TIMEOUT))
      until curl -sf --max-time 5 "${url}" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
          echo "[remote-decoupled/docker] ERROR: ${url} still unreachable after ${HEALTH_TIMEOUT}s" >&2
          echo "[remote-decoupled/docker] run scripts/serving/launch_remote_vllm_pool.sh start first" >&2
          echo "[remote-decoupled/docker] confirm EC2 SG inbound 8100-8103 from this public IP" >&2
          exit 1
        fi
        sleep 2
      done
      echo "[remote-decoupled/docker]   :${port} healthy"
    done
    echo "[remote-decoupled/docker] remote pool ${REMOTE_DNS}:8100-8103 healthy"

    cd /workspace

    # Fresh run — same discipline as Stage 1 Cut B: a resumed run would
    # skip the initial steps we gate on and silently reuse old
    # weights/optimizer state.
    STAGE1_OUT=/workspace/outputs/ProAgent/ProAgent-Qwen3-4B-instruct-training-GRPO-remote-decoupled
    echo "[remote-decoupled/docker] clearing $STAGE1_OUT for a fresh run"
    rm -rf "$STAGE1_OUT"

    bash trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_remote_decoupled.sh \
      trainer.total_epochs=3 \
      ++trainer.total_training_steps=20 \
      trainer.save_freq=10 \
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
      ++actor_rollout_ref.rollout.custom.rollout_save_dir=/workspace/outputs/rollout_data_remote_decoupled \
      "$@"
  ' 2>&1 | tee /tmp/s1-remote.log
