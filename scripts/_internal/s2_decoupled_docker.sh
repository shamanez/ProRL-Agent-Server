#!/bin/bash
# Stage 2: 20-step DECOUPLED GRPO run inside the verl docker image (v0.8 + vLLM 0.18).
#
# Decoupled vs s0_baseline_docker.sh:
#   - Trainer sees only GPUs 0-3 (via --gpus "device=0,1,2,3"; the NVIDIA
#     runtime renumbers them to 0-3 inside the container). vLLM actor spawn
#     is short-circuited by the EXTERNAL BYPASS ACTIVE path in
#     trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py.
#   - Rollouts go to an external pool launched on GPUs 4-7 at :8100-:8103
#     (see scripts/serving/launch_external_vllm_pool.sh; start that in a
#     separate terminal BEFORE running this script).
#   - ProRL on host :8006 is reused unchanged (Stage 0 launcher).
#
# Architecture:
#   - Host
#     - ProRL FastAPI :8006 (Terminal 1 — scripts/_internal/s0_prorl.sh)
#     - vLLM pool    :8100-:8103 (Terminal 2 — launch_external_vllm_pool.sh on GPUs 4-7)
#   - This container (Terminal 3)
#     - FSDP trainer on GPUs 0-3
#     - rollouts → external pool via host network
#
# verl version: 0.8.0.dev (shamanez/verl main branch)
# Docker image: verlai/verl:vllm018.dev1 (vLLM 0.18, PyTorch 2.6+)
set -eo pipefail
source /home/ubuntu/.prorl_creds.env

REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
IMG=verlai/verl:vllm018.dev1
CNAME=s2-decoupled

echo "[decoupled/docker] starting $(date -u +%FT%TZ)"
echo "[decoupled/docker] image: $IMG"

# Clean up any stale container from a prior attempt.
docker rm -f "$CNAME" >/dev/null 2>&1 || true

docker run --rm --name "$CNAME" \
  --gpus '"device=0,1,2,3"' \
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
  "$IMG" \
  bash -c '
    set -eo pipefail
    source /creds.env
    echo "[in-container] python: $(which python3) $(python3 --version)"
    echo "[in-container] decoupled pkgs:"
    python3 -c "import vllm, torch, transformers, ray; print(\"  vllm=\"+vllm.__version__); print(\"  torch=\"+torch.__version__); print(\"  transformers=\"+transformers.__version__); print(\"  ray=\"+ray.__version__)"

    # Install verl (shamanez/verl main, mounted at /opt/verl) in editable
    # mode so any future edits take effect without rebuild. --no-deps because
    # the image already ships the heavy runtime stack.
    pip install --no-deps -e /opt/verl >/dev/null
    # Install our verl_custom extension on top.
    pip install --no-deps -e /workspace/trainer_integration/verl >/dev/null

    # Any extra deps verl_custom needs that are not in the image.
    pip install scipy math_verify tabulate absl-py async_generator codetiming 2>/dev/null || true

    python3 -c "import verl, verl_custom; print(\"verl=\"+verl.__version__); print(\"verl_custom ok\")"

    # Pre-flight: wait for the external pool to report healthy before the
    # trainer starts. Fail loud here rather than deep inside the rollout loop.
    # Qwen3-4B cold-load on vLLM 0.18 takes 30-90s; without a retry loop a
    # freshly-launched pool would false-negative this check.
    HEALTH_TIMEOUT=180
    for port in 8100 8101 8102 8103; do
      echo "[decoupled/docker] waiting for :${port}/health (timeout ${HEALTH_TIMEOUT}s)..."
      deadline=$((SECONDS + HEALTH_TIMEOUT))
      until curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
          echo "[decoupled/docker] ERROR: :${port}/health still unreachable after ${HEALTH_TIMEOUT}s" >&2
          echo "[decoupled/docker] run scripts/serving/launch_external_vllm_pool.sh first" >&2
          exit 1
        fi
        sleep 2
      done
      echo "[decoupled/docker]   :${port} healthy"
    done
    echo "[decoupled/docker] external pool :8100-:8103 healthy"

    cd /workspace

    # Stage 2 must be single-use: auto-resume from a prior attempt would skip
    # the initial steps we care about gating on (global_step >= 20 from
    # scratch) and silently reuse old weights/optimizer state. Clean the
    # resolved output directory before launch and pass resume_mode=disable.
    STAGE2_OUT=/workspace/outputs/ProAgent/ProAgent-Qwen3-4B-instruct-training-GRPO-decoupled
    echo "[decoupled/docker] clearing $STAGE2_OUT for a fresh run"
    rm -rf "$STAGE2_OUT"

    bash trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_decoupled.sh \
      trainer.total_epochs=1 \
      ++trainer.total_training_steps=20 \
      trainer.save_freq=10 \
      trainer.resume_mode=disable \
      data.max_prompt_length=8192 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
      +actor_rollout_ref.actor.calculate_entropy=false \
      data.train_files=[/data/SkyRL-v0-293/train.filtered.parquet] \
      data.val_files=[/data/SkyRL-v0-293/validation.filtered.parquet] \
      trainer.default_local_dir="$STAGE2_OUT" \
      ++actor_rollout_ref.rollout.custom.rollout_save_dir=/workspace/outputs/rollout_data_decoupled
  ' 2>&1 | tee /tmp/s2-decoupled.log
