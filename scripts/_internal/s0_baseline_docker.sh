#!/bin/bash
# Stage 0: 20-step colocated GRPO baseline inside the verl docker image (v0.8 + vLLM 0.18).
#
# Why docker:
#   /opt/pytorch has torch 2.10 + vllm 0.19. verl + verl_custom need a
#   matching vllm inside the container. The verlai/verl image ships the
#   correct combo baked in.
#
# Architecture:
#   - ProRL FastAPI: runs on HOST at :8006 (sees singularity_images; spawns SIFs)
#   - Trainer + vLLM (colocated): runs INSIDE container, reaches ProRL via
#     --network=host → localhost:8006.
#
# verl version: 0.8.0.dev (shamanez/verl main branch)
# Docker image: verlai/verl:vllm018.dev1 (vLLM 0.18, PyTorch 2.6+)
set -eo pipefail
source /home/ubuntu/.prorl_creds.env

REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
IMG=verlai/verl:vllm018.dev1
CNAME=s0-baseline

echo "[baseline/docker] starting $(date -u +%FT%TZ)"
echo "[baseline/docker] image: $IMG"

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
  "$IMG" \
  bash -c '
    set -eo pipefail
    source /creds.env
    echo "[in-container] python: $(which python3) $(python3 --version)"
    echo "[in-container] baseline pkgs:"
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

    # Smoke test: verify critical imports resolve after verl v0.8 upgrade.
    python3 -c "
from verl.utils.profiler.performance import _timer
from verl.utils.vllm import is_version_ge
from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker
print(\"import compat: OK\")
"

    cd /workspace
    bash trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh \
      trainer.total_epochs=1 \
      ++trainer.total_training_steps=20 \
      trainer.save_freq=10 \
      data.max_prompt_length=16384 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
      +actor_rollout_ref.actor.calculate_entropy=false \
      data.train_files=[/data/SkyRL-v0-293/train.filtered.parquet] \
      data.val_files=[/data/SkyRL-v0-293/validation.filtered.parquet] \
      trainer.default_local_dir=/workspace/outputs/ProAgent/ProAgent-Qwen3-4B-instruct-training-GRPO \
      ++actor_rollout_ref.rollout.custom.rollout_save_dir=/workspace/outputs/rollout_data
  ' 2>&1 | tee /tmp/s0-baseline.log
