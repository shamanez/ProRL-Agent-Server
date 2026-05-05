# prorl/verl-trainer — VERL FSDP trainer for the RolloutFabric
#
# This image is a pure TRAINING consumer. It never generates rollout data.
# All training data arrives via LiveStoreClient.get_batch() from the LiveStore
# service (UDS /tmp/prorl_live_store.sock, mounted at container start via -v /tmp:/tmp).
# After each checkpoint, PolicyRegistryClient.publish_policy_version() fans out
# the new LoRA weights to all vLLM pool endpoints.
#
# Build:
#   cd ProRL-Agent-Server
#   docker build -f trainers/verl/Dockerfile -t prorl/verl-trainer:latest .
#
# Run: see trainers/verl/scripts/start.sh
#
# Architecture role: TrainerAdapter (BC-15)
#   - Connects ONLY to LiveStore (gRPC UDS) and PolicyRegistry (gRPC UDS)
#   - No parquet dataset, no ProRL URL, no vLLM URL
#   - Repo mounted at /workspace; PYTHONPATH=/workspace/core for rollout_fabric.*

FROM verlai/verl:vllm018.dev1

# Pre-install packages that would otherwise be pip-installed at container start.
# Baking them into the image removes the startup latency on every training run.
RUN pip install --no-cache-dir \
        scipy \
        math_verify \
        tabulate \
        absl-py \
        async_generator \
        codetiming \
        peft \
        "daytona-api-client==0.20.0" \
    && pip cache purge

# Default environment — overridden by docker run -e flags
ENV PYTHONPATH=/workspace/core \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    RAY_memory_usage_threshold=0.98 \
    RAY_memory_monitor_refresh_ms=250 \
    RAY_object_store_memory=21474836480

WORKDIR /workspace
