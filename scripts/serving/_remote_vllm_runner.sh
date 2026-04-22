#!/bin/bash
# Stage 1 Cut C — remote-side runner. Executed on vllm-instance under nohup by
# launch_remote_vllm_pool.sh; keeps all heredoc complexity out of the
# orchestrator.
#
# Usage (remote): _remote_vllm_runner.sh <gpu> <port> <model> <gpu_mem_util> <max_model_len>
#
# Writes own PID to ~/vllm-pool/pid-<port>.pid. Child stdout/stderr are
# redirected by the caller (`nohup … > ~/vllm-pool/child-<port>.log 2>&1 &`),
# so this script must not redirect them again.
set -eo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <gpu> <port> <model> <gpu_mem_util> <max_model_len>" >&2
  exit 2
fi

GPU="$1"
PORT="$2"
MODEL="$3"
GPU_MEM_UTIL="$4"
MAX_MODEL_LEN="$5"

# Honour REMOTE_POOL_DIR exported by the orchestrator so an operator override
# (e.g. REMOTE_POOL_DIR=/opt/vllm-pool) doesn't silently read stale artifacts
# out of the default path.
POOL_DIR="${REMOTE_POOL_DIR:-${HOME}/vllm-pool}"
VENV="${POOL_DIR}/venv"
CHILD="${POOL_DIR}/_vllm_child.py"
PIDFILE="${POOL_DIR}/pid-${PORT}.pid"

if [[ ! -f "$CHILD" ]]; then
  echo "missing $CHILD (bootstrap incomplete)" >&2
  exit 3
fi
if [[ ! -d "$VENV" ]]; then
  echo "missing $VENV (bootstrap incomplete)" >&2
  exit 3
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

export CUDA_VISIBLE_DEVICES="$GPU"
# Match the Docker-image environment used by the trainer so allocator behaviour
# is identical on both sides of the endpoint.
export PYTORCH_ALLOC_CONF="expandable_segments:True"
# Must match the HF_HOME used during bootstrap's `hf download`. Default
# ~/.cache/huggingface on this AMI is root-owned; pin to the pool dir.
export HF_HOME="${POOL_DIR}/hf-cache"

echo $$ > "$PIDFILE"

exec python "$CHILD" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 128 \
  --enable-lora \
  --max-loras 8 \
  --max-lora-rank 32 \
  --max-cpu-loras 16
