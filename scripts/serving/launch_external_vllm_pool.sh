#!/bin/bash
# Stage 1 external vLLM pool launcher.
#
# Fans out one Docker container per (GPU, port) pair. Each container runs
# scripts/serving/vllm_launcher.py, which in turn spawns a child vLLM
# /generate server. ProRL registers the supervisor port via /add_llm_server.
#
# Why Docker: vLLM 0.18 is only installed inside verlai/verl:vllm018.dev1.
# The host poetry venv (used by scripts/_internal/s0_prorl.sh) does not
# carry vllm. Keeping the pool in Docker also guarantees the exact same
# vLLM stack as Stage 0 training.
#
# Ports: supervisor=<N>, child=<N+1000>. Child log at /tmp/vllm-child-<N>.log
# (written from inside the container via /tmp bind mount).
#
# Usage:
#   bash scripts/serving/launch_external_vllm_pool.sh --gpus 0,1 --ports 8100,8101
#   bash scripts/serving/launch_external_vllm_pool.sh --gpus 4,5,6,7 --ports 8100,8101,8102,8103
#
# Teardown:
#   for p in 8100 8101; do docker stop vllm-sup-$p; done
set -eo pipefail

REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
IMG=verlai/verl:vllm018.dev1
HF_CACHE=/home/ubuntu/.cache/huggingface

GPUS=""
PORTS=""
MODEL=""
GPU_MEM_UTIL=0.45
MAX_MODEL_LEN=17920
MAX_NUM_BATCHED_TOKENS=8192
DEFAULT_MODEL_REPO='Qwen/Qwen3-4B-Instruct-2507'

usage() {
  cat <<'EOF' >&2
usage: launch_external_vllm_pool.sh --gpus G1,G2,... --ports P1,P2,... [--model PATH]
                                    [--gpu-memory-utilization F] [--max-model-len N]
                                    [--max-num-batched-tokens N]

Launches one Dockerized vLLM supervisor per (GPU, port) pair.
EOF
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --ports) PORTS="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done

[[ -z "$GPUS" || -z "$PORTS" ]] && usage

# Default model is the HF repo id — vLLM resolves it via the cache that this
# launcher bind-mounts at /root/.cache/huggingface (HF_HOME in the container).
# Using the repo id (not a host-absolute snapshot path) keeps the model arg
# valid inside the container and matches how s0_baseline_docker.sh calls vLLM.
# Verify the snapshot is already present so we fail before spawning Docker.
if [[ -z "$MODEL" ]]; then
  shopt -s nullglob
  SNAPS=( "$HF_CACHE"/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/ )
  if [[ ${#SNAPS[@]} -eq 0 ]]; then
    echo "error: no Qwen3-4B-Instruct-2507 snapshot found in $HF_CACHE/hub/" >&2
    echo "       run: huggingface-cli download $DEFAULT_MODEL_REPO" >&2
    exit 3
  fi
  MODEL="$DEFAULT_MODEL_REPO"
fi

IFS=',' read -ra GPU_ARR <<< "$GPUS"
IFS=',' read -ra PORT_ARR <<< "$PORTS"
if [[ ${#GPU_ARR[@]} -ne ${#PORT_ARR[@]} ]]; then
  echo "error: --gpus has ${#GPU_ARR[@]} entries but --ports has ${#PORT_ARR[@]}" >&2
  exit 2
fi

echo "[pool] image=$IMG"
echo "[pool] model=$MODEL"
echo "[pool] pairs: $(paste -d= <(printf '%s\n' "${GPU_ARR[@]}") <(printf '%s\n' "${PORT_ARR[@]}") | tr '\n' ' ')"

for i in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$i]}"
  PORT="${PORT_ARR[$i]}"
  CHILD_PORT=$((PORT + 1000))
  CNAME="vllm-sup-${PORT}"
  SUP_LOG="/tmp/vllm-sup-${PORT}.log"

  # Remove any stale container from a previous run. Host cannot rm the
  # root-owned pid/log files from the previous container, but the new
  # container's write_text() truncates on open, so stale content is replaced.
  docker rm -f "$CNAME" >/dev/null 2>&1 || true

  echo "[pool] launching GPU=$GPU supervisor=:$PORT child=:$CHILD_PORT container=$CNAME"

  # shellcheck disable=SC2086
  docker run --rm --name "$CNAME" \
    --gpus "device=${GPU}" \
    --network host \
    --ipc host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$REPO":/workspace \
    -v /tmp:/tmp \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -w /workspace \
    -e PYTHONPATH=/workspace \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e HF_HOME=/root/.cache/huggingface \
    "$IMG" \
    python3 scripts/serving/vllm_launcher.py \
      --port "$PORT" \
      --child-port "$CHILD_PORT" \
      --model "$MODEL" \
      -- \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --enforce-eager \
      --enable-chunked-prefill \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    > "$SUP_LOG" 2>&1 &

  CLIENT_PID=$!
  disown "$CLIENT_PID" 2>/dev/null || true
  echo "[pool]   host docker-client pid=$CLIENT_PID (teardown: docker stop $CNAME)"
  echo "[pool]   supervisor log=$SUP_LOG  child log=/tmp/vllm-child-${PORT}.log"
done

echo
# Liveness gate: `docker run ... &` drops the exit status on the floor, so a
# fast failure (image pull, bind-mount missing, port collision) would otherwise
# be reported as "launched" while /health never comes up. Sleep briefly, then
# fail loud if any container is not Running.
sleep 3
FAILED=0
for PORT in "${PORT_ARR[@]}"; do
  CNAME="vllm-sup-${PORT}"
  STATE=$(docker inspect --format='{{.State.Running}} {{.State.ExitCode}}' "$CNAME" 2>/dev/null || echo 'MISSING')
  if [[ "$STATE" != "true 0" ]]; then
    echo "[pool] ERROR: container $CNAME not running (state=$STATE)" >&2
    echo "[pool]   supervisor log tail (/tmp/vllm-sup-${PORT}.log):" >&2
    tail -n 20 "/tmp/vllm-sup-${PORT}.log" 2>&1 | sed 's/^/[pool]     /' >&2 || true
    FAILED=1
  fi
done
if [[ "$FAILED" -ne 0 ]]; then
  echo "[pool] one or more supervisors failed to start; see logs above" >&2
  exit 1
fi

echo "[pool] launched ${#GPU_ARR[@]} supervisor(s). Wait for health:"
for PORT in "${PORT_ARR[@]}"; do
  printf '  until curl -sf http://localhost:%s/health >/dev/null; do sleep 2; done\n' "$PORT"
done
