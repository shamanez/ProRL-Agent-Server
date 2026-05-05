#!/bin/bash
# Single-node standalone SWE-bench evaluation script for Qwen3-4B-Instruct-2507.
# Usage: bash run_standalone_swebench_test_qwen3_4b_instruct_single_node.sh
#
# This script launches ProRL Agent Server + vLLM servers locally, then runs evaluation.
# All background processes are cleaned up on exit (Ctrl-C or natural completion).

set -x  # Enable debug output

# ==================== Configuration ====================
ProRL_Agent_WORKDIR=/path/to/ProRL-Agent-Server
LOG_DIR="${ProRL_Agent_WORKDIR}/logs/standalone_test_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${ProRL_Agent_WORKDIR}/results/standalone_test_$(date +%Y%m%d_%H%M%S)"

# Model configuration
MODEL_PATH='Qwen/Qwen3-4B-Instruct-2507'
TOKENIZER_PATH='Qwen/Qwen3-4B-Instruct-2507'

# Data configuration
DATA_PATH='/path/to/data/swe-bench-verified.parquet'

# Server configuration
GPUS_PER_NODE=8
TP_SIZE=4
GPU_MEM_UTIL=0.8
NUM_SERVERS=$((GPUS_PER_NODE / TP_SIZE))
VLLM_BASE_PORT=8100
ProRL_Agent_Server_PORT=8006
ProRL_Agent_NUM_WORKERS=64

# Evaluation configuration
NUM_TRAJECTORIES=1
TEMPERATURE=0.0
TOP_P=1.0
MAX_ITERATIONS=50
MAX_OUTPUT_TOKENS=1536
MAX_MODEL_LEN=32768
TIMEOUT=1500
HINT_MODE=none
TOKEN_LEVEL_GENERATION=true  # set to true for token-level generation

# ==================== Cleanup trap ====================
PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing PID $pid"
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
        fi
    done
    echo "Cleanup done."
}

trap cleanup EXIT INT TERM

# ==================== Setup ====================
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

NODE_IP=$(hostname --ip-address)
# Convert to ipv4 if needed
if [[ "$NODE_IP" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$NODE_IP"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        NODE_IP=${ADDR[1]}
    else
        NODE_IP=${ADDR[0]}
    fi
fi
echo "Node IP: $NODE_IP"

# ==================== Start ProRL Agent Server ====================
echo "Starting ProRL Agent Server..."

cd "$ProRL_Agent_WORKDIR"
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/path/to/singularity_images
export OVERWRITE_OPENHANDS_DIR="$ProRL_Agent_WORKDIR"
export PYTHONPATH="${ProRL_Agent_WORKDIR}:${PYTHONPATH}"
export LOG_LEVEL=ERROR
export DEBUG=False

python scripts/start_server_thread.py \
    --max-init-workers 70 \
    --max-run-workers "$ProRL_Agent_NUM_WORKERS" \
    --timeout 9999999 \
    > "$LOG_DIR/ProRL_Agent_Server.out" 2> "$LOG_DIR/ProRL_Agent_Server.err" &
PIDS+=($!)
echo "ProRL Agent Server PID: ${PIDS[-1]}"

ProRL_Agent_Server_urls="http://${NODE_IP}:${ProRL_Agent_Server_PORT}"
echo "ProRL Agent Server URL: $ProRL_Agent_Server_urls"

cd "$WORKDIR"

# ==================== Start vLLM servers ====================
echo "Starting $NUM_SERVERS vLLM server(s)..."
llm_server_urls=""

for server_idx in $(seq 0 $((NUM_SERVERS - 1))); do
    gpu_start=$((server_idx * TP_SIZE))
    gpu_end=$((gpu_start + TP_SIZE - 1))
    cuda_devices=$(seq -s, "$gpu_start" "$gpu_end")
    port=$((VLLM_BASE_PORT + server_idx))

    echo "  Server $server_idx: GPUs=$cuda_devices, port=$port"

    if [ "$TOKEN_LEVEL_GENERATION" = "true" ]; then
        CUDA_VISIBLE_DEVICES=$cuda_devices python "$ProRL_Agent_WORKDIR/scripts/tests/vllm_api_server.py" \
            --model "$MODEL_PATH" \
            --tensor-parallel-size "$TP_SIZE" \
            --port "$port" \
            --host 0.0.0.0 \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            > "$LOG_DIR/vllm_server_${server_idx}.out" 2> "$LOG_DIR/vllm_server_${server_idx}.err" &
    else
        CUDA_VISIBLE_DEVICES=$cuda_devices python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --tensor-parallel-size "$TP_SIZE" \
            --port "$port" \
            --host 0.0.0.0 \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            > "$LOG_DIR/vllm_server_${server_idx}.out" 2> "$LOG_DIR/vllm_server_${server_idx}.err" &
    fi
    PIDS+=($!)
    echo "  vLLM server $server_idx PID: ${PIDS[-1]}"

    if [ -z "$llm_server_urls" ]; then
        llm_server_urls="http://${NODE_IP}:${port}"
    else
        llm_server_urls="${llm_server_urls}+http://${NODE_IP}:${port}"
    fi
done

echo "LLM Server URLs: $llm_server_urls"

# ==================== Wait for vLLM servers to be healthy ====================
echo "Waiting for vLLM servers to become healthy..."
IFS='+' read -ra LLM_URLS <<< "$llm_server_urls"
all_healthy=true
for url in "${LLM_URLS[@]}"; do
    healthy=false
    for attempt in $(seq 1 120); do
        if curl -s -o /dev/null -w "%{http_code}" "${url}/health" 2>/dev/null | grep -q "200"; then
            echo "  vLLM server $url is healthy (attempt $attempt)"
            healthy=true
            break
        fi
        sleep 5
    done
    if [ "$healthy" = false ]; then
        echo "ERROR: vLLM server $url did not become healthy after 10 minutes"
        echo "  Check logs: $LOG_DIR/vllm_server_*.err"
        all_healthy=false
    fi
done

if [ "$all_healthy" = false ]; then
    echo "Some vLLM servers failed to start. Aborting."
    exit 1
fi

echo "All vLLM servers are healthy."


# ==================== Run evaluation ====================
echo ""
echo "=========================================="
echo "Starting standalone SWE-bench evaluation"
echo "=========================================="
echo "  ProRL Agent Server URLs:   $ProRL_Agent_Server_urls"
echo "  LLM Server URLs:  $llm_server_urls"
echo "  Data:             $DATA_PATH"
echo "  Output:           $OUTPUT_DIR"
echo "  Logs:             $LOG_DIR"
echo "=========================================="

TOKEN_LEVEL_FLAG=""
if [ "$TOKEN_LEVEL_GENERATION" = "true" ]; then
    TOKEN_LEVEL_FLAG="--token_level_generation"
fi

cd "$ProRL_Agent_WORKDIR"
export PYTHONPATH="${ProRL_Agent_WORKDIR}:${PYTHONPATH}"

python scripts/tests/standalone_swebench_test.py \
    --data_path "$DATA_PATH" \
    --ProRL_Agent_Server_urls "$ProRL_Agent_Server_urls" \
    --llm_server_urls "$llm_server_urls" \
    --model_name "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_trajectories "$NUM_TRAJECTORIES" \
    --num_workers_per_server "$ProRL_Agent_NUM_WORKERS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_iterations "$MAX_ITERATIONS" \
    --max_output_tokens "$MAX_OUTPUT_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --timeout "$TIMEOUT" \
    --hint_mode "$HINT_MODE" \
    --custom_tokenizer "$TOKENIZER_PATH" \
    $TOKEN_LEVEL_FLAG

echo ""
echo "Evaluation completed! Results saved to: $OUTPUT_DIR"
echo "Logs available at: $LOG_DIR"
