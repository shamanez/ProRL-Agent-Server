#!/bin/bash
# Multi-node standalone SWE-bench evaluation via SLURM for Qwen3-4B-Instruct-2507.
# Usage: sbatch run_standalone_swebench_test_qwen3_4b_instruct.sh
#
# This script submits a SLURM job that starts ProRL Agent Server + vLLM on multiple
# nodes, then runs evaluation. For single-node local run without SLURM, use
# run_standalone_swebench_test_qwen3_4b_instruct_single_node.sh instead.

#SBATCH --job-name=standalone-swebench-test
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000G
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=4:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --output=/path/to/ProRL-Agent-Server/logs/slurm-%A_%a.out
#SBATCH --error=/path/to/ProRL-Agent-Server/logs/slurm-%A_%a.err

set -x  # Enable debug output

# ==================== Configuration ====================
ProRL_Agent_WORKDIR=/path/to/ProRL-Agent-Server
RESULTS_DIR="${ProRL_Agent_WORKDIR}/results/standalone_test_$(date +%Y%m%d_%H%M%S)"
container_name=/path/to/your/container.sqsh
MOUNTS="--container-mounts=/path/to/data:/path/to/data"

# Model configuration
MODEL_PATH='Qwen/Qwen3-4B-Instruct-2507'
TOKENIZER_PATH='Qwen/Qwen3-4B-Instruct-2507'

# Data configuration
DATA_PATH='/path/to/data/swe-bench-verified.parquet'
OUTPUT_DIR="${RESULTS_DIR}/standalone_swebench_test_${SLURM_JOB_ID}"

# Server configuration
GPUS_PER_NODE=8
TP_SIZE=4
GPU_MEM_UTIL=0.8
NUM_SERVERS_PER_NODE=$((GPUS_PER_NODE / TP_SIZE))
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

# ==================== Node Setup ====================
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
NNODES=$SLURM_NNODES

mkdir -p "$RESULTS_DIR"

# ==================== Resolve Node IPs ====================
declare -a node_ips
for i in "${!nodes_array[@]}"; do
    node=${nodes_array[$i]}
    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node" hostname --ip-address)
    # Convert to ipv4 if needed
    if [[ "$node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$node_ip"
        if [[ ${#ADDR[0]} -gt 16 ]]; then
            node_ip=${ADDR[1]}
        else
            node_ip=${ADDR[0]}
        fi
    fi
    node_ips[$i]=$node_ip
    echo "Node $i: ${nodes_array[$i]} -> IP: $node_ip"
done

# ==================== Start ProRL Agent Server on all nodes ====================
echo "Starting ProRL Agent Server on all nodes..."
ProRL_Agent_Server_urls=""

for i in "${!nodes_array[@]}"; do
    node=${nodes_array[$i]}
    node_ip=${node_ips[$i]}

    echo "Starting ProRL Agent Server on node $node (IP: $node_ip)"

    srun --nodes=1 --ntasks=1 -w "$node" \
        -o "$RESULTS_DIR/output-%A_%a-ProRL_Agent-node-$i.out" \
        -e "$RESULTS_DIR/output-%A_%a-ProRL_Agent-node-$i.err" \
        --container-image="$container_name" $MOUNTS \
        bash -c "cd $ProRL_Agent_WORKDIR \
        && export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/path/to/singularity_images \
        && export OVERWRITE_OPENHANDS_DIR=$ProRL_Agent_WORKDIR \
        && export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\$PATH \
        && export PYTHONPATH=$ProRL_Agent_WORKDIR:\$PYTHONPATH \
        && export LOG_LEVEL=ERROR \
        && export DEBUG=False \
        && nohup /usr/bin/python scripts/start_server_thread.py --max-init-workers 70 --max-run-workers $ProRL_Agent_NUM_WORKERS --timeout 9999999" &

    # Build the ProRL Agent Server URLs string
    if [ -z "$ProRL_Agent_Server_urls" ]; then
        ProRL_Agent_Server_urls="http://$node_ip:$ProRL_Agent_Server_PORT"
    else
        ProRL_Agent_Server_urls="$ProRL_Agent_Server_urls+http://$node_ip:$ProRL_Agent_Server_PORT"
    fi
done

echo "ProRL Agent Server URLs: $ProRL_Agent_Server_urls"

# ==================== Start vLLM servers on all nodes ====================
echo "Starting vLLM servers on all nodes..."
llm_server_urls=""

for i in "${!nodes_array[@]}"; do
    node=${nodes_array[$i]}
    node_ip=${node_ips[$i]}

    echo "Starting $NUM_SERVERS_PER_NODE vLLM server(s) on node $node (IP: $node_ip)"

    for server_idx in $(seq 0 $((NUM_SERVERS_PER_NODE - 1))); do
        gpu_start=$((server_idx * TP_SIZE))
        gpu_end=$((gpu_start + TP_SIZE - 1))
        cuda_devices=$(seq -s, $gpu_start $gpu_end)
        port=$((VLLM_BASE_PORT + server_idx))

        if [ "$TOKEN_LEVEL_GENERATION" = "true" ]; then
            # Token-level generation: use custom vllm_api_server.py
            vllm_cmd="CUDA_VISIBLE_DEVICES=$cuda_devices python $ProRL_Agent_WORKDIR/scripts/tests/vllm_api_server.py \
                --model $MODEL_PATH \
                --tensor-parallel-size $TP_SIZE \
                --port $port \
                --host 0.0.0.0 \
                --gpu-memory-utilization $GPU_MEM_UTIL \
                --max-model-len $MAX_MODEL_LEN"
        else
            # Standard mode: use OpenAI-compatible vLLM server
            vllm_cmd="CUDA_VISIBLE_DEVICES=$cuda_devices python -m vllm.entrypoints.openai.api_server \
                --model $MODEL_PATH \
                --tensor-parallel-size $TP_SIZE \
                --port $port \
                --host 0.0.0.0 \
                --gpu-memory-utilization $GPU_MEM_UTIL \
                --max-model-len $MAX_MODEL_LEN"
        fi

        srun --overlap --nodes=1 --ntasks=1 -w "$node" \
            -o "$RESULTS_DIR/output-%A_%a-vllm-node-$i-server-$server_idx.out" \
            -e "$RESULTS_DIR/output-%A_%a-vllm-node-$i-server-$server_idx.err" \
            --container-image="$container_name" $MOUNTS \
            bash -c "$vllm_cmd" &

        # Build the LLM server URLs string
        if [ -z "$llm_server_urls" ]; then
            llm_server_urls="http://$node_ip:$port"
        else
            llm_server_urls="$llm_server_urls+http://$node_ip:$port"
        fi
    done
done

echo "LLM Server URLs: $llm_server_urls"

# ==================== Wait for servers to be ready ====================
echo "Waiting for servers to start up..."
sleep 120

# Health check for vLLM servers
echo "Checking vLLM server health..."
IFS='+' read -ra LLM_URLS <<< "$llm_server_urls"
for url in "${LLM_URLS[@]}"; do
    for attempt in $(seq 1 60); do
        if curl -s -o /dev/null -w "%{http_code}" "$url/health" | grep -q "200"; then
            echo "vLLM server $url is healthy"
            break
        fi
        if [ $attempt -eq 60 ]; then
            echo "WARNING: vLLM server $url did not become healthy after 5 minutes"
        fi
        sleep 5
    done
done

# ==================== Build evaluation command args ====================
TOKEN_LEVEL_FLAG=""
if [ "$TOKEN_LEVEL_GENERATION" = "true" ]; then
    TOKEN_LEVEL_FLAG="--token_level_generation"
fi

# ==================== Run standalone evaluation ====================
echo "Starting standalone SWE-bench evaluation..."
echo "  ProRL Agent Server URLs: $ProRL_Agent_Server_urls"
echo "  LLM Server URLs: $llm_server_urls"

srun --overlap --nodes=1 --ntasks=1 -w "${nodes_array[0]}" \
    --cpus-per-task=8 \
    --mem=16G \
    -o "$RESULTS_DIR/output-%A_%a-evaluation.out" \
    -e "$RESULTS_DIR/output-%A_%a-evaluation.err" \
    --container-image="$container_name" $MOUNTS \
    bash -c "cd $ProRL_Agent_WORKDIR \
    && export PYTHONPATH=$ProRL_Agent_WORKDIR:\$PYTHONPATH \
    && python scripts/tests/standalone_swebench_test.py \
        --data_path $DATA_PATH \
        --ProRL_Agent_Server_urls '$ProRL_Agent_Server_urls' \
        --llm_server_urls '$llm_server_urls' \
        --model_name $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --num_trajectories $NUM_TRAJECTORIES \
        --num_workers_per_server $ProRL_Agent_NUM_WORKERS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --max_iterations $MAX_ITERATIONS \
        --max_output_tokens $MAX_OUTPUT_TOKENS \
        --max_model_len $MAX_MODEL_LEN \
        --timeout $TIMEOUT \
        --hint_mode $HINT_MODE \
        --custom_tokenizer $TOKENIZER_PATH \
        $TOKEN_LEVEL_FLAG"

echo "Evaluation completed! Results saved to: $OUTPUT_DIR"
