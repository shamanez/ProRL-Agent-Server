#!/bin/bash

#SBATCH --account nvr_lpr_agentic
#SBATCH --partition interactive,batch_short,backfill,batch_block1
#SBATCH --time 02:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=8
#SBATCH --job-name llm-judge-qwen30b
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --exclusive
#SBATCH --dependency=singleton

set -x

max_response_length=8192

# vLLM Server Configuration
VLLM_NODES=1  # Number of nodes to use for vLLM servers (from the end)
VLLM_MODEL="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/jianh/data/models/Qwen3-30B-A3B-Instruct-2507-FP8"  # Model to load
VLLM_PORT=8000  # Base port for vLLM servers

export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export WANDB_API_KEY=${WANDB_API_KEY}

HOME_DIR="/lustre/fsw/portfolios/nvr/users/jianh"
HOME_DIR2="/lustre/fsw/portfolios/nvr/users/mingjiel"
HOME_DIR3="/lustre/fsw/portfolios/nvr/users/sdiao"
MODEL_PATH="/lustre/fsw/portfolios/nvr/users/mingjiel/models/DeepSeek-R1-Distill-Qwen-1.5B"
GPFS=$HOME_DIR3/verl_internal

PROJECT="llm-judge"
EXPNAME="llm-judge-qwen30b"
CKPT_DIR="$HOME_DIR3/results/$PROJECT/$EXPNAME/ckpt"
RESULTS_DIR="$HOME_DIR3/results/$PROJECT/$EXPNAME/$SLURM_JOB_ID"
mkdir -p $RESULTS_DIR
mkdir -p $CKPT_DIR

# TODO: add image name
container_name="$HOME_DIR/data/images/nvidian+nemo+verl_v2+vllm0.10dev.sqsh"

MOUNTS="--container-mounts=${GPFS}:${GPFS},/lustre:/lustre,${GPFS}:/verl"
export HF_HOME="$HOME_DIR3/.cache/huggingface"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Calculate node allocation
vllm_nodes=$VLLM_NODES

echo "vLLM nodes: $vllm_nodes"

# Create arrays for different node types
vllm_nodes_array=${nodes_array[0]}

echo "vLLM nodes: ${vllm_nodes_array[*]}"

# Start vLLM servers on dedicated nodes
echo "Starting vLLM servers on dedicated nodes: ${vllm_nodes_array[*]}"

# Set environment variables for llm_judge
# Create comma-separated list of vLLM server URLs
vllm_urls=""
for ((i = 0; i < vllm_nodes; i++)); do
    node_i=${vllm_nodes_array[$i]}
    port=$((VLLM_PORT + i))
    vllm_urls="$vllm_urls,http://$node_i:$port/v1"
done

export OPENAI_BASE_URL="$vllm_urls"
export OPENAI_API_KEY="dummy_key"
export OPENAI_MODEL="$VLLM_MODEL"
export OPENAI_TEMPERATURE="0"
export OPENAI_TOP_P="1.0"
export OPENAI_TOP_K="-1"
export OPENAI_MAX_TOKENS="1024"
export OPENAI_MAX_MODEL_LEN="$((max_response_length + OPENAI_MAX_TOKENS))"

# Start vLLM servers on each dedicated node
for ((i = 0; i < vllm_nodes; i++)); do
    node_i=${vllm_nodes_array[$i]}
    port=$((VLLM_PORT + i))
    echo "Starting vLLM server on node $node_i with port $port"

    srun --nodes=1 --ntasks=1 -w $node_i -o "$RESULTS_DIR/vllm-server-%j-node-$i.out" -e "$RESULTS_DIR/vllm-server-%j-node-$i.err" --no-container-mount-home --container-image="$container_name" $MOUNTS bash -c \
    "cd /verl && vllm serve \
        $VLLM_MODEL \
        --host 0.0.0.0 \
        --port $port \
        --data-parallel-size 8 \
        --tensor-parallel-size 1 \
        --enable-expert-parallel  \
        --gpu-memory-utilization 0.95 \
        --max-model-len $OPENAI_MAX_MODEL_LEN \
        --trust-remote-code \
        --enforce-eager \
        --api-key $OPENAI_API_KEY" &
done

# Wait for vLLM servers to be ready
echo "Waiting for vLLM servers to start..."
for ((i = 0; i < vllm_nodes; i++)); do
    node_i=${vllm_nodes_array[$i]}
    port=$((VLLM_PORT + i))

    echo "Checking vLLM server on $node_i:$port..."
    while ! curl -s --connect-timeout 5 --max-time 10 "http://$node_i:$port/health" > /dev/null 2>&1; do
        echo "vLLM server on $node_i:$port not ready yet, waiting..."
        sleep 10
    done
    echo "vLLM server on $node_i:$port is ready!"
done

echo "All vLLM servers are ready!"

wait
