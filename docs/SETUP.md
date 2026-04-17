# Setup Guide — 8 × A100-SXM4-40GB

## What runs where

| Component | Where | Purpose |
|---|---|---|
| **ProRL Server** (:8006) | Host, Poetry venv | Drives the OpenHands coding agent inside Singularity sandboxes. Sends token-IDs to vLLM over HTTP. |
| **Trainer + vLLM** | Docker container | GRPO training (verl + FSDP) with colocated vLLM 0.8.5. Reaches ProRL at `localhost:8006` via `--network=host`. |

Two separate environments because verl requires vLLM 0.8.x Python internals, but the host has vLLM 0.19+. The Docker image bundles the correct combo.

---

## Step 1: Credentials

```bash
cat > ~/.prorl_creds.env <<'EOF'
export WANDB_API_KEY='<your wandb key>'
export HF_TOKEN='<your hf token>'
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export SINGULARITY_DOCKER_USERNAME='<dockerhub user>'
export SINGULARITY_DOCKER_PASSWORD='<dockerhub PAT>'
export OH_RUNTIME_SINGULARITY_IMAGE_REPO="$HOME/de-coupled-rollouts-rl/ProRL-Agent-Server/singularity_images"
EOF
chmod 600 ~/.prorl_creds.env
source ~/.prorl_creds.env
```

## Step 2: Build Singularity images

The training dataset (SkyRL-v0-293) contains 293 software engineering tasks from real GitHub repos (e.g. "fix this bug in django", "add this feature to flask"). Each task runs inside its own isolated Singularity container (`.sif` file) that reproduces the exact repo state at the time of the issue. Building these containers converts Docker images from SWE-Bench into Singularity format.

```bash
source ~/.prorl_creds.env
REPO=$HOME/de-coupled-rollouts-rl/ProRL-Agent-Server
mkdir -p "$REPO/singularity_images"

# Build the first 50 SIF images (takes 1-4 hours, ~1-5 min each)
# 50 is enough to start training — the trainer only uses tasks whose SIFs exist
$REPO/scripts/_internal/s0_build_sifs.sh 1 50

# Check progress
ls "$REPO/singularity_images/"*.sif | wc -l

# Build the remaining 243 (can run while training is in progress)
$REPO/scripts/_internal/s0_build_sifs.sh 51 293
```

The numbers `1 50` and `51 293` are row ranges in the parquet file — row 1 through 50, then 51 through 293. You can build all 293 at once, but starting with 50 lets you begin training sooner.

## Step 3: Download training data

```bash
source ~/.prorl_creds.env
mkdir -p ~/data
huggingface-cli download NovaSky-AI/SkyRL-v0-293-data \
  --repo-type dataset --local-dir ~/data/SkyRL-v0-293
```

This downloads `train.parquet` and `validation.parquet` (~2 MB total). The dataset is [SkyRL-v0-293](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-293-data) — a curated subset of 293 real software engineering tasks from [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym). Each row describes a GitHub issue: the repo, the commit, the failing test, and the gold patch. The RL agent learns to solve these tasks by interacting with the codebase through bash and editor tools.

Filter the parquet to only tasks whose SIF images are built:

```bash
python scripts/_internal/filter_parquet_to_built_sifs.py \
  --source ~/data/SkyRL-v0-293/train.parquet \
  --sif-dir "$REPO/singularity_images" \
  --dest ~/data/SkyRL-v0-293/train.filtered.parquet
```

## Step 4: Set up the ProRL server (Poetry)

```bash
cd $HOME/de-coupled-rollouts-rl/ProRL-Agent-Server

# Install Poetry if not present
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
poetry install --with dev,test,runtime,evaluation
poetry run pip install httpx
poetry run pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git
poetry run pip install git+https://github.com/R2E-Gym/R2E-Gym.git
```

Verify the fakeroot fix (required for Apptainer 1.4.5 — without it, every sandbox hangs on `su root`):

```bash
grep -n "run_as_fakeroot" openhands/nvidia/swe_agent/utils.py
# Expected: 174:    sandbox_config.run_as_fakeroot = True
```

## Step 5: Set up the trainer (Docker)

### 5a. Pull the Docker image

```bash
docker pull verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
```

~77 GB. Contains torch 2.6, vLLM 0.8.5, ray, transformers — everything the trainer needs except verl itself.

### 5b. Clone verl (pinned commit)

```bash
git clone https://github.com/verl-project/verl.git /tmp/verl
cd /tmp/verl && git checkout 60138ebd
```

This is verl v0.4-dev (commit `60138ebd` — "[worker] fix: do not break dynamic bsz in dp critic"). The Docker script mounts it at `/opt/verl` inside the container.

### 5c. How the patching works

The repo has a patch package at `trainer_integration/verl/verl_custom/` that extends verl with:
- Custom rollout workers (async vLLM server manager)
- Custom reward managers (SWE-Bench reward scoring)
- Custom training scripts (Hydra config for Qwen3-4B + GRPO)

Inside the container, `s0_baseline_docker.sh` automatically runs:

1. `pip install --no-deps -e /opt/verl` — installs the pinned verl source (no extra deps, the image has them)
2. `pip install --no-deps -e /workspace/trainer_integration/verl` — installs `verl_custom` on top, which imports from `verl.*` and overrides rollout/trainer behavior

You don't run these manually — the Docker launch script handles both.

## Step 6: Run training

Two separate terminals:

```bash
REPO=$HOME/de-coupled-rollouts-rl/ProRL-Agent-Server

# Terminal 1: ProRL server (host)
$REPO/scripts/_internal/s0_prorl.sh

# Terminal 2: Trainer (Docker)
$REPO/scripts/_internal/s0_baseline_docker.sh
```

`s0_baseline_docker.sh` runs `docker run` with:
- `--gpus all --network host --ipc host --shm-size=16g`
- Mounts: repo at `/workspace`, verl at `/opt/verl`, data at `/data`
- Installs verl + verl_custom (see 5c)
- Runs training with 40GB-tuned Hydra overrides

### 40GB Hydra overrides (baked into `s0_baseline_docker.sh`)

| Override | Why |
|---|---|
| `++trainer.total_training_steps=20` | Real loop-exit key. `+trainer.max_steps` is a silent no-op (wrong key name). |
| `trainer.save_freq=10` | Each checkpoint ~63 GB. `save_freq=1` fills 991 GB disk at step 8. |
| `data.max_prompt_length=16384` | H200 default 31232 doesn't fit at SP=2 on 40 GB. |
| `actor_rollout_ref.rollout.gpu_memory_utilization=0.6` | Leaves 16 GB headroom for FSDP backward pass. |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size=2` | Halves per-GPU activation memory by splitting sequence across 2 GPUs. |

> Do NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — crashes vLLM's CuMemAllocator.

### Success criteria

| Metric | Pass |
|---|---|
| `step` | >= 20 |
| `actor/grad_norm` | finite, > 0, < 1e6 |
| `critic/rewards/mean` | not identically zero |
| Advantage variance | > 0 |
| `actor/kl` | finite |

Reference run: [wandb xncnwaie](https://wandb.ai/shamanework-pl/ProAgent/runs/xncnwaie) — rewards 0.375 -> 0.500, 20/20 steps, 2h08m wall clock.

---

## Version reference

| Component | Version | Notes |
|---|---|---|
| **verl** | commit `60138ebd` (v0.4-dev) | Cloned to `/tmp/verl` |
| **Docker image** | `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2` | torch 2.6, vLLM 0.8.5, Python 3.10 |
| **verl_custom** | `trainer_integration/verl/verl_custom/` | Patch package on top of verl |
| **Training script** | `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 40GB overrides in `s0_baseline_docker.sh` |
| **Dataset** | [SkyRL-v0-293](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-293-data) | 293 SWE tasks from real GitHub repos |

### Docker image compatibility (tested 2026-04-16)

| Image | vLLM | Works? | Why not |
|---|---|---|---|
| `app-verl0.4-vllm0.8.5-*` | 0.8.5 | **Yes** | Current |
| `vllm017.latest` | 0.17.0 | No | `vllm.lora.models`, `vllm.worker.worker_base`, `vllm.config.EngineConfig`, `vllm.model_executor.layers.sampler.SamplerOutput` removed/moved |
| `vllm018.dev1` | 0.18.x | No | Same breakage expected |

Upgrading requires porting `verl_custom` (16 files, 25 verl submodule imports) to verl 0.7+ API.
