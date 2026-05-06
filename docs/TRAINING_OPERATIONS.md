# Training Operations Guide

This guide covers everything needed to start a training run from zero, monitor it,
and recover from the most common failure modes. An agent reading this should be able
to execute all steps without additional context.

All paths are relative to the repo root:
`/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server`

---

## 0. From-Scratch Checklist

Run this checklist **once on a fresh machine**. Skip steps that are already done.

```bash
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
source /home/ubuntu/.prorl_creds.env   # Must exist — see section 2.1
```

### 0.1 Build the trainer Docker image (one-time, ~5 min)

```bash
docker build -f trainers/verl/Dockerfile -t prorl/verl-trainer:vllm018 .
```

Requires Docker with GPU support and internet access to pull `verlai/verl:vllm018.dev1`
from Docker Hub. The resulting local image is `prorl/verl-trainer:vllm018` — this is
what `trainers/verl/scripts/start.sh` uses by default.

### 0.2 Install Python environments (one-time, ~5 min total)

```bash
# Fabric-core (LiveStore, PolicyRegistry, RolloutManager, ReplayArchive)
cd core && poetry install && cd ..

# EnvironmentProvider (ProRL FastAPI server — OpenHands + litellm + Singularity)
cd environments/prorl_openhands && poetry install && cd ../..

# Out-of-band git dep required by SweAgentHandler (not on PyPI):
$(cd environments/prorl_openhands && poetry env info --path)/bin/pip install \
    "git+https://github.com/SWE-Gym/SWE-Bench-Package.git"
```

### 0.3 Export Python paths (every session)

```bash
export ROLLOUT_FABRIC_PYTHON=$(cd core && poetry env info --path)/bin/python
export PRORL_OPENHANDS_PYTHON=$(cd environments/prorl_openhands && poetry env info --path)/bin/python
```

Add these to your shell profile or a `.env` file so they survive across sessions.
`start_all.sh` will auto-resolve them if unset, but having them pre-exported is safer.

### 0.4 Build Singularity images (one-time, 3–5 min/image)

See section 1.1. Skip if `.sif` files already exist in `singularity_images/`.

### 0.5 Filter parquet dataset (one-time, after each new SIF batch)

See section 1.2. Skip if `train.ready.parquet` already exists and SIF list hasn't changed.

### 0.6 Set required env vars then start

```bash
export DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.ready.parquet"
export POLICY_ID="qwen3-4b-skyrl"
export ENVIRONMENT_ID="swe_agent"

# Step 1: vLLM pool (if not already running on the remote EC2)
bash inference/vllm/scripts/launch_remote_vllm_pool.sh start

# Steps 2–5: everything else in order with health gates
bash ops/services/start_all.sh
```

`start_all.sh` handles steps 2–5 in strict dependency order and blocks until the
BC-16 warm-up gate passes before starting the trainer.

---

## 1. Prerequisites

### 1.1 Singularity images

SWE-Bench tasks run inside Singularity (Apptainer) containers. Each task image must
be built as a `.sif` file before training can use it.

Images live at: `singularity_images/`

To build images (run in tmux — 3-5 min per image, ~15h total for all 293):

```bash
source /home/ubuntu/.prorl_creds.env
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server

CACHE_BASE="$(pwd)/scripts/_singularity_cache"
mkdir -p "${CACHE_BASE}/apptainer_tmpdir" "${CACHE_BASE}/apptainer_localcachedir"

APPTAINER_CACHEDIR="${CACHE_BASE}/apptainer_cachedir" \
APPTAINER_LOCALCACHEDIR="${CACHE_BASE}/apptainer_localcachedir" \
APPTAINER_TMPDIR="${CACHE_BASE}/apptainer_tmpdir" \
APPTAINER_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME}" \
APPTAINER_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD}" \
SINGULARITY_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME}" \
SINGULARITY_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD}" \
poetry run python scripts/pull_swe_images.py \
  --parquet-file /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --dest-dir singularity_images \
  --log-name build_all.log
# Monitor: ls singularity_images/*.sif | wc -l
```

### 1.2 Dataset filter

The RolloutManager only dispatches tasks that have a built SIF image. Run once after
any new SIFs are added or after first build:

```bash
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
python ops/data/filter_parquet_to_built_sifs.py \
  --input /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --sif-dir singularity_images \
  --output /home/ubuntu/data/SkyRL-v0-293/train.ready.parquet
```

This writes `train.ready.parquet` — the file used by `DATA_FILES`.

### 1.3 Python environments

Two host venvs are required. The Docker trainer environment is set up at container
start, not on the host.

**Fabric-core** (LiveStore, PolicyRegistry, RolloutManager, ReplayArchive — 13 packages):

```bash
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/core
poetry install
export ROLLOUT_FABRIC_PYTHON=$(poetry env info --path)/bin/python
cd ..
```

**EnvironmentProvider** (full OpenHands + litellm + Singularity stack, ~200 packages):

```bash
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/environments/prorl_openhands
poetry install
export PRORL_OPENHANDS_PYTHON=$(poetry env info --path)/bin/python
cd ../..
```

**Out-of-band git dep** — required for `SweAgentHandler` (not on PyPI):

```bash
# Run once after poetry install, into the EnvironmentProvider venv:
$(cd environments/prorl_openhands && poetry env info --path)/bin/pip install \
    "git+https://github.com/SWE-Gym/SWE-Bench-Package.git"
```

> `ROLLOUT_FABRIC_PYTHON` and `PRORL_OPENHANDS_PYTHON` must be exported before
> running any service script, or add them to your shell profile / `.env` file.
> Resolve them fresh each session — never hardcode the hash portion of the venv path.

**TrainerAdapter** — never install on host. The Docker image installs it at container
start:

```
Docker image: verlai/verl:vllm018.dev1
```

---

## 2. Environment Variables

### 2.1 Source credentials file

```bash
source /home/ubuntu/.prorl_creds.env
```

This file contains secrets. Never commit it. It defines at minimum:

| Variable | What it is |
|---|---|
| `WANDB_API_KEY` | W&B API key for training metrics logging |
| `HF_TOKEN` | HuggingFace token for model weights download |
| `HUGGING_FACE_HUB_TOKEN` | Alias for `HF_TOKEN` (some tools read this) |
| `REMOTE_DNS` | Hostname/IP of the remote EC2 running vLLM (e.g. `ec2-3-87-168-160.compute-1.amazonaws.com`) |
| `SINGULARITY_DOCKER_USERNAME` | Docker credentials for Apptainer pulls |
| `SINGULARITY_DOCKER_PASSWORD` | Docker credentials for Apptainer pulls |

### 2.2 Set before starting services

```bash
export ROLLOUT_FABRIC_PYTHON=$(cd core && poetry env info --path)/bin/python
export PRORL_OPENHANDS_PYTHON=$(cd environments/prorl_openhands && poetry env info --path)/bin/python
export DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.ready.parquet"
export POLICY_ID="qwen3-4b-skyrl"
export ENVIRONMENT_ID="swe_agent"
# Required if calling `python -m rollout_fabric.*` directly (the ops/services/ scripts
# set this internally, but you need it in your shell for ad-hoc commands):
export PYTHONPATH=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/core
```

### 2.3 Full knob table (defaults shown)

| Variable | Default | Service | Description |
|---|---|---|---|
| `DATA_FILES` | _(required)_ | RolloutManager | Space-separated parquet paths |
| `POLICY_ID` | `qwen3-4b-skyrl` | RolloutManager, LiveStore | Policy identifier |
| `ENVIRONMENT_ID` | `swe_agent` | RolloutManager, LiveStore | Environment type |
| `LIVE_STORE_SOCKET` | `/tmp/prorl_live_store.sock` | LiveStore, Trainer | UDS path |
| `POLICY_REGISTRY_SOCKET` | `/tmp/prorl_policy_registry.sock` | PolicyRegistry, Trainer | UDS path |
| `LIVE_STORE_MAX_SIZE` | `256` | LiveStore | Max groups in buffer |
| `STALENESS_CUTOFF_K` | `32` | LiveStore, Trainer | Max step age for groups |
| `NO_PROGRESS_TIMEOUT` | `1800` | LiveStore | Seconds before abort if no new push |
| `VLLM_POOL_ENDPOINTS` | auto from `REMOTE_DNS` | PolicyRegistry | Space-separated vLLM endpoint URLs |
| `GROUP_SIZE` | `16` | RolloutManager | GRPO group size (N siblings per task) |
| `PRORL_URL` | `http://localhost:8006` | RolloutManager | EnvironmentProvider URL |
| `PRORL_PORT` | `8006` | EnvironmentProvider | Port the server listens on |
| `BATCH_SIZE` | `4` | Trainer | Groups drawn per training step |
| `GEN_BATCH_SIZE` | `4 × BATCH_SIZE` | Trainer | Parquet rows per dataloader batch |
| `TOTAL_TRAINING_STEPS` | `500` | Trainer | Stop after N steps |
| `SAVE_FREQ` | `1` | Trainer | Publish LoRA every N steps |
| `REPLAY_ENABLE` | `True` | Trainer | Enable replay buffer |
| `BUFFER_SIZE` | `64` | Trainer | Trainer-side buffer size |
| `USE_TEMPORAL_IS` | `False` | Trainer | Importance sampling correction |
| `SWAP_PROTOCOL` | `pinning` | Trainer | vLLM weight swap protocol |
| `REPLAY_ARCHIVE_DISABLED` | `0` | start_all.sh | Set to `1` to skip ReplayArchive |
| `ROLLOUT_FABRIC_PYTHON` | fabric-core venv python | fabric services | Override python for fabric services |
| `PRORL_OPENHANDS_PYTHON` | openhands venv python | EnvironmentProvider | Override python for env provider |

---

## 3. Startup Sequence

Run steps in strict order. Each step has a health gate — do not proceed until it passes.

The orchestrated launcher `ops/services/start_all.sh` runs all steps in order,
enforces the BC-16 warm-up gate, and handles clean shutdown on Ctrl-C:

```bash
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
source /home/ubuntu/.prorl_creds.env
export DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.ready.parquet"
export POLICY_ID="qwen3-4b-skyrl"
export ENVIRONMENT_ID="swe_agent"
bash ops/services/start_all.sh
```

For manual step-by-step execution, follow the subsections below.

### Step 1 — InferenceBackend (vLLM pool on remote EC2)

```bash
bash inference/vllm/scripts/launch_remote_vllm_pool.sh start
```

**Health gate:** all 4 ports return HTTP 200:

```bash
for port in 8100 8101 8102 8103; do
  curl -sf --max-time 3 "http://${REMOTE_DNS}:${port}/health" && echo ":${port} OK"
done
```

All four must print `OK` before proceeding.

### Step 2 — EnvironmentProvider (ProRL :8006)

Requires `REMOTE_DNS` set and `PRORL_OPENHANDS_PYTHON` pointing to the full OpenHands
venv (or using the default poetry env).

```bash
nohup bash environments/prorl_openhands/scripts/start.sh > /tmp/s0-prorl.log 2>&1 &
echo $! > /tmp/prorl.pid
```

**Health gate:**

```bash
for i in $(seq 1 30); do
  curl -sf http://localhost:8006/status -o /dev/null && { echo "ProRL up"; break; }
  sleep 1
done
```

Then activate the agent server (must POST /start after the health check passes):

```bash
curl -sf -X POST http://localhost:8006/start -H "Content-Type: application/json" -d '{}'
```

Verify it is running:

```bash
curl -sf http://localhost:8006/status
# Expected: {"status":"running", ...}
```

### Steps 3a + 3b — LiveStore and PolicyRegistry (run in parallel)

**LiveStore (Step 3a):**

```bash
nohup bash ops/services/start_live_store.sh > /tmp/live_store.log 2>&1 &
echo $! > /tmp/live_store.pid
```

Environment variables respected: `LIVE_STORE_SOCKET`, `LIVE_STORE_MAX_SIZE`,
`STALENESS_CUTOFF_K`, `NO_PROGRESS_TIMEOUT`, `ROLLOUT_FABRIC_PYTHON`.

**Health gate:**

```bash
[[ -S /tmp/prorl_live_store.sock ]] && echo "LiveStore socket OK"
```

**PolicyRegistry (Step 3b):**

```bash
# Set endpoints from REMOTE_DNS (or override VLLM_POOL_ENDPOINTS directly):
export VLLM_POOL_ENDPOINTS="http://${REMOTE_DNS}:8100 http://${REMOTE_DNS}:8101 http://${REMOTE_DNS}:8102 http://${REMOTE_DNS}:8103"
nohup bash ops/services/start_policy_registry.sh > /tmp/policy_registry.log 2>&1 &
echo $! > /tmp/policy_registry.pid
```

Environment variables respected: `POLICY_REGISTRY_SOCKET`, `POLICY_REGISTRY_DB`,
`VLLM_POOL_ENDPOINTS`, `ROLLOUT_FABRIC_PYTHON`.

**Health gate:**

```bash
[[ -S /tmp/prorl_policy_registry.sock ]] && echo "PolicyRegistry socket OK"
```

Both sockets must exist before proceeding to Step 4.

### Step 4 — RolloutManager

**BC-14 enforced:** `DATA_FILES` must be set — the worker owns the dataset, not the
trainer.

```bash
nohup bash ops/services/start_rollout_manager.sh > /tmp/rollout_manager.log 2>&1 &
echo $! > /tmp/rollout_manager.pid
```

Environment variables respected: `DATA_FILES` (required), `LIVE_STORE_SOCKET`,
`PRORL_URL`, `POLICY_ID`, `ENVIRONMENT_ID`, `GROUP_SIZE`, `POLICY_MANIFEST_PATH`,
`REPLAY_ARCHIVE_ROOT`, `REPLAY_ARCHIVE_DISABLED`, `FILTER_ZERO_VARIANCE`,
`ROLLOUT_FABRIC_PYTHON`.

**Health gate — BC-16 warm-up:** Wait until LiveStore has at least 1 group before
starting the trainer. This prevents burning the 1800s no-progress timer during
warm-up:

```bash
PYTHONPATH=./core python - <<'PYEOF'
import sys, time
from rollout_fabric.live_store.client import LiveStoreClient
import os
cli = LiveStoreClient(
    '/tmp/prorl_live_store.sock',
    policy_id=os.environ.get('POLICY_ID', 'qwen3-4b-skyrl'),
    environment_id=os.environ.get('ENVIRONMENT_ID', 'swe_agent'),
)
deadline = time.monotonic() + 300  # 5 min timeout for wait
while time.monotonic() < deadline:
    n = cli.num_groups()
    if n >= 1:
        print(f'LiveStore has {n} group(s) — OK to start trainer')
        cli.close(); sys.exit(0)
    time.sleep(5)
print('TIMEOUT: no groups appeared in 5 min — check rollout_manager.log')
cli.close(); sys.exit(1)
PYEOF
```

Do not start the trainer until this returns 0.

### Step 5 — TrainerAdapter (Docker VERL)

**Only after BC-16 warm-up passes (Step 4 gate).**

```bash
bash trainers/verl/scripts/start.sh
# Logs: /tmp/s3-fullasync.log (and stdout of the docker run command)
```

The script runs Docker container `verlai/verl:vllm018.dev1` with `--gpus all`.
Inside, it installs the trainer package and runs the Hydra training launch script.

Key training knobs that can be overridden before calling `start.sh`:

```bash
export BATCH_SIZE=4               # groups per training step (32 for prod)
export TOTAL_TRAINING_STEPS=500   # stop after N steps
export SAVE_FREQ=1                # publish LoRA every N steps
export SWAP_PROTOCOL=pinning      # vLLM weight swap: pinning or quiesce
export STALENESS_CUTOFF_K=4       # max step age before group eviction
export USE_TEMPORAL_IS=False      # importance sampling correction
bash trainers/verl/scripts/start.sh
```

**Health gate:** The trainer has no HTTP endpoint. Confirm it is working by watching
for `get_batch` calls in the trainer log:

```bash
grep -i "get_batch\|loss\|step" /tmp/s3-fullasync.log | tail -5
```

### Step 6 — Training Monitor

Run in a separate tmux pane. Writes `current_training_progress.md` every 30s:

```bash
PYTHONPATH=./core python ops/services/training_monitor.py --interval 30
```

Options:

```bash
# Single pass (useful for scripted checks):
PYTHONPATH=./core python ops/services/training_monitor.py --once

# Metrics only — disable auto-rescue:
PYTHONPATH=./core python ops/services/training_monitor.py --no-rescue

# Custom output file:
PYTHONPATH=./core python ops/services/training_monitor.py \
  --progress-file /tmp/my_progress.md

# Override REMOTE_DNS for vLLM health checks:
PYTHONPATH=./core python ops/services/training_monitor.py \
  --remote-dns ec2-3-87-168-160.compute-1.amazonaws.com
```

---

## 4. Health Gates Summary

| Step | Service | Gate command | Pass condition |
|---|---|---|---|
| 1 | InferenceBackend | `curl -sf http://${REMOTE_DNS}:8100/health` | HTTP 200 on all 4 ports |
| 2 | EnvironmentProvider | `curl -sf http://localhost:8006/status` | JSON with `status:running` |
| 3a | LiveStore | `[[ -S /tmp/prorl_live_store.sock ]]` | Socket file exists |
| 3b | PolicyRegistry | `[[ -S /tmp/prorl_policy_registry.sock ]]` | Socket file exists |
| 4 | RolloutManager | `LiveStoreClient.num_groups() >= 1` | At least 1 group in buffer |
| 5 | TrainerAdapter | `grep "loss" /tmp/s3-fullasync.log` | Loss values appear in log |
| all | All services | `bash tests/harness/smoke_test.sh` | All 6 gates PASS |

The smoke test harness validates all contract boundaries at once:

```bash
source /home/ubuntu/.prorl_creds.env
PYTHONPATH=./core bash tests/harness/smoke_test.sh
# Log: /tmp/training_bootstrap.log
```

Gates checked:
- GATE-1: HTTP health on EnvironmentProvider + vLLM pool + UDS sockets
- GATE-2: LiveStore has ≥1 group (BC-16 warm-up)
- GATE-3: token IDs are `list[int]` not `list[str]` (BC-1)
- GATE-4: `group_uid` consistent across siblings in one `get_batch` result
- GATE-5: `behavior_policy_version` is `int`, not `None`
- GATE-6: PolicyRegistry reachable and queryable

---

## 5. Monitoring

### 5.1 Training monitor

```bash
PYTHONPATH=./core python ops/services/training_monitor.py --interval 30
```

The monitor collects from 5 sub-agents every 30s:

| Agent | What it checks |
|---|---|
| `MetricsAgent` | Parses `PRODUCER_ITER` JSON lines from `/tmp/rollout_manager.log` |
| `WeightSyncAgent` | Scans `/tmp/policy_registry.log` for `reload_lora` events and `endpoints_failed` (BC-9) |
| `RolloutAgent` | HTTP health + latency on vLLM pool ports 8100-8103 |
| `LiveStoreAgent` | Socket existence + `num_groups` + `total_pushes` via gRPC client |
| `TrainerAgent` | Scans `/tmp/trainer.log` for loss values and NaN patterns |

Auto-rescue fires when any service is unhealthy (disable with `--no-rescue`).

### 5.2 Progress file

`current_training_progress.md` (repo root) is rewritten every monitor tick. It shows:

- Overall status: `HEALTHY | DEGRADED | RESCUED | STALLED`
- Policy version, groups pushed, groups filtered, throughput (groups/hr)
- Boundary condition checks: BC-0, BC-5, BC-9, BC-16
- Per-service health table
- Last N weight sync events and rescue events

A stall (no new push for >300s) shows `STALLED`. NaN in trainer log shows `DEGRADED`.

### 5.3 Log files

| Log | Service |
|---|---|
| `/tmp/s0-prorl.log` | EnvironmentProvider (ProRL) |
| `/tmp/live_store.log` | LiveStore |
| `/tmp/policy_registry.log` | PolicyRegistry |
| `/tmp/rollout_manager.log` | RolloutManager |
| `/tmp/s3-fullasync.log` | TrainerAdapter (Docker) |
| `/tmp/training_bootstrap.log` | Smoke test harness |

Tail any log with `tail -f /tmp/<name>.log`.

---

## 6. Stopping

Stop in reverse order — trainer first, InferenceBackend last. Do not kill LiveStore
or PolicyRegistry before the trainer, or the trainer's next `get_batch` / `publish`
will fail.

```bash
# Trainer (Docker container name is s3-fullasync):
docker stop s3-fullasync || true

# RolloutManager:
kill $(cat /tmp/rollout_manager.pid) 2>/dev/null || pkill -f rollout_manager.main

# LiveStore + PolicyRegistry (in parallel):
kill $(cat /tmp/live_store.pid) 2>/dev/null || true
kill $(cat /tmp/policy_registry.pid) 2>/dev/null || true

# EnvironmentProvider:
kill $(cat /tmp/prorl.pid) 2>/dev/null || pkill -f start_server.py

# InferenceBackend (on remote EC2):
bash inference/vllm/scripts/launch_remote_vllm_pool.sh stop
```

If using `ops/services/start_all.sh`, press Ctrl-C. The trap handler shuts down in
reverse order automatically.

---

## 7. Common Errors and Fixes

### Error 1: `ModuleNotFoundError: No module named 'evaluation'`

**Context:** EnvironmentProvider startup, or importing anything under
`environments/prorl_openhands/`.

**Cause:** The vendor path containing `evaluation/` is not on `PYTHONPATH`. The
`start.sh` script adds it automatically, but manual runs omit it.

**Fix:** The vendor path must be in `PYTHONPATH`:

```bash
VENDOR_ROOT="/home/ubuntu/unextractable-agentic-rl/vendor/ProRL-Agent-Server"
export PYTHONPATH="${REPO_ROOT}/environments/prorl_openhands:${REPO_ROOT}/core:${VENDOR_ROOT}:${PYTHONPATH:-}"
```

The script `environments/prorl_openhands/scripts/start.sh` sets this correctly.
Use the script rather than invoking Python directly.

### Error 2: `TypeError: 'NoneType' object is not callable` at TrajectoryStore / LiveStore

**Context:** Trainer starts but crashes immediately on first `get_batch` attempt.

**Cause:** `LIVE_STORE_SOCKET` env var is not set inside the Docker container. The
trainer tries to instantiate `LiveStoreClient` with `socket_path=None`.

**Fix:** Verify that `start.sh` passes `-e LIVE_STORE_SOCKET=/tmp/prorl_live_store.sock`
to `docker run`. Check with:

```bash
docker inspect s3-fullasync | python3 -m json.tool | grep LIVE_STORE
```

If missing, the `trainers/verl/scripts/start.sh` already passes this by default.
Do not override `LIVE_STORE_SOCKET` to empty.

### Error 3: `FileNotFoundError: /path/to/data/parquet/train.parquet`

**Context:** Trainer logs show it cannot open the parquet file at startup.

**Cause:** `DATA_PATH` env var is set to a placeholder path inside the Docker
container (`/data/SkyRL-v0-293` is the mounted path, but the file expected is
`train.parquet` not `train.ready.parquet`). The trainer uses parquet only for Hydra
schema inference, not for rollout data — but the file must exist.

**Fix:** The Docker volume mount in `trainers/verl/scripts/start.sh` mounts
`/home/ubuntu/data` as `/data:ro`. The trainer expects the original `train.parquet`
(not the filtered one) at `--data.train_files`. The file `/home/ubuntu/data/SkyRL-v0-293/train.parquet`
must exist. Do not delete the original parquet after filtering.

### Error 4: `daytona_api_client: cannot import name 'WorkspaceState'`

**Context:** EnvironmentProvider startup when the OpenHands poetry env has an
incompatible version of `daytona-api-client`.

**Cause:** OpenHands imports `WorkspaceState` from a newer API, but an older version
of `daytona-api-client` is installed.

**Fix:**

```bash
cd environments/prorl_openhands
poetry run pip install daytona-api-client==0.20.0
```

Or add to `environments/prorl_openhands/pyproject.toml` under `[tool.poetry.dependencies]`:
`daytona-api-client = "0.20.0"`. Do not widen the pin without checking compatibility.

### Error 5: `poetry env info failed` exits script early

**Context:** `start.sh` or `start_all.sh` uses `set -e` and the `poetry env info`
fallback command fails (e.g. on a machine where poetry was not set up for this
directory).

**Cause:** `set -eo pipefail` is active and `poetry env info --path` returns non-zero
when no venv exists for that project directory.

**Fix:** Use `|| true` to suppress the error, or set `ROLLOUT_FABRIC_PYTHON` and
`PRORL_OPENHANDS_PYTHON` explicitly before running any script:

```bash
export ROLLOUT_FABRIC_PYTHON=$(cd core && poetry env info --path)/bin/python
export PRORL_OPENHANDS_PYTHON=$(cd environments/prorl_openhands && poetry env info --path)/bin/python
```

The scripts check `ROLLOUT_FABRIC_PYTHON` / `PRORL_OPENHANDS_PYTHON` first and only
fall back to `poetry env info`. Use the dynamic form above — never hardcode the venv
hash path.

### Error 6: `Server is already running` / HTTP 400 from trainer on ProRL `/start`

**Context:** After a restart, `start_all.sh` sends `POST /start` and ProRL returns
HTTP 400 with a message like `Server is already running`.

**Cause:** ProRL was not killed before the restart, so the agent server is already
running. The `POST /start` endpoint rejects double-starts.

**Fix:** This is harmless. The ProRL server is running and will accept tasks. The
training run can proceed. If you need a clean restart, kill ProRL first:

```bash
kill $(cat /tmp/prorl.pid) 2>/dev/null || pkill -f start_server.py
sleep 2
bash environments/prorl_openhands/scripts/start.sh &
```

### Error 7: `ray.exceptions.RayTaskError` — vLLM pool unreachable at trainer start

**Context:** TrainerAdapter container starts, waits for vLLM pool health, and times
out after 300s.

**Cause:** InferenceBackend (Step 1) is not running or the `REMOTE_DNS` variable is
wrong inside the container.

**Fix:** The Docker `start.sh` passes `-e REMOTE_DNS="$REMOTE_DNS"`. Verify the value:

```bash
echo $REMOTE_DNS
# Should be the EC2 hostname, e.g.: ec2-3-87-168-160.compute-1.amazonaws.com
```

Then confirm vLLM is up before starting the trainer:

```bash
for port in 8100 8101 8102 8103; do
  curl -sf "http://${REMOTE_DNS}:${port}/health" && echo ":${port} OK"
done
```

### Error 8: `endpoints_failed > 0` — PolicyRegistry hard aborts

**Context:** Trainer logs show `PublishFailedError` / `ABORT: registry publish failed`.

**Cause:** One or more vLLM endpoints in the pool returned non-200/non-409 during
`reload_lora`. BC-9: this is a hard abort, not a degraded mode.

**Fix:** Check which vLLM port is down:

```bash
for port in 8100 8101 8102 8103; do
  curl -sf "http://${REMOTE_DNS}:${port}/health" && echo ":${port} OK" || echo ":${port} FAIL"
done
```

Restart the failed vLLM instance, then restart the training run from a checkpoint:

```bash
bash inference/vllm/scripts/launch_remote_vllm_pool.sh restart
# Then restart trainer with trainer.resume_mode=auto
```

---

## 8. Key Configuration Knobs

The trainer is configured via env vars passed to `trainers/verl/scripts/start.sh`.
All Hydra overrides are also accepted as positional args.

| Knob | Default | Effect |
|---|---|---|
| `BATCH_SIZE` | `4` | Number of groups per training step. Increase to 32 for production. |
| `GEN_BATCH_SIZE` | `4 × BATCH_SIZE` | Parquet rows the VERL dataloader reads per step. Must be ≤ parquet row count. Set to 3 if only a few SIFs are built. |
| `TOTAL_TRAINING_STEPS` | `500` | Hard stop after N gradient steps. |
| `SAVE_FREQ` | `1` | Publish a LoRA checkpoint via PolicyRegistry every N steps. |
| `STALENESS_CUTOFF_K` | `4` | Groups older than `current_step - K` are evicted from LiveStore as stale. |
| `REPLAY_ENABLE` | `True` | Allow replaying non-fresh groups from the buffer. Set False for pure on-policy. |
| `BUFFER_SIZE` | `64` | Trainer-side buffer size in groups. |
| `USE_TEMPORAL_IS` | `False` | Apply importance sampling correction for stale replay. |
| `SWAP_PROTOCOL` | `pinning` | vLLM LoRA swap protocol: `pinning` (faster) or `quiesce` (safer). |
| `FILTER_ZERO_VARIANCE` | `1` | (RolloutManager) Filter groups where all siblings have identical rewards. |
| `GROUP_SIZE` | `16` | GRPO/DAPO group size — N episodes dispatched per task. |
| `LIVE_STORE_MAX_SIZE` | `256` | Maximum number of groups in the LiveStore FIFO. |
| `NO_PROGRESS_TIMEOUT` | `1800` | Seconds the LiveStore waits for a new push before aborting. |

---

## 9. Reference

- Full BC table (BC-0 through BC-16): `plans-n-solutions/rollout_fabric.md`
- Design rationale: `plans-n-solutions/rollout_fabric.md`
- Service dependency footprint: `docs/service-envs.md`
- Deployment topology: `docs/topology.md`
- Protocol contracts: `core/rollout_fabric/schemas/protocols/PLUGGING_IN.md`
