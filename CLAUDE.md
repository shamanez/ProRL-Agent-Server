# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project overview

ProRL-Agent-Server is a contract-first agentic RL training fabric. Five independent
services are wired by gRPC/HTTP contracts; every service is independently replaceable
by swapping the adapter implementation behind that slot's typed `Protocol`.

The forward design and full boundary-condition table live in
`plans-n-solutions/rollout_fabric.md`. The operational companion — what is running,
how to start it, what breaks and why — is in
`plans-n-solutions/rollout_fabric_progress.md`. Read both before touching the
rollout/store/trainer wiring.

---

## The 5-service architecture

| Service | Script | Health | Role | Boundary |
|---|---|---|---|---|
| EnvironmentProvider | `scripts/_internal/s0_prorl.sh` | `GET :8006/status` → `{"status":"running"}` | Runs OpenHands agent loop inside Singularity sandbox; returns token IDs + logprobs. | Must not own training data. Must not call LiveStore. |
| InferenceBackend | `scripts/serving/launch_remote_vllm_pool.sh start` | `GET :8100-8103/health` → 200 each | Token-level generation; LoRA hot-reload via path-versioned pinning. | Must not know policy version semantics. Must not call LiveStore. |
| LiveStore | inline — see startup sequence | `[[ -S /tmp/prorl_live_store.sock ]]` | Bounded hot FIFO between RolloutManager and trainer. Pop-on-sample; server-side blocking get_batch; staleness eviction. | Never returns without popping (no with-replacement). Never exposes internal deque directly. |
| PolicyRegistry | inline — see startup sequence | `[[ -S /tmp/prorl_policy_registry.sock ]]` | Single source of truth for LoRA version and adapter URI. Fans out `/reload_lora` to pool children. Hard abort if `endpoints_failed > 0`. | Never degrades silently on publish failure. |
| RolloutManager | `python -m rollout_manager.main` | Worker logs first push to LiveStore | Reads parquet, dispatches episodes to EnvironmentProvider, stamps policy version, pushes groups to LiveStore. | Zero VERL/OpenHands imports. Owns parquet dataloader (BC-14). |
| TrainerAdapter | `scripts/_internal/s3_fullasync_docker.sh` | Begins `get_batch` calls without timeout | Consumes groups from LiveStore; pads locally; runs FSDP forward/backward; publishes policy versions to PolicyRegistry. | No parquet. No dataloader. No ProRL address. No vLLM address. Connects only to LiveStore + PolicyRegistry. |

---

## Startup sequence

Run in strict order. Each step must pass its health gate before the next begins.

```bash
# 0. Prerequisites
source /home/ubuntu/.prorl_creds.env
export DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.ready.parquet"
export POLICY_ID="qwen3-4b-skyrl"
export ENVIRONMENT_ID="swe_agent"
export PYTHONPATH=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
POETRY_PYTHON=$(poetry run python -c "import sys; print(sys.executable)")
```

**Before starting the worker**, the parquet must be filtered to tasks with built SIF
images. Run once after any new SIF images are added:

```bash
poetry run python scripts/filter_parquet_to_built_sifs.py \
  --input /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --sif-dir singularity_images \
  --output /home/ubuntu/data/SkyRL-v0-293/train.ready.parquet
```

### Step 1 — InferenceBackend (remote EC2 vLLM pool)

```bash
bash scripts/serving/launch_remote_vllm_pool.sh start
# Health gate: all 4 ports return 200
for port in 8100 8101 8102 8103; do
  curl -sf --max-time 3 "http://${REMOTE_DNS}:${port}/health" && echo ":${port} OK"
done
```

### Step 2 — EnvironmentProvider (ProRL :8006)

```bash
nohup bash scripts/_internal/s0_prorl.sh > /tmp/s0-prorl.log 2>&1 &
echo $! > /tmp/prorl.pid
# Health gate:
for i in $(seq 1 30); do
  curl -sf http://localhost:8006/status -o /dev/null && { echo "ProRL up"; break; }; sleep 1
done
curl -sf -X POST http://localhost:8006/start -H "Content-Type: application/json" -d '{}'
```

### Step 3a — LiveStore (parallel with 3b)

```bash
nohup $POETRY_PYTHON -c "
import logging, signal, sys
sys.path.insert(0, '/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server')
logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s live_store: %(message)s')
from live_store.server import serve
server = serve(socket_path='/tmp/prorl_live_store.sock',
               max_size=256, staleness_cutoff_k=4, no_progress_timeout_s=1800)
print('[live_store] healthy', flush=True)
def _stop(s, f): server.stop(grace=2.0); sys.exit(0)
signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)
server.wait_for_termination()
" > /tmp/live_store.log 2>&1 &
echo $! > /tmp/live_store.pid
# Health gate: [[ -S /tmp/prorl_live_store.sock ]]
```

### Step 3b — PolicyRegistry (parallel with 3a)

```bash
ENDPOINTS_PY="['http://${REMOTE_DNS}:8100','http://${REMOTE_DNS}:8101','http://${REMOTE_DNS}:8102','http://${REMOTE_DNS}:8103']"
nohup $POETRY_PYTHON -c "
import logging, signal, sys
sys.path.insert(0, '/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server')
logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s policy_registry: %(message)s')
from policy_registry.server import serve
server = serve(socket_path='/tmp/prorl_policy_registry.sock',
               db_path='/tmp/prorl_policy_registry.db',
               pool_endpoints=${ENDPOINTS_PY})
print('[policy_registry] healthy', flush=True)
def _stop(s, f): server.stop(grace=2.0); sys.exit(0)
signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)
server.wait_for_termination()
" > /tmp/policy_registry.log 2>&1 &
echo $! > /tmp/policy_registry.pid
# Health gate: [[ -S /tmp/prorl_policy_registry.sock ]]
```

### Step 4 — RolloutManager

```bash
nohup $POETRY_PYTHON -m rollout_manager.main \
  --live-store-socket /tmp/prorl_live_store.sock \
  --prorl-url http://localhost:8006 \
  --policy-id "${POLICY_ID}" \
  --environment-id "${ENVIRONMENT_ID}" \
  --data-files "${DATA_FILES}" \
  --group-size 4 \
  --policy-manifest-path /tmp/prorl_policy_manifest.json \
  --archive-root /home/ubuntu/replay_archive \
  --filter-zero-variance \
  --archive-disabled \
  > /tmp/rollout_manager.log 2>&1 &
echo $! > /tmp/rollout_manager.pid
# Health gate: wait for >=1 group in LiveStore before starting trainer
```

### Step 5 — TrainerAdapter

```bash
# Start ONLY after the rollout manager has pushed >=1 group to LiveStore (BC-16).
bash scripts/_internal/s3_fullasync_docker.sh
```

**Orchestrated startup:** `bash scripts/services/start_all.sh` runs steps 1–5 in order,
enforces the BC-16 warm-up gate, and handles clean shutdown in reverse order on Ctrl-C.

**Stop order:** trainer → rollout manager → LiveStore + PolicyRegistry → EnvironmentProvider + InferenceBackend.

---

## Key invariants (load-bearing)

**BC-0 — One PolicyVersionSnapshot per group.**
All N sibling episodes in a group are dispatched with the same snapshot. No trajectory
may span two policy versions. Violating this invalidates advantage computation.

**BC-1 — Token IDs as `int` on every wire.**
Token arrays are packed int32-LE bytes in protobuf, list[int] in Python.
Never `str`. KL/entropy goes NaN within 2 training steps if this is violated.

**BC-9 — `endpoints_failed > 0` is a hard abort, not degraded mode.**
`PolicyRegistryClient.publish_policy_version()` raises `PublishFailedError` on any
pool child non-200/non-409. A warm replay buffer must not mask a broken pool.

**BC-13 — RolloutManager imports zero VERL/OpenHands code.**
The worker calls ProRL via plain HTTP. Framework coupling destroys pluggability.

**BC-14 — RolloutManager owns the parquet dataloader.**
The trainer has no `data.train_files`, no `StatefulDataLoader`, no producer thread.

**BC-15 — Trainer connects only to LiveStore and PolicyRegistry.**
The trainer has no ProRL address and no vLLM address. Swapping VERL for another
trainer requires only changing the Docker image and the Hydra command.

---

## Token-in / token-out (INVARIANT — DO NOT MODIFY)

`openhands/llm/nvidia/qwen3.py` and `qwen2_5_vl.py` communicate with vLLM in token IDs,
not text. Re-tokenizing decoded text across turns shifts token boundaries; actor vs
reference diverges; KL/entropy go NaN; PPO/GRPO/DAPO collapses. These files are frozen.

**Frozen files — never edit, make siblings:**
- `openhands/llm/nvidia/qwen3.py`
- `openhands/llm/nvidia/qwen2_5_vl.py`
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh`
- `scripts/serving/_vllm_child.py`
- `openhands/nvidia/async_server.py`

---

## What NOT to do

- Do not add any VERL or OpenHands import to `rollout_manager/` (BC-13).
- Do not give the trainer a parquet path or a `StatefulDataLoader` (BC-14, BC-15).
- Do not give the trainer a ProRL address or a vLLM address (BC-15).
- Do not call `push_group` twice for the same group — double-push corrupts
  `behavior_policy_version` / `created_at_step` tracking under pop-on-sample.
- Do not decode + re-tokenize token IDs across turns. The store stores int arrays.
- Do not store decoded text strings anywhere in the replay pipeline.
- Do not fall back to the in-process `TrajectoryStore` / `ContinuousRolloutProducer`
  path — that coupling is the thing we are migrating away from.
- Do not touch `/tmp/verl` — pinned read-only upstream reference.
- Do not edit the frozen files listed above. Make siblings.
- Do not commit `outputs/`, `wandb/`, `/tmp/*.log`, `singularity_images`, or
  anything under `/home/ubuntu/.prorl_creds.env`.
- Do not re-download the SkyRL-v0-293 dataset.
- Do not commit with `--no-verify`. Fix the hook.
- Do not `git push --force` or push without explicit approval.
- Do not run `/codex:*` without a concrete diff in scope.
- Do not widen pinned dependencies in `pyproject.toml` without reading the pin comment.

---

## Edit groups (read files together)

- `live_store/store_core.py` ↔ `live_store/server.py` ↔ `live_store/client.py` ↔ `live_store/codec.py`
- `policy_registry/server.py` ↔ `policy_registry/client.py` ↔ `policy_registry/fanout.py`
- `rollout_manager/loop.py` ↔ `rollout_manager/episode_builder.py` ↔ `rollout_manager/policy_subscription.py`
- `schemas/protocols/` ↔ `schemas/proto/*.proto` ↔ `schemas/_gen/`
- `trainer_adapters/verl/pad.py` ↔ trainer's `sample_mini_batch` seam in `ray_trainer_dapo.py`
- `openhands/nvidia/registry.py` ↔ `openhands/nvidia/async_server.py` ↔ any concrete `AgentHandler`
- `openhands/llm/nvidia/qwen3.py` ↔ `openhands/llm/nvidia/qwen2_5_vl.py` (paired, frozen)
- Same-basename collision: `openhands/nvidia/async_server.py` ≠ `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`

---

## Conventions

- Linter/formatter/type-checker configs: `dev_config/python/` (`ruff.toml`, `mypy.ini`, pre-commit). Do not modify without explicit approval.
- Run `make lint` before committing.
- `pyproject.toml` pins `litellm` narrowly — read the comment before widening any pin.
- SWE-Bench / SWE-Gym / R2E-Gym packages install from git (not PyPI).
- Pre-commit autoflake strips imports it thinks are unused. Imports referenced only in
  decorators or late-bound methods need `# noqa: PLC0415`.

---

## Working principles

1. **Think before coding.** Surface assumptions; ask before guessing.
2. **Simplicity first.** Minimum code, no speculative abstractions, no premature error-handling.
3. **Surgical changes.** Only touch required lines; match existing style.
4. **Goal-driven.** Define verifiable success criteria with tests before you start.
