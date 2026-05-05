# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project overview

ProRL-Agent-Server is an agentic RL training fabric for Qwen3-4B on SWE-Bench tasks
(GRPO/DAPO). Six independent services are wired by gRPC/HTTP contracts; every service
sits behind a typed `Protocol` and is independently replaceable.

Design rationale and BC definitions: `plans-n-solutions/rollout_fabric.md`
Operational details (SIF build, rescue team): `plans-n-solutions/rollout_fabric_progress.md`
Deployment topology + co-location rules: `docs/topology.md`
Per-service Python env footprint: `docs/service-envs.md`
How to plug in a new environment or trainer: `schemas/protocols/PLUGGING_IN.md`

**Two `pyproject.toml` files:**
- `./pyproject.toml` — fabric-core (5 packages: grpcio, protobuf, pyarrow, httpx, requests)
- `./openhands/pyproject.toml` — EnvironmentProvider (OpenHands + litellm + docker + e2b...)

---

## Service architecture

Six services in data-flow order. Read the diagram, then the table.

```
  SkyRL parquet
       │ (ParquetDataLoader — RolloutManager owns this, BC-14)
       ▼
  RolloutManager ──POST /process──► EnvironmentProvider (ProRL :8006 + Singularity)
       │                                    │ (calls vLLM for token generation)
       │                                    ▼
       │                             InferenceBackend (vLLM :8100-8103 EC2)
       │                                    ▲
  gRPC push_group                    POST /reload_lora
       │                                    │
       ▼                             PolicyRegistry (UDS /tmp/prorl_policy_registry.sock)
  LiveStore ──────gRPC get_batch──►        ▲
  (UDS /tmp/prorl_live_store.sock)  gRPC publish_policy_version
                                           │
                                    TrainerAdapter (VERL FSDP, Docker 8×A100)
```

| Service | Socket / Port | Script | Health check | Boundary |
|---|---|---|---|---|
| EnvironmentProvider | `:8006` | `scripts/adapters/start_env_prorl.sh` | `GET :8006/status` → `{"status":"running"}` | Must not own training data. Must not call LiveStore. |
| InferenceBackend | `:8100-8103` (EC2) | `scripts/inference/launch_remote_vllm_pool.sh` | `GET :810N/health` → 200 each | Must not know policy version semantics. |
| LiveStore | `/tmp/prorl_live_store.sock` | inline (see Step 3a) | socket exists | Pop-on-sample. Server-side blocking `get_batch`. Never returns without popping. |
| PolicyRegistry | `/tmp/prorl_policy_registry.sock` | inline (see Step 3b) | socket exists | Hard abort if `endpoints_failed > 0`. Never degrades silently. |
| RolloutManager | no port (client only) | `python -m rollout_manager.main` | first push logged | Zero VERL/OpenHands imports. Owns parquet dataloader (BC-14). |
| TrainerAdapter | Docker internal | `scripts/adapters/start_trainer_verl.sh` | begins `get_batch` calls | Connects ONLY to LiveStore + PolicyRegistry. No parquet. No ProRL address. No vLLM address. |

---

## Startup sequence

Run in strict order. Each step must pass its health gate before the next begins.

```bash
# Prerequisites
source /home/ubuntu/.prorl_creds.env
export DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.ready.parquet"
export POLICY_ID="qwen3-4b-skyrl"
export ENVIRONMENT_ID="swe_agent"
export PYTHONPATH=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server

# ── Two poetry environments (three total including Docker) ───────────────────
#
#  1. Fabric-core  (pyproject.toml in repo root — 5 packages, fast install)
#     poetry install
#     ROLLOUT_FABRIC_PYTHON=$(poetry env info --path)/bin/python
#
#  2. EnvironmentProvider  (openhands/pyproject.toml — full OpenHands stack)
#     cd openhands && poetry install && cd ..
#     PRORL_OPENHANDS_PYTHON=$(cd openhands && poetry env info --path)/bin/python
#
#  3. TrainerAdapter  (verlai/verl Docker — installed at container start, not baked in)
#     NEVER use the host poetry env for trainer deps.
#     /tmp/verl is bind-mounted as /opt/verl inside the container.
#
# On this machine the pre-built envs are still present as fallback:
#   ROLLOUT_FABRIC_PYTHON=/home/ubuntu/.cache/pypoetry/virtualenvs/openhands-ai-342rfuwh-py3.12/bin/python
#   PRORL_OPENHANDS_PYTHON=/home/ubuntu/.cache/pypoetry/virtualenvs/openhands-ai-342rfuwh-py3.12/bin/python
# ─────────────────────────────────────────────────────────────────────────────
```

**Before starting the worker**, filter the parquet to tasks with built SIF images.
Run once after any new SIFs are added:

```bash
python scripts/data/filter_parquet_to_built_sifs.py \
  --input /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --sif-dir singularity_images \
  --output /home/ubuntu/data/SkyRL-v0-293/train.ready.parquet
```

### Step 1 — InferenceBackend

```bash
bash scripts/inference/launch_remote_vllm_pool.sh start
# Health gate: all 4 ports return 200
for port in 8100 8101 8102 8103; do
  curl -sf --max-time 3 "http://${REMOTE_DNS}:${port}/health" && echo ":${port} OK"
done
```

### Step 2 — EnvironmentProvider

```bash
nohup bash scripts/adapters/start_env_prorl.sh > /tmp/s0-prorl.log 2>&1 &
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
# Health gate: wait for >=1 group in LiveStore before starting trainer (BC-16)
```

### Step 5 — TrainerAdapter

```bash
# Start ONLY after RolloutManager has pushed >=1 group (BC-16).
bash scripts/adapters/start_trainer_verl.sh
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
Token arrays are packed int32-LE bytes in protobuf, `list[int]` in Python.
Never `str`. KL/entropy goes NaN within 2 training steps if violated.

**BC-9 — `endpoints_failed > 0` is a hard abort, not degraded mode.**
`PolicyRegistryClient.publish_policy_version()` raises `PublishFailedError` on any
pool child non-200/non-409. A warm replay buffer must not mask a broken pool.

**BC-13 — RolloutManager imports zero VERL/OpenHands code.**
The worker calls ProRL via plain `httpx`. Framework coupling destroys pluggability.

**BC-14 — RolloutManager owns the parquet dataloader.**
The trainer has no `data.train_files`, no `StatefulDataLoader`, no producer thread.

**BC-15 — Trainer connects only to LiveStore and PolicyRegistry.**
The trainer has no ProRL address and no vLLM address. Swapping VERL for another
trainer requires only changing the Docker image and the Hydra command.

**BC-16 — LiveStore warm-up: start trainer AFTER worker pushes ≥1 group.**
The no-progress detector (1800s) fires only if the worker stops entirely — not because
it is slow. Start the trainer too early and you burn the timer during warm-up.

Full BC table (BC-0 through BC-15): `plans-n-solutions/rollout_fabric.md`

---

## Token-in / token-out (INVARIANT — DO NOT MODIFY)

`openhands/llm/nvidia/qwen3.py` and `qwen2_5_vl.py` communicate with vLLM in token IDs,
not text. Re-tokenizing decoded text across turns shifts token boundaries; actor vs
reference diverges; KL/entropy go NaN; PPO/GRPO/DAPO collapses. These files are frozen.

**Frozen files — never edit, make siblings:**
- `openhands/llm/nvidia/qwen3.py`
- `openhands/llm/nvidia/qwen2_5_vl.py`
- `scripts/inference/_vllm_child.py`
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
