# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project overview

ProRL-Agent-Server is an agentic RL training fabric for Qwen3-4B on SWE-Bench tasks
(GRPO/DAPO). Six independent services are wired by gRPC/HTTP contracts; every service
sits behind a typed `Protocol` and is independently replaceable.

Design rationale and BC definitions: `plans-n-solutions/rollout_fabric.md`
Operational details (SIF build, rescue team): `plans-n-solutions/rollout_fabric_progress.md`
End-to-end training flow (all 6 services, groups, staleness): `docs/TRAINING_FLOW.md`
Deployment topology + co-location rules: `docs/topology.md`
Per-service Python env footprint: `docs/service-envs.md`
Operational runbook (startup, monitoring, errors): `docs/TRAINING_OPERATIONS.md`
How to plug in a new environment or trainer: `core/rollout_fabric/schemas/protocols/PLUGGING_IN.md` and `docs/PLUGGING_IN_NEW_TRAINER_OR_ENVIRONMENT.md`

**Four `pyproject.toml` files:**
- `./pyproject.toml` — dev workspace (ruff, mypy, pytest, pre-commit)
- `./core/pyproject.toml` — fabric-core installable (5 runtime deps: grpcio, protobuf, pyarrow, httpx, requests)
- `./environments/prorl_openhands/pyproject.toml` — EnvironmentProvider (OpenHands + litellm + docker + e2b...)
- `./trainers/verl/pyproject.toml` — VERL TrainerAdapter (setuptools, Docker-only)

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
| EnvironmentProvider | `:8006` | `environments/prorl_openhands/scripts/start.sh` | `GET :8006/status` → `{"status":"running"}` | Must not own training data. Must not call LiveStore. |
| InferenceBackend | `:8100-8103` (EC2) | `inference/vllm/scripts/launch_remote_vllm_pool.sh` | `GET :810N/health` → 200 each | Must not know policy version semantics. |
| LiveStore | `/tmp/prorl_live_store.sock` | inline (see Step 3a) | socket exists | Pop-on-sample. Server-side blocking `get_batch`. Never returns without popping. |
| PolicyRegistry | `/tmp/prorl_policy_registry.sock` | inline (see Step 3b) | socket exists | Hard abort if `endpoints_failed > 0`. Never degrades silently. |
| RolloutManager | no port (client only) | `python -m rollout_fabric.rollout_manager.main` | first push logged | Zero VERL/OpenHands imports. Owns parquet dataloader (BC-14). |
| TrainerAdapter | Docker internal | `trainers/verl/scripts/start.sh` | begins `get_batch` calls | Connects ONLY to LiveStore + PolicyRegistry. No parquet. No ProRL address. No vLLM address. |

---

## Startup sequence

> **Authoritative guide:** `docs/TRAINING_OPERATIONS.md` — Section 0 (from-scratch checklist) and Section 3 (step-by-step with health gates, knob table, error recovery).

**From scratch — run once:**

```bash
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
source /home/ubuntu/.prorl_creds.env
docker build -f trainers/verl/Dockerfile -t prorl/verl-trainer:vllm018 .
cd core && poetry install && cd ..
cd environments/prorl_openhands && poetry install && cd ../..
$(cd environments/prorl_openhands && poetry env info --path)/bin/pip install \
    "git+https://github.com/SWE-Gym/SWE-Bench-Package.git"
```

**Every session:**

```bash
source /home/ubuntu/.prorl_creds.env
export ROLLOUT_FABRIC_PYTHON=$(cd core && poetry env info --path)/bin/python
export PRORL_OPENHANDS_PYTHON=$(cd environments/prorl_openhands && poetry env info --path)/bin/python
export DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.ready.parquet"
export POLICY_ID="qwen3-4b-skyrl"
export ENVIRONMENT_ID="swe_agent"
bash inference/vllm/scripts/launch_remote_vllm_pool.sh start  # skip if already running
bash ops/services/start_all.sh
```

`start_all.sh` starts services 1–5 in dependency order, enforces the BC-16 warm-up gate (trainer starts only after ≥1 group in LiveStore), and shuts down in reverse on Ctrl-C. Stop order: trainer → rollout manager → LiveStore + PolicyRegistry → EnvironmentProvider + InferenceBackend.

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

`environments/prorl_openhands/openhands/llm/nvidia/qwen3.py` and `qwen2_5_vl.py`
communicate with vLLM in token IDs, not text. Re-tokenizing decoded text across turns
shifts token boundaries; actor vs reference diverges; KL/entropy go NaN;
PPO/GRPO/DAPO collapses. These files are frozen.

**Frozen files — never edit, make siblings:**
- `environments/prorl_openhands/openhands/llm/nvidia/qwen3.py`
- `environments/prorl_openhands/openhands/llm/nvidia/qwen2_5_vl.py`
- `inference/vllm/scripts/_vllm_child.py`
- `environments/prorl_openhands/openhands/nvidia/async_server.py`

---

## What NOT to do

- Do not add any VERL or OpenHands import to `core/rollout_fabric/rollout_manager/` (BC-13).
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

- `core/rollout_fabric/live_store/store_core.py` ↔ `core/rollout_fabric/live_store/server.py` ↔ `core/rollout_fabric/live_store/client.py` ↔ `core/rollout_fabric/live_store/codec.py`
- `core/rollout_fabric/policy_registry/server.py` ↔ `core/rollout_fabric/policy_registry/client.py` ↔ `core/rollout_fabric/policy_registry/fanout.py`
- `core/rollout_fabric/rollout_manager/loop.py` ↔ `core/rollout_fabric/rollout_manager/episode_builder.py` ↔ `core/rollout_fabric/rollout_manager/policy_subscription.py`
- `core/rollout_fabric/schemas/protocols/` ↔ `core/rollout_fabric/schemas/proto/*.proto` ↔ `core/rollout_fabric/schemas/_gen/`
- `trainers/verl/verl_custom/fabric_adapter/pad.py` ↔ trainer's `sample_mini_batch` seam in `ray_trainer_dapo.py`
- `environments/prorl_openhands/openhands/nvidia/registry.py` ↔ `environments/prorl_openhands/openhands/nvidia/async_server.py` ↔ any concrete `AgentHandler`
- `environments/prorl_openhands/openhands/llm/nvidia/qwen3.py` ↔ `environments/prorl_openhands/openhands/llm/nvidia/qwen2_5_vl.py` (paired, frozen)
- Same-basename collision: `environments/prorl_openhands/openhands/nvidia/async_server.py` ≠ `trainers/verl/verl_custom/nvidia/rollout/async_server.py`

---

## Conventions

- Linter/formatter/type-checker configs: `dev_config/python/` (`ruff.toml`, `mypy.ini`, pre-commit). Do not modify without explicit approval.
- Run `make lint` before committing.
- `environments/prorl_openhands/pyproject.toml` pins `litellm` narrowly — read the comment before widening any pin.
- SWE-Bench / SWE-Gym / R2E-Gym packages install from git (not PyPI).
- Pre-commit autoflake strips imports it thinks are unused. Imports referenced only in
  decorators or late-bound methods need `# noqa: PLC0415`.

---

## Working principles

1. **Think before coding.** Surface assumptions; ask before guessing.
2. **Simplicity first.** Minimum code, no speculative abstractions, no premature error-handling.
3. **Surgical changes.** Only touch required lines; match existing style.
4. **Goal-driven.** Define verifiable success criteria with tests before you start.
