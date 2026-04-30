# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project overview

ProRLAgent Server is a scalable multi-turn rollout service for training and evaluating RL agents. It is a fork of OpenHands — the upstream `openhands/` tree is kept largely intact, and the new RL-serving code lives under `openhands/nvidia/`, `openhands/llm/nvidia/`, `scripts/`, and `trainer_integration/verl/`. Agent jobs flow through a FastAPI server that dispatches to pluggable handlers and talks to vLLM via token-level I/O.

The forward design is in **[`plans-n-solutions/producer_as_a_service.md`](plans-n-solutions/producer_as_a_service.md)** — read that before changing the rollout/store/trainer wiring. This file covers the architectural invariants of the *current* in-process system you'll be migrating away from.

## Topology — three processes, two machines

| Role | Machine | Launcher |
|---|---|---|
| ProRL FastAPI server (`:8006`) | trainer box (host, poetry venv) | `bash scripts/_internal/s0_prorl.sh` |
| Remote vLLM pool (4 children, `:8100–:8103`) | EC2 `vllm-instance` (SSH alias) | `bash scripts/serving/launch_remote_vllm_pool.sh start` |
| Decoupled GRPO/DAPO trainer (Docker, 8×A100 FSDP) | trainer box | `bash scripts/_internal/s3_fullasync_docker.sh` |

Run in order — the trainer pre-flight probes pool `/health`. Stop in reverse. `WANDB_API_KEY` and other secrets are pinned in `/home/ubuntu/.prorl_creds.env` and sourced by all three launchers — do not re-export inline. Dataset is at `/home/ubuntu/data/SkyRL-v0-293/` (293 train / 23 val) — do not re-download.

## Architecture

### Three-stage async pipeline

`openhands/nvidia/async_server.py` implements `OpenHandsServer`, a producer-consumer pipeline with three stages, each with its own queue + worker pool + exception handler:

1. **init** — build `runtime`, `metadata`, `config` for a job
2. **run** — execute the agent; `runtime.close()` is called immediately after this stage
3. **eval** — score the run's output

Jobs flow as `JobDetails` (dataclass in `openhands/nvidia/registry.py`) that accumulates state across stages. Per-job `PausableTimer` ensures queue-waiting time doesn't count against the timeout.

### Registry + AgentHandler

Each job's `instance.data_source` keys into a registry (`openhands/nvidia/registry.py`) that dispatches to an `AgentHandler` implementing `init`, `run`, `eval`, matching `*_exception` hooks, and `final_result`. Two parallel registries exist (default, `reasoning=True`); `JobDetails.is_reasoning_task` selects.

Built-in handlers: `swe_agent/`, `math_coder/`, `stem_agent/`, `gui_agent/`. New task type: subclass `AgentHandler`, implement the seven methods, call `register_agent_handler(MyHandler())`. Handler `name` must match the `instance["data_source"]`.

### Token-in / token-out LLM path (INVARIANT — DO NOT MODIFY)

`openhands/llm/nvidia/qwen3.py` and `qwen2_5_vl.py` communicate with vLLM in **token IDs**, not text. Multi-turn RL stability: re-tokenizing decoded text across turns shifts boundaries, actor vs reference diverges, KL/entropy go NaN, PPO/GRPO collapses. The exact token IDs from each turn must be reused on the next turn. The replay store stores token IDs, not strings.

### Runtime

Default sandbox is **Singularity/Apptainer**, not Docker — rootless single-file `.sif`, Slurm-compatible. `OH_RUNTIME_SINGULARITY_IMAGE_REPO` points at the image cache. Tests: `TEST_RUNTIME=singularity`.

### RL trainer integration (current in-process loop)

`trainer_integration/verl/` is a patch package on top of `verlai/verl:vllm018.dev1` (`shamanez/verl` main, v0.8.0.dev). The current loop is fully-async decoupled agentic RL, all in one Python process inside the trainer container:

- **Producer** (`continuous_producer.py`) — daemon thread; calls ProRL → vLLM, eager-pushes filtered groups into the in-process replay store.
- **Replay store** (`trajectory_store.py`) — bounded FIFO deque, single `threading.Lock`, pop-on-sample, K-staleness eviction.
- **Trainer** (`ray_trainer.py` / `ray_trainer_dapo.py`) — samples from the store on its own cadence with clipped temporal IS correction; publishes rank-32 LoRA adapters to the pool on `save_freq` via `_publish_lora_adapter`.
- **Pool child** (`scripts/serving/_vllm_child.py`) — pinning swap protocol, `/v{N}/generate` path-pins to a policy version, `/reload_lora` installs new adapters with refcount-based eviction.

The producer-as-a-service design splits these three roles into separate services. See the design doc for the migration path.

## Critical gotchas

1. **Token-in/token-out is non-negotiable.** Never decode + re-tokenize across turns. The store stores token IDs.
2. **Frozen files — never edit, make siblings.** `scripts/_internal/s2_weightsync_docker.sh` and `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh` are kept as a matched-`global_steps` lock-step baseline for A/B.
3. **Remote EC2 DNS is hardcoded** in `run_proagent_qwn3_4B_instruct_fullasync.sh` and the weightsync sibling. Parameterize via `REMOTE_DNS=` before handing to a new environment. The SSH alias `vllm-instance` must exist in `~/.ssh/config` on the trainer box for `launch_remote_vllm_pool.sh` to work.
4. **`endpoints_failed > 0` abort contract is load-bearing.** A warm replay buffer must NOT mask a broken pool; publish failure raises from `_publish_lora_adapter` and the producer thread is stopped as part of trainer exit. Preserve this contract in any refactor.
5. **Sampling is consume-on-sample (queue semantics), not with-replacement.** `sample_mini_batch` pops chosen groups before returning. Whole groups are sampled intact (GRPO `compute_advantage` needs the `n` siblings present); `sample_mini_batch` never splits a group.
6. **Eager-push seam is the DAPO manager's job under `filter_groups=True`.** The continuous producer skips its terminal `push_from_dataproto` via `meta_info['eager_pushed_all']`. Never call `store.push_from_dataproto(out_batch)` unconditionally — double-push corrupts `behavior_policy_version` / `created_at_step` tracking under pop-on-sample.
7. **Buffer is ephemeral — not checkpointed.** On trainer resume (`trainer.resume_mode=auto`), the store starts empty and re-warms from scratch. Pre-resume entries would be maximally stale anyway.
8. **`policy_version` is read across threads without a lock.** Producer (daemon) reads, trainer (main) writes inside `_publish_lora_adapter`. Relies on CPython GIL atomicity of single-int load/store. The benign race is exactly the TIS correction's input — do NOT rewrite as a lock or `threading.Event`.

## Conventions

- Linter/formatter/type-checker configs live in `dev_config/python/` (`ruff.toml`, `mypy.ini`, pre-commit). **Do not modify** without explicit approval. Run `make lint` before committing.
- `pyproject.toml` pins `litellm` narrowly — several versions have known bugs or a supply-chain CVE. Read the comment before widening any pin.
- SWE-Bench / SWE-Gym / R2E-Gym packages install from git (not PyPI-usable form).
- **Never commit with `--no-verify`.** Fix the hook.
- **Never `git push --force` or push at all without explicit approval.**
- Pre-commit autoflake strips module-level imports it thinks are unused. Imports referenced only in decorators, late-bound methods, or generated strings need an inline import with `# noqa: PLC0415`. Precedent: `verl_custom/workers/fsdp_workers.py` (`_build_compute_log_prob`), `verl_custom/trainer/ppo/ray_trainer.py:1242` (inline `local_mkdir_safe`).

## Working principles

1. **Think before coding.** Surface assumptions; ask before guessing.
2. **Simplicity first.** Minimum code, no speculative abstractions, no premature error-handling.
3. **Surgical changes.** Only touch required lines; match existing style.
4. **Goal-driven.** Define verifiable success criteria with tests before you start.

## Edit groups (read files together)

- `openhands/nvidia/registry.py` ↔ `async_server.py` ↔ any concrete `AgentHandler`
- `openhands/llm/nvidia/qwen3.py` ↔ `qwen2_5_vl.py` (paired token-level clients)
- Same-basename collision: `openhands/nvidia/async_server.py` ≠ `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`
- `trainer_integration/verl/verl_custom/replay/{trajectory_store,continuous_producer}.py` ↔ the trainer's sample seam in `ray_trainer{,_dapo}.py`

## What NOT to do

- Don't touch `/tmp/verl` — pinned read-only upstream reference. Fork customizations live in `trainer_integration/verl/verl_custom/`.
- Don't touch `openhands/llm/nvidia/qwen3.py` or `qwen2_5_vl.py` (token-in/token-out invariant).
- Don't edit the frozen files in the topology table. Make siblings.
- Don't store decoded text across steps in the replay buffer.
- Don't commit `outputs/`, `wandb/`, `/tmp/*.log`, `singularity_images`, or anything under `/home/ubuntu/.prorl_creds.env`.
- Don't re-download the SkyRL-v0-293 dataset.
- Don't run `/codex:*` without a concrete diff in scope (codified in `.claude/rules/codex-usage.md`).
