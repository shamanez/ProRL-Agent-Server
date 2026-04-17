# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project overview

ProRLAgent Server is a scalable multi-turn rollout service for training/evaluating RL agents. It is a fork of OpenHands — the upstream `openhands/` tree is kept largely intact, and the new RL-serving code lives under `openhands/nvidia/`, `openhands/llm/nvidia/`, `scripts/`, and `trainer_integration/verl/`. Agent jobs flow through a FastAPI server that dispatches to pluggable handlers and talks to vLLM via token-level I/O.

## Commands

Python 3.12, Poetry for deps. The Makefile wraps most setup.

```bash
# Env
make build                                   # poetry install + pre-commit install
pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git
pip install git+https://github.com/R2E-Gym/R2E-Gym.git

# Lint (pre-commit: ruff + mypy + pyproject-fmt)
make lint                 # openhands/**, evaluation/**, tests/**
make lint-scripts         # scripts/**

# Fast test loop
pytest -m "not integration and not slow and not real_data" tests/ -q

# Runtime tests (need env vars)
TEST_RUNTIME=singularity RUN_AS_OPENHANDS=False PYTHONPATH=. \
    pytest tests/runtime/test_browsing.py -v -s

# NVIDIA subtree
pytest -m "not integration and not slow" tests/nvidia/
pytest --cov=openhands.nvidia --cov-report=term-missing tests/nvidia/
```

Running the rollout server and training:

```bash
# 1) Start vLLM (see README.md step 2)
# 2) Start rollout server — REQUIRES image repo env var
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/path/to/singularity_images
python scripts/start_server.py --host 0.0.0.0 --port 8006 \
    --max-init-workers 64 --max-run-workers 64 --timeout 300

# 3) Register LLM + /start (see README.md step 5)
# 4) Bulk eval / RL loop
python scripts/run_swe.py --dataset-path ... --llm-addresses http://.../v1 ...
bash trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh
```

## Architecture

### Three-stage async pipeline

`openhands/nvidia/async_server.py` implements `OpenHandsServer`, a producer-consumer pipeline with three stages, each with its own queue + worker pool + exception handler:

1. **init** — build `runtime`, `metadata`, `config` for a job
2. **run** — execute the agent; `runtime.close()` is called immediately after this stage
3. **eval** — score the run's output

Jobs flow as `JobDetails` (dataclass in `openhands/nvidia/registry.py`) that accumulates state across stages. Completion is signaled via `job_details.event.set()`. Per-job `PausableTimer` ensures queue-waiting time doesn't count against the timeout. See `openhands/nvidia/README.md` for the full lifecycle.

### Registry + AgentHandler

Each job's `instance.data_source` keys into a registry (`openhands/nvidia/registry.py`) that dispatches to an `AgentHandler` implementing `init`, `run`, `eval`, matching `*_exception` hooks, and `final_result`. Two parallel registries exist (default, `reasoning=True`); `JobDetails.is_reasoning_task` selects. `add_name_mapping(name, mapped_name)` routes multiple `data_source` values to one handler.

Built-in handlers:
- `openhands/nvidia/swe_agent/` — SWE-Gym / SWE-Bench / SWE-Bench-Multimodal / R2E-Gym
- `openhands/nvidia/math_coder/` — math + code (incl. `prorl_handler`)
- `openhands/nvidia/stem_agent/`, `openhands/nvidia/gui_agent/`

New task type: subclass `AgentHandler`, implement the seven methods, call `register_agent_handler(MyHandler())`. Handler `name` must match the `instance["data_source"]`.

### Token-in / token-out LLM path (INVARIANT)

`openhands/llm/nvidia/` (primarily `qwen3.py`, `qwen2_5_vl.py`) communicates with vLLM in **token IDs**, not text. Multi-turn RL stability: re-tokenizing decoded text across turns shifts boundaries, actor vs reference diverges, KL/entropy go NaN, PPO/GRPO collapses. Preserve this invariant — exact token IDs from each turn must be reused on the next turn. See `openhands/llm/nvidia/README.md`.

The custom Qwen3 chat template keeps thinking content inside the `content` field (matters for tool-calling flows).

### Runtime

Default sandbox is **Singularity/Apptainer**, not Docker — rootless single-file `.sif`, Slurm-compatible. `OH_RUNTIME_SINGULARITY_IMAGE_REPO` points at the image cache. `scripts/pull_swe_images.py` converts SWE-Bench Docker refs into `.sif` artifacts. Tests: `TEST_RUNTIME=singularity`. The upstream Docker runtime still exists but isn't the default.

### RL trainer integration

`trainer_integration/verl/` is a patch package on top of a pinned verl checkout. **Current stack (post Stage 0.1):** `shamanez/verl` main (v0.8.0.dev, commit `910ba344`) at `/tmp/verl` + Docker image `verlai/verl:vllm018.dev1` (vLLM 0.18, PyTorch 2.6+). Training runs inside the container; ProRL runs on the host at `:8006`. See `plans-n-solutions/stages/stage0_1.md` for the upgrade record and cold-start bootstrap.

## Conventions

- Linter/formatter/type-checker configs live in `dev_config/python/` (`ruff.toml`, `mypy.ini`, pre-commit config). **Do not modify** without explicit approval. Run `make lint` before committing.
- `pyproject.toml` pins `litellm` narrowly — several versions have known bugs or a supply-chain CVE. Read the comment before widening.
- SWE-Bench / SWE-Gym / R2E-Gym packages install from git (not PyPI-usable form).
- **Never commit with `--no-verify`.** Fix the hook.
- Pre-commit autoflake strips module-level imports it thinks are unused. Imports referenced only in decorators, late-bound methods, or generated strings need a closure or an inline import with `# noqa: PLC0415`. Precedent: `verl_custom/workers/fsdp_workers.py` (`_build_compute_log_prob`), `verl_custom/trainer/ppo/ray_trainer.py:1242` (inline `local_mkdir_safe`).

## Working principles

1. **Think before coding.** Surface assumptions; ask before guessing.
2. **Simplicity first.** Minimum code, no speculative abstractions, no premature error-handling.
3. **Surgical changes.** Only touch required lines; match existing style.
4. **Goal-driven.** Define verifiable success criteria with tests before you start.

## Edit groups (read files together)

- `openhands/nvidia/registry.py` ↔ `async_server.py` ↔ any concrete `AgentHandler`
- `openhands/llm/nvidia/qwen3.py` ↔ `qwen2_5_vl.py` (paired token-level clients)
- Same-basename collision: `openhands/nvidia/async_server.py` ≠ `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`
- Schema ↔ loader ↔ fixture: `config.template.toml` ↔ `openhands/core/config/` ↔ any `tests/**/conftest.py`
