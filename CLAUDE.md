# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

ProRLAgent Server is a scalable multi-turn rollout service for training/evaluating RL agents. It is a fork of OpenHands — the upstream `openhands/` tree is kept largely intact, and the new RL-serving code lives under `openhands/nvidia/`, `openhands/llm/nvidia/`, `scripts/`, and `trainer_integration/verl/`. Agent jobs flow through a FastAPI server that dispatches to pluggable handlers and talks to vLLM via token-level I/O.

## Commands

Dependencies are managed with Poetry (Python 3.12). The Makefile wraps most setup.

```bash
# Full env setup: checks deps, installs via poetry, installs pre-commit hooks
make build

# Install only what is needed for running rollouts + evaluation
poetry install --with dev,test,runtime,evaluation
pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git
pip install git+https://github.com/R2E-Gym/R2E-Gym.git

# Lint (pre-commit over openhands/**, evaluation/**, tests/**)
make lint                  # == lint-backend
make lint-scripts          # pre-commit over scripts/**

# Tests (pytest config is in pytest.ini; no warnings; per-function asyncio loop)
TEST_RUNTIME=singularity RUN_AS_OPENHANDS=False PYTHONPATH='.' \
    pytest tests/runtime/test_browsing.py -v -s

# NVIDIA module tests with markers
pytest -m "not integration and not slow" tests/nvidia/
pytest -m integration tests/nvidia/
pytest -m real_data tests/nvidia/     # skipped if the SWE-Gym parquet is missing
pytest --cov=openhands.nvidia --cov-report=term-missing tests/nvidia/

# Run a single test
pytest tests/nvidia/test_async_server.py::test_job_lifecycle -v
```

Running the rollout server and training:

```bash
# 1) Start a vLLM server yourself (see README.md step 2)
# 2) Start the FastAPI rollout server — REQUIRES this env var for Singularity images
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/path/to/singularity_images
python scripts/start_server.py --host 0.0.0.0 --port 8006 \
    --max-init-workers 64 --max-run-workers 64 --timeout 300

# 3) Register an LLM endpoint, then /start the worker (see README.md step 5)
# 4) Bulk SWE-Bench eval / RL loop:
python scripts/run_swe.py --dataset-path ... --llm-addresses http://.../v1 ...
bash trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh
```

## Architecture

### Three-stage async pipeline (the core abstraction)

`openhands/nvidia/async_server.py` implements `OpenHandsServer`, which runs a producer-consumer pipeline with three stages, each with its own queue, worker pool, and exception handler:

1. **init** — build `runtime`, `metadata`, `config` for a job
2. **run** — execute the agent; `runtime.close()` is called immediately after this stage
3. **eval** — score the run's output

Jobs are represented by `JobDetails` (a dataclass in `openhands/nvidia/registry.py`) that accumulates state as it flows through the stages (`runtime`, `metadata`, `config` → `run_results` → `eval_results`). Completion is signaled via `job_details.event.set()`. Timeouts are managed by a `PausableTimer` attached to each job so that queue-waiting time doesn't count against the budget. `openhands/nvidia/README.md` has a complete walkthrough of the pipeline and `JobDetails` lifecycle.

### Registry + AgentHandler (how new task types are added)

Each job's `instance.data_source` is used as a key into a registry (`openhands/nvidia/registry.py`) that dispatches to an `AgentHandler`. A handler implements `init`, `run`, `eval`, matching `*_exception` hooks, and `final_result`. There are two parallel registries: the default one and a `reasoning=True` one; `JobDetails.is_reasoning_task` selects between them. `add_name_mapping(name, mapped_name)` lets multiple `data_source` values route to the same underlying handler (e.g. all `deepcoder*` names auto-map to the `deepcoder` handler).

Built-in handlers live in:
- `openhands/nvidia/swe_agent/` — SWE-Gym / SWE-Bench / SWE-Bench-Multimodal / R2E-Gym
- `openhands/nvidia/math_coder/` — math + code (incl. a `prorl_handler`)
- `openhands/nvidia/stem_agent/`, `openhands/nvidia/gui_agent/`

To add a task type: subclass `AgentHandler`, implement the seven methods, and call `register_agent_handler(MyHandler())`. The handler's `name` must match the `instance["data_source"]` value that clients send.

### Token-in / token-out LLM path

`openhands/llm/nvidia/` (primarily `qwen3.py`, `qwen2_5_vl.py`) is the NVIDIA-specific LLM path that communicates with vLLM in **token IDs**, not text. This is a deliberate design choice for multi-turn RL stability: re-tokenizing decoded text across turns can shift token boundaries so that actor and reference models diverge, producing NaN KL/entropy and collapsing PPO/GRPO updates. When touching the LLM layer, preserve this invariant — exact token IDs from each turn must be reused on the next turn. See `openhands/llm/nvidia/README.md` for the full rationale.

A custom Qwen3 chat template in this module also keeps thinking content inside the `content` field, which matters for tool-calling flows.

### Runtime

The default sandbox here is **Singularity/Apptainer**, not Docker — chosen for rootless, single-file `.sif` execution and Slurm compatibility. `OH_RUNTIME_SINGULARITY_IMAGE_REPO` points at the local image cache. `scripts/pull_swe_images.py` converts Docker images referenced by SWE-Bench parquet files into `.sif` artifacts. Tests pick the runtime via `TEST_RUNTIME=singularity`. The upstream Docker runtime in `openhands/runtime/` still exists but is not the default path for rollouts.

### RL trainer integration

`trainer_integration/verl/` is a patch package installed on top of a pinned verl checkout (`git checkout 60138ebd` per README). It provides custom reward managers, reward scorers, rollout, and training scripts under `verl_custom/nvidia/`. Training scripts expect the FastAPI rollout server from step 2 to already be running.

## Conventions

- Linting and type-checking configs live in `dev_config/python/` (`ruff.toml`, `mypy.ini`, and the pre-commit config referenced by `make lint` and the `lint.yml` GitHub workflow). Run `make lint` before committing changes under `openhands/`, `evaluation/`, or `tests/`.
- `pyproject.toml` pins `litellm` to a narrow range — several versions have known bugs or a supply-chain CVE. Don't widen this without reading the comment there.
- The SWE-Bench / SWE-Gym / R2E-Gym packages must be installed from git (see `make build` section above); they are not on PyPI in the required form.

## Working Principles (Karpathy)

Four principles that reduce common LLM-assisted coding mistakes. Full rationale: `skills/karpathy-guidelines/SKILL.md`.

1. **Think before coding.** Surface assumptions; ask before guessing.
2. **Simplicity first.** Minimum code, no speculative abstractions, no premature error-handling.
3. **Surgical changes.** Only touch required lines; match existing style; don't drive-by refactor.
4. **Goal-driven execution.** Define verifiable success criteria with tests before you start.

## Claude Code Commands (quick-ref)

| Area | Commands |
|------|----------|
| Core | `/plan` `/tdd` `/verify` `/code-review` `/python-review` `/build-fix` |
| Testing | `/tdd` `/test-coverage` `/quality-gate` |
| Hygiene | `/refactor-clean` `/update-docs` `/docs` |
| Session | `/checkpoint` `/save-session` `/resume-session` `/sessions` `/aside` |
| Learning | `/learn` `/learn-eval` `/skill-create` |
| Context | `/context-budget` `/harness-audit` |
| Runtime | `/monitor-run --url <endpoint>` |

## Skills index (auto-load on match)

| When editing | Skill |
|--------------|-------|
| any `**/*.py` | `python-patterns`, `python-conventions` (rule) |
| `tests/**/*.py` | `python-testing`, `testing-conventions` (rule) |
| Unfamiliar module | `repo-architecture` |
| Long session | `strategic-compact` |
| Before verifying | `verification-loop` |
| Reviewing security of new code | `security-review` |
| Designing an eval | `eval-harness` |

## Edit groups (read files together)

- `openhands/nvidia/registry.py` ↔ `async_server.py` ↔ any concrete `AgentHandler`
- `openhands/llm/nvidia/qwen3.py` ↔ `qwen2_5_vl.py` (paired token-level clients)
- Same-basename collision: `openhands/nvidia/async_server.py` ≠ `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`
- Schema ↔ loader ↔ fixture: `config.template.toml` ↔ `openhands/core/config/` ↔ any `tests/**/conftest.py`
