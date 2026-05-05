# Service Python Environments

## Two `pyproject.toml` files — three environments total

| `pyproject.toml` | Venv name | What it covers |
|---|---|---|
| `./pyproject.toml` | `rollout-fabric-*` | LiveStore, PolicyRegistry, ReplayArchive, RolloutManager, schemas |
| `./openhands/pyproject.toml` | `prorl-environment-provider-*` | ProRL FastAPI server (OpenHands + litellm + docker + e2b...) |
| `./trainer_integration/verl/pyproject.toml` | inside Docker only | TrainerAdapter (never installed on host) |

```bash
# Fabric-core (fast — 5 packages):
poetry install
ROLLOUT_FABRIC_PYTHON=$(poetry env info --path)/bin/python

# EnvironmentProvider (full OpenHands stack):
cd openhands && poetry install && cd ..
PRORL_OPENHANDS_PYTHON=$(cd openhands && poetry env info --path)/bin/python
```

## Why NOT one env for everything

`litellm`, `docker`, `e2b`, `aiohttp`, `anthropic` previously lived in the root
`[tool.poetry.dependencies]` main group alongside fabric deps. They're now in
`openhands/pyproject.toml` where they belong. LiveStore does not import any of them;
deploying the EnvironmentProvider env on a machine running only LiveStore wastes ~2 GB.

## Per-service dependency footprint

| Service | Machine | Python env | Minimal deps |
|---------|---------|-----------|--------------|
| LiveStore | trainer host | fabric-core venv | grpcio, pyarrow, protobuf |
| PolicyRegistry | trainer host | fabric-core venv | grpcio, requests |
| ReplayArchive | trainer host | fabric-core venv | grpcio, pyarrow |
| RolloutManager | trainer host (or remote) | fabric-core venv | grpcio, httpx, pyarrow |
| EnvironmentProvider | trainer host (or remote) | full poetry env | openhands + litellm + fastapi + Singularity |
| TrainerAdapter (VERL) | trainer host | Docker (verlai/verl) | NEVER use host venv |
| TrainerAdapter (slime) | trainer host | Docker (slime image) | NEVER use host venv |
| TrainerAdapter (ROLL) | trainer host | Docker (ROLL image) | NEVER use host venv |
| InferenceBackend (vLLM) | EC2 remote | separate venv | vllm + fastapi |

## Environment variables

| Variable | Used by | Points at |
|---|---|---|
| `ROLLOUT_FABRIC_PYTHON` | LiveStore, PolicyRegistry, ReplayArchive, RolloutManager | fabric-core venv or full poetry env |
| `PRORL_OPENHANDS_PYTHON` | EnvironmentProvider (ProRL) | full poetry env (must have openhands) |
| *(neither)* | TrainerAdapter | Docker container — never the host env |

## Current single-machine default

All Python services (except the Docker trainer and remote vLLM) share the same
`rollout-fabric-*` poetry env on the current machine. This is a deployment
convenience ONLY. The env variables above override it per-script.

## Fabric-core venv (for multi-machine deployment)

On a machine running only fabric services (no EnvironmentProvider, no trainer):

```bash
python -m venv .venv-fabric
.venv-fabric/bin/pip install -r requirements/fabric-core.txt
export ROLLOUT_FABRIC_PYTHON="$(pwd)/.venv-fabric/bin/python"
bash scripts/services/start_live_store.sh
```

## Trainer Docker isolation

Each trainer uses its own Docker image (verlai/verl, slime image, ROLL image). The
`trainer_integration/{trainer}/` package is installed inside the container:

```bash
pip install -e /workspace/trainer_integration/verl
```

The host poetry env NEVER receives trainer or CUDA dependencies. Never run the trainer
on the host venv.

## Setting up on a fresh machine

```bash
cd /path/to/ProRL-Agent-Server
poetry install                          # populates rollout-fabric-* env (full install)
POETRY_PYTHON=$(poetry env info --path)/bin/python
# or for fabric-only:
python -m venv .venv-fabric && .venv-fabric/bin/pip install -r requirements/fabric-core.txt
export ROLLOUT_FABRIC_PYTHON="$(pwd)/.venv-fabric/bin/python"
```
