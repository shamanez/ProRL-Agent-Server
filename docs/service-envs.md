# Service Python Environments

## Four `pyproject.toml` files — three runtime environments

| `pyproject.toml` | Venv | What it covers |
|---|---|---|
| `./pyproject.toml` | dev workspace | ruff, mypy, pytest, pre-commit — dev tools only |
| `./core/pyproject.toml` | `fabric-core-*` | LiveStore, PolicyRegistry, ReplayArchive, RolloutManager, schemas (5 runtime deps) |
| `./environments/prorl_openhands/pyproject.toml` | `openhands-*` | ProRL FastAPI server (OpenHands + litellm + docker + e2b…) |
| `./trainers/verl/pyproject.toml` | inside Docker only | TrainerAdapter — never installed on host |

```bash
# Fabric-core (fast — 5 packages):
cd core && poetry install && cd ..
ROLLOUT_FABRIC_PYTHON=$(cd core && poetry env info --path)/bin/python

# EnvironmentProvider (full OpenHands stack):
cd environments/prorl_openhands && poetry install && cd ../..
PRORL_OPENHANDS_PYTHON=$(cd environments/prorl_openhands && poetry env info --path)/bin/python
```

## Why NOT one env for everything

`litellm`, `docker`, `e2b`, `aiohttp`, `anthropic` previously lived alongside fabric deps. They're now in `environments/prorl_openhands/pyproject.toml` where they belong. LiveStore does not import any of them; deploying the EnvironmentProvider env on a machine running only LiveStore wastes ~2 GB.

## Per-service dependency footprint

| Service | Machine | Python env | Minimal deps |
|---|---|---|---|
| LiveStore | trainer host | fabric-core venv | grpcio, pyarrow, protobuf |
| PolicyRegistry | trainer host | fabric-core venv | grpcio, requests |
| ReplayArchive | trainer host | fabric-core venv | grpcio, pyarrow |
| RolloutManager | trainer host (or remote) | fabric-core venv | grpcio, httpx, pyarrow |
| EnvironmentProvider | trainer host (or remote) | openhands venv | openhands + litellm + fastapi + Singularity |
| TrainerAdapter (VERL) | trainer host | Docker (`verlai/verl`) | NEVER use host venv |
| TrainerAdapter (slime) | trainer host | Docker (slime image) | NEVER use host venv |
| TrainerAdapter (ROLL) | trainer host | Docker (ROLL image) | NEVER use host venv |
| InferenceBackend (vLLM) | EC2 remote | separate venv | vllm + fastapi |

## Environment variables

| Variable | Used by | Points at |
|---|---|---|
| `ROLLOUT_FABRIC_PYTHON` | LiveStore, PolicyRegistry, ReplayArchive, RolloutManager | fabric-core venv |
| `PRORL_OPENHANDS_PYTHON` | EnvironmentProvider (ProRL) | openhands venv (must have openhands) |
| *(neither)* | TrainerAdapter | Docker container — never the host env |

## Current single-machine default

All Python services (except the Docker trainer and remote vLLM) can share the same openhands poetry env on this machine as a deployment convenience. The env variables above override it per-script.

Pre-built envs on this machine:
```bash
ROLLOUT_FABRIC_PYTHON=/home/ubuntu/.cache/pypoetry/virtualenvs/openhands-ai-342rfuwh-py3.12/bin/python
PRORL_OPENHANDS_PYTHON=/home/ubuntu/.cache/pypoetry/virtualenvs/openhands-ai-342rfuwh-py3.12/bin/python
```

## Trainer Docker isolation

Each trainer uses its own Docker image. The `trainers/{trainer}/` package is installed inside the container at start:

```bash
pip install --no-deps -e /opt/verl            # upstream VERL from /tmp/verl bind-mount
pip install --no-deps -e /workspace/trainers/verl   # our patch package
```

The host poetry env NEVER receives trainer or CUDA dependencies.
