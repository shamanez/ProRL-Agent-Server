# ProRL / OpenHands — EnvironmentProvider

One concrete implementation of the `EnvironmentProvider` protocol for SWE-Bench tasks.
It is **not** the framework — other environments plug in by implementing the same
`POST /process` HTTP contract without touching this directory.

## Contract

Accepts `POST /process` with a task instance, runs an OpenHands agent inside a
Singularity sandbox, calls vLLM per assistant turn using token IDs only (BC-1), and
returns `{messages, reward, resolved, finish}`. Must not own training data or call
LiveStore (BC-14).

## Install (host running this service)

```bash
cd environments/prorl_openhands && poetry install
# Out-of-band git dep required by SweAgentHandler:
$(poetry env info --path)/bin/pip install "git+https://github.com/SWE-Gym/SWE-Bench-Package.git"
```

## Start

```bash
bash environments/prorl_openhands/scripts/start.sh
```

Health gate: `GET http://localhost:8006/status` → `{"status":"running"}`

## Frozen files

`openhands/llm/nvidia/qwen3.py` and `async_server.py` implement the token-in/token-out
vLLM protocol. Never edit — make siblings instead. See root `CLAUDE.md`.
