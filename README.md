# ProRLAgent Server

A scalable multi-turn rollout service for training and evaluating RL agents on agentic tasks (SWE-Bench, math/code, STEM, GUI). Fork of [OpenHands](https://github.com/All-Hands-AI/OpenHands): the upstream tree is kept largely intact and a FastAPI dispatch layer drives a remote vLLM pool and feeds a fully-async decoupled GRPO/DAPO trainer.

## System shape

Three processes on two machines, glued by an in-process replay store:

- **ProRL FastAPI** (`:8006`, trainer host) — accepts rollout jobs, dispatches to per-task `AgentHandler`s, talks to vLLM in **token IDs** (not text — the multi-turn invariant).
- **Remote vLLM pool** (4 children on ports `8100–8103`, EC2) — generation backend; LoRA adapters hot-reloaded on `save_freq`.
- **Decoupled trainer** (Docker, 8×A100 FSDP) — samples a bounded replay store with clipped temporal IS correction, publishes rank-16 LoRA adapters back to the pool.

The producer fills the buffer; the trainer drains it at its own cadence. They share nothing else.

## Where to start

| You are… | Read |
|---|---|
| Running training | [`plans-n-solutions/stages/how_to_run.md`](plans-n-solutions/stages/how_to_run.md) |
| Implementing / debugging the system | [`CLAUDE.md`](CLAUDE.md) → [`plans-n-solutions/architecture-walkthrough.md`](plans-n-solutions/architecture-walkthrough.md) (doc map) → [`plans-n-solutions/handsoff.md`](plans-n-solutions/handsoff.md) |
| Picking up open work | [`plans-n-solutions/stages/current_bottlenecks_and_problems.md`](plans-n-solutions/stages/current_bottlenecks_and_problems.md) |

## Repo layout

| Path | What's there |
|---|---|
| `openhands/` | Upstream OpenHands tree, mostly untouched |
| `openhands/nvidia/` | RL-serving FastAPI server, registry, agent handlers |
| `openhands/llm/nvidia/` | Token-in / token-out vLLM clients (Qwen3, Qwen2.5-VL) |
| `trainer_integration/verl/` | Patch package on top of a pinned `verl` checkout |
| `scripts/_internal/` | Canonical launchers (`s0_prorl.sh`, `s3_fullasync_docker.sh`) |
| `plans-n-solutions/` | Architecture handsoff + stage docs |
| `tests/` | pytest suites; markers declared in `pytest.ini` |

## License

See [`LICENSE`](LICENSE). Inherits from upstream OpenHands.
