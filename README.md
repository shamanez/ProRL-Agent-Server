# ProRLAgent Server

A scalable multi-turn rollout service for training and evaluating RL agents on agentic tasks (SWE-Bench, math/code, STEM, GUI). Fork of [OpenHands](https://github.com/All-Hands-AI/OpenHands): the upstream tree is kept largely intact and a FastAPI dispatch layer drives a remote vLLM pool and feeds a fully-async decoupled GRPO/DAPO trainer.

## System shape

Three processes on two machines, glued by an in-process replay store today; the [rollout fabric design](plans-n-solutions/rollout_fabric.md) decouples them into independently replaceable adapter slots.

- **ProRL FastAPI** (`:8006`, trainer host) — accepts rollout jobs, dispatches to per-task `AgentHandler`s, talks to vLLM in **token IDs** (not text — the multi-turn invariant).
- **Remote vLLM pool** (4 children on ports `8100–8103`, EC2) — generation backend; LoRA adapters hot-reloaded on `save_freq` via path-versioned pinning.
- **Decoupled trainer** (Docker, 8×A100 FSDP) — samples a bounded replay store with clipped temporal IS correction, publishes rank-32 LoRA adapters back to the pool.

The producer fills the buffer; the trainer drains it at its own cadence. They share nothing else.

## How to run

Three terminals, in order — the trainer pre-flight probes pool `/health`, so don't parallelise startup.

```bash
# Terminal 1 — trainer box (host, NOT Docker) — ProRL FastAPI
bash scripts/_internal/s0_prorl.sh
# Wait for: "Uvicorn running on http://0.0.0.0:8006"

# Terminal 2 — trainer box (host) — remote vLLM pool orchestration
source /home/ubuntu/.prorl_creds.env
bash scripts/serving/launch_remote_vllm_pool.sh start
# SSHs into vllm-instance, boots 4 children on 8100-8103.
# Wait for: 4× "ready" from /health.

# Terminal 3 — trainer box (Docker) — fully-async trainer
bash scripts/_internal/s3_fullasync_docker.sh
```

Stop order is reverse: kill the trainer container, then `launch_remote_vllm_pool.sh stop`, then kill ProRL.

### First run — smoke test (default)

The trainer launcher ships with smoke-test defaults so a functional check finishes in minutes, not hours:

| Knob | Smoke-test default | Production |
|---|---|---|
| `VAL_BEFORE_TRAIN` | `False` (skip up-front val) | `True` |
| `TEST_FREQ` | `-1` (disable in-training val) | `5` |
| `BATCH_SIZE` (trainer groups/step) | `4` | `32` |
| `GEN_BATCH_SIZE` (DAPO survivors/producer call) | `16` | `128` |

Smoke run (just hit Terminal 3):
```bash
bash scripts/_internal/s3_fullasync_docker.sh
```

Production run:
```bash
VAL_BEFORE_TRAIN=True TEST_FREQ=1 BATCH_SIZE=32 GEN_BATCH_SIZE=128 \
  bash scripts/_internal/s3_fullasync_docker.sh
```

All other env knobs the launcher reads (defaults in `scripts/_internal/s3_fullasync_docker.sh`): `TOTAL_TRAINING_STEPS`, `SAVE_FREQ`, `NUM_TRAJ`, `FILTER_GROUPS`, `OPENHANDS_NUM_WORKERS`, `REMOTE_DNS`, `SWAP_PROTOCOL`. Read the comments in that file before changing anything — the knobs are interlocked.

Pool health check:

```bash
for p in 8100 8101 8102 8103; do curl -sf http://$REMOTE_DNS:$p/health | jq .; done
```

## Where to start

| You are… | Read |
|---|---|
| Implementing the next architecture | [`plans-n-solutions/rollout_fabric.md`](plans-n-solutions/rollout_fabric.md) |
| Touching code in the existing system | [`CLAUDE.md`](CLAUDE.md) — invariants, gotchas, edit groups |

## Repo layout

| Path | What's there |
|---|---|
| `openhands/` | Upstream OpenHands tree, mostly untouched |
| `openhands/nvidia/` | RL-serving FastAPI server, registry, agent handlers |
| `openhands/llm/nvidia/` | Token-in / token-out vLLM clients (Qwen3, Qwen2.5-VL) |
| `trainer_integration/verl/` | Patch package on top of a pinned `verl` checkout |
| `scripts/_internal/` | Canonical launchers (`s0_prorl.sh`, `s3_fullasync_docker.sh`) |
| `scripts/serving/` | vLLM pool runner + orchestrator (`_vllm_child.py`, `launch_remote_vllm_pool.sh`) |
| `plans-n-solutions/` | Forward architecture design |
| `tests/` | pytest suites; markers declared in `pytest.ini` |

## License

See [`LICENSE`](LICENSE). Inherits from upstream OpenHands.
