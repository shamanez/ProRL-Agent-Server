# ProRLAgent Server

A scalable multi-turn rollout service for training and evaluating RL agents on agentic tasks (SWE-Bench, math/code, STEM, GUI). Fork of [OpenHands](https://github.com/All-Hands-AI/OpenHands): the upstream tree is kept largely intact and a FastAPI dispatch layer drives a remote vLLM pool and feeds a fully-async decoupled GRPO/DAPO trainer.

## System shape

Three processes on two machines, glued by an in-process replay store:

- **ProRL FastAPI** (`:8006`, trainer host) — accepts rollout jobs, dispatches to per-task `AgentHandler`s, talks to vLLM in **token IDs** (not text — the multi-turn invariant).
- **Remote vLLM pool** (4 children on ports `8100–8103`, EC2) — generation backend; LoRA adapters hot-reloaded on `save_freq`.
- **Decoupled trainer** (Docker, 8×A100 FSDP) — samples a bounded replay store with clipped temporal IS correction, publishes rank-16 LoRA adapters back to the pool.

The producer fills the buffer; the trainer drains it at its own cadence. They share nothing else.

## Current state — what runs today

The system shipping on the `producer-as-a-service` branch is three processes
across two machines, glued by an in-process replay store inside a Docker
container.

```
┌───────────────────────── trainer box (host) ─────────────────────────┐
│                                                                       │
│  ProRL FastAPI :8006  (scripts/_internal/s0_prorl.sh)                 │
│    - OpenHands agent dispatcher (registry + AgentHandler)             │
│    - Singularity sandbox lifecycle                                    │
│    - Three-stage pipeline: init → run → eval                          │
│    - Calls remote vLLM children per assistant turn (token IDs only)   │
│                                                                       │
│  ┌─────────────────── Docker container (s3_fullasync_docker.sh) ─────┐│
│  │                                                                    ││
│  │  DATA LOADER                                                       ││
│  │    SkyRL-v0-293 parquet → StatefulDataLoader                       ││
│  │    [trainer-owned today; this is the data-ownership leak]          ││
│  │                                                                    ││
│  │  CONTINUOUS PRODUCER (daemon thread)                                ││
│  │    - Pulls prompts from data_loader                                 ││
│  │    - Calls AsyncLLMServerManagerDAPO.generate_sequences_dapo()     ││
│  │    - That hits ProRL :8006 → vLLM pool                             ││
│  │    - Eager-pushes survivors into TrajectoryStore                    ││
│  │    - Tags each group with behavior_policy_version                   ││
│  │                                                                    ││
│  │  TRAJECTORY STORE (in-process, threading.Lock)                      ││
│  │    - deque(maxlen=256) of groups                                    ││
│  │    - pop-on-sample (queue semantics)                                ││
│  │    - staleness eviction (K=4)                                       ││
│  │    - re-pad to sample-local max at pack time                        ││
│  │                                                                    ││
│  │  TRAINER (RayPPOTrainerDAPO, 8×A100 FSDP)                          ││
│  │    - sample_mini_batch(n_groups) from store                         ││
│  │    - compute_reward → compute_old_log_prob → compute_advantage      ││
│  │    - update_actor (PPO/GRPO/DAPO)                                   ││
│  │    - save_checkpoint → _publish_lora_adapter → pool /reload_lora    ││
│  │    - _validate: pauses producer, runs val via ProRL, resumes        ││
│  │                                                                    ││
│  └────────────────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────────────┘
                              │ HTTP
                              ▼
┌──────────────────────── EC2 vllm-instance ────────────────────────────┐
│  4× _vllm_child.py  :8100 :8101 :8102 :8103                          │
│  Qwen3-4B-Instruct + LoRA, pinning swap protocol                     │
│  /v{N}/generate pins to policy version N                              │
│  /reload_lora installs new adapter, never removes old (LRU eviction)  │
└───────────────────────────────────────────────────────────────────────┘
```

Source files for the current implementation:
- Launcher: `scripts/_internal/s3_fullasync_docker.sh`
- ProRL server: `openhands/nvidia/async_server.py` and the `AgentHandler`
  registry at `openhands/nvidia/registry.py`
- Token-level vLLM clients: `openhands/llm/nvidia/qwen3.py`,
  `openhands/llm/nvidia/qwen2_5_vl.py`
- Live store: `trainer_integration/verl/verl_custom/replay/trajectory_store.py`
- Producer: `trainer_integration/verl/verl_custom/replay/continuous_producer.py`
- DAPO trainer: `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py`
- Pool child: `scripts/serving/_vllm_child.py`

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
