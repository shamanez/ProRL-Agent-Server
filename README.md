# RolloutFabric

Six independent services wired by gRPC/HTTP contracts. Plug in any environment,
any agent framework, any trainer — the fabric stays the same.
Currently running: SWE-Bench tasks (OpenHands + Singularity) with VERL FSDP on 8×A100.

## Architecture

```
  dataset (parquet)
       │
       ▼
  RolloutManager ──POST /process──► EnvironmentProvider  (:8006)
       │                                    │  any env + agent framework
       │                                    ▼
       │                             InferenceBackend    (:8100-8103, EC2)
       │                                    ▲  any inference server
  gRPC push_group              POST /reload_lora
       │                                    │
       ▼                             PolicyRegistry     (UDS)
  LiveStore ──gRPC get_batch──► TrainerAdapter          (Docker)
  (UDS)                                any trainer framework
```

Data flows down: dataset → groups → gradient steps → LoRA publishes.
Policies flow up: trainer → registry fans out → inference pool reloads.

## Pluggable adapters

| Role | Current | Swap by |
|---|---|---|
| EnvironmentProvider | [OpenHands / SWE-Bench](environments/prorl_openhands/README.md) | Implementing `POST /process` |
| TrainerAdapter | [VERL FSDP](trainers/verl/README.md) | Connecting to LiveStore + PolicyRegistry (BC-15) |

See `docs/PLUGGING_IN_NEW_TRAINER_OR_ENVIRONMENT.md` for the step-by-step contract guide.

## Quick start

```bash
source /home/ubuntu/.prorl_creds.env
export DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.ready.parquet"
export POLICY_ID="qwen3-4b-skyrl"
bash ops/services/start_all.sh
```

`start_all.sh` starts all services in order with health gates, enforces the BC-16
warm-up gate, and shuts down cleanly on Ctrl-C.
Full setup, health gates, monitoring, and error recovery: `docs/TRAINING_OPERATIONS.md`.

## Boundary conditions

| BC | Rule | Breaks if violated |
|---|---|---|
| BC-0 | One policy snapshot per group — all siblings stamped before dispatch | Siblings see different policies → invalid advantages |
| BC-1 | Token IDs as `int` on every wire — never decode and re-tokenize | KL/entropy NaN within 2 training steps |
| BC-9 | `endpoints_failed > 0` = hard abort, never degraded mode | Mixed-version pool → silent gradient corruption |
| BC-13 | RolloutManager imports zero trainer/env framework code | Swapping trainer requires rewriting the worker |
| BC-14 | RolloutManager owns the dataset — trainer has no parquet path | Trainer becomes hidden orchestrator |
| BC-15 | Trainer connects only to LiveStore + PolicyRegistry | Swapping trainer requires more than a Docker swap |
| BC-16 | Start trainer only after ≥1 group in LiveStore | No-progress timer burns during warm-up |

## Tests

```bash
PYTHONPATH=core poetry run pytest tests/invariants/ tests/contracts/ tests/slots/ -q
pytest -m "not integration and not slow and not real_data" tests/ -q
```

## Repo layout

```
├── core/rollout_fabric/          # LiveStore, PolicyRegistry, RolloutManager, schemas
├── environments/                 # EnvironmentProvider adapters (prorl_openhands + others)
├── trainers/                     # TrainerAdapter adapters (verl, slime, roll)
├── inference/vllm/               # vLLM InferenceBackend (remote EC2)
├── ops/services/                 # start_all.sh, rescue_team.py, monitoring
├── docs/                         # TRAINING_FLOW, TRAINING_OPERATIONS, topology, service-envs
└── tests/
```

## License

See [`LICENSE`](LICENSE).
