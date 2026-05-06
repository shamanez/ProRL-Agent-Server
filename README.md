# RolloutFabric

A contract-first agentic RL training fabric. Six independent services wired by
gRPC/HTTP contracts; currently training Qwen3-4B on SWE-Bench tasks via GRPO/DAPO.
Every service sits behind a typed `Protocol` and is independently replaceable.

## What this repo is

**RolloutFabric** separates rollout generation, sandbox execution, hot storage,
training, inference, and policy publishing into swappable service boundaries.
See `docs/topology.md` for the deployment diagram and `docs/service-envs.md` for
the per-service dependency footprint.

## What `environments/prorl_openhands/` is

`environments/prorl_openhands/` is the **SWE-Bench EnvironmentProvider implementation** — one concrete
implementation of the `core/rollout_fabric/schemas/protocols/environment_provider.EnvironmentProvider`
Protocol. It is NOT the framework. Other environments (ROCK, OpenReward) plug in by
implementing the same `POST /process` contract. See `environment_providers/README.md`.

## What `trainers/verl/` is

The VERL FSDP TrainerAdapter — one concrete implementation running inside a Docker
container. Other trainers (slime, ROLL) plug in via the same LiveStore + PolicyRegistry
contracts.

**How VERL is installed at container start (not baked into the image):**

```
Host /tmp/verl  ──bind-mount──►  /opt/verl  (inside container)
                                      │
                              pip install --no-deps -e /opt/verl   ← upstream VERL
                              pip install --no-deps -e /workspace/trainers/verl  ← our patch
```

VERL is never baked into the Docker image — it's always installed from the host's
`/tmp/verl` checkout at container start. This means you can update VERL by changing
`/tmp/verl` on the host without rebuilding the image. The `verl_custom` patch package
(`trainers/verl/pyproject.toml`) adds the LiveStore consumer seam and
PolicyRegistry publish hook on top.

## How to plug in a new environment or trainer

See `core/rollout_fabric/schemas/protocols/PLUGGING_IN.md` for the step-by-step guide.

## Python environments

Two host venvs (`fabric-core` and `prorl_openhands`); trainer runs inside Docker; vLLM on remote EC2. See `docs/service-envs.md` for the full dependency footprint and install commands.

---

## Architecture

```
  SkyRL parquet
       │ (ParquetDataLoader — RolloutManager owns this)
       ▼
  RolloutManager ──POST /process──► EnvironmentProvider (ProRL :8006 + Singularity)
       │                                    │ (token generation per assistant turn)
       │                                    ▼
       │                             InferenceBackend (vLLM :8100-8103 EC2)
       │                                    ▲
  gRPC push_group                    POST /reload_lora
       │                                    │
       ▼                             PolicyRegistry (UDS)
  LiveStore ──────gRPC get_batch──►        ▲
  (UDS)                             gRPC publish_policy_version
                                           │
                                    TrainerAdapter (VERL FSDP, Docker 8×A100)
```

Data flows down: parquet → groups → training steps → LoRA publishes.
Policies flow up: trainer publishes → registry fans out → vLLM pool reloads.
RolloutManager polls the registry at 1Hz to stamp `behavior_policy_version` on each group.

---

## Quick start

**Prerequisites:** 8×A100 trainer box, EC2 `vllm-instance` reachable via SSH alias,
`/home/ubuntu/.prorl_creds.env` with `REMOTE_DNS` + credentials, dataset at
`/home/ubuntu/data/SkyRL-v0-293/`.

### 1. Build Singularity images (first time only)

```bash
source /home/ubuntu/.prorl_creds.env
PRORL_OPENHANDS_PYTHON=$(cd environments/prorl_openhands && poetry env info --path)/bin/python

APPTAINER_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME}" \
APPTAINER_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD}" \
"${PRORL_OPENHANDS_PYTHON}" ops/data/pull_skyrl_data.py \
  --parquet-file /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --dest-dir singularity_images
# ~3-5 min per image
```

### 2. Filter parquet to built SIFs

```bash
ROLLOUT_FABRIC_PYTHON=$(cd core && poetry env info --path)/bin/python
"${ROLLOUT_FABRIC_PYTHON}" ops/data/filter_parquet_to_built_sifs.py \
  --input /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --sif-dir singularity_images \
  --output /home/ubuntu/data/SkyRL-v0-293/train.ready.parquet
```

Re-run whenever new SIFs are added.

### 3. Start all services

```bash
source /home/ubuntu/.prorl_creds.env
DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.ready.parquet" \
POLICY_ID="qwen3-4b-skyrl" \
  bash ops/services/start_all.sh
```

`start_all.sh` starts services in dependency order, health-probes each, waits for
the RolloutManager to push ≥1 group (BC-16 warm-up gate), then starts the trainer.
Stop with Ctrl-C; services shut down in reverse order.

For the manual step-by-step sequence, health gates, monitoring, and error recovery
see `docs/TRAINING_OPERATIONS.md`.

---

## Service scripts

| Script | Socket / Port | Env | Notes |
|---|---|---|---|
| `environments/prorl_openhands/scripts/start.sh` | `:8006` | `PRORL_OPENHANDS_PYTHON` | ProRL FastAPI + Singularity |
| `inference/vllm/scripts/launch_remote_vllm_pool.sh` | `:8100-8103` (EC2) | EC2 venv | SSH alias `vllm-instance` must exist |
| `ops/services/start_live_store.sh` | `/tmp/prorl_live_store.sock` | `ROLLOUT_FABRIC_PYTHON` | gRPC UDS, pop-on-sample |
| `ops/services/start_policy_registry.sh` | `/tmp/prorl_policy_registry.sock` | `ROLLOUT_FABRIC_PYTHON` | gRPC UDS, fanout to vLLM |
| `ops/services/start_rollout_manager.sh` | no port | `ROLLOUT_FABRIC_PYTHON` | owns `train.ready.parquet` |
| `trainers/verl/scripts/start.sh` | Docker internal | `verlai/verl` Docker | VERL FSDP, 8×A100 |
| `ops/services/rescue_team.py` | n/a | either | `--check` / `--watch` / `--rescue <service>` |

---

## Key configuration

| Variable | Where set | What it controls |
|---|---|---|
| `REMOTE_DNS` | `.prorl_creds.env` | EC2 hostname for vLLM pool and SSH |
| `DATA_FILES` | export before start | Path to `train.ready.parquet` (filtered, SIF-verified) |
| `POLICY_ID` | export before start | Policy identifier stamped on all groups and registry entries |
| `SAVE_FREQ` | `trainers/verl/scripts/start.sh` env var | Steps between LoRA publishes (default: 1) |
| `WANDB_API_KEY` | `.prorl_creds.env` | WandB logging |
| `SINGULARITY_DOCKER_USERNAME/PASSWORD` | `.prorl_creds.env` | Apptainer registry auth for SIF builds |

---

## Key boundary conditions

| BC | What it protects | What breaks if violated |
|---|---|---|
| BC-0: one `PolicyVersionSnapshot` per group | Consistent advantage computation across siblings | Invalid advantages; NaN loss within steps |
| BC-1: token IDs as `int` everywhere | Multi-turn RL stability | KL/entropy NaN within 2 training steps |
| BC-9: `endpoints_failed > 0` = hard abort | Prevents mixed-version vLLM pool | IS weights become lies; silent gradient corruption |
| BC-13: zero VERL/OpenHands in RolloutManager | Trainer pluggability | Swapping trainer requires rewriting worker |
| BC-14: worker owns parquet | Trainer cannot be hidden orchestrator | Trainer becomes implicit coordinator |
| BC-15: trainer connects only to LiveStore + PolicyRegistry | Trainer pluggability | Swapping VERL for another trainer requires more than a Docker image swap |

Full BC table (BC-0 through BC-15): `plans-n-solutions/rollout_fabric.md`

---

## Tests

```bash
# Fast loop — no real services needed (~36 tests, ~7s)
PYTHONPATH=core poetry run pytest tests/invariants/ tests/contracts/ tests/slots/ -q

# Full suite excluding integration/slow/real_data
pytest -m "not integration and not slow and not real_data" tests/ -q
```

---

## Repo layout

```
ProRL-Agent-Server/
├── core/rollout_fabric/    # fabric services (LiveStore, PolicyRegistry, RolloutManager, ReplayArchive, schemas)
├── environments/prorl_openhands/  # OpenHands EnvironmentProvider
├── trainers/verl/          # VERL TrainerAdapter (Docker)
├── inference/vllm/         # vLLM InferenceBackend (remote EC2)
├── ops/services/           # orchestration, rescue, monitoring
├── ops/data/               # data utilities
├── docs/                   # TRAINING_FLOW, TRAINING_OPERATIONS, PLUGGING_IN_*, topology, service-envs
├── tests/                  # test suite
└── pyproject.toml          # dev workspace
```

| Path | What's there |
|---|---|
| `core/rollout_fabric/live_store/` | gRPC LiveStore service |
| `core/rollout_fabric/policy_registry/` | PolicyRegistry gRPC service + fanout |
| `core/rollout_fabric/rollout_manager/` | Standalone RolloutManager (zero VERL/OpenHands) |
| `core/rollout_fabric/replay_archive/` | Append-only Parquet + SQLite archive (tee, pre-filter) |
| `core/rollout_fabric/schemas/` | Typed Protocols, wire schemas, proto bindings, invariant tests |
| `environments/prorl_openhands/openhands/` | Upstream OpenHands tree (mostly untouched) |
| `environments/prorl_openhands/openhands/nvidia/` | ProRL FastAPI server, registry, AgentHandlers |
| `environments/prorl_openhands/openhands/llm/nvidia/` | Token-in / token-out vLLM clients (frozen) |
| `trainers/verl/` | Patch package on top of pinned verl checkout |
| `trainers/verl/verl_custom/fabric_adapter/` | VERL bridge: `pad.py` unpads LiveStore batches |
| `inference/vllm/scripts/` | vLLM pool runner |
| `ops/services/` | Service start scripts + rescue team + orchestrated start_all |
| `ops/data/` | Data utilities (filter parquet, pull skyrl data) |
| `plans-n-solutions/` | Design rationale (`rollout_fabric.md`) + operations (`rollout_fabric_progress.md`) |
| `tests/` | pytest suites; fast-loop markers in `pytest.ini` |

---

## License

See [`LICENSE`](LICENSE). Inherits from upstream OpenHands.
