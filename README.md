# RolloutFabric

A contract-first agentic RL training fabric. Six independent services wired by
gRPC/HTTP contracts; currently training Qwen3-4B on SWE-Bench tasks via GRPO/DAPO.
Every service sits behind a typed `Protocol` and is independently replaceable.

## What this repo is

**RolloutFabric** separates rollout generation, sandbox execution, hot storage,
training, inference, and policy publishing into swappable service boundaries.
See `docs/topology.md` for the deployment diagram and `docs/service-envs.md` for
the per-service dependency footprint.

## What `openhands/` is

`openhands/` is the **SWE-Bench EnvironmentProvider implementation** — one concrete
implementation of the `schemas/protocols/environment_provider.EnvironmentProvider`
Protocol. It is NOT the framework. Other environments (ROCK, OpenReward) plug in by
implementing the same `POST /process` contract. See `environment_providers/README.md`.

## What `trainer_integration/verl/` is

The VERL FSDP TrainerAdapter — one concrete implementation running inside a Docker
container. Other trainers (slime, ROLL) plug in via the same LiveStore + PolicyRegistry
contracts. See `trainer_adapters/README.md`.

**How VERL is installed at container start (not baked into the image):**

```
Host /tmp/verl  ──bind-mount──►  /opt/verl  (inside container)
                                      │
                              pip install --no-deps -e /opt/verl   ← upstream VERL
                              pip install --no-deps -e /workspace/trainer_integration/verl  ← our patch
```

VERL is never baked into the Docker image — it's always installed from the host's
`/tmp/verl` checkout at container start. This means you can update VERL by changing
`/tmp/verl` on the host without rebuilding the image. The `verl_custom` patch package
(`trainer_integration/verl/pyproject.toml`) adds the LiveStore consumer seam and
PolicyRegistry publish hook on top.

## How to plug in a new environment or trainer

See `schemas/protocols/PLUGGING_IN.md` for the step-by-step guide.

## Python environments — three, cleanly separated

| Environment | `pyproject.toml` | What runs in it |
|---|---|---|
| **Fabric-core** | `./pyproject.toml` | LiveStore, PolicyRegistry, ReplayArchive, RolloutManager, schemas |
| **EnvironmentProvider** | `./openhands/pyproject.toml` | ProRL FastAPI server :8006 (OpenHands + litellm + docker + e2b...) |
| **TrainerAdapter** | `./trainer_integration/verl/pyproject.toml` | VERL FSDP inside `verlai/verl` Docker image |

```bash
# Fabric-core venv (fast — 5 packages):
poetry install                          # from repo root
ROLLOUT_FABRIC_PYTHON=$(poetry env info --path)/bin/python

# EnvironmentProvider venv (full openhands stack):
cd openhands && poetry install && cd ..
PRORL_OPENHANDS_PYTHON=$(cd openhands && poetry env info --path)/bin/python

# TrainerAdapter: pip install -e inside Docker (see trainer_integration/verl/)
# vLLM pool (EC2): pip install -r scripts/inference/requirements-remote.txt
```

See `docs/service-envs.md` for per-service details.

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
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server

CACHE_BASE="$(pwd)/scripts/_singularity_cache"
mkdir -p "${CACHE_BASE}/apptainer_tmpdir" "${CACHE_BASE}/apptainer_localcachedir"

APPTAINER_CACHEDIR="${CACHE_BASE}/apptainer_cachedir" \
APPTAINER_LOCALCACHEDIR="${CACHE_BASE}/apptainer_localcachedir" \
APPTAINER_TMPDIR="${CACHE_BASE}/apptainer_tmpdir" \
APPTAINER_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME}" \
APPTAINER_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD}" \
PRORL_OPENHANDS_PYTHON=$(cd openhands && poetry env info --path)/bin/python
"${PRORL_OPENHANDS_PYTHON}" scripts/dev/pull_swe_images.py \
  --parquet-file /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --dest-dir singularity_images
# ~3-5 min per image; 232 GB OCI blobs are pre-cached locally
```

### 2. Filter parquet to built SIFs

```bash
ROLLOUT_FABRIC_PYTHON=$(poetry env info --path)/bin/python
"${ROLLOUT_FABRIC_PYTHON}" scripts/data/filter_parquet_to_built_sifs.py \
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
  bash scripts/services/start_all.sh
```

`start_all.sh` starts services in dependency order, health-probes each, waits for
the RolloutManager to push ≥1 group (BC-16 warm-up gate), then starts the trainer.
Stop with Ctrl-C; services shut down in reverse order.

For manual step-by-step startup, see `CLAUDE.md`.

---

## Service scripts

| Script | Socket / Port | Env | Notes |
|---|---|---|---|
| `scripts/adapters/start_env_prorl.sh` | `:8006` | `PRORL_OPENHANDS_PYTHON` | ProRL FastAPI + Singularity |
| `scripts/inference/launch_remote_vllm_pool.sh` | `:8100-8103` (EC2) | EC2 venv | SSH alias `vllm-instance` must exist |
| `scripts/services/start_live_store.sh` | `/tmp/prorl_live_store.sock` | `ROLLOUT_FABRIC_PYTHON` | gRPC UDS, pop-on-sample |
| `scripts/services/start_policy_registry.sh` | `/tmp/prorl_policy_registry.sock` | `ROLLOUT_FABRIC_PYTHON` | gRPC UDS, fanout to vLLM |
| `scripts/services/start_rollout_manager.sh` | no port | `ROLLOUT_FABRIC_PYTHON` | owns `train.ready.parquet` |
| `scripts/adapters/start_trainer_verl.sh` | Docker internal | `verlai/verl` Docker | VERL FSDP, 8×A100 |
| `scripts/services/rescue_team.py` | n/a | either | `--check` / `--watch` / `--rescue <service>` |

---

## Key configuration

| Variable | Where set | What it controls |
|---|---|---|
| `REMOTE_DNS` | `.prorl_creds.env` | EC2 hostname for vLLM pool and SSH |
| `DATA_FILES` | export before start | Path to `train.ready.parquet` (filtered, SIF-verified) |
| `POLICY_ID` | export before start | Policy identifier stamped on all groups and registry entries |
| `SAVE_FREQ` | `scripts/adapters/start_trainer_verl.sh` env var | Steps between LoRA publishes (default: 1) |
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
PYTHONPATH=. poetry run pytest tests/invariants/ tests/contracts/ tests/slots/ -q

# Full suite excluding integration/slow/real_data
pytest -m "not integration and not slow and not real_data" tests/ -q
```

---

## Repo layout

| Path | What's there |
|---|---|
| `openhands/` | Upstream OpenHands tree (mostly untouched) |
| `openhands/nvidia/` | ProRL FastAPI server, registry, AgentHandlers |
| `openhands/llm/nvidia/` | Token-in / token-out vLLM clients (frozen) |
| `live_store/` | gRPC LiveStore service |
| `policy_registry/` | PolicyRegistry gRPC service + fanout |
| `rollout_manager/` | Standalone RolloutManager (zero VERL/OpenHands) |
| `replay_archive/` | Append-only Parquet + SQLite archive (tee, pre-filter) |
| `trainer_adapters/verl/` | VERL bridge: `pad.py` unpads LiveStore batches |
| `schemas/` | Typed Protocols, wire schemas, proto bindings, invariant tests |
| `trainer_integration/verl/` | Patch package on top of pinned verl checkout |
| `scripts/_internal/` | Canonical service launchers |
| `scripts/serving/` | vLLM pool runner |
| `scripts/services/` | Service start scripts + rescue team + orchestrated start_all |
| `plans-n-solutions/` | Design rationale (`rollout_fabric.md`) + operations (`rollout_fabric_progress.md`) |
| `tests/` | pytest suites; fast-loop markers in `pytest.ini` |

---

## License

See [`LICENSE`](LICENSE). Inherits from upstream OpenHands.
