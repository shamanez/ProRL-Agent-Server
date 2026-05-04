# ProRL-Agent-Server

A scalable multi-turn agentic RL training fabric. Five independent services wired by
gRPC/HTTP contracts; currently training Qwen3-4B on SWE-Bench tasks via GRPO/DAPO.
Built on VERL + OpenHands + vLLM; every service sits behind a typed `Protocol` and is
independently replaceable.

---

## Architecture

```
  SkyRL-v0-293/train.ready.parquet
         |  (ParquetDataLoader — RolloutWorker owns it, §3.8 / BC-14)
         |  filter_parquet_to_built_sifs.py keeps only rows with a .sif
         v
  +-------------------------------------------------------------+
  |  RolloutWorker  (scripts/services/start_rollout_worker.sh)  |
  |  Zero VERL / OpenHands imports (BC-13)                      |
  |                                                             |
  |  for each task in dataloader:                               |
  |    snap = policy_cache.snapshot()    <- ONE read per group  |
  |    for i in range(group_size):       <- all N use snap.ver  |
  |      ep = prorl_client.POST /process (token IDs, §3.1)      |
  |    samples = build_group(episodes, snap)  <- BC-0 / §3.2    |
  |    archive.submit(record)            <- tee pre-filter, BC-12|
  |    if not zero_variance: store.push_group(samples)  <- §3.7 |
  +-------------------------------------------------------------+
         |  gRPC push_group (packed int32 bytes, BC-1, BC-2)
         v
  +------------------------------------------------------------+
  |  LiveStore  (scripts/services/start_live_store.sh)         |
  |  Bounded FIFO, pop-on-sample (§3.6 / BC-3)                 |
  |  Server-side blocking get_batch (BC-4, BC-5)               |
  |  Staleness eviction by created_at_step (§3.6)              |
  +------------------------------------------------------------+
         |  gRPC get_batch -> unpadded TrainingSample list (BC-11)
         v
  +------------------------------------------------------------+
  |  TrainerAdapter  (scripts/_internal/s3_fullasync_docker.sh)|
  |  VERL FSDP inside Docker, 8x A100                          |
  |  Connects ONLY to LiveStore + PolicyRegistry (BC-15)        |
  |  sample_mini_batch() -> pad locally -> FSDP forward/backward|
  |  No dataloader, no producer thread (§3.8)                  |
  |  After save_freq steps -> publish_policy_version()          |
  +------------------------------------------------------------+
         |  gRPC publish_policy_version (§3.3 abort gate, BC-9)
         v
  +------------------------------------------------------------+
  |  PolicyRegistry  (scripts/services/start_policy_registry.sh)|
  |  Single source of truth for LoRA version + adapter URI      |
  |  Fans out /reload_lora to all pool children synchronously   |
  |  endpoints_failed > 0 -> hard abort (BC-9)                 |
  +------------------------------------------------------------+
         |  HTTP POST /reload_lora (multipart, §3.4 pinning)
         v
  +------------------------------------------------------------+
  |  InferenceBackend  (vLLM pool :8100-8103 on EC2)           |
  |  Frozen through S4 -- _vllm_child.py unchanged             |
  |  /v{N}/generate pins each trajectory to dispatch-time PV   |
  |  LRU eviction of old LoRA slots                            |
  +------------------------------------------------------------+
         ^
  +--------------------------------------------------------------+
  |  EnvironmentProvider  (ProRL FastAPI :8006)                  |
  |  Frozen through S4 -- async_server.py unchanged              |
  |  POST /process -> Singularity sandbox -> tool loop           |
  |  Returns token IDs + logprobs (§3.1 token-in/token-out)     |
  +--------------------------------------------------------------+
```

---

## Quick-start training

### Prerequisites

- Trainer box: 8x A100, Docker, Poetry, Singularity/Apptainer
- EC2 `vllm-instance` reachable via SSH alias `~/.ssh/config` (alias: `vllm-instance`)
- Secrets file: `/home/ubuntu/.prorl_creds.env` (sets `REMOTE_DNS`, `SINGULARITY_DOCKER_*`, `WANDB_API_KEY`, etc.)
- Dataset already at `/home/ubuntu/data/SkyRL-v0-293/` — do not re-download

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
poetry run python scripts/pull_swe_images.py \
  --parquet-file /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --dest-dir singularity_images
# ~3-5 min per image; 232 GB OCI blobs are pre-cached locally
```

### 2. Filter parquet to built SIFs

```bash
poetry run python scripts/filter_parquet_to_built_sifs.py \
  --input /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --sif-dir singularity_images \
  --output /home/ubuntu/data/SkyRL-v0-293/train.ready.parquet
```

This must be re-run whenever new SIFs are added. The worker refuses to start without
`train.ready.parquet`.

### 3. Start all services (orchestrated)

```bash
source /home/ubuntu/.prorl_creds.env
DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.ready.parquet" \
POLICY_ID="qwen3-4b-skyrl" \
  bash scripts/services/start_all.sh
```

`start_all.sh` starts services in dependency order, health-probes each, waits for the
RolloutWorker to push at least one group (BC-16 warm-up gate), then starts the trainer.

### 4. Or start manually in order

```bash
# Step 1: vLLM pool (remote EC2)
bash scripts/serving/launch_remote_vllm_pool.sh start

# Step 2: ProRL environment provider
nohup bash scripts/_internal/s0_prorl.sh > /tmp/s0-prorl.log 2>&1 &

# Step 3a + 3b: LiveStore and PolicyRegistry (parallel)
# See CLAUDE.md startup sequence for inline commands.

# Step 4: RolloutWorker
python -m rollout_worker.main --live-store-socket /tmp/prorl_live_store.sock ...

# Step 5: Trainer (after worker warm-up)
bash scripts/_internal/s3_fullasync_docker.sh
```

Stop in reverse order: trainer → worker → LiveStore + PolicyRegistry → ProRL + vLLM.

---

## Service scripts table

| Script | Port / Socket | Health check | Notes |
|---|---|---|---|
| `scripts/_internal/s0_prorl.sh` | `:8006` | `GET :8006/status` | vLLM addresses baked in via `--llm-server-address` |
| `scripts/serving/launch_remote_vllm_pool.sh` | `:8100-8103` (EC2) | `GET :810N/health` | SSH alias `vllm-instance` must exist |
| inline — see CLAUDE.md | `/tmp/prorl_live_store.sock` | socket exists | `live_store.server.serve()` |
| inline — see CLAUDE.md | `/tmp/prorl_policy_registry.sock` | socket exists | `policy_registry.server.serve()` |
| `python -m rollout_worker.main` | no port (client only) | first push logged | owns `train.ready.parquet` |
| `scripts/_internal/s3_fullasync_docker.sh` | Docker internal | begins `get_batch` | VERL FSDP, 8x A100 |
| `scripts/services/rescue_team.py` | n/a | `--check` flag | probe + diagnose + auto-fix loop |

---

## Key configuration

| Variable | Where set | What it controls |
|---|---|---|
| `REMOTE_DNS` | `.prorl_creds.env` | EC2 hostname for vLLM pool and SSH |
| `DATA_FILES` | export before start | Path to `train.ready.parquet` (filtered, SIF-verified) |
| `POLICY_ID` | export before start | Policy identifier stamped on all groups and registry entries |
| `LIVE_STORE_SOCKET` | env for trainer | Unix domain socket path for LiveStore gRPC |
| `SAVE_FREQ` | trainer Hydra config | Steps between LoRA publishes (currently 5) |
| `WANDB_API_KEY` | `.prorl_creds.env` | WandB logging; do not re-export inline |
| `SINGULARITY_DOCKER_USERNAME` / `_PASSWORD` | `.prorl_creds.env` | Apptainer registry auth for SIF builds |

---

## Boundary conditions

| BC | What it protects | What breaks if violated |
|---|---|---|
| BC-0: one PolicyVersionSnapshot per group | Consistent advantage computation across siblings | Invalid advantages; NaN loss within steps |
| BC-1: token IDs as `int` on every wire | Multi-turn RL stability | KL/entropy NaN within 2 training steps |
| BC-9: `endpoints_failed > 0` = hard abort | Prevents mixed-version pool | IS weights become lies; silent gradient corruption |
| BC-13: zero VERL/OpenHands in RolloutWorker | Trainer pluggability | Swapping trainer requires rewriting worker |
| BC-15: trainer connects only to LiveStore + PolicyRegistry | Trainer pluggability | Adding a new trainer requires changing orchestration |

---

## Tests

```bash
# Fast loop — no real services needed (~36 tests, ~7s)
PYTHONPATH=. poetry run pytest tests/invariants/ tests/contracts/ tests/slots/ -q

# Full suite excluding integration/slow/real_data
pytest -m "not integration and not slow and not real_data" tests/ -q
```

Tests cover: all 16 boundary conditions (BC-0 through BC-15), all seven service
`Protocol` surfaces, LiveStore gRPC round-trips, and packed-bytes token-ID codec.

---

## Current training status

- Model: Qwen3-4B-Instruct, rank-32 LoRA
- Task: SWE-Bench Verified (293 train / 23 val instances)
- Training: step 3+/500, GRPO/DAPO, 8x A100 FSDP
- Solve rate: 18.75% on SWE-Bench at step 2
- Policy publishes at `SAVE_FREQ=5` steps
- SIF images: building (232 GB OCI blobs cached; ~15h for all 293)
- Trainer migration to LiveStoreClient: next step (see `plans-n-solutions/rollout_fabric_progress.md` Section 4)

---

## Repo layout

| Path | What's there |
|---|---|
| `openhands/` | Upstream OpenHands tree (mostly untouched) |
| `openhands/nvidia/` | ProRL FastAPI server, registry, AgentHandlers |
| `openhands/llm/nvidia/` | Token-in / token-out vLLM clients (frozen) |
| `live_store/` | Extracted gRPC LiveStore service |
| `policy_registry/` | PolicyRegistry gRPC service + fanout |
| `rollout_worker/` | Standalone RolloutWorker (zero VERL/OpenHands) |
| `replay_archive/` | Append-only Parquet + SQLite archive (tee, pre-filter) |
| `trainer_adapters/verl/` | VERL bridge: pad.py unpads LiveStore batches |
| `schemas/` | Typed Protocols, wire schemas, proto bindings, invariant tests |
| `trainer_integration/verl/` | Patch package on top of pinned verl checkout |
| `scripts/_internal/` | Canonical launchers (frozen baseline files noted) |
| `scripts/serving/` | vLLM pool runner + orchestrator |
| `scripts/services/` | Service start scripts + rescue team |
| `plans-n-solutions/` | Architecture design + operational runbook |
| `tests/` | pytest suites; fast-loop markers in `pytest.ini` |

---

## License

See [`LICENSE`](LICENSE). Inherits from upstream OpenHands.
