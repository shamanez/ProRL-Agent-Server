# Rollout Fabric — Implementation Progress & Runbook

**Branch:** `producer-as-a-service`
**Author:** Architecture + implementation session, 2026-05-04
**Status:** S0.5–S2 fully implemented; S3–S4 service stubs complete; SIF build in progress.

Read `rollout_fabric.md` for the full design rationale before reading this doc.
This doc is the **operational companion**: what exists, how to start it, what breaks and why.

---

## 0.1 The Architecture (One-Paragraph Thesis)

The rollout fabric is **five independent services wired by three contracts**.
No service knows the internals of another. Every boundary condition is enforced
in code and tested in `tests/invariants/` and `tests/slots/`.

```
  SkyRL-v0-293/train.ready.parquet
         │  (ParquetDataLoader — RolloutManager owns it, §3.8 / BC-14)
         │  filter_parquet_to_built_sifs.py keeps only rows with a .sif
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  RolloutManager  (scripts/services/start_rollout_manager.sh)  │
  │  Zero VERL / OpenHands imports (BC-13)                       │
  │                                                             │
  │  for each task in dataloader:                               │
  │    snap = policy_cache.snapshot()    ← ONE read per group   │
  │    for i in range(group_size):       ← all N use snap.ver   │
  │      ep = prorl_client.POST /process (token IDs, §3.1)      │
  │    samples = build_group(episodes, snap)  ← BC-0 / §3.2     │
  │    archive.submit(record)            ← tee pre-filter, BC-12│
  │    if not zero_variance: store.push_group(samples)  ← §3.7  │
  └─────────────────────────────────────────────────────────────┘
         │  gRPC push_group (packed int32 bytes, BC-1, BC-2)
         ▼
  ┌────────────────────────────────────────────────────────────┐
  │  LiveStore  (scripts/services/start_live_store.sh)          │
  │  Bounded FIFO, pop-on-sample (§3.6 / BC-3)                  │
  │  Server-side blocking get_batch (BC-4, BC-5)                │
  │  Staleness eviction by created_at_step (§3.6)               │
  └────────────────────────────────────────────────────────────┘
         │  gRPC get_batch → unpadded TrainingSample list (BC-11)
         ▼
  ┌────────────────────────────────────────────────────────────┐
  │  TrainerAdapter  (scripts/_internal/s3_fullasync_docker.sh) │
  │  VERL FSDP inside Docker, 8× A100                           │
  │  Connects ONLY to LiveStore + PolicyRegistry (BC-15)        │
  │  sample_mini_batch() → pad locally → FSDP forward/backward  │
  │  No dataloader, no producer thread (§3.8)                   │
  │  After save_freq steps → publish_policy_version()           │
  └────────────────────────────────────────────────────────────┘
         │  gRPC publish_policy_version (§3.3 abort gate, BC-9)
         ▼
  ┌────────────────────────────────────────────────────────────┐
  │  PolicyRegistry  (scripts/services/start_policy_registry.sh)│
  │  Single source of truth for LoRA version + adapter URI       │
  │  Fans out /reload_lora to all pool children synchronously    │
  │  endpoints_failed > 0 → hard abort (BC-9)                   │
  └────────────────────────────────────────────────────────────┘
         │  HTTP POST /reload_lora (multipart, §3.4 pinning)
         ▼
  ┌────────────────────────────────────────────────────────────┐
  │  InferenceBackend  (vLLM pool :8100-8103 on EC2)            │
  │  Frozen through S4 — _vllm_child.py unchanged               │
  │  /v{N}/generate pins each trajectory to dispatch-time PV    │
  │  LRU eviction of old LoRA slots                             │
  └────────────────────────────────────────────────────────────┘
         ↑
  ┌──────────────────────────────────────────────────────────┐
  │  EnvironmentProvider  (ProRL FastAPI :8006)                │
  │  Frozen through S4 — async_server.py unchanged             │
  │  POST /process → Singularity sandbox → tool loop           │
  │  Returns token IDs + logprobs (§3.1 token-in/token-out)    │
  └──────────────────────────────────────────────────────────┘
```

**The invariant that holds this together (BC-0):**
One `PolicyVersionSnapshot` is read at the START of each group dispatch.
All N sibling episodes are submitted with the same `policy_version`.
All N resulting samples are stamped with the same `behavior_policy_version`.
No trajectory in one group ever spans two policies.

**The invariant that makes the trainer pluggable (BC-15):**
The trainer connects to exactly two services: LiveStore (read) and
PolicyRegistry (write). It has no parquet files, no dataloader, no ProRL
address, and no vLLM address. Swapping VERL for ROLL or slime requires
only changing the Docker image and the Hydra command — nothing else.

---

## 1. What Was Implemented

### Stage S0.5 — Substrate (schemas + protocols + invariant tests)

All seven slot contracts from Appendix A of `rollout_fabric.md` now exist as typed
Python `Protocol` classes. Every boundary condition documented in the plan file has a
corresponding test.

| Package | What it contains |
|---|---|
| `schemas/policy_version.py` | `PolicyVersionSnapshot` + `PolicyVersionCache` — the cleverest primitive for §3.5. Immutable frozen snapshot + atomic STORE_ATTR swap. One LOAD_ATTR per group dispatch; no locks on the read path. |
| `schemas/training_sample.py` | §6.2 `TrainingSample` / `TrainingGroup` — unpadded wire schema. |
| `schemas/episode_record.py` | §6.1 `EpisodeRecord` — canonical archive record with `TrustLevel`. |
| `schemas/protocols/` | Seven `Protocol` classes: `EnvironmentProvider`, `InferenceBackend`, `RolloutManager`, `LiveStore`, `ReplayArchive`, `TrainerAdapter`, `PolicyRegistry`. |
| `schemas/proto/*.proto` | gRPC schema for LiveStore (slot 5.4) and PolicyRegistry (slot 5.7). Token arrays as packed `bytes` (int32-LE) — never `string`. |
| `schemas/_gen/` | Pre-compiled protobuf Python bindings (grpcio 1.71.0). |
| `tests/invariants/` | 5 fast-loop tests pinning the 12 boundary conditions (BC-1 through BC-12). |
| `tests/contracts/` | Protocol importability + method surface tests. |

**Key invariant: PolicyVersionSnapshot-per-group (BC-0 + §3.2 + §3.5)**

```python
snap = policy_cache.snapshot()          # ONE atomic LOAD_ATTR before dispatching all N siblings
for episode in run_group(task, snap.version):
    sample = build_training_sample(episode, snap)   # snap.version stamped on every row
# All N samples have the same behavior_policy_version — no trajectory spans two policies.
```

---

### Stage S1 — LiveStore as gRPC Service

Extracted `TrajectoryStore` into a standalone gRPC service on a Unix domain socket.

| File | Role |
|---|---|
| `live_store/store_core.py` | In-process FIFO bounded buffer (deque + Condition). Server-side blocking in `get_batch` replaces trainer-side `wait_until_with_progress` busy-loop (BC-5). |
| `live_store/codec.py` | `TrainingSample` ↔ protobuf wire codec. Token IDs as packed int32-LE bytes (BC-1). |
| `live_store/server.py` | gRPC service: `PushGroup`, `GetBatch`, `GetMetrics`, `NotifyPolicyVersion`. |
| `live_store/client.py` | gRPC client drop-in for `TrajectoryStore`. `sample_mini_batch()` calls `get_batch` + local pad (BC-16). |

**Critical boundary conditions verified by `tests/slots/live_store/`:**
- BC-1: token IDs survive packed-bytes round-trip as `int`, never `str`
- BC-2: `push_group` is atomic — 0 or N records, never partial
- BC-3: `get_batch` pops under lock before sending response (pop-on-sample)
- BC-4: blocking predicate is `num_fresh_groups` not `num_groups` (staleness-aware)
- BC-5: `NoProgressError` after `no_progress_timeout_s` if producer stops pushing
- BC-11: wire is unpadded; `trainer_adapters/verl/pad.py` pads locally

**`_pack()` migration:** Extracted from `trajectory_store.py` into
`trainer_adapters/verl/pad.py` as `pack_unpadded_groups`. This is an atomic
coupling — LiveStore extraction and trainer adapter pad migration ship together.

---

### Stage S2 — RolloutManager as Independent Process (BC-13 + BC-14)

**Critical design decision: zero VERL/OpenHands imports in the worker.**

The prior implementation had `generate_fn=async_rollout_manager.generate_sequences_dapo`
— a VERL function. This violated the boundary. The new worker calls ProRL via plain HTTP.

| File | Role |
|---|---|
| `rollout_manager/prorl_client.py` | Thin `httpx` client for `POST /process`. No OpenHands imports. |
| `rollout_manager/dataloader.py` | `ParquetDataLoader` — worker owns the dataset (BC-14 / §3.8). Simple pyarrow reader with `state_dict()` for resume. |
| `rollout_manager/episode_builder.py` | Converts `ProRLEpisodeResult` → `TrainingSample`. Stamps `snap.version` on every row. `is_zero_variance_group()` filter (§3.7). |
| `rollout_manager/loop.py` | Main loop: read task → snapshot policy → dispatch N siblings → archive tee → filter → push to LiveStore. All N siblings get the same snapshot (BC-0). |
| `rollout_manager/policy_subscription.py` | `FilePollingPolicySubscription` (S2 / 1Hz) + `GrpcStreamingPolicySubscription` (S4). Feeds `PolicyVersionCache` via atomic ref-swap. |
| `rollout_manager/main.py` | Entry point. Wires all dependencies. No VERL, no OpenHands. |

**How the worker calls ProRL (EnvironmentProvider):**
```
Worker → POST localhost:8006/process
Body: {"instance": {..., "policy_version": N},
       "sampling_params": {"token_level_generation": true, ...}}
Response: {"messages": [{..., "token_ids": [...], "logprobs": [...]}],
           "resolved": bool, "success": bool, "finish": bool, "reward": float}
```

ProRL remains frozen and unchanged. The worker is ProRL-version-agnostic.

---

### Stage S3 — ReplayArchive as Tee

| File | Role |
|---|---|
| `replay_archive/server.py` | `ArchiveServer` — append-only Parquet + SQLite index. |
| `replay_archive/writer.py` | `ReplayArchiveWriter` — async queue + retry + dead-letter. Non-blocking from worker hot path (BC-12). |
| `replay_archive/query.py` | Offline SQL query against the SQLite index. |
| `replay_archive/derive.py` | Re-derive `TrainingSample` from `EpisodeRecord` with `TokenizerMismatchError` (§3.1 corollary). |

**BC-12 (tee is pre-filter):** Worker calls `archive.submit()` BEFORE
`filter_easy_hard_instance`. Filtered groups appear in the archive but NOT in the
LiveStore. `archive.episode_count >= live_store.total_pushes` after any run.

---

### Stage S4 — PolicyRegistry as Single Source of Truth

| File | Role |
|---|---|
| `policy_registry/file_registry.py` | S2 minimal: atomic-rename JSON manifest the trainer writes; worker polls at 1Hz. |
| `policy_registry/fanout.py` | `fanout_to_pool()` — §3.3 abort gate. Any pool child non-200/non-409 → `success=False`. |
| `policy_registry/server.py` | S4 gRPC service: `PublishPolicyVersion` (fanout + SQLite commit + subscriber notify), `GetLatestVersion`, `SubscribeVersionUpdates` (server-streaming). |
| `policy_registry/client.py` | `PolicyRegistryClient` — trainer-side `publish_policy_version()` raises `PublishFailedError` on `endpoints_failed > 0` (BC-9). Worker-side `stream_version_updates()`. |

---

### Trainer Adapter (VERL bridge, unchanged internals)

| File | Role |
|---|---|
| `trainer_adapters/verl/pad.py` | `pack_unpadded_groups()` — pads unpadded `TrainingSample` list from `LiveStore.get_batch()` into `SampledMiniBatch` (DataProto-ready tensors). Trainer calls this AFTER `get_batch`. |

**BC-15 (trainer connects only to LiveStore + PolicyRegistry):** After S2, the
trainer's `ray_trainer_dapo.py` is modified to replace `self.trajectory_store` with
`LiveStoreClient`, remove dataloader, and remove internal producer construction.
This migration is the **next step** after SIF images are available for testing.

---

### Rescue Agent Team

`scripts/services/rescue_team.py` — four-role team:

| Agent | Job |
|---|---|
| `ProbeAgent` | Health-checks all services. HTTP probes, socket existence, LiveStore push count. |
| `LogAgent` | Maps error signatures (10 patterns) to `Diagnosis(error_class, root_cause, suggested_fix)`. |
| `FixAgent` | Per-error-class fix: restart service, clean stale socket, wait for upstream, etc. |
| `RescueCoordinator` | Runs probe → diagnose → fix → verify loop, max 3 retries before escalation. |

```bash
# One-shot health check
REMOTE_DNS=ec2-54-159-82-213.compute-1.amazonaws.com PYTHONPATH=. \
  poetry run python scripts/services/rescue_team.py --check

# Continuous watch + auto-rescue
REMOTE_DNS=... PYTHONPATH=. \
  poetry run python scripts/services/rescue_team.py --watch

# Rescue a specific service
REMOTE_DNS=... PYTHONPATH=. \
  poetry run python scripts/services/rescue_team.py --rescue live_store
```

---

## 2. Six-Service Startup Sequence

**Read this entire section before starting training. Every step must succeed before the
next begins.**

### Prerequisites

```bash
# 1. Source credentials (sets REMOTE_DNS, SINGULARITY_DOCKER_*, etc.)
source /home/ubuntu/.prorl_creds.env

# 2. Export for worker
export DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.parquet"
export POLICY_ID="qwen3-4b-skyrl"
export ENVIRONMENT_ID="swe_agent"
export PYTHONPATH=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server

# 3. Find the poetry venv python (needed for service scripts)
POETRY_PYTHON=$(poetry run python -c "import sys; print(sys.executable)")
```

### Step 1 — InferenceBackend (remote EC2 vLLM pool)

```bash
# Health check first — may already be running
for port in 8100 8101 8102 8103; do
  curl -sf --max-time 3 "http://${REMOTE_DNS}:${port}/health" && echo ":${port} OK"
done

# Start if not running
bash scripts/serving/launch_remote_vllm_pool.sh start
# Wait for all 4 health probes to pass (takes ~2min for model load)
```

**Health gate:** All 4 of `:8100-8103` return HTTP 200.

### Step 2 — EnvironmentProvider (ProRL :8006)

```bash
# Start — vLLM addresses baked in via --llm-server-address (fixed in s0_prorl.sh)
nohup bash scripts/_internal/s0_prorl.sh > /tmp/s0-prorl.log 2>&1 &
echo $! > /tmp/prorl.pid

# Wait for startup (up to 30s)
for i in $(seq 1 30); do
  curl -sf http://localhost:8006/status -o /dev/null && { echo "ProRL up"; break; }; sleep 1
done

# Initialize server (creates job queues)
curl -sf -X POST http://localhost:8006/start -H "Content-Type: application/json" -d '{}'

# Verify vLLM endpoints registered (baked into s0_prorl.sh now)
curl -sf http://localhost:8006/status
```

**Health gate:** `GET :8006/status` returns `{"status": "running"}`.

**Why we fixed `s0_prorl.sh`:** Previously, vLLM addresses were registered via
`POST /add_llm_server` after startup and stored in an in-memory Python list. Every
ProRL restart wiped them. The fix adds `--llm-server-address` CLI args that
pre-populate the list at process start — addresses survive restarts.

### Step 3a — LiveStore (gRPC UDS, same-machine)

```bash
nohup $POETRY_PYTHON -c "
import logging, os, signal, sys
sys.path.insert(0, '/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server')
logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s live_store: %(message)s')
from live_store.server import serve
server = serve(socket_path='/tmp/prorl_live_store.sock',
               max_size=256, staleness_cutoff_k=4, no_progress_timeout_s=1800)
print('[live_store] healthy', flush=True)
def _stop(s, f): server.stop(grace=2.0); sys.exit(0)
signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)
server.wait_for_termination()
" > /tmp/live_store.log 2>&1 &
echo $! > /tmp/live_store.pid
```

**Health gate:** `[[ -S /tmp/prorl_live_store.sock ]]`

**BC-16 note:** LiveStore starts empty. `get_batch` blocks server-side with a 1800s
no-progress timeout. Start the trainer AFTER the RolloutManager has pushed ≥1 group.

### Step 3b — PolicyRegistry (gRPC UDS, parallel with 3a)

```bash
ENDPOINTS_PY="['http://${REMOTE_DNS}:8100','http://${REMOTE_DNS}:8101','http://${REMOTE_DNS}:8102','http://${REMOTE_DNS}:8103']"

nohup $POETRY_PYTHON -c "
import logging, os, signal, sys
sys.path.insert(0, '/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server')
logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s policy_registry: %(message)s')
from policy_registry.server import serve
server = serve(socket_path='/tmp/prorl_policy_registry.sock',
               db_path='/tmp/prorl_policy_registry.db',
               pool_endpoints=${ENDPOINTS_PY})
print('[policy_registry] healthy', flush=True)
def _stop(s, f): server.stop(grace=2.0); sys.exit(0)
signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)
server.wait_for_termination()
" > /tmp/policy_registry.log 2>&1 &
echo $! > /tmp/policy_registry.pid
```

**Health gate:** `[[ -S /tmp/prorl_policy_registry.sock ]]`

### Step 4 — RolloutManager (BC-14: owns the dataset)

```bash
# SIF images must be built first (see Section 3 below)
# Verify at least one SIF exists:
ls singularity_images/*.sif | wc -l   # must be > 0

nohup $POETRY_PYTHON -m rollout_manager.main \
  --live-store-socket /tmp/prorl_live_store.sock \
  --prorl-url http://localhost:8006 \
  --policy-id "${POLICY_ID}" \
  --environment-id "${ENVIRONMENT_ID}" \
  --data-files "${DATA_FILES}" \
  --group-size 4 \
  --policy-manifest-path /tmp/prorl_policy_manifest.json \
  --archive-root /home/ubuntu/replay_archive \
  --filter-zero-variance \
  --archive-disabled \
  > /tmp/rollout_manager.log 2>&1 &
echo $! > /tmp/rollout_manager.pid
```

**Health gate (BC-16 warm-up):** Wait for LiveStore to have ≥1 group:
```bash
$POETRY_PYTHON -c "
import sys, time
sys.path.insert(0, '/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server')
from live_store.client import LiveStoreClient
cli = LiveStoreClient('/tmp/prorl_live_store.sock',
    policy_id='${POLICY_ID}', environment_id='${ENVIRONMENT_ID}')
for _ in range(120):
    if cli.total_pushes() > 0:
        print('Worker producing — store has groups')
        sys.exit(0)
    time.sleep(5)
print('TIMEOUT: worker not producing after 10min')
sys.exit(1)
"
```

**What the worker does (no VERL/OpenHands):**
```
Task → POST :8006/process → ProRLEpisodeResult → TrainingSample
→ group_uid bound across N siblings → archive tee → zero-variance filter → push to LiveStore
```

### Step 5 — TrainerAdapter (VERL Docker)

```bash
# Only start AFTER step 4 warm-up gate passes.
# The trainer must use LiveStoreClient (not the in-process TrajectoryStore).
# This modification to ray_trainer_dapo.py is the NEXT migration step (see Section 4).
bash scripts/_internal/s3_fullasync_docker.sh
```

### Using `start_all.sh` (orchestrator)

```bash
source /home/ubuntu/.prorl_creds.env
DATA_FILES="/home/ubuntu/data/SkyRL-v0-293/train.parquet" \
VLLM_POOL_ENDPOINTS="http://${REMOTE_DNS}:8100 ..." \
POLICY_ID="qwen3-4b-skyrl" \
  bash scripts/services/start_all.sh
```

`start_all.sh` runs all steps in order, health-probes each, enforces the BC-16 warm-up
gate (waits for worker to push ≥1 group before starting trainer), and handles clean
shutdown in reverse order on Ctrl-C.

---

## 3. Building Singularity Images (SWE-Bench)

The 232GB of OCI blob layers are pre-cached at
`scripts/_singularity_cache/apptainer_cachedir/cache/blob/` (329 manifests).
Converting them to `.sif` format is required before ProRL can run SWE-Bench episodes.

### Quick build (first N images for testing)

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
SINGULARITY_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME}" \
SINGULARITY_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD}" \
poetry run python scripts/pull_swe_images.py \
  --parquet-file /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --dest-dir singularity_images \
  --start-index 1 --end-index 10 \
  --log-name build.log
```

**Time estimate:** ~3-5 min per image (layers are cached; no download needed).
**All 293 train images:** `--start-index 1` (no `--end-index`) — ~15h total.

### Run full build in background (tmux recommended)

```bash
tmux new -s sif_build
# Inside tmux:
source /home/ubuntu/.prorl_creds.env
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
CACHE_BASE="$(pwd)/scripts/_singularity_cache"
mkdir -p "${CACHE_BASE}/apptainer_tmpdir" "${CACHE_BASE}/apptainer_localcachedir"
APPTAINER_CACHEDIR="${CACHE_BASE}/apptainer_cachedir" \
APPTAINER_LOCALCACHEDIR="${CACHE_BASE}/apptainer_localcachedir" \
APPTAINER_TMPDIR="${CACHE_BASE}/apptainer_tmpdir" \
APPTAINER_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME}" \
APPTAINER_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD}" \
SINGULARITY_DOCKER_USERNAME="${SINGULARITY_DOCKER_USERNAME}" \
SINGULARITY_DOCKER_PASSWORD="${SINGULARITY_DOCKER_PASSWORD}" \
poetry run python scripts/pull_swe_images.py \
  --parquet-file /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --dest-dir singularity_images \
  --log-name build_all.log
# Detach: Ctrl-B D
```

Monitor progress: `ls singularity_images/*.sif | wc -l`

---

## 4. What Still Needs to Be Done (Trainer Migration)

The trainer (`trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py`)
still uses the in-process `TrajectoryStore` and the internal `ContinuousRolloutProducer`.
To complete the S2 migration:

### Change 1: Replace `TrajectoryStore` with `LiveStoreClient`

In `ray_trainer_dapo.py` `__init__`:
```python
# OLD
self.trajectory_store = TrajectoryStore(
    max_size=replay_cfg.max_size,
    staleness_cutoff_k=replay_cfg.staleness_cutoff_k,
    pad_token_id=..., prompt_length_cap=..., response_length_cap=...,
)
# NEW
from live_store.client import LiveStoreClient
self.trajectory_store = LiveStoreClient(
    socket_path=os.environ.get('LIVE_STORE_SOCKET', '/tmp/prorl_live_store.sock'),
    policy_id=self.config.actor_rollout_ref.model.get('policy_id', 'qwen3-4b-skyrl'),
    environment_id='swe_agent',
    pad_token_id=..., prompt_length_cap=..., response_length_cap=...,
)
```

### Change 2: Remove internal producer

Remove calls to:
- `_make_continuous_producer()`
- `_start_continuous_producer_if_needed()`
- `_stop_continuous_producer_if_needed()`
- `dataloader` / `data.train_files` initialization

The trainer then does only `self.trajectory_store.sample_mini_batch(n_groups, global_steps)`
which under the new architecture calls `LiveStoreClient.sample_mini_batch` → blocks
server-side until N fresh groups are available → returns padded `SampledMiniBatch`.

### Change 3: Update policy version publish

After `_publish_lora_adapter()` succeeds:
```python
# Write manifest for worker polling (S2)
from policy_registry.file_registry import PolicyManifest, write_manifest
write_manifest(PolicyManifest(
    policy_id=self.config.actor_rollout_ref.model.policy_id,
    version=self.policy_version,
    adapter_uri=f'file://{local_checkpoint_folder}',
    trainer_id='trainer-0',
    published_at=time.time(),
))
```

At S4, replace this with `PolicyRegistryClient.publish_policy_version(...)`.

---

## 5. Major Boundary Conditions (Quick Reference)

| BC | Boundary | What breaks if violated |
|---|---|---|
| BC-0 | One `PolicyVersionSnapshot` per group dispatch | Siblings in a group see different policy versions → invalid advantage computation |
| BC-1 | Token IDs as `int` on every wire | KL/entropy NaN within 2 steps |
| BC-2 | `push_group` is atomic (0 or N records) | Partial group → wrong advantage denominator → NaN |
| BC-3 | Pop-on-sample in `get_batch` | Duplicate training → policy collapse |
| BC-4 | `get_batch` blocks on `num_fresh_groups`, not `num_groups` | Stale groups returned → `InsufficientTrajectoriesError` wedge |
| BC-5 | No-progress detector in `get_batch` server-side | Trainer hangs forever when worker is wedged |
| BC-6 | `eager_pushed_all` flag prevents double-push | Double-push corrupts `behavior_policy_version` / `created_at_step` |
| BC-7 | Worker polls policy version (≤1s latency) | IS correction uses wrong version → silent gradient bias |
| BC-8 | `created_at_step` from registry `trainer_step` field | Severely stale steps → all groups immediately evicted |
| BC-9 | `endpoints_failed > 0` = hard abort (not degraded) | Mixed-version pool → IS weights are lies |
| BC-10 | vLLM pinning protocol unchanged | Mid-trajectory adapter swap → IS weights are lies |
| BC-11 | LiveStore returns unpadded; trainer pads locally | `torch.stack` shape mismatch → crash |
| BC-12 | Archive tee is pre-filter; LiveStore is post-filter | Filtered groups lost forever from archive |
| BC-13 | RolloutManager imports zero VERL/OpenHands | Framework coupling breaks pluggability |
| BC-14 | Worker owns parquet dataloader | Trainer becomes hidden orchestrator (§3.8 violation) |
| BC-15 | Trainer connects only to LiveStore + PolicyRegistry | Trainer sneaks back into orchestrator role |
| BC-16 | `get_batch` warm-up: start trainer AFTER worker pushes ≥1 group | Trainer times out during buffer warm-up |

---

## 6. Fast Test Loop

```bash
# Run invariant + contract tests (no real services needed)
PYTHONPATH=. poetry run pytest tests/invariants/ tests/contracts/ -q
# Expected: 27 passed in ~1s

# Run LiveStore slot tests (starts in-process gRPC server)
PYTHONPATH=. poetry run pytest tests/slots/live_store/ -q
# Expected: 9 passed in ~5s
```

---

## 7. Rescue Agent Team Usage

```bash
# Health check all services
source /home/ubuntu/.prorl_creds.env
REMOTE_DNS=${REMOTE_DNS} PYTHONPATH=. \
  poetry run python scripts/services/rescue_team.py --check

# Auto-watch + rescue (run in a tmux pane during training)
REMOTE_DNS=${REMOTE_DNS} PYTHONPATH=. \
  poetry run python scripts/services/rescue_team.py --watch

# Manual rescue of a specific service
REMOTE_DNS=${REMOTE_DNS} PYTHONPATH=. \
  poetry run python scripts/services/rescue_team.py --rescue rollout_manager
```

The rescue team runs the probe → diagnose → fix loop (max 3 retries) per failing
service. Known error classes: `import_error`, `port_conflict`, `grpc_dead`, `oom`,
`nan_loss`, `pool_publish_fail`, `producer_wedged`. Unknown errors surface the log tail
for human inspection.

---

## 8. Current Status (2026-05-04)

| Service | Status | Notes |
|---|---|---|
| vLLM pool (:8100-8103) | ✓ Running | Remote EC2, all 4 healthy |
| ProRL (:8006) | ✓ Running | `s0_prorl.sh` fixed — vLLM addresses baked in |
| LiveStore (UDS) | ✓ Running | gRPC healthy, socket at `/tmp/prorl_live_store.sock` |
| PolicyRegistry (UDS) | ✓ Running | gRPC healthy, socket at `/tmp/prorl_policy_registry.sock` |
| RolloutManager | ⚠ ProRL 500 → fixed | Worker runs, but SIF images needed for episodes |
| SIF images | 🔄 Building | 1 image building; 232GB blobs cached; ~15h for all 293 |
| Trainer → LiveStore | ✗ Not yet | Next step: modify `ray_trainer_dapo.py` (Section 4) |
| Full training run | ✗ Blocked on SIF + trainer migration | After SIF build + trainer change |
