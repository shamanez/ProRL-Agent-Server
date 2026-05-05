# Rollout Fabric — Operations

This doc covers what is NOT in `CLAUDE.md`: SIF image build, rescue team, fast test
loop, and what was implemented in each stage. For startup commands and invariants,
see `CLAUDE.md`. For design rationale and BC definitions, see `rollout_fabric.md`.

---

## Implementation status

All four stages are complete. Training is running.

| Stage | What was built |
|---|---|
| S0.5 — Schemas + protocols | `schemas/policy_version.py`, `schemas/training_sample.py`, `schemas/episode_record.py`, `schemas/protocols/` (7 Protocol classes), `schemas/proto/*.proto`, `schemas/_gen/` (pre-compiled gRPC bindings), `tests/invariants/`, `tests/contracts/` |
| S1 — LiveStore gRPC service | `live_store/store_core.py`, `live_store/codec.py`, `live_store/server.py`, `live_store/client.py`, `trainer_adapters/verl/pad.py` |
| S2 — RolloutManager as standalone process | `rollout_manager/prorl_client.py`, `rollout_manager/dataloader.py`, `rollout_manager/episode_builder.py`, `rollout_manager/loop.py`, `rollout_manager/policy_subscription.py`, `rollout_manager/main.py` |
| S3 — ReplayArchive as tee | `replay_archive/server.py`, `replay_archive/writer.py`, `replay_archive/query.py`, `replay_archive/derive.py` |
| S4 — PolicyRegistry as single source of truth | `policy_registry/file_registry.py`, `policy_registry/fanout.py`, `policy_registry/server.py`, `policy_registry/client.py` |

Key fixes made during implementation:
- `s0_prorl.sh`: vLLM addresses baked in via `--llm-server-address` — survive ProRL restarts
- `ray_trainer_dapo.py`: internal producer removed; `trajectory_store` wired to `LiveStoreClient`
- `ray_trainer.py`: `_publish_lora_adapter` replaced with `PolicyRegistryClient.publish_policy_version()`
- `openhands/nvidia/__init__.py`: `add_name_mapping('swe-gym', 'swebench')` so SkyRL parquet routes to `SweAgentHandler`

---

## Building Singularity images (SWE-Bench)

The 232 GB of OCI blob layers are pre-cached at
`scripts/_singularity_cache/apptainer_cachedir/cache/blob/` (329 manifests).
Converting them to `.sif` format is required before ProRL can run episodes.

**Filter parquet after each new batch of SIFs:**

```bash
poetry run python scripts/filter_parquet_to_built_sifs.py \
  --input /home/ubuntu/data/SkyRL-v0-293/train.parquet \
  --sif-dir singularity_images \
  --output /home/ubuntu/data/SkyRL-v0-293/train.ready.parquet
```

**Build SIF images (run in tmux — takes ~3-5 min per image):**

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
  --log-name build_all.log
# Monitor: ls singularity_images/*.sif | wc -l
# All 293 images: ~15h total
```

---

## Rescue team

`scripts/services/rescue_team.py` runs a probe → diagnose → fix loop (max 3 retries)
for each failing service.

```bash
source /home/ubuntu/.prorl_creds.env
PYTHONPATH=.

# One-shot health check
poetry run python scripts/services/rescue_team.py --check

# Continuous watch + auto-rescue (run in a tmux pane during training)
poetry run python scripts/services/rescue_team.py --watch

# Rescue a specific service
poetry run python scripts/services/rescue_team.py --rescue rollout_manager
```

Known error classes handled automatically: `import_error`, `port_conflict`,
`grpc_dead`, `oom`, `nan_loss`, `pool_publish_fail`, `producer_wedged`.
Unknown errors surface the log tail for human inspection.

---

## Fast test loop

```bash
# Invariant + contract tests — no real services needed (~1s)
PYTHONPATH=. poetry run pytest tests/invariants/ tests/contracts/ -q

# LiveStore slot tests — starts an in-process gRPC server (~5s)
PYTHONPATH=. poetry run pytest tests/slots/live_store/ -q

# Full fast loop (excludes integration/slow/real_data)
pytest -m "not integration and not slow and not real_data" tests/ -q
```

Tests cover all 16 boundary conditions (BC-0 through BC-15), all seven Protocol
surfaces, LiveStore gRPC round-trips, and the packed-bytes token-ID codec.

---

## Policy sync chain (what happens at each LoRA publish)

When the trainer completes `SAVE_FREQ` steps:

1. FSDP saves `global_step_N/actor/lora_adapter/*.safetensors` under `/workspace/outputs/`
2. `_publish_lora_adapter` calls `PolicyRegistryClient.publish_policy_version(version=N, adapter_uri=file://...)`
3. PolicyRegistry fans out `POST /reload_lora` to all 4 vLLM children synchronously
4. If any child returns non-200/non-409 → `PublishFailedError` raised → trainer aborts (BC-9)
5. PolicyRegistry writes `/tmp/prorl_policy_manifest.json` (atomic rename)
6. RolloutManager's 1Hz `FilePollingPolicySubscription` detects mtime change
7. `PolicyVersionCache` atomically swaps to new snapshot
8. Next group dispatch: all 4 siblings use `/v{N}/generate` on vLLM (BC-0)
