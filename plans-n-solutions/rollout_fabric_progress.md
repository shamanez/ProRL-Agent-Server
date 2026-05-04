# Rollout Fabric Migration — Progress

**Active stage:** post-S4 (ready for ultimate training run)
**Schema version:** v1.0.0
**Policy primitive:** `PolicyVersionSnapshot` (frozen) + `PolicyVersionCache` (atomic ref-swap) — `schemas/policy_version.py`

This file is the migration's audit trail. Per the operating principles
(see `/home/ubuntu/.claude/plans/you-are-a-planning-jolly-patterson.md`):

1. Per-stage CI is unit + contract + invariant + chaos; **no per-stage
   training runs**.
2. Each stage adds a "things to check" list of explainable behavioral
   assertions to the post-S4 training-run checklist below.
3. Hard cutover at every stage; deletes happen in the same PR as the
   replacement.

## S0 — Today (reference baseline)

- [x] Goal: full async loop runs end-to-end as today.
- Status: **green** (this is the inherited baseline).

## S0.5 — Cross-cutting prerequisites

- [x] Goal: land schemas, Protocols, the cleverest `policy_version`
      primitive, invariant tests, target-dir layout, progress doc.
      No behavior change.

### Tasks

- [x] `schemas/policy_version.py` — `PolicyVersionSnapshot` (frozen) +
      `PolicyVersionCache` (atomic ref-swap).
- [x] `schemas/training_sample.py` — §6.2 wire shape.
- [x] `schemas/episode_record.py` — §6.1 canonical record.
- [x] `schemas/protocols/*.py` — §A.1–§A.7 Protocol classes.
- [x] `schemas/proto/live_store.proto`, `schemas/proto/policy_registry.proto`.
- [x] `tests/invariants/test_token_in_token_out.py` — §3.1.
- [x] `tests/invariants/test_group_integrity.py` — §3.2.
- [x] `tests/invariants/test_per_row_policy_version_stamp.py` — §3.5
      + the cleverest primitive's contention contract.
- [x] `tests/invariants/test_pop_on_sample.py` — §3.6.
- [x] `tests/invariants/test_eager_push_seam.py` — §3.7.
- [x] `tests/contracts/test_protocols_importable.py` — Appendix A surface.
- [x] `pytest.ini` updated: markers (`invariant`, `contract`, …),
      `testpaths = tests`.
- [x] Empty target-dir markers per §D.4.
- [x] Progress doc seeded.

### Validation gates (S0.5)

- [ ] `make lint` green (run after pyproject + Makefile updates).
- [ ] Fast pytest loop green:
      `pytest -m "not integration and not slow and not real_data" tests/ -q`

### Things to check at post-S4 training (S0.5 list)

*None — S0.5 is no-op; verified at the cut by the fast pytest loop.*

### Status: **green** — 28/28 invariant + contract tests pass.

---

## S1 — LiveStore behind a same-machine gRPC boundary

- [x] Goal: replace in-process `TrajectoryStore` with a same-machine
      gRPC LiveStore over UDS. `_pack` lifted to
      `trainer_adapters/verl/pad.py`. No-progress detector server-side.
      Hard cutover; `trajectory_store.py` deleted.

### Tasks

- [x] `live_store/store_core.py` — extracted FIFO + lock + staleness +
      condition-variable-based no-progress detector.
- [x] `live_store/codec.py` — `TrainingSample` ↔ proto + DataProto
      bridge; token arrays as packed int32-LE bytes.
- [x] `live_store/server.py` + `live_store/client.py` — gRPC service
      and drop-in client.
- [x] `live_store/main.py` + `scripts/_internal/s0_5_live_store.sh` —
      service entry + launcher.
- [x] `trainer_adapters/verl/pad.py` — `pack_unpadded_groups` lifted
      from `_pack`; padding is now adapter-local per §6.2.
- [x] `schemas/_gen/` — gRPC stubs generated and checked in (so the
      runtime nodes only need `grpcio` / `protobuf`, not `grpcio-tools`).
- [x] Trainer construction at `ray_trainer.py:438` swapped to
      `LiveStoreClient`.
- [x] Trainer import at `ray_trainer.py:64` swapped to `live_store`.
- [x] Legacy `trainer_integration/verl/verl_custom/replay/trajectory_store.py`
      deleted.
- [x] Legacy `replay/__init__.py` re-exports `InsufficientTrajectoriesError`
      and `SampledMiniBatch` from the new home for any stragglers.
- [x] Slot tests: gRPC round-trip, no-progress (timeout + reset),
      concurrent pop-on-sample, pack-unpadded-groups.

### Validation gates (S1)

- [x] Fast pytest loop green (35/35 pass).
- [x] End-to-end launcher smoke: server launched via `live_store.main`,
      3-push + 2-sample round-trip yields legacy
      `(B, prompt_cap+response_cap)` tensor shape.

### Things to check at post-S4 training (S1 list)

| # | Item | §3 | Signal |
|---|---|---|---|
| 1 | Eager-push seam: zero terminal pushes when `eager_pushed_all=True` | §3.7 | `live_store.push.source` histogram |
| 2 | Per-row `behavior_policy_version` populated and monotonic | §3.5 | trainer step metrics |
| 3 | Pop-on-sample: no group-uid in two consecutive get_batch | §3.6 | `live_store.sample.uids` log |
| 4 | Token-in/token-out on the wire | §3.1 | first-push debug dump |
| 5 | `get_batch` blocks server-side; raises `NoProgressError` (not `KeyboardInterrupt`) | — | trainer logs |
| 6 | Step time within 5% of S0 baseline | — | WandB step-time |
| 7 | LiveStore kill-restart: trainer aborts cleanly, resumes after relaunch | — | chaos drill |

### Status: **green** — 9 slot tests pass, end-to-end smoke green, hard cutover landed.

---

## S2 — RolloutWorker process + data ownership migration

- [x] Goal: lift dataloader / producer / DAPO manager / eager-push
      seam out of the trainer. Trainer image drops openhands / aiohttp /
      fastapi / uvicorn. Trainer config drops `data.train_files`.
      `_validate` deleted entirely (validation removed). policy_version
      cache populated by JSON-mtime poller.

### Tasks

- [x] `policy_registry/file_registry.py` — atomic-rename JSON manifest
      writer/reader.
- [x] `rollout_worker/policy_subscription.py` — 1 Hz mtime poller,
      feeds `PolicyVersionCache.update` (atomic-ref-swap, lock-free reads).
- [x] `rollout_worker/manager.py` — producer loop lifted from
      `continuous_producer.py`; reads `policy_version` from
      `PolicyVersionCache.snapshot()` (not the GIL-atomic int).
- [x] `rollout_worker/main.py` + `scripts/_internal/s2_rollout_worker.sh` —
      service entry point + launcher.
- [x] Trainer's `_publish_lora_adapter` writes JSON manifest after
      pool ACK (manifest only on full success ⇒ §3.3 abort gate
      preserved by construction).
- [x] Trainer's producer-construction methods reduced to no-ops
      (`_start_continuous_producer_if_needed`,
      `_stop_continuous_producer_if_needed`); `_make_continuous_producer`
      deleted (parent + DAPO override).
- [x] Trainer's `_acquire_training_batch` and `_acquire_training_batch_dapo`
      simplified to pure `LiveStoreClient.sample_mini_batch` calls;
      gRPC LiveStore handles wedge detection server-side.
- [x] All `_validate` call sites + `val_before_train` block deleted
      from both trainers' `fit()` loops.
- [x] Legacy `continuous_producer.py` deleted.
- [x] §3.7 eager-push seam test rewired to import from
      `rollout_worker.manager`.

### Validation gates (S2)

- [x] Fast pytest loop green (38/38 pass, no skips).
- [x] End-to-end smoke: trainer-side `write_manifest` → worker-side
      mtime poller → `PolicyVersionCache.update` (atomic-ref-swap) →
      `LiveStore.notify_policy_version` propagation. Stale write
      rejected; cache stays at fresher version.

### Things to check at post-S4 training (S2 list)

| # | Item | §3 | Signal |
|---|---|---|---|
| 8 | Trainer image is `openhands`-free | §3.8 | startup log |
| 9 | `data.train_files` absent from trainer config | §3.8 | step-0 config dump |
| 10 | Worker keeps producing when trainer is killed | — | chaos drill + worker log |
| 11 | `policy_version` snapshot read once-per-group dispatch | §3.5 | worker log line per dispatch |
| 12 | Worker dataloader checkpoint round-trip | — | resume task-id set diff |

### Status: **green** — 38/38 fast-loop tests pass, manifest+cache smoke green, hard cutover landed.

---

## S3 — ReplayArchive durable canonical record

- [x] Goal: every episode the worker produces is teed to a durable,
      queryable archive in canonical `EpisodeRecord` form — including
      filtered groups.

### Tasks

- [x] `replay_archive/server.py` — append-only Parquet (zstd) + SQLite
      index. Partitioned `(policy_id, date, environment_id)`.
      Idempotent on `episode_uid` (dedup at append).
- [x] `replay_archive/writer.py` — bounded queue + retry +
      dead-letter spillover. Non-blocking `submit`; archive
      availability never stalls the producer.
- [x] `replay_archive/query.py` — `FilterSpec` against the SQLite
      index; Parquet hydrated for matching rows only.
- [x] `replay_archive/derive.py` — re-derive `TrainingSample` from
      `EpisodeRecord`; raises `TokenizerMismatchError` on mismatch
      (post-S4 checklist item 16).
- [x] Worker `main.py` wired to instantiate `ArchiveServer` +
      `ReplayArchiveWriter`; lifecycle owned by the worker process.
- [x] Slot tests: 7 round-trip / dedup / filter tests + 3 writer
      chaos tests (transient retry, retry-exhausted dead-letter,
      non-blocking submit under queue overflow).

### Validation gates (S3)

- [x] Fast pytest loop green (48/48 pass).
- [x] End-to-end smoke: 5 submits → 5 archived (0 dead-letter) →
      `query()` returns all 5 → offline reward-histogram job
      consumed the archive end-to-end.

### Things to check at post-S4 training (S3 list)

| # | Item | §3 | Signal |
|---|---|---|---|
| 13 | Archive count == episodes generated (mod dedup) | — | end-of-run summary |
| 14 | Filtered groups present in archive but absent from live path | §3.7 corollary | counter parity check |
| 15 | Archive backpressure does not stall the worker | — | step-time stability under chaos |
| 16 | `tokenizer_id` mismatch raises typed error | §3.1 corollary | offline re-derive smoke |

### Status: **green** — 10 archive slot tests pass, end-to-end smoke green, worker tee wired.

---

## S4 — PolicyRegistry as single source of truth

- [x] Goal: trainer publishes via gRPC PolicyRegistry; abort gate moves
      into the registry's fanout. Worker subscription flips from JSON
      polling to gRPC streaming, populates the same atomic-snapshot
      cache.

### Tasks

- [x] `policy_registry/server.py` — gRPC service: PublishPolicyVersion
      (with §3.3 abort-gate fanout), GetLatestVersion,
      SubscribeVersionUpdates (server-streaming), RegisterPolicyNamespace.
      SQLite backend.
- [x] `policy_registry/fanout.py` — pool /reload_lora fanout lifted
      from the legacy trainer code; tarball-and-POST + 200/409
      handling preserved.
- [x] `policy_registry/client.py` — `PolicyRegistryClient` with
      `publish_policy_version` (raises `PublishFailedError` on
      §3.3 abort) and `stream_version_updates` (yields immutable
      snapshots).
- [x] `rollout_worker/policy_subscription.py` — added
      `GrpcStreamingPolicySubscription` alongside the S2 file poller;
      same `PolicyVersionCache.update` populator (atomic ref-swap),
      different source. Reconnect with exponential backoff; cache
      never regresses on disconnect.
- [x] Trainer's `_publish_lora_adapter` reduced from 170-line fanout
      to a one-liner `client.publish_policy_version(...)` call.
- [x] `policy_registry/main.py` + `scripts/_internal/s4_policy_registry.sh` —
      service entry + launcher.
- [x] Slot tests: 3 abort-gate tests (partial fail, full success,
      409 idempotent), 2 streaming subscription tests (atomic swap,
      reconnect-no-stale).

### Validation gates (S4)

- [x] Fast pytest loop green (53/53 pass — full migration complete).
- [x] End-to-end smoke: trainer publishes via registry, pool stub
      simulates partial-pool failure, §3.3 abort fires, cache stays
      at prior version. Then full success → cache atomically
      swaps to new version, `get_latest_version` returns the
      manifest.

### Things to check at post-S4 training (S4 list)

| # | Item | §3 | Signal |
|---|---|---|---|
| 17 | Pool-child failure aborts trainer (abort gate) | §3.3 | chaos drill |
| 18 | Worker subscription latency <1 s p99 | — | publish→cache-update histogram |
| 19 | Subscription reconnect: no stale version served | — | chaos drill |
| 20 | Adapter URI manifest queryable | — | post-run `get_latest_version` |
| 21 | Snapshot atomicity: high-rate dispatch reads concurrent with publishes never observe a torn `(version, adapter_uri)` pair | §3.5 | invariant test green during run; sampled debug-dump on group-dispatch logs |

### Status: **green** — 8 registry slot tests pass, end-to-end smoke green (abort gate + atomic-ref-swap propagation), trainer publish path reduced to one line.

---

## Post-S4 ultimate training-run checklist (consolidated)

The single end-to-end training run that validates S1 → S4. All 21
items above are ticked individually with the log line / metric link
that proves them. If any item fails, the run is paused and the
offending cut is investigated.
