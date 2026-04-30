# Producer-as-a-Service Architecture

Design doc for fully decoupling the rollout producer, trajectory store, and
trainer into independently deployable services. This is the forward-looking
architecture that evolves the current in-process `ContinuousRolloutProducer` +
`TrajectoryStore` + `RayPPOTrainerDAPO` stack into a network-separated
producer-consumer topology.

**Status:** Design-only. No code changes.
**Author:** Architecture review, 2026-04-30.
**References:** `CLAUDE.md`, `handsoff.md`, `full_async_system_overview.md`,
                `replay_dynamics.md`, and the files enumerated in each section.

---

## 1. Current state — what lives where today

```
┌───────────────────────── trainer box (host) ─────────────────────────┐
│                                                                       │
│  ProRL FastAPI :8006  (scripts/_internal/s0_prorl.sh)                 │
│    - OpenHands agent dispatcher                                       │
│    - Singularity sandbox lifecycle                                    │
│    - Multi-stage pipeline: init → run → eval                          │
│    - Calls vLLM children per assistant turn                           │
│                                                                       │
│  ┌─────────────────── Docker container (s3_fullasync_docker.sh) ─────┐│
│  │                                                                    ││
│  │  DATA LOADER                                                       ││
│  │    SkyRL-v0-293 parquet → StatefulDataLoader                       ││
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

**Key coupling today:** The producer daemon thread, the trajectory store, and
the trainer all live in the same Python process inside the Docker container.
They share memory, a `threading.Lock`, a `StepCounter`, and a direct function
pointer (`_push_fn`). The producer reads `rollout_manager.policy_version`
across threads (GIL-atomic int). The dataloader is a `StatefulDataLoader` in
the same process.

**What works about this:** Zero serialization overhead. The `_pack` method
produces tensors directly. Push and sample are ~1ms under the lock. No network
latency between producer and store.

**What does not scale:**
- Cannot run multiple producers (different machines, different instance pools).
- Cannot run multiple trainers against the same trajectory feed.
- A producer crash kills the trainer (same process).
- Buffer is ephemeral — not checkpointable across restarts.
- Producer throughput is limited to one Python process's concurrency.

---

## 2. Target architecture — producer-as-a-service

```
┌──────── Producer Service A (machine P1) ──────────┐
│  Data Loader (SkyRL-v0 train.parquet)              │
│  AsyncLLMServerManagerDAPO                         │
│  filter_easy_hard_instance (zero-variance drop)    │
│  ProRL client → ProRL :8006 → vLLM pool            │
│  gRPC push_group() → Trajectory Store Service      │
│  Subscribes to policy_version from Coord Service   │
└────────────────────────────────────────────────────┘

┌──────── Producer Service B (machine P2) ──────────┐    ← future
│  (same shape, different data shard or prompt set)  │
│  gRPC push_group() → same Store Service            │
└────────────────────────────────────────────────────┘

┌──────── Trajectory Store Service (machine S1) ────┐
│  gRPC server                                       │
│  Bounded FIFO buffer (groups)                      │
│  Staleness eviction (K-cutoff)                     │
│  Sample-and-pop (queue semantics)                  │
│  Re-pad + pack at sample time                      │
│  Metrics endpoint (/metrics, Prometheus)            │
│  Coordination: receives policy_version updates     │
│  Backpressure: push blocks or rejects when full    │
└────────────────────────────────────────────────────┘

┌──────── Trainer Service (machine T1) ─────────────┐
│  RayPPOTrainerDAPO (8×A100 FSDP)                   │
│  gRPC get_batch(n_groups, current_step)             │
│    → receives SampledMiniBatch-shaped response      │
│  compute_reward → old_log_prob → advantage          │
│  update_actor                                       │
│  save_checkpoint                                    │
│  gRPC publish_version(v, adapter_uri) → Coord       │
│  Validation: gRPC get_val_batch() from Store or     │
│    direct ProRL call (see §9)                       │
└────────────────────────────────────────────────────┘

┌──────── Coordination Service (lightweight) ───────┐
│  gRPC / HTTP                                       │
│  Source of truth: latest policy_version             │
│  Fanout: notifies pool, store, all producers       │
│  Adapter registry: version → URI (S3/NFS path)     │
└────────────────────────────────────────────────────┘

┌──────── vLLM Pool (unchanged) ────────────────────┐
│  4× _vllm_child.py :8100-8103                      │
│  /reload_lora now triggered by Coord Service        │
│  /v{N}/generate unchanged                           │
│  /health unchanged                                  │
└────────────────────────────────────────────────────┘

┌──────── ProRL FastAPI :8006 (unchanged) ──────────┐
│  Agent dispatcher, sandbox lifecycle               │
│  No changes needed                                  │
└────────────────────────────────────────────────────┘
```

### Arrow summary

```
Producer(s)  ──push_group()──►  Store Service
Trainer      ──get_batch()───►  Store Service
Trainer      ──publish()─────►  Coordination Service
Coord Svc    ──notify()──────►  Producer(s), Store, Pool
Producer(s)  ──HTTP──────────►  ProRL :8006 ──HTTP──► vLLM Pool
```

---

## 3. Boundary contracts — RPC surfaces

### 3.1 Producer → Trajectory Store: `PushGroup`

```protobuf
service TrajectoryStoreService {
  rpc PushGroup(PushGroupRequest) returns (PushGroupResponse);
  rpc GetBatch(GetBatchRequest) returns (GetBatchResponse);
  rpc GetStoreMetrics(MetricsRequest) returns (MetricsResponse);
  rpc NotifyPolicyVersion(PolicyVersionNotification) returns (Empty);
}

message TrajectoryRecord {
  // Token-level data (the hot path — these are the large payloads)
  bytes prompt_ids = 1;           // varint-packed int32 sequence
  bytes response_ids = 2;         // varint-packed int32 sequence
  bytes response_loss_mask = 3;   // varint-packed int32 (0/1)
  bytes response_log_probs = 4;   // packed float32 sequence

  // Scalar metadata
  float reward = 5;
  float advantage = 6;
  int32 behavior_policy_version = 7;
  int32 created_at_step = 8;      // trainer step at push time
  string prompt_uid = 9;
  string group_uid = 10;
  bool resolved = 11;
  bool success = 12;
  bool finish = 13;
  bool is_padded = 14;
  string error = 15;              // empty string = no error

  // Structured metadata — serialized as JSON bytes
  bytes instance_json = 16;       // dict → JSON
  bytes prompt_extras_json = 17;  // dict → JSON (data_source, ability, etc.)
}

message PushGroupRequest {
  repeated TrajectoryRecord records = 1;
  string group_uid = 2;
  int32 producer_id = 3;         // identifies which producer
}

message PushGroupResponse {
  bool accepted = 1;
  int32 store_size = 2;          // current group count post-push
  int32 backpressure_ms = 3;     // 0 if no backpressure; >0 = suggested wait
}
```

**Wire format rationale:** Token sequences dominate the payload
(~50K int32s per trajectory at max length). Using `bytes` with packed encoding
keeps the protobuf envelope small. The store deserializes only at pack/sample
time, not on ingest.

**Field mapping from current `TrajectoryRecord` dataclass
(`trajectory_store.py:43-74`):** Every field of the frozen dataclass appears
above. `prompt_extras` is serialized as JSON bytes because its schema varies
per data source — protobuf `map<string, bytes>` would work but adds complexity
for a field that is ~200 bytes per row.

### 3.2 Trainer → Trajectory Store: `GetBatch`

```protobuf
message GetBatchRequest {
  int32 n_groups = 1;
  int32 current_step = 2;         // for staleness eviction
  int32 staleness_cutoff_k = 3;   // override or 0 = use server default
  int32 timeout_ms = 4;           // server-side blocking if insufficient
}

message GetBatchResponse {
  // Tensors: serialized as raw bytes with shape metadata.
  // The store re-pads before serialization (same as _pack today).
  map<string, TensorPayload> tensors = 1;
  map<string, NonTensorColumn> non_tensors = 2;

  // Meta-info for IS correction
  repeated int32 behavior_policy_versions = 3;
  repeated int32 created_at_steps = 4;
  repeated int32 sample_ages = 5;

  StoreMetrics pre_sample_metrics = 6;
  StoreMetrics post_sample_metrics = 7;
}

message TensorPayload {
  repeated int64 shape = 1;
  string dtype = 2;              // "int64", "float32", "bool"
  bytes data = 3;                // raw tensor bytes (row-major)
}

message NonTensorColumn {
  repeated bytes values = 1;     // per-row JSON-encoded values
}
```

**Why the store packs before sending:** The current `_pack` method
(`trajectory_store.py:470-621`) already does re-padding and tensor
construction. Moving this to the server side means the trainer receives
ready-to-use tensors — no redundant deserialization + re-padding on the
client. The `TensorPayload` is a zero-copy envelope; the trainer calls
`torch.frombuffer(data, dtype=dtype).reshape(shape)`.

**Blocking semantics:** If the store has fewer than `n_groups` non-stale
groups, the server blocks up to `timeout_ms` (polling internally). This
replaces the current `wait_until_with_progress` busy-loop in
`ray_trainer_dapo.py:113-118`. The store server implements the no-progress
detector internally: if `total_pushes` does not increase for
`no_progress_timeout_s`, the RPC returns with an error code.

### 3.3 Trainer → Coordination Service: `PublishPolicyVersion`

```protobuf
service CoordinationService {
  rpc PublishPolicyVersion(PublishRequest) returns (PublishResponse);
  rpc GetLatestVersion(Empty) returns (VersionInfo);
  rpc SubscribeVersionUpdates(Empty) returns (stream VersionInfo);
}

message PublishRequest {
  int32 policy_version = 1;
  string adapter_uri = 2;         // S3 path or NFS path to the tarball
  int32 trainer_step = 3;
}

message PublishResponse {
  bool success = 1;
  int32 endpoints_ok = 2;
  int32 endpoints_failed = 3;
  float publish_latency_s = 4;
}

message VersionInfo {
  int32 policy_version = 1;
  string adapter_uri = 2;
  int64 timestamp_ms = 3;
}
```

**What the coordination service does on `PublishPolicyVersion`:**

1. Stores `(version, adapter_uri)` in its registry.
2. POSTs `/reload_lora` to every pool child (same logic as
   `ray_trainer.py:_publish_lora_adapter:1413-1544`). Preserves the abort
   contract: `endpoints_failed > 0` → returns failure, trainer aborts.
3. Pushes `VersionInfo` to all connected `SubscribeVersionUpdates` streams
   (producers and the store).

**Why not trainer → pool direct?** It still can be, and cut 0/1 keep it that
way. The coordination service is cut 3+, when multiple producers or trainers
need a single source of truth for the active version set. In the interim the
trainer keeps its direct `/reload_lora` fanout and the coordination service is
just a version registry the producers poll.

### 3.4 Producer → Coordination: subscribe version updates

Producers call `SubscribeVersionUpdates()` — a server-streaming RPC. Each
`VersionInfo` message tells the producer: "dispatch new rollouts against
`policy_version=N` using path-versioned `/v{N}/generate`."

The producer reads this before constructing each DAPO batch. If no update has
arrived, it reuses the last known version. The benign race where a publish
lands mid-batch is the same as today's GIL-atomic `policy_version` read
(gotcha #20 in `handsoff.md`): the batch finishes tagged with the old version;
TIS correction handles it.

---

## 4. State migration — what moves out of the trainer

| State | Current location | Target location | Migration complexity |
|---|---|---|---|
| `TrajectoryStore` (deque + lock + counters) | In-process, `ray_trainer.py` | Store Service (new process) | Medium — core logic unchanged, add gRPC envelope |
| `_pack()` re-padding logic | `trajectory_store.py:470-621` | Store Service, inside `GetBatch` handler | Low — copy as-is |
| Staleness eviction (`_evict_stale_locked`) | `trajectory_store.py:406-418` | Store Service | Low |
| `filter_easy_hard_instance` (zero-variance drop) | `async_server_dapo.py:766-784` | Stays producer-side | None — already on the right side |
| `StepCounter` | `continuous_producer.py:50-70` | Replaced by `current_step` field in `GetBatchRequest` | Low |
| `policy_version` (monotonic int) | Trainer attribute + GIL-atomic cross-thread read | Coordination Service registry | Medium |
| `_push_fn` closure (eager-push seam) | `ray_trainer.py:1810-1817` | Producer calls `store_client.push_group()` directly | Low |
| Adapter tarball build + fanout | `ray_trainer.py:_publish_lora_adapter` | Phase 1: stays in trainer. Phase 2: Coordination Service | Medium |
| `wait_until_with_progress` (no-progress detector) | `continuous_producer.py:357-392` | Store Service (server-side blocking on `GetBatch`) | Low |
| Backpressure (`_store_full` check) | `continuous_producer.py:305-328` | `PushGroupResponse.backpressure_ms` | Low |
| Dataloader (`StatefulDataLoader`) | Trainer process | Producer Service | Low — already used by DAPO manager |
| Validation dispatch | Trainer calls `_validate()` → `generate_sequences(val_mode=True)` | See section 9 | Medium |

---

## 5. Wire schema for trajectories

Complete field inventory crossing the producer→store boundary, derived from
`TrajectoryRecord` (`trajectory_store.py:43-74`) and
`_convert_results_to_dataproto_token` (`async_server.py:1300-1479`):

| Field | Python type | Wire type | Bytes/row (typical) | Notes |
|---|---|---|---|---|
| `prompt_ids` | `tuple[int, ...]` | packed int32 | ~12 KB (3000 tokens) | Left-padded to `max_starting_message_length` in current system; stored unpadded in `TrajectoryRecord` |
| `response_ids` | `tuple[int, ...]` | packed int32 | ~48 KB (12000 tokens) | Up to `total_len = 47616` tokens max |
| `response_loss_mask` | `tuple[int, ...]` | packed int32 (0/1) | same as response_ids | 1 on assistant turns only |
| `response_log_probs` | `tuple[float, ...]` | packed float32 | ~48 KB | Per-token log-prob from vLLM |
| `reward` | `float` | float32 | 4 B | Scalar, often 0.0 at push time (recomputed by trainer) |
| `advantage` | `float` | float32 | 4 B | 0.0 at push time (computed by trainer on sample) |
| `behavior_policy_version` | `int` | int32 | 4 B | Per-row stamp from `instance['policy_version']` |
| `created_at_step` | `int` | int32 | 4 B | Trainer step at push time |
| `prompt_uid` | `str` | string | ~36 B (UUID) | Groups rows by prompt |
| `group_uid` | `str` | string | ~36 B (UUID) | Same as prompt_uid in DAPO |
| `resolved` | `bool` | bool | 1 B | Did the agent solve the task? |
| `success` | `bool` | bool | 1 B | Task completion status |
| `finish` | `bool` | bool | 1 B | Finish action detected |
| `is_padded` | `bool` | bool | 1 B | Was this a padding row? |
| `error` | `str \| None` | string | 0-200 B | Error message or empty |
| `instance` | `dict[str, Any]` | JSON bytes | ~500 B | Instance metadata |
| `prompt_extras` | `dict[str, Any]` | JSON bytes | ~200 B | `data_source`, `ability`, `reward_model`, `extra_info`, `index` |

**Per-group wire size estimate (n=8 siblings, avg 8K response tokens):**
- Token data: 8 × (12 + 32 + 32 + 32) KB = ~864 KB
- Metadata: 8 × ~1 KB = ~8 KB
- Total: ~870 KB per group, ~7 MB per push (8 groups)
- gRPC default max message size is 4 MB; raise to 16 MB or stream in chunks.

**What crosses the store→trainer boundary (`GetBatchResponse`):**
All of the above, but re-padded into tensors matching the current
`SampledMiniBatch` shape (`trajectory_store.py:90-101`):

| Tensor key | Shape | dtype |
|---|---|---|
| `input_ids` | `(B, prompt_cap + response_cap)` | int64 |
| `responses` | `(B, response_cap)` | int64 |
| `attention_mask` | `(B, prompt_cap + response_cap)` | int64 |
| `position_ids` | `(B, prompt_cap + response_cap)` | int64 |
| `loss_mask` | `(B, response_cap)` | int64 |
| `rollout_log_probs` | `(B, response_cap)` | float32 |
| `is_padded` | `(B,)` | bool |
| `error_mask` | `(B,)` | bool |
| `reward` | `(B,)` | float32 |
| `advantage` | `(B,)` | float32 |

Where `B = n_groups × n` and caps are `prompt_length_cap=31232`,
`response_length_cap=47616` from the Hydra config.

Non-tensor columns: `uid`, `success`, `error`, `resolved`, `finish`,
`instance`, plus all `prompt_extras` keys — same as
`trajectory_store.py:604-617`.

---

## 6. Failure modes and isolation properties

### 6.1 Producer crashes

**Symptom:** No new `PushGroup` RPCs arrive at the store.

**Store behavior:** Continues serving `GetBatch` from existing buffer.
Groups age and are evicted by the K-staleness cutoff. Eventually the store
drains to zero groups.

**Trainer behavior:** `GetBatch` blocks up to `timeout_ms`. The no-progress
detector (implemented server-side on the store) fires after
`no_progress_timeout_s` with no new pushes. Trainer gets an error response,
logs it, and can decide to: (a) wait for producer restart, (b) save checkpoint
and exit, (c) switch to a backup producer.

**Recovery:** Restart the producer. It calls `SubscribeVersionUpdates` on
the coordination service, learns the current `policy_version`, and resumes
dispatching rollouts. The store buffer warms up normally.

**Improvement over today:** Today a producer crash kills the trainer (same
process, `_exception` propagated via `check_background_error`). In the new
architecture the trainer is isolated.

### 6.2 Trainer crashes

**Symptom:** No new `GetBatch` RPCs arrive at the store.

**Store behavior:** Buffer fills to `max_size`. FIFO eviction drops oldest
groups. Producer's `PushGroup` calls continue succeeding (store silently
evicts). If we want backpressure instead, the store returns
`backpressure_ms > 0` and the producer throttles (see section 7).

**Producer behavior:** Keeps running. Its `SubscribeVersionUpdates` stream
stops receiving new versions (no publishes happening), so it keeps dispatching
on the last-known version. This is correct — the trajectories will be stale
by the time the trainer restarts, but K-staleness eviction handles that.

**Recovery:** Restart the trainer. It calls `GetBatch` with its
`global_steps` (from checkpoint). Stale groups are evicted. Producer's
fresh groups fill the buffer. Training resumes.

### 6.3 Store crashes

**Impact:** Catastrophic but recoverable. Buffer is ephemeral (not
checkpointed, same as today — `handsoff.md` gotcha #24).

**Producer behavior:** `PushGroup` RPCs fail. Producer logs errors, backs off,
retries with exponential backoff. It does not crash — just pauses.

**Trainer behavior:** `GetBatch` RPCs fail. Trainer saves checkpoint, enters
wait-with-backoff mode.

**Recovery:** Restart the store service. Both producer and trainer reconnect.
Buffer warms from scratch. Warm-up time = `N / producer_throughput`.

**Mitigation:** The store is a stateless service (buffer is ephemeral).
Run it with a process supervisor (systemd, K8s deployment with
`restartPolicy: Always`). Startup time is ~1 second. For additional
resilience, WAL the push stream to disk and replay on restart (cut 4+).

### 6.4 Network partition between trainer and store

**Trainer behavior:** `GetBatch` times out. Trainer retries with exponential
backoff. If `no_progress_timeout_s` elapses with no successful sample, trainer
saves checkpoint and exits (same behavior as today's no-progress detector).

**Store behavior:** Keeps accepting pushes from the producer. Buffer fills,
FIFO evicts, or backpressure kicks in.

**Recovery:** Network heals. Trainer's next `GetBatch` succeeds. Stale groups
are evicted. Fresh groups are sampled.

### 6.5 Network partition between producer and store

**Producer behavior:** `PushGroup` RPCs fail. Producer keeps generating
rollouts and discards them (or buffers a small local queue, see section 7).
Logs warnings.

**Store behavior:** Buffer drains as the trainer keeps sampling.

**Recovery:** Network heals. Producer's pushes resume. Buffer refills.

### 6.6 Weight-sync race under the migration

**Concern:** Producer dispatches rollout on `policy_version=v` while the pool
is mid-`/reload_lora` for version `v+1`.

**Current handling (preserved):** The pinning swap protocol
(`_vllm_child.py:60-86`) never removes in-flight adapters. A request pinned
to `/v{N}/generate` uses `LoRARequest(lora_int_id=N)` regardless of whether
a newer version has been loaded. The pool's LRU handles GPU slot management.
This protocol is entirely within the vLLM pool and ProRL, which are unchanged.

**New concern:** With the coordination service broadcasting version updates,
could a producer learn about version `v+1` before the pool has finished
loading it? Yes — but the producer only uses the version for tagging the
`behavior_policy_version` on the trajectory. The actual LoRA used for
generation is determined by the `/v{N}/generate` path, and the pool returns
a generation error if version N is not yet loaded (the child returns 404 for
unknown versions). The producer handles this by falling back to the
last-known-good version or retrying.

**Recommendation:** The coordination service should only broadcast after
receiving success from all pool children. This is exactly the current
`endpoints_failed == 0` gate in `_publish_lora_adapter`
(`ray_trainer.py:1500-1506`).

---

## 7. Backpressure model

**Today:** `ContinuousRolloutProducer._store_full()` checks
`store.num_groups() >= store._max_size` and sleeps `poll_interval_s` before
retrying (`continuous_producer.py:239-243`). This is cooperative, in-process,
zero-cost.

**Options for the networked case:**

| Strategy | Pros | Cons |
|---|---|---|
| **A. Server-side blocking on push** | Simple client. Store controls flow. | Ties up producer's gRPC channel. Multi-producer fairness is hard. |
| **B. Push always succeeds, FIFO evicts** | Producer never stalls. | Wasted compute: producer generates trajectories the store will throw away. |
| **C. Push returns backpressure hint** | Producer can throttle itself. Store stays non-blocking. Composable with multiple producers. | Slightly more complex client. |
| **D. Producer pre-queries free slots** | Producer can skip dispatch if full. | Extra RPC per iteration. Race between query and push. |

**Recommendation: Option C — push with backpressure hint.**

The `PushGroupResponse` includes `backpressure_ms`. When the store is at
capacity, it FIFO-evicts and returns `backpressure_ms > 0` as a hint. The
producer sleeps `backpressure_ms` before its next dispatch cycle. When the
store is below high-water (e.g., 80% full), it returns `backpressure_ms = 0`.

This matches the current behavior: the store's `deque(maxlen=N)` silently
drops the oldest group on overflow. The producer's sleep is advisory, not
mandatory. Wasted rollout compute is bounded by one batch per overflow.

**Why not blocking?** Multi-producer fairness. If producer A fills the buffer
and its push blocks, producer B's pushes also block — head-of-line blocking
across independent producers. The advisory model lets each producer
independently decide whether to throttle or discard.

For the single-producer case (cut 1-2), blocking is simpler and wastes less
compute. The advisory model is forward-compatible with multi-producer (cut 4+).

---

## 8. Weight-publish topology

**Today:** Trainer → pool direct (`_publish_lora_adapter` POSTs `/reload_lora`
to each child endpoint). Trainer writes `self.policy_version` after all
endpoints ACK. Producer reads it via GIL-atomic int.

**Options for the decoupled case:**

| Topology | Pros | Cons |
|---|---|---|
| **A. Trainer → pool direct (unchanged), producer polls coordination** | Minimal change. Pool publish is battle-tested. | Two sources of truth (trainer int + coordination registry). |
| **B. Trainer → coordination → fanout to pool + producers + store** | Single source of truth. Clean observer pattern. | New SPOF (coordination service). Extra latency on publish. |
| **C. Trainer → pool direct + trainer writes version to shared store** | Simple. No new service. | Store becomes the coordination layer — mixed concerns. |

**Recommendation: Incremental migration.**

- **Cut 1-2:** Option A. Trainer keeps direct pool publish. Adds a lightweight
  HTTP endpoint or Redis key where the trainer writes the current
  `policy_version` after successful pool publish. Producer polls this.
  Simple, no new service.

- **Cut 3+:** Option B. Coordination service takes over pool publish. Trainer
  calls `PublishPolicyVersion(v, adapter_uri)`. Coordination:
  1. Stores adapter tarball at `adapter_uri` (S3 or NFS).
  2. POSTs `/reload_lora` to all pool children with the tarball.
  3. Waits for all ACKs. Abort on any failure.
  4. Broadcasts `VersionInfo` to all subscribers (producers, store).

**Why B wins long-term:** With multiple trainers (federated aggregation),
each trainer publishes independently. The coordination service serializes
version increments and broadcasts atomically. Without it, N trainers racing
to POST `/reload_lora` on the same pool creates conflicts (pool rejects
non-monotonic versions).

**Adapter storage:** Move from in-memory tarball (current: `buf = io.BytesIO()`
in `_publish_lora_adapter`) to a durable URI (S3 bucket or NFS mount). The
coordination service holds the manifest `{version: uri}`. Pool children
download from the URI on `/reload_lora`. This decouples adapter creation
(trainer) from adapter distribution (coordination).

---

## 9. Validation flow

**Today:** Validation pauses the producer
(`_stop_continuous_producer_if_needed`), runs `_validate()` which calls
`generate_sequences(val_mode=True)` through the same ProRL/pool path, then
resumes the producer. The producer and validation share one OpenHands session
(`handsoff.md` gotcha #25).

**Options:**

| Option | Description |
|---|---|
| **A. Trainer drives validation directly** | Trainer calls ProRL/pool for val, same as today. Producer paused during val. |
| **B. Dedicated val producer** | A separate producer instance runs validation prompts and pushes to a val-specific store or directly returns results. |
| **C. Val-as-a-service endpoint on the producer** | Producer service exposes a `RunValidation(val_batch)` RPC. Trainer calls it. Producer pauses train dispatch, runs val, returns results, resumes. |

**Recommendation: Option A for cut 1-2, migrate to Option C for cut 3+.**

Option A is simplest: the trainer already owns the FSDP actor weights needed
for `compute_old_log_prob` on validation batches. Validation is infrequent
(`test_freq` steps apart) and short (~23 instances vs 293 train).

For cut 3+, Option C is cleaner: the producer service already owns the ProRL
client and pool connection. Validation is just another dispatch with
`val_mode=True`. The producer pauses train dispatch, runs val, returns the
`DataProto` over gRPC, resumes. This avoids the trainer needing a ProRL
client at all.

**Key constraint:** Validation still needs the trainer's FSDP actor to
compute `old_log_prob` and `ref_log_prob` on the validation batch (for
computing perplexity correlation metrics). The rollout itself (ProRL → pool)
does not need FSDP. So the flow is:

```
Trainer: "run validation please"
   → Producer Service: dispatches val batch through ProRL/pool
   ← Producer Service: returns rollout DataProto (token_ids, log_probs)
Trainer: computes old_log_prob, reward, advantage locally
Trainer: logs val metrics
```

---

## 10. Decentralization angle

### 10.1 Multiple producers

**Enabled by:** Each producer is an independent service with its own data
shard. All push to the same store. Each subscribes to version updates from
coordination.

**Data sharding:** The `StatefulDataLoader` today iterates all 293 train
prompts. With N producers, each gets a disjoint shard (e.g., producer 0 gets
prompts 0-146, producer 1 gets 147-293). Or: all producers iterate the full
dataset but the store deduplicates by `group_uid` (UUID collision is
astronomically unlikely). Sharding is simpler and avoids redundant rollouts.

**Heterogeneous producers:** Producer A runs SWE-Bench instances (expensive,
multi-turn). Producer B runs math instances (cheap, single-turn). Both push
to the same store. The trainer samples a mix. This is the natural extension
of the current multi-`data_source` registry.

**A/B prompt strategies:** Producer A uses prompt template v1. Producer B uses
template v2. `prompt_extras` carries a `prompt_strategy` tag. The trainer
can filter or weight by strategy at sample time (add a `filter_fn` to
`GetBatchRequest`).

### 10.2 Multiple trainers

**Enabled by:** Each trainer calls `GetBatch` independently from the store.
Pop-on-sample semantics mean two trainers drawing simultaneously get disjoint
groups (the lock inside `sample_mini_batch` serializes the pops).

**Concern:** Two trainers each publishing different policy versions. The pool
can only serve one version at a time (per-child, path-versioned pinning
supports multiple, but the coordination service must agree on version
ordering). Solution: each trainer publishes to its own adapter namespace.
Pool children load multiple adapters concurrently (already supported:
`--max-loras 8`). Producers tag which adapter to use per rollout.

**Federated aggregation:** Trainer A produces adapter delta A. Trainer B
produces adapter delta B. An aggregation service (FedAvg, etc.) combines
them into adapter C and publishes C to the pool via coordination. This is
beyond cut 5 and requires adapter-arithmetic support, but the architecture
does not preclude it.

### 10.3 Externally-contributed trajectories

**Enabled by:** Any client that speaks the `PushGroup` gRPC contract can feed
trajectories into the store. A partner runs their own rollout infrastructure,
generates trajectories in the wire schema (section 5), and pushes them.

**Requirements:**
- Partner must use the same tokenizer (Qwen3-4B-Instruct-2507). Token IDs
  must be exact; the token-in/token-out invariant
  (`openhands/llm/nvidia/qwen3.py`) is non-negotiable.
- `behavior_policy_version` must be set to 0 (or a known sentinel) indicating
  "external, not from our policy." The IS correction must handle this:
  external trajectories get `tis_imp_ratio = 1.0` (no correction, treated
  as on-policy or excluded from the ratio).
- `prompt_extras` must include `data_source` so the reward manager can route
  to the correct scorer.

### 10.4 Multi-region deployment

```
Region A (us-east-1):
  Producer A → Store A → Trainer A → Pool A
  
Region B (eu-west-1):
  Producer B → Store B → Trainer B → Pool B

Cross-region aggregation:
  Trainer A publishes adapter-A → Aggregation Service
  Trainer B publishes adapter-B → Aggregation Service
  Aggregation Service → combined adapter → Pool A + Pool B
```

The store is region-local (latency-sensitive). Aggregation is async and
tolerant of ~100ms cross-region latency. This is the "futuristic" vision.

---

## 11. What does NOT change

1. **vLLM pool** — stays as-is. `_vllm_child.py`, `/v{N}/generate`,
   `/reload_lora`, pinning swap protocol, `--max-loras 8`. No code changes.

2. **ProRL FastAPI server** — stays as-is. `openhands/nvidia/async_server.py`,
   the three-stage pipeline (init → run → eval), the agent handler registry.
   The producer calls ProRL the same way it does today (HTTP to `:8006`).

3. **Token-in/token-out invariant** — `openhands/llm/nvidia/qwen3.py`,
   `qwen2_5_vl.py`. Exact token IDs round-trip across turns. Never
   decode/re-tokenize. The wire schema stores token IDs as integers.

4. **Chat template logic** — `chat_template_manager.py`. Unchanged.

5. **`filter_easy_hard_instance`** — stays producer-side. The zero-variance
   drop (`resolved == 0` or `resolved == n`) runs before push. The store
   never sees zero-variance groups.

6. **GRPO advantage computation** — stays trainer-side. `compute_advantage`
   runs on the sampled batch after `GetBatch`. Requires whole groups intact
   (the store never splits groups — same as today).

7. **Temporal IS correction** — `core_algos.py:664-690`. The ratio
   `exp(old_log_prob - rollout_log_probs)` is computed trainer-side on the
   sampled batch. `rollout_log_probs` comes from the store (behavior policy).
   `old_log_prob` is recomputed by the FSDP actor. No change.

8. **Reward computation** — stays trainer-side. `reward_fn` is a Ray remote.

9. **`_save_checkpoint`** — stays trainer-side. FSDP actor weights.

---

## 12. Migration path — incremental cuts

### Cut 0: Today (no changes)

Everything in-process inside the Docker container. Producer daemon thread,
in-process `TrajectoryStore`, trainer loop. This is the `full-async-
optimization-final-cut` branch.

### Cut 1: Extract the store to a separate process (same machine)

**Scope:** Replace the in-process `TrajectoryStore` with a gRPC server running
as a separate process on the trainer box. The trainer and producer both connect
to it via localhost gRPC.

**Changes:**
- New file: `trajectory_store_server.py` — wraps `TrajectoryStore` in a gRPC
  service. Implements `PushGroup`, `GetBatch`, `GetStoreMetrics`.
- New file: `trajectory_store_client.py` — a thin client that the trainer's
  `_acquire_training_batch_dapo` and the producer's push path call instead of
  the in-process store.
- `ray_trainer.py` / `ray_trainer_dapo.py`: replace `self.trajectory_store`
  with `TrajectoryStoreClient(host='localhost', port=50051)`.
- `continuous_producer.py`: replace `self._store.push_from_dataproto(...)` and
  `self._store.num_groups()` with client calls.
- `s3_fullasync_docker.sh`: launch `trajectory_store_server.py` as a sidecar
  before the trainer.

**What this proves:** The gRPC envelope works. Serialization latency is
acceptable. The store can be restarted independently.

**Risk:** Serialization overhead on the hot path. Mitigate by benchmarking
`_pack` + serialize vs in-process `_pack`. Expected overhead: ~10-50ms per
`GetBatch` call (dominated by tensor serialization for B=256 rows at 47K
tokens each). Acceptable given the trainer step is ~60s.

**Launcher change:**
```bash
# Add to s3_fullasync_docker.sh, before the trainer starts:
python3 trajectory_store_server.py \
  --port 50051 \
  --max-size $BUFFER_SIZE \
  --staleness-cutoff-k $STALENESS_CUTOFF_K \
  --pad-token-id 151643 \
  --prompt-length-cap 31232 \
  --response-length-cap 47616 &
STORE_PID=$!
```

### Cut 2: Extract the producer to a separate process (same machine)

**Scope:** The producer becomes a standalone process that reads from the
dataloader, calls ProRL, and pushes to the store service. It no longer lives
inside the trainer's Docker container.

**Changes:**
- New file: `producer_service.py` — a long-running process that:
  - Loads the train dataset.
  - Instantiates `AsyncLLMServerManagerDAPO` (or a simplified version that
    only does dispatch, not trainer-side concerns).
  - Runs `generate_sequences_dapo()` in a loop.
  - Pushes survivors to the store via gRPC.
  - Polls coordination (or a simple version-file) for `policy_version`.
- Trainer no longer instantiates `ContinuousRolloutProducer`,
  `AsyncLLMServerManagerDAPO`, or the `_push_fn` closure. It only calls
  `store_client.get_batch(n_groups, current_step)`.
- `s3_fullasync_docker.sh` is split into `s3_trainer.sh` (trainer only) and
  `s4_producer.sh` (producer only).

**What this proves:** Producer and trainer are fully decoupled. Either can
restart independently. The producer can be scaled by running multiple
instances (cut 4).

**Dependencies removed from trainer:**
- `openhands` package (no more ProRL client in trainer).
- `aiohttp`, `fastapi`, `uvicorn` (these were only needed for the rollout
  manager).
- `async_generator` (only used by rollout manager).

This is a significant dependency cleanup for the trainer image.

### Cut 3: Coordination service + adapter storage

**Scope:** Introduce the coordination service. Trainer publishes to
coordination; coordination fans out to pool + producers + store.

**Changes:**
- New file: `coordination_service.py` — gRPC server implementing
  `PublishPolicyVersion`, `GetLatestVersion`, `SubscribeVersionUpdates`.
- `_publish_lora_adapter` in `ray_trainer.py`: instead of POSTing to each
  pool child, call `coordination.PublishPolicyVersion(v, adapter_uri)`.
  Coordination handles the pool fanout.
- Adapter tarballs stored at a durable URI (S3 bucket or NFS).
- Producer subscribes to version updates via gRPC stream.
- Store receives version notifications (for metrics, not for eviction — the
  store already uses `created_at_step`, not `policy_version`, for eviction).

### Cut 4: Multi-producer

**Scope:** Run N producer instances, each with a data shard.

**Changes:**
- `producer_service.py` accepts `--shard-id` and `--num-shards`. Dataloader
  uses a `DistributedSampler`-style offset.
- Store handles concurrent `PushGroup` from multiple producers (already
  thread-safe; gRPC server is inherently concurrent).
- Coordination broadcasts to all connected producers.

### Cut 5: Multi-trainer (federated)

**Scope:** Multiple trainers draw from the same store. Each trains
independently. An aggregation service merges adapter deltas.

**Changes:**
- `GetBatch` supports multiple concurrent callers (pop-on-sample ensures
  disjoint draws — existing semantics).
- Each trainer publishes to its own version namespace in coordination.
- Aggregation service: periodic merge of adapter deltas → combined adapter
  → coordination publishes to pool.

**This is the "futuristic decentralised training" vision.** It requires
adapter arithmetic (averaging LoRA deltas), which is well-studied but not
yet wired in this codebase.

### Cut 6: Store persistence + WAL

**Scope:** Add a write-ahead log to the store so it survives restarts without
losing buffer contents. On restart, replay the WAL to reconstruct the buffer.

**Changes:**
- `PushGroup` appends a WAL entry before ACKing.
- On startup, replay WAL entries newer than `max_age` seconds.
- WAL is a simple append-only file (or a Redis stream, or a Kafka topic —
  see section 13).

**Why this is late in the cut sequence:** The buffer is ephemeral today and
the system works. WAL adds durability but also complexity (compaction, replay
correctness, disk I/O). Only worth it when the buffer holds high-value
trajectories (e.g., expensive multi-turn SWE-Bench runs that take 45 min
each).

---

## 13. Technology selection

### Hot path: Trainer ↔ Store

**Recommendation: gRPC with protobuf.**

| Option | Latency | Schema evolution | Multi-language | Durability | Complexity |
|---|---|---|---|---|---|
| gRPC + protobuf | ~5-20ms (localhost), ~50-100ms (cross-machine) | Excellent (field numbers) | Excellent | None (add WAL separately) | Medium |
| HTTP + JSON | ~20-50ms | Poor (stringly typed) | Excellent | None | Low |
| HTTP + MessagePack | ~15-30ms | Fair | Good | None | Low |
| Redis Streams | ~1-5ms | None (opaque bytes) | Good | Redis AOF | Medium |
| Kafka | ~10-50ms | Excellent (schema registry) | Excellent | Excellent | High |

**Why gRPC:**
- The `GetBatch` response is large (~100 MB for 256 rows at max token length).
  gRPC handles large messages well with streaming and flow control.
- Schema evolution via protobuf field numbers: adding a new field to
  `TrajectoryRecord` is backward-compatible (old producers ignore it, old
  stores skip it).
- Python gRPC is well-supported (`grpcio`, `grpcio-tools`).
- Streaming RPCs for `SubscribeVersionUpdates`.
- Bidirectional streaming could later serve as the backpressure channel.

**Why not Kafka:** Overkill. One producer, one trainer, one store. Kafka's
partition/consumer-group model adds operational complexity (ZooKeeper/KRaft,
topic management, offset tracking) with no benefit at this scale. If we later
need durability, a WAL file or Redis stream is simpler.

**Why not Redis Streams:** Good fit for the push/pop semantic, but: (a) the
`GetBatch` response needs tensor packing (re-padding), which must happen
server-side; Redis cannot run custom packing logic. (b) Staleness eviction
needs the `current_step` from the trainer; Redis has no built-in
age-by-external-clock eviction.

### Coordination: HTTP + JSON

The coordination service is low-frequency (one publish per `save_freq` steps,
~1 per minute). gRPC is fine but HTTP + JSON is simpler. A single
`/publish` POST and a `/version` GET (polled by producers every ~1s) suffices
for cut 3. Upgrade to gRPC streaming when multi-producer latency matters.

### Adapter storage: S3 or NFS

Adapter tarballs are ~80 MiB (rank-32 LoRA for Qwen3-4B). S3 is the natural
choice for multi-region; NFS for single-region simplicity. The coordination
service stores `{version: uri}` and the pool children download from the URI.

---

## 14. Open questions, risks, and non-goals

1. **Latency tolerance on `GetBatch`.** The trainer's step time is ~60s (FSDP
   forward + backward + optimizer). Can it tolerate ~100ms gRPC latency per
   `GetBatch`? Almost certainly yes — it is 0.17% of the step. But benchmark
   the tensor serialization for the max payload (256 groups × 8 rows × 47K
   tokens) to confirm it does not blow up. **User should answer: is the 60s
   step time stable, or does it vary with batch composition?**

2. **Tensor serialization format.** Protobuf is not zero-copy for large byte
   fields. For the `GetBatch` response (~100 MB), consider:
   - gRPC server-side streaming (chunk the response into N messages of ~10 MB
     each).
   - Shared-memory transport for same-machine (cut 1). Use `multiprocessing
     .shared_memory` to pass tensor buffers without copying.
   - Arrow IPC or Plasma for cross-machine. Overkill for now.
   **User should decide: same-machine or cross-machine for the store in the
   first deployment?**

3. **`filter_groups` at the store vs producer boundary.** Today, zero-variance
   filtering runs producer-side (`async_server_dapo.py:766-784`). An
   alternative: push all groups and let the store filter. Pro: simpler
   producer. Con: wasted network bandwidth. **Recommendation: keep it
   producer-side (current behavior). The filter is cheap (one boolean check
   per group) and saves ~30% of push bandwidth (typical zero-variance rate
   on SkyRL-v0).**

4. **Store process placement.** Same machine as trainer (cut 1) or same
   machine as producer (latency on push is lower) or dedicated machine?
   **Recommendation: same machine as trainer.** The hot path is `GetBatch`
   (trainer → store), which is called once per step (~60s). The cold path is
   `PushGroup` (producer → store), called once per eager-push (~every few
   seconds). Minimizing `GetBatch` latency matters more. Cross-machine
   push adds ~1ms latency, which is negligible for the push path.

5. **gRPC max message size.** Default is 4 MB. A single `GetBatchResponse`
   with 256 rows at max length is ~100 MB. Must set
   `grpc.max_send_message_length` and `grpc.max_receive_message_length` to
   ~200 MB on both client and server. Or use server-side streaming to chunk
   the response. **User should decide: chunked streaming or large single
   message?** Recommendation: single large message with raised limit for
   simplicity; switch to streaming if memory spikes are observed.

6. **Store high-availability.** The store is an SPOF. Options: (a) accept it
   (current system has no HA for the replay buffer either), (b) replicate to
   a hot standby, (c) make the store stateless with external storage (Redis,
   PostgreSQL). **Recommendation: (a) for cuts 1-4. The buffer is ephemeral;
   a restart loses ~5 minutes of rollouts. HA is a cut 6+ concern.**

7. **Authentication and authorization.** The gRPC services are currently
   assumed to run in a trusted network (same VPC, security group). For
   externally-contributed trajectories (section 10.3), add mTLS and
   per-producer API keys. **Non-goal for cuts 1-3.**

8. **Monitoring migration.** The current `trajectory_store.metrics()` feeds
   WandB via the trainer's `logger.log()`. With the store as a service,
   metrics must be either: (a) returned in `GetBatchResponse` (current
   approach — `pre_sample_metrics`, `post_sample_metrics`), or (b) scraped
   from a Prometheus `/metrics` endpoint on the store service. **Cut 1 uses
   (a). Cut 3+ adds (b).**

9. **Dataloader state across producer restarts.** Today `StatefulDataLoader`
   state is checkpointed by the trainer (`ray_trainer.py:1301-1307`). With
   the producer owning the dataloader, the producer must checkpoint its own
   dataloader state. On restart, it loads the last checkpoint to avoid
   repeating prompts. **Cut 2 concern. Use the same `state_dict()` /
   `load_state_dict()` API the trainer uses today.**

10. **Validation requires pausing the producer.** With the producer as a
    separate service, the trainer cannot just call `producer.stop()`. It
    must send an RPC: `PauseProduction()` / `ResumeProduction()`. The
    producer drains its current dispatch batch and ACKs. Simpler: run
    validation on a dedicated ProRL session (separate port) so the producer
    never needs to pause. **User should decide: shared session (requires
    pause protocol) or dedicated val session (no coordination needed)?
    Recommendation: dedicated val session for simplicity, especially since
    the 23-instance val set is small.**

---

## 15. Summary of decisions

| Decision | Choice | Rationale |
|---|---|---|
| Hot-path transport | gRPC + protobuf | Schema evolution, streaming, large messages |
| Coordination transport | HTTP + JSON (cut 1-2), gRPC (cut 3+) | Low frequency, simplicity first |
| Backpressure model | Push with advisory hint + FIFO evict | Multi-producer compatible, producer never blocks |
| Weight publish topology | Trainer → pool direct (cut 1-2), coordination fanout (cut 3+) | Incremental migration |
| Store placement | Same machine as trainer | Minimize `GetBatch` latency |
| `filter_groups` placement | Producer-side (unchanged) | Saves bandwidth, already works |
| Validation | Trainer-driven via ProRL (cut 1-2), producer-driven RPC (cut 3+) | Minimal change first |
| Store persistence | None (cut 1-5), WAL (cut 6) | Buffer is ephemeral; HA is late-stage |
| Adapter storage | Local filesystem (cut 1-2), S3/NFS (cut 3+) | Cross-machine access needed at cut 3 |

---

## Appendix A: gRPC service definitions (complete)

```protobuf
syntax = "proto3";

package prorl.replay;

// ─── Trajectory Store Service ───────────────────────────────

service TrajectoryStoreService {
  // Producer pushes one group (n sibling trajectories from one prompt).
  rpc PushGroup(PushGroupRequest) returns (PushGroupResponse);

  // Trainer draws n_groups from the buffer (pop-on-sample).
  // Blocks server-side if insufficient groups, up to timeout_ms.
  rpc GetBatch(GetBatchRequest) returns (GetBatchResponse);

  // Introspection.
  rpc GetMetrics(GetMetricsRequest) returns (GetMetricsResponse);

  // Coordination pushes version updates so the store can log
  // per-group version-age metrics.
  rpc NotifyPolicyVersion(PolicyVersionUpdate) returns (Empty);
}

message Empty {}

message TrajectoryRecord {
  bytes prompt_ids = 1;
  bytes response_ids = 2;
  bytes response_loss_mask = 3;
  bytes response_log_probs = 4;
  float reward = 5;
  float advantage = 6;
  int32 behavior_policy_version = 7;
  int32 created_at_step = 8;
  string prompt_uid = 9;
  string group_uid = 10;
  bool resolved = 11;
  bool success = 12;
  bool finish = 13;
  bool is_padded = 14;
  string error = 15;
  bytes instance_json = 16;
  bytes prompt_extras_json = 17;
}

message PushGroupRequest {
  repeated TrajectoryRecord records = 1;
  string group_uid = 2;
  int32 producer_id = 3;
}

message PushGroupResponse {
  bool accepted = 1;
  int32 store_size = 2;
  int32 backpressure_ms = 3;
}

message GetBatchRequest {
  int32 n_groups = 1;
  int32 current_step = 2;
  int32 staleness_cutoff_k = 3;
  int32 timeout_ms = 4;
  int32 no_progress_timeout_ms = 5;
}

message TensorPayload {
  repeated int64 shape = 1;
  string dtype = 2;
  bytes data = 3;
}

message NonTensorColumn {
  repeated bytes values = 1;
}

message StoreMetrics {
  float store_size = 1;
  float store_fill_ratio = 2;
  float store_num_trajectories = 3;
  float store_age_p50 = 4;
  float store_age_p95 = 5;
  float dropped_by_staleness_total = 6;
  float sample_age_steps_p50 = 7;
  float sample_age_steps_p95 = 8;
}

message GetBatchResponse {
  map<string, TensorPayload> tensors = 1;
  map<string, NonTensorColumn> non_tensors = 2;
  repeated int32 behavior_policy_versions = 3;
  repeated int32 created_at_steps = 4;
  repeated int32 sample_ages = 5;
  StoreMetrics pre_sample_metrics = 6;
  StoreMetrics post_sample_metrics = 7;
  bool success = 8;
  string error_message = 9;
}

message GetMetricsRequest {
  int32 current_step = 1;
}

message GetMetricsResponse {
  StoreMetrics metrics = 1;
  int32 total_pushes = 2;
}

message PolicyVersionUpdate {
  int32 policy_version = 1;
  string adapter_uri = 2;
}

// ─── Coordination Service ───────────────────────────────────

service CoordinationService {
  rpc PublishPolicyVersion(PublishRequest) returns (PublishResponse);
  rpc GetLatestVersion(Empty) returns (VersionInfo);
  rpc SubscribeVersionUpdates(Empty) returns (stream VersionInfo);
}

message PublishRequest {
  int32 policy_version = 1;
  string adapter_uri = 2;
  int32 trainer_step = 3;
  bytes adapter_tarball = 4;  // inline for small adapters; URI for large
}

message PublishResponse {
  bool success = 1;
  int32 endpoints_ok = 2;
  int32 endpoints_failed = 3;
  float publish_latency_s = 4;
}

message VersionInfo {
  int32 policy_version = 1;
  string adapter_uri = 2;
  int64 timestamp_ms = 3;
}
```

---

## Appendix B: Sequence diagrams

### B.1 Normal training step (cut 2+)

```
Trainer                Store Service           Producer             ProRL/Pool
   │                        │                      │                    │
   │  GetBatch(32, step=5)  │                      │                    │
   │───────────────────────>│                      │                    │
   │                        │ (has >=32 fresh)     │                    │
   │                        │ evict stale          │                    │
   │                        │ sample 32 groups     │                    │
   │                        │ pack tensors         │                    │
   │  GetBatchResponse      │                      │                    │
   │<───────────────────────│                      │                    │
   │                        │                      │                    │
   │ compute_reward         │                      │                    │
   │ compute_old_log_prob   │                      │                    │
   │ compute_advantage      │                      │                    │
   │ update_actor           │                      │                    │
   │ save_checkpoint        │                      │                    │
   │                        │                      │                    │
   │ PublishPolicyVersion   │                      │                    │
   │────────────────────────┼──────────────────────┼───────────────────>│
   │                        │                      │  /reload_lora      │
   │                        │                      │                    │
   │                        │  PushGroup(grp_42)   │                    │
   │                        │<─────────────────────│ (runs in parallel) │
   │                        │  accepted, bp=0      │                    │
   │                        │─────────────────────>│                    │
   │                        │                      │                    │
   │                        │  PushGroup(grp_43)   │                    │
   │                        │<─────────────────────│                    │
```

### B.2 Producer crash and recovery

```
Trainer                Store Service           Producer
   │                        │                      │
   │  GetBatch(32, step=10) │                      │
   │───────────────────────>│                      │
   │                        │ (has 28 fresh)       │  ✗ CRASH
   │                        │ blocks...            │
   │                        │ no_progress timeout  │
   │  error: no_progress    │                      │
   │<───────────────────────│                      │
   │                        │                      │
   │ save_checkpoint        │                      │
   │ wait + backoff         │                      │
   │                        │                      │  ✓ RESTART
   │                        │  PushGroup(grp_50)   │
   │                        │<─────────────────────│
   │                        │                      │
   │  GetBatch(32, step=10) │                      │
   │───────────────────────>│                      │
   │                        │ (filling...)         │
```

---

## Appendix C: Configuration mapping

Current env vars (`s3_fullasync_docker.sh`) → new service config:

| Current env var | Current consumer | New consumer |
|---|---|---|
| `BUFFER_SIZE=256` | `TrajectoryStore(max_size=)` | Store Service `--max-size` |
| `STALENESS_CUTOFF_K=4` | `TrajectoryStore(staleness_cutoff_k=)` | Store Service `--staleness-cutoff-k` |
| `USE_TEMPORAL_IS=True` | `core_algos.py` gated | Trainer (unchanged) |
| `CONTINUOUS_PRODUCER=True` | `_continuous_producer_mode()` | Implicit (producer is always a service) |
| `FILTER_GROUPS=True` | `main_ppo.py` class selector | Producer Service config |
| `BATCH_SIZE=32` | Trainer `train_batch_size` | Trainer `GetBatchRequest.n_groups` |
| `GEN_BATCH_SIZE=128` | Producer `gen_batch_size` | Producer Service `--gen-batch-size` |
| `SAVE_FREQ=1` | Trainer checkpoint + publish | Trainer (unchanged) |
| `SWAP_PROTOCOL=pinning` | Pool children | Pool (unchanged) |
| `REMOTE_DNS` | Pool addresses | Producer Service + Coordination Service config |
| `OPENHANDS_NUM_WORKERS=32` | ProRL worker count | ProRL (unchanged) |
```

