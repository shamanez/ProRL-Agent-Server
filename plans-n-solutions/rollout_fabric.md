# Rollout Fabric Architecture

A contract-first design for evolving the current ProRL + OpenHands + vLLM +
VERL full-async stack into a decentralized agentic-RL fabric where
environments, inference backends, rollout workers, live stores, replay
archives, trainers, and policy coordination are independently replaceable
adapter slots.

**Status:** Design-only. No code changes implied by this document.
**Author:** Architecture review, 2026-04-30.

**Reading order:**
1. §1 Vision — the one-paragraph thesis.
2. §2 Current state — what runs in this repo right now.
3. §3 Architectural invariants — what cannot change across the migration.
4. §4 Pluggability principles — the four design rules.
5. §5 The slot model — seven adapter slots with current and future
   implementations named.
6. §6 Wire schemas — the two data contracts (episode and training sample).
7. §7 Live store vs durable replay archive — why they are different
   systems from day one.
8. §8 Component placement — where each slot naturally runs.
9. §9 Stage-wise migration plan — S0 (today) through S8 (federation).
10. §10 What does NOT change.
11. §11 Non-goals.
12. §12 Open questions for the planning agent.
13. Appendices — slot interfaces, code-to-slot mapping, external-framework
    mapping.

**What this document is for.** It is the input to a downstream planning
agent. The planner's job is to take the contracts and stages here and
produce an implementation plan: pick transports, pick storage, sequence
service extraction, write migration scripts, and define the test matrix. The
planner is explicitly authorized to choose between options when this
document leaves them open. The planner is explicitly *not* authorized to
soften or skip any of the invariants in §3 or the principles in §4.

---

## 1. Vision

The current ProRL + OpenHands + VERL stack runs end-to-end agentic RL with a
fully-async producer-consumer loop. It works in the form: one Python process,
one Docker container, one trainer machine, one EC2 vLLM pool. That is the
right shape for proving the algorithm. It is the wrong shape for everything
that comes next.

The vision is to turn this stack into a **rollout fabric** — a small set of
stable contracts between independently replaceable services. Many environment
providers generate verifiable, tool-rich episodes. Many rollout workers
execute those episodes against versioned policies. One or more trainers
consume a clean training-sample stream. Policy versions are published back to
inference backends without binding the system to any one trainer, environment
runtime, or serving engine. Trajectories live in a durable archive that
outlives any single training run and is queryable for offline RL,
distillation, audit, and curation.

The first implementation should be built with the current stack because that
is what runs today, but every component must sit behind a small replaceable
contract. ROCK, GEM, ORS/OpenReward, ROLL, slime, vLLM, SGLang, and future
systems must be pluggable as adapter implementations with bounded local
changes. **If adding a new environment or a new trainer requires rewriting
the rollout-store-trainer core, this architecture has failed.**

This document is contract-first. Transport choices (gRPC vs HTTP vs Ray vs
shared memory), storage choices (Parquet vs Iceberg vs Postgres), exact
service extraction order, and deployment topology are deferred to a later
implementation plan. What this document fixes is: the slot map, the wire
schemas, the invariants the migration must preserve, and the stage-by-stage
goals that prove each slot is real.

---

## 2. Current state — what runs today

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

**What works about this shape.** Zero serialization overhead. Push and sample
are ~1 ms under a single `threading.Lock`. No network latency between
producer and store. The pinning swap protocol means in-flight rollouts never
see an adapter swap mid-trajectory. The `endpoints_failed > 0` abort gate
makes the trainer fail loud rather than silently train on a broken pool.
DAPO's eager-push seam keeps `behavior_policy_version` and
`created_at_step` per row, so temporal IS correction can be exact.

**What does not scale.**
- The trainer process owns the dataloader, the producer thread, the live
  store, the actor, and the LoRA publish path. It is the implicit
  orchestrator of everything.
- The producer cannot be fanned out across machines or across heterogeneous
  environment providers. There is one producer thread.
- A producer crash kills the trainer (same process, exception propagates).
- The buffer is ephemeral and not checkpointable.
- There is no record of the trajectories the buffer threw away — they are
  lost forever, which precludes offline replay, distillation, and audit.
- Adding a second trainer (federated, A/B, distillation) requires rebuilding
  the producer-store loop a second time.
- The trainer image carries `openhands`, `aiohttp`, `fastapi`, and
  `uvicorn` only because the rollout manager runs in-process. None of these
  are training-time concerns.

These limitations are the rollout fabric's reason to exist.

---

## 3. Architectural invariants

These are the load-bearing rules that any future stage must preserve. The
planner is not authorized to soften any of them. Each invariant cites the
file or contract in the current code where it lives so the migration can
verify preservation per stage.

### 3.1 Token-in / token-out across turns

`openhands/llm/nvidia/qwen3.py` and `openhands/llm/nvidia/qwen2_5_vl.py`
communicate with vLLM in **token IDs**, not text. Multi-turn RL stability
requires that the exact token IDs from each turn are reused on the next
turn. Re-tokenizing decoded text shifts boundaries; actor and reference
diverge; KL/entropy go NaN; PPO/GRPO collapses. The replay store stores
token IDs, not strings. This invariant survives the migration: every wire
schema between services carries token IDs, never text.

### 3.2 Group integrity for GRPO/DAPO

`compute_advantage` requires the `n` siblings of a group to be present at
sample time so that group-relative advantages can be computed without a
running normalizer. The store never splits a group; sampling pops whole
groups. Filter decisions (zero-variance drop in
`async_server_dapo.py:766-784`) operate on whole groups. This invariant
holds in every successor live store and every trainer adapter.

### 3.3 `endpoints_failed > 0` is a hard abort

The current `_publish_lora_adapter`
(`ray_trainer.py:1413-1544`) raises if any pool child fails to ACK
`/reload_lora`. A warm replay buffer must not mask a broken pool. Whatever
service eventually owns policy publication (today: trainer; cuts S4+:
coordination) preserves this gate — partial publish is failure, not
degraded mode.

### 3.4 Pinning swap protocol on the inference backend

`_vllm_child.py:60-86` pins each rollout to the policy version active at
dispatch time via `/v{N}/generate`. `/reload_lora` installs new adapters
and never removes adapters with in-flight requests. LRU eviction handles
GPU slot management. This is a correctness invariant for async RL — without
it, a trajectory can begin on policy v and end on policy v+1, making
`behavior_policy_version` per-row meaningless. Any new inference backend
adapter (SGLang, TGI, hosted) must implement pinning or an equivalent
guarantee that a single rollout sees one policy.

### 3.5 `behavior_policy_version` stamped per row, GIL-atomic write

The producer thread reads `policy_version` and stamps it on each emitted
trajectory; the trainer thread writes it inside `_publish_lora_adapter`
after pool ACK. The cross-thread read is GIL-atomic on a single int. The
benign race (a publish lands mid-batch; the batch finishes tagged with the
old version) is exactly the input the temporal IS correction is designed
for. Any future coordination protocol must keep `behavior_policy_version`
correct per row, not per batch — this is a row-level invariant.

### 3.6 Pop-on-sample (queue semantics) on the live store

`sample_mini_batch` pops chosen groups before returning. This is queue
semantics, not sample-with-replacement. Two independent trainers drawing
concurrently get disjoint groups. Future live store implementations
preserve this — pop-on-sample is the property that makes multi-trainer
work (S5+) without coordination beyond a single lock or its equivalent.

### 3.7 Eager-push seam under filter_groups=True

The DAPO manager's `generate_sequences_dapo` is the only path that pushes
each survivor group into the store the moment it clears
`filter_easy_hard_instance`. The continuous producer skips its terminal
`push_from_dataproto` via `meta_info['eager_pushed_all']`. Calling
`store.push_from_dataproto(out_batch)` unconditionally double-pushes and
corrupts `behavior_policy_version` / `created_at_step` tracking under
pop-on-sample. The eager-push seam moves into the rollout-worker slot and
is preserved across the migration; the planner is not authorized to revert
to terminal-push under `filter_groups=True`.

### 3.8 Data ownership boundary — the trainer never holds the dataset

This is the invariant that prevents the trainer from sneaking back into
being the orchestrator after the migration. State this precisely:

> Training and validation **task datasets** belong to the environment
> provider or the rollout scheduler. The trainer **does not sample raw
> tasks directly.** The trainer samples `TrainingGroup` records from the
> live store. Validation is expressed as **evaluation episodes against
> named task splits**, dispatched by the rollout worker against the
> environment provider — not as trainer-local token batches drawn from a
> trainer-local parquet file.

Consequences and tests of this invariant:

- The trainer image after migration **does not import a dataloader.** The
  current `data.train_files` and `data.val_files` Hydra fields become
  rollout-worker config, not trainer config.
- The trainer **does not know task IDs.** A `TrainingGroup` references the
  task by `task_id` and `environment_id`/`environment_version` (provenance
  fields), but the trainer never resolves a `task_id` back to a raw task
  description.
- Validation runs are RPCs from the trainer to the rollout worker:
  *"please run an evaluation pass against env=X, split=val, return the
  resulting `TrainingGroup` records."* The trainer scores; the rollout
  worker dispatches. Where the FSDP actor must compute `old_log_prob` /
  `ref_log_prob` on validation tokens, it does so on records returned by
  the rollout worker, not on records loaded from local parquet.
- A second trainer (S6 onward) plugs in **without bringing its own dataset
  loader.** It subscribes to the same live store. The dataset stays where
  it belongs: at the environment provider.
- A new environment provider (S5: ROCK, GEM, ORS) brings its own dataset
  / task registry / split definitions. The trainer never knows.

If this invariant is violated, the architecture has failed: the trainer
becomes the hidden orchestrator, and pluggability collapses to
"swap one slot if you also rewrite the trainer's data path." This is the
single most load-bearing invariant in the document.

---

## 4. Pluggability principles

Four design rules. Every other section is consistent with these; if a
later section appears to violate one, the later section is wrong.

### 4.1 Contract-first

Define wire schemas and method signatures **before** picking transports or
extracting services. ProRL, vLLM, VERL, and the in-process store are
implementation details. The architecture boundaries are: the environment
contract, the inference contract, the episode and training-sample
contracts, the live-store and archive contracts, the trainer-adapter
contract, and the policy-registry contract. A new ROCK environment, a new
SGLang backend, or a new ROLL/slime trainer is an adapter behind a slot —
not a redesign of the system.

### 4.2 Two-tier data contract

The system has two distinct data records, with different lifetimes and
different consumers (§6 specifies them in full):

```
EpisodeRecord  →  TrainingTrajectory  →  TrainingGroup  →  TrainerAdapter batch
   durable          intermediate           live-hot         per-trainer-shape
   archive                                  store
```

`EpisodeRecord` is environment-agnostic, durable, audit-friendly, and
preserved in the replay archive. `TrainingSample` / `TrainingGroup` is
compact, derived from episodes, and shaped to satisfy ROLL, slime, VERL,
and SFT/distillation trainers without per-trainer wire variants.

Starting the contract at `TrainingGroup` is sufficient for the trainer
hot path but insufficient as a system-wide contract: it precludes
external contributions, offline RL, distillation, and audit. The
two-tier contract is non-negotiable.

### 4.3 Live store and durable replay are different systems

The current draft treats persistence as a "later WAL feature." This
document treats them as **separate products from the start**, even if the
first archive implementation is a simple append-only log of
`EpisodeRecord`s. §7 makes the property comparison explicit. The summary:
the live store is bounded RAM, FIFO, pop-on-sample, ephemeral, optimized
for trainer `GetBatch` latency. The archive is unbounded durable storage,
append-only, queryable, optimized for offline use, weeks-to-months horizon.
WAL is a live-store recovery mechanism. The archive is a separate product
surface.

### 4.4 Trainer adapters compute algorithm-specific fields

Wire schemas carry **rewards**, not advantages. `behavior_log_probs` are
default fields (provenance even when not consumed). `ref_log_probs`,
`advantages`, `returns`, `KL penalties`, `IS ratios`, and `value targets`
are computed by the trainer adapter. Without this rule, ROLL's pre-computed
advantages and slime's trainer-side computation become incompatible at the
wire layer; with it, both are adapters on the same `TrainingGroup` shape.

### 4.5 (Restated as a principle) — Data ownership stays with the rollout side

This is invariant 3.8 stated as a design rule: the trainer never holds
task datasets. Listed here so the planner sees it next to the other
principles.

---

## 5. The slot model

Seven adapter slots. For each: the minimum interface (vision-level, no
transport), the current concrete adapter that ships in this repo, the named
future adapters that should be plug-compatible, and the state that lives
inside vs outside the slot. Per-slot interface signatures are in
Appendix A; per-slot file mappings are in Appendix B.

The summary table:

| # | Slot | Today's adapter | Future adapters | Boundary |
|---|---|---|---|---|
| 5.1 | EnvironmentProvider | ProRL FastAPI :8006 with OpenHands inside | ROCK, GEM, ORS/OpenReward, Gymnasium, browser/code-exec sandboxes | `list_tasks / create_episode / get_prompt / act / close` over typed tool calls |
| 5.2 | InferenceBackend | Remote vLLM child pool :8100-8103 | SGLang, TGI, TRT-LLM, hosted APIs (where logprobs available) | `Generate(policy_ref, tokenized_prompt, sampling) → token_ids + logprobs + metadata` |
| 5.3 | RolloutWorker | `ContinuousRolloutProducer` + `AsyncLLMServerManagerDAPO` | Multi-machine producer fleet, mixed-environment producers, partner producers | Reads tasks from EnvProvider; dispatches via InferenceBackend; emits `EpisodeRecord` and `TrainingGroup`; subscribes to PolicyRegistry |
| 5.4 | LiveStore | In-process `TrajectoryStore` (deque + lock) | Colocated gRPC/Ray/shared-memory hot store | `push_group / get_batch / get_metrics / notify_policy_version` with bounded FIFO + staleness + pop-on-sample |
| 5.5 | ReplayArchive | Does not exist today | Parquet/Iceberg on S3, Postgres index, object store + metadata catalog | Append-only `EpisodeRecord` log; queryable `TrainingSample` derivation by env, split, policy, reward, time |
| 5.6 | TrainerAdapter | VERL `RayPPOTrainerDAPO` | ROLL, slime/Megatron, DeepSpeed/FSDP, single-GPU PEFT, SFT/distillation pipelines | Consumes `TrainingGroup`; computes alg-specific fields locally; publishes policy versions |
| 5.7 | PolicyRegistry / Coordination | Trainer-owned `policy_version` int + direct `/reload_lora` fanout | Dedicated registry with adapter URIs, pub/sub for version updates, optional aggregation service | `publish_version / get_latest_version / subscribe_version_updates` |

Each slot below: minimum interface, current adapter, future adapters,
state ownership.

### 5.1 EnvironmentProvider

**Role.** Owns task registry, task splits, episode lifecycle, ground
truth, reward computation, sandbox/runtime isolation, and termination
logic. The agent-side runtime (tool execution, browser harness, code
sandbox, file system) is internal to this slot.

**Minimum interface.** Five operations, validated by the convergence of
ROCK, GEM, and ORS/OpenReward:
- `list_tasks(split: str) → list[task_id]`
- `create_episode(task_id: str, **opts) → episode_handle`
- `get_prompt(episode_handle) → list[ContentBlock]` (text and/or image)
- `act(episode_handle, ToolCall(name, input_dict)) → StepResult{
   observation: list[ContentBlock], reward: float, done: bool, info: dict}`
- `close(episode_handle)`

Actions are typed tool calls (function name + input dict), not raw
strings. Observations are `list[ContentBlock]` (text or image blocks) to
handle both text-only and multimodal environments. This interface is the
conceptual common denominator; transport (Python SDK, HTTP+SSE, gRPC) is
a planner choice.

**Current adapter.** ProRL FastAPI on `:8006` with OpenHands as the
internal agent runtime, Singularity as the sandbox runtime, registry +
`AgentHandler` selecting `swe_agent` / `math_coder` / `stem_agent` /
`gui_agent`. The three-stage pipeline (init → run → eval) is internal to
this adapter. The token-in/token-out invariant is owned by this adapter
(via `openhands/llm/nvidia/qwen3.py`).

**Future adapters.**
- **ROCK** (Alibaba): sandbox provisioning + GEM-compatible `make/reset/step`.
  Adds resource scheduling and pooled isolation.
- **GEM** (axon-rl): the standard agentic-LLM Gymnasium contract. Async
  vectorized rollout, observation/tool wrappers, turn-level rewards.
- **ORS / OpenReward**: HTTP+SSE protocol. Sessions, tool calls, streamed
  reward updates. Strong fit for partner-contributed environments.
- **Browser/code-exec adapters**: anything implementing the five-method
  interface above.

**State ownership.** The provider owns task state, ground truth, reward
function, sandbox processes, and the verifier. The provider does **not**
own the agent's policy (that lives at the InferenceBackend) or the
trajectory (which lives at the RolloutWorker / LiveStore / ReplayArchive).

### 5.2 InferenceBackend

**Role.** Produces tokens and per-token logprobs from a policy. Owns
model weights, LoRA cache, KV cache, and serving infrastructure.

**Minimum interface.**
- `generate(policy_ref, tokenized_prompt, sampling_params) → {
   token_ids, logprobs, finish_reason, metadata}`
- `reload_policy(policy_id, policy_version, adapter_uri_or_blob) →
   {ok, error}`
- `health() → status`

`policy_ref` is opaque from the caller's side: it might be a path-versioned
URL like `/v{N}/generate`, an explicit `policy_id` argument, a session
binding, or any equivalent that produces the **pinning** guarantee that a
single generate call sees one policy.

**Current adapter.** vLLM child pool on `:8100-8103`, with the pinning
swap protocol (`_vllm_child.py`), `--max-loras 8`, refcount-based eviction
on `/reload_lora`, and `/v{N}/generate` for path-versioned pinning.

**Future adapters.**
- **SGLang** with sgl-router: same interface; verify it returns per-token
  logprobs. slime uses SGLang as its inference backend, so an SGLang
  adapter unlocks easier slime trainer plug-in (S6/S7).
- **TGI**: similar.
- **TRT-LLM**: similar; logprob support varies by build.
- **Hosted APIs** (Anthropic, OpenAI, etc.): only for eval/non-RL flows
  unless the API exposes per-token logprobs.

**Constraint.** Any future adapter must provide either pinning or an
equivalent guarantee that a single rollout sees one policy. Without this,
invariant 3.4 breaks and per-row `behavior_policy_version` becomes a lie.

**State ownership.** The backend owns model weights, KV cache, LoRA slots.
It does **not** own policy version semantics (that is the
PolicyRegistry's job). It is told *which* policy to load and *which* to
serve per request.

### 5.3 RolloutWorker

**Role.** Executes the agent loop against EnvironmentProvider and
InferenceBackend, produces episodes, derives training groups, applies
producer-side filters (zero-variance drop), tags provenance and policy
version, and pushes results to LiveStore and ReplayArchive. **Owns the
training/validation task dataset** (per invariant 3.8).

**Minimum interface (what it does, not what it exposes — a worker is
mostly a callee, not a callable):**
- `run(env_provider, inference_backend, policy_subscription, store_client,
   archive_client, dataset_or_task_source, gen_batch_size,
   filter_strategy)`
- Internally, it loops:
  - read next task batch from its own dataloader (or task source)
  - look up current policy version from PolicyRegistry
  - dispatch n parallel episodes through EnvironmentProvider, with each
    turn calling InferenceBackend
  - assemble `EpisodeRecord` per completed episode
  - derive `TrainingGroup` per group of n siblings (DAPO sense)
  - apply producer-side filters (zero-variance drop, length cuts)
  - eager-push surviving groups to LiveStore tagged with
    `behavior_policy_version` and `created_at_step`
  - tee `EpisodeRecord` to ReplayArchive (S3+)

**RPC surface exposed to the trainer.**
- `pause_production() → ack`
- `resume_production() → ack`
- `run_validation(env_id, split, options) → list[TrainingGroup]` (the
  validation flow per invariant 3.8 — trainer asks worker to run an eval
  pass and returns the resulting groups for trainer-side scoring)
- `get_dataloader_state() / load_dataloader_state()` (for the worker's
  own resume)

**Current adapter.** `ContinuousRolloutProducer` (daemon thread) +
`AsyncLLMServerManagerDAPO`. Today it lives in the trainer process and
shares `policy_version` cross-thread via GIL-atomic int. It eager-pushes
through a `_push_fn` closure rather than a network call.

**Future adapters.** Multi-machine producer fleet; per-environment
specialized workers (SWE-Bench worker on high-CPU box with sandbox,
math/code worker on cheaper box, GUI worker with browser); external
partner workers contributing to the same store.

**State ownership.** The worker owns the dataloader, the inflight episodes,
and the producer-side filters. It does **not** own the trainer step
counter, the actor weights, or any algorithm-specific computation. Per
invariant 3.8, the worker — not the trainer — owns the training and
validation **task datasets**.

### 5.4 LiveStore

**Role.** A bounded, low-latency, hot buffer between RolloutWorker and
TrainerAdapter. Operates in groups (n siblings together), pops on sample,
evicts by staleness, returns batches sized for the trainer step.

**Minimum interface.**
- `push_group(records: list[TrainingSample], group_uid, producer_id) →
   {accepted, store_size, backpressure_hint}`
- `get_batch(n_groups, current_step, staleness_cutoff_k, timeout_ms) →
   {tensors, non_tensors, behavior_policy_versions, created_at_steps,
    sample_ages, metrics_pre, metrics_post}`
- `get_metrics(current_step) → store_metrics`
- `notify_policy_version(version, adapter_uri) → ack`

`get_batch` blocks server-side up to `timeout_ms` if the store has fewer
than `n_groups` non-stale groups. The no-progress detector (invariant
analog from `continuous_producer.py:357-392`) lives inside the store: if
`total_pushes` does not increase for `no_progress_timeout_s`, the call
returns an error. The trainer is no longer responsible for the busy-loop.

**Current adapter.** In-process `TrajectoryStore`: `deque(maxlen=256)`,
single `threading.Lock`, K-staleness eviction, re-pad to sample-local max
inside `_pack`.

**Future adapters.** Colocated gRPC server (Stage 1), Ray actor store,
shared-memory store for same-machine deployments. The hot path is
`get_batch`, which is called once per trainer step; minimizing this
latency is the placement constraint (§8).

**Pop-on-sample is mandatory** (invariant 3.6). Push semantics are queue,
not pub/sub. Staleness eviction uses `created_at_step`, not policy version
(versions can advance for reasons other than fresh data).

**State ownership.** The store owns the bounded buffer, the lock, the
metrics counters, and the no-progress detector. It does **not** own
`policy_version` semantics (that is PolicyRegistry's job — the store
receives notifications for metrics tagging only).

### 5.5 ReplayArchive

**Role.** Append-only, durable, queryable record of every episode the
fabric ever produced. Source of truth for offline RL, distillation,
curation, audit, and reproducibility.

**Today: this slot does not exist.** The current system loses every
trajectory the live store evicts. This is fine for proving the algorithm,
not for everything else.

**Minimum interface.**
- `append_episodes(records: list[EpisodeRecord]) → {accepted, episode_uids}`
- `query(filter_spec) → iterable[EpisodeRecord | TrainingSample]`
  where `filter_spec` is e.g. environment_id, split, time_range,
  policy_id, reward predicate, trust_level, data_source.
- `derive_training_samples(episode_uids, schema_version) →
   list[TrainingSample]` (re-derivation from canonical records)

**Future adapters.**
- Parquet / Iceberg on S3 with a metadata index (Postgres or DynamoDB).
- Append-only object store + metadata catalog.
- Database-backed storage for smaller/early deployments.

**Why this is a separate product from the live store:**
- Different horizon (weeks/months vs minutes).
- Different durability guarantees (no loss vs loss-tolerant).
- Different access pattern (point-and-range queries vs bounded FIFO).
- Different consumer set (offline training, distillation, audit, curation
  vs only the live trainer).
- Different optimization targets (queryability and durability vs
  `get_batch` latency).

**Trust and provenance.** Records carry `trust_level` (own-fabric,
partner-validated, partner-untrusted) and full provenance fields
(§6.1). External-trajectory routing (§10.3 of the old draft, §6.3 here)
depends on these fields being first-class, not tacked on.

**State ownership.** Owns durable storage, indexes, and query plans. Does
**not** own live training (no `get_batch` latency target — that's the
LiveStore's job).

### 5.6 TrainerAdapter

**Role.** Consumes `TrainingGroup` records, computes algorithm-specific
fields locally (advantages, KL, IS ratios, value targets), runs the
optimizer step, and publishes new policy versions to the PolicyRegistry.

**Minimum interface (what the trainer adapter implements):**
- `request_batch(n_groups, step) → TrainingGroup` (calls LiveStore)
- `step(training_group) → {loss, metrics, gradient_norm, ...}`
- `save_checkpoint(step, dir) → checkpoint_uri`
- `publish_policy_version(step, adapter_uri, policy_id) → publish_result`
   (calls PolicyRegistry)
- `request_validation(env_id, split) → list[TrainingGroup]`
   (calls RolloutWorker per invariant 3.8)
- `score_validation(groups) → val_metrics`

**Current adapter.** VERL `RayPPOTrainerDAPO`. Currently owns the
dataloader; in the migration that ownership moves to RolloutWorker (S2),
leaving this adapter focused on optimizer math and policy publication.

**Future adapters.**
- **ROLL**: AsyncController + DeepSpeed/Megatron/FSDP2. Already tracks
  `behavior_policy_version` per sample, supports six off-policy IS
  variants. Likely path: a ROLL-side adapter that consumes `TrainingGroup`
  via the same `get_batch` protocol that VERL uses.
- **slime / Megatron + SGLang**: separates training, rollout, data buffer.
  Its data-buffer abstraction maps onto LiveStore; its rollout module
  maps onto RolloutWorker. Plugging slime as a TrainerAdapter requires
  bridging its Ray-object-ref data path to the LiveStore's `get_batch`.
- **DeepSpeed/FSDP single-trainer adapters**, single-GPU PEFT trainers,
  pure SFT trainers, distillation pipelines.

Per principle 4.4, adapters compute their own algorithm-specific fields.
Per invariant 3.8, no adapter owns a task dataset.

**State ownership.** The adapter owns optimizer state (FSDP shards,
Megatron tensor parallelism, ZeRO stages), actor weights, the LoRA delta
producer for publish, and per-step compute. It does **not** own task data,
task schedule, or rollout dispatch.

### 5.7 PolicyRegistry / Coordination

**Role.** Source of truth for the active policy version set. Receives
publishes from TrainerAdapter, fans out to InferenceBackend (`/reload_lora`),
LiveStore (for metrics tagging), and RolloutWorkers (for next-batch
version selection).

**Minimum interface.**
- `publish_policy_version(version, adapter_uri, policy_id, trainer_id) →
   {success, endpoints_ok, endpoints_failed, latency_s}`
- `get_latest_version(policy_id) → version_info`
- `subscribe_version_updates(policy_id) → stream version_info`
- `register_policy_namespace(policy_id, base_model_id, tokenizer_id,
   adapter_storage_root) → ack`

**Today.** Trainer-owned `policy_version` (a Python int) plus direct
`/reload_lora` fanout from `_publish_lora_adapter`. The producer thread
reads `policy_version` cross-thread (GIL-atomic). There is no
out-of-process registry.

**Cuts S1/S2.** The trainer keeps its direct pool publish; the registry is
a thin version-and-URI store the producers poll. Two sources of truth
exist intentionally for the migration window.

**Cuts S3+.** The registry takes over publish. Trainer calls
`publish_policy_version`. Registry stores `(version, adapter_uri)`,
fans out to all pool children with the abort gate (invariant 3.3),
broadcasts to subscribers. Single source of truth.

**Cuts S5+.** Multiple namespaces (per-trainer policies for federation),
adapter aggregation hooks, multi-region fanout.

**State ownership.** The registry owns the version → URI manifest and the
fanout state. It does **not** own adapter binaries (those live in adapter
storage — local FS, S3, or NFS depending on deployment) or the policy
math (that's the trainer adapter's job).

---

## 6. Wire schemas

Two contracts: the canonical `EpisodeRecord` and the derived
`TrainingSample` / `TrainingGroup`. Both are described as field
inventories with semantics and provenance. **Bytes-on-the-wire (protobuf
vs MessagePack vs Arrow vs JSON), tensor serialization, and chunking are
deferred to the planner.**

### 6.1 EpisodeRecord (canonical, durable, audit-friendly)

The unit written to the ReplayArchive. Captures everything needed to
reconstruct or audit an episode without further reference to the
environment provider's internal state. Field inventory derived from
ORS/OpenReward, ROCK/GEM, and the current `TrajectoryRecord` dataclass at
`trajectory_store.py:43-74`.

Fields:

| Field | Type | Notes |
|---|---|---|
| `episode_uid` | str | Stable opaque ID; UUID. |
| `task_id` | str | Provider-assigned; identifies the task within an env. |
| `split` | str | `train` / `val` / `test` / `partner` / `eval`. |
| `environment_provider` | str | e.g. `prorl`, `rock`, `ors-acme`. |
| `environment_id` | str | Provider-internal env name (e.g. `swe_agent`, `math_coder`). |
| `environment_version` | str | Semver or commit SHA of the env definition. |
| `verifier_version` | str | Version of the reward/eval logic (separate from env). |
| `reward_spec_id` | str | Logical ID of the reward function/contract. |
| `policy_id` | str | Logical policy name (e.g. `qwen3-4b-skyrl`). |
| `policy_version` | int | Monotonic per `policy_id`. |
| `base_model_id` | str | e.g. `Qwen/Qwen3-4B-Instruct-2507`. |
| `tokenizer_id` | str | Identifies the exact tokenizer used. |
| `inference_backend` | str | e.g. `vllm-pinning`, `sglang`. |
| `sampling_params` | dict | Temperature, top_p, etc. |
| `created_at_step` | int | Trainer step at episode start (or 0 if external). |
| `started_at` | timestamp | Wall-clock UTC. |
| `finished_at` | timestamp | Wall-clock UTC. |
| `termination_reason` | str | `done`, `truncated`, `error`, `timeout`. |
| `events` | list[Event] | Ordered events: tool calls, observations, agent actions, reward updates. Each event carries token IDs (when LLM-emitted), tool name + input + output (when tool), or reward delta + provenance. |
| `messages_or_turns` | list[Message] | Optional structured turn-level view; redundant with events but useful for trainers that want a turn-level shape. |
| `total_reward` | float | Episode-final reward. |
| `reward_events` | list[RewardEvent] | Per-event reward deltas with provenance (verifier ID, judge model ID if any). |
| `behavior_log_probs` | list[float] OR null | Per-token log-probabilities from the inference backend at generation time. Required for async RL correction; `null` only if the backend cannot provide them (in which case `trust_level` and routing are restricted; see §6.3). |
| `prompt_token_ids` | list[int] | Initial prompt tokens. |
| `response_token_ids` | list[int] | All model-emitted tokens across turns, concatenated. |
| `response_loss_mask` | list[int] (0/1) | 1 on assistant turns; 0 on tool/observation tokens. |
| `tool_calls` | list[ToolCall] | Structured representation of the tool calls made by the agent. |
| `provenance` | dict | `{worker_id, worker_version, dataset_uri, dispatch_time, ...}`. |
| `trust_level` | enum | `own-fabric`, `partner-validated`, `partner-untrusted`, `external-eval-only`. Routing per §6.3 depends on this. |
| `schema_version` | str | EpisodeRecord schema version (semver). |

`events` is the load-bearing field: an EpisodeRecord can always be
reconstructed from its event stream. The flat fields above are summaries
and indexes for query.

**Tokenizer constraint.** The token-in/token-out invariant (§3.1) requires
that `tokenizer_id` matches across all consumers of the record. An archive
record with a different `tokenizer_id` than the trainer's expected
tokenizer is routed to mismatched-tokenizer handling (drop, re-tokenize
flag, or alternate trainer adapter), never silently consumed.

### 6.2 TrainingSample / TrainingGroup (derived, hot-path)

The unit pushed to LiveStore and consumed by TrainerAdapter. Compact,
shaped to satisfy ROLL, slime, VERL, and SFT/distillation trainers
simultaneously (per the trainer-side research).

A `TrainingGroup` is `n` `TrainingSample` records sharing a `group_uid`
(GRPO/DAPO sense). The store operates in groups (invariant 3.2); the
trainer adapter samples in groups; the wire format groups them together.

**TrainingSample fields:**

| Field | Type | Notes |
|---|---|---|
| `sample_uid` | str | Per-row UUID. |
| `group_uid` | str | Per-group UUID; n samples share. |
| `episode_uid` | str | Back-pointer to canonical EpisodeRecord. |
| `prompt_token_ids` | list[int] OR packed bytes | Unpadded. Re-padding is the LiveStore's `get_batch` job, not the wire's. |
| `response_token_ids` | list[int] OR packed bytes | Same. |
| `response_loss_mask` | list[int] (0/1) OR packed bytes | 1 on assistant tokens. |
| `behavior_log_probs` | list[float] OR packed bytes | Per-token logprobs from the inference backend at generation time. Default-required (per §4.4 and the algorithm matrix in §6.3). |
| `reward` | float | Episode-final reward, or aggregated. Trainers compute advantages from this; not from a pre-computed `advantage`. |
| `raw_reward` | float | Pre-normalization reward (slime/ROLL carry this; VERL DataProto today does not — add it). |
| `truncated` | bool | True if the rollout was cut by length (slime/ROLL carry this; useful for reward shaping decisions). |
| `behavior_policy_version` | int | Per-row stamp (invariant 3.5). |
| `created_at_step` | int | Trainer step at push time. |
| `task_id` | str | Provenance only — trainer must not use this to re-load the task from a local dataset (invariant 3.8). |
| `split` | str | `train` / `val` / `test` / etc. — provenance. |
| `policy_id` | str | Logical policy name. |
| `environment_id` | str | Provenance. |
| `environment_version` | str | Provenance. |
| `verifier_version` | str | Provenance. |
| `trust_level` | enum | Inherited from EpisodeRecord. |
| `sample_indices` | list[int] OR null | Optional back-pointer to a structured offset within the episode (slime carries this; VERL DataProto today does not — add it). |
| `instance` | dict (JSON) | Compact instance metadata (data_source, ability, reward_model, extra_info, index — preserves the current `prompt_extras` payload). |
| `error` | str OR null | Error message if dispatch failed; non-null implies row-padding behavior at the trainer adapter. |
| `is_padded` | bool | Padding row flag (preserves current behavior). |

**What the schema deliberately does not carry:**
- `advantage` — computed by trainer adapter (per §4.4).
- `returns` — same.
- `KL penalty / ref_log_probs` — recomputed by FSDP/Megatron actor on
  sample.
- Any per-trainer normalization constants — those belong to the adapter.
- Raw task description / problem statement — that lives at the
  EnvironmentProvider behind `task_id` (invariant 3.8). Trainer never
  resolves `task_id` back to a problem statement.

**What gets re-padded server-side at `get_batch`:**

The current `_pack` method (`trajectory_store.py:470-621`) re-pads to
sample-local max at pack time. The successor LiveStore preserves this:
the wire schema stores token sequences unpadded, the `get_batch` response
returns padded tensors. Padding caps come from the trainer's request, not
from the wire schema.

The current tensor shapes the trainer adapter receives (from
`trajectory_store.py:90-101`):

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
| `raw_reward` | `(B,)` | float32 (NEW, per §6.2) |
| `truncated` | `(B,)` | bool (NEW, per §6.2) |

These are adapter-shape tensors, not part of the wire schema — the
trainer adapter could reshape them per its own preference; the wire only
guarantees the field inventory and dtypes.

### 6.3 Algorithm-fields matrix and trust routing

Per algorithm or pipeline, what fields a `TrainingSample` must carry, and
which trust levels each algorithm admits:

| Algorithm or pipeline | Needs groups? | Behavior logprobs policy | Needs ref policy? | Notes |
|---|---:|---:|---:|---|
| REINFORCE | No | Yes for async RL | No | Without behavior logprobs, stale samples become a biased on-policy approximation. |
| RLOO | Yes | Yes for async RL | No | Sibling grouping load-bearing; behavior logprobs needed when samples come from older policy versions. |
| GRPO/DAPO | Yes | Yes | Optional | Current path; group integrity is invariant 3.2. |
| SFT on filtered trajectories | No | Not directly | No | SFT does not consume behavior logprobs, but archived rollouts still retain them when available. |
| On-policy distillation | No | Yes for audit/provenance | Teacher policy | Uses fresh/current-policy rollouts or teacher outputs; behavior logprobs prove provenance. |
| Normal distillation | No | Optional, useful for filtering | Teacher policy | Trains from teacher-generated outputs without RL correction; retains provenance. |
| Offline distillation | No or group | Optional, useful for filtering | Optional teacher | Provenance and quality filters needed. |

**Default required fields on every TrainingSample:**
- token IDs
- response/action loss mask
- reward (or reward events)
- `group_uid` or `pair_uid`
- behavior policy ID/version
- behavior logprobs by default for any async RL rollout; if the inference
  backend cannot provide them, the sample is marked `behavior_logprobs:
  null` and `trust_level` is restricted to `external-eval-only` or
  `partner-untrusted`. **Never silently set `IS=1.0` for missing logprobs.**
- `created_at_step` / `created_at_time`
- environment / task / provenance metadata

**External / partner trajectories.** Per principle 4.5:

> External trajectories are untrusted by default. Do not silently set IS
> weight to 1.0 for all external data. Route to explicit modes: offline
> RL, SFT, rejection learning, eval-only, or corrected RL if behavior
> logprobs are available.

The `trust_level` field plus the routing matrix above (which algorithms
admit which trust levels) is the policy mechanism. The planner specifies
the per-trainer-adapter routing rules, not this document.

**Validation samples.** Validation `TrainingGroup`s (per invariant 3.8)
carry `split=val` and are tagged so the trainer adapter routes them to
`score_validation`, not `step`. The trainer adapter's `request_validation`
RPC to the RolloutWorker is the one path that produces `split=val`
records; trainer-local val-loaders are abolished.

---

## 7. Live store vs durable replay archive

The two systems described in §5.4 and §5.5 are different products, with
different optimization targets. Design rule 4.3 makes this explicit.
Stating the property comparison directly:

| Property | Live store (slot 5.4) | Replay archive (slot 5.5) |
|---|---|---|
| Bound | Bounded RAM (e.g. 256 groups) | Unbounded durable storage |
| Order | FIFO with K-staleness eviction | Append-only by ingest time |
| Read pattern | `get_batch(n_groups)` pop-on-sample | Range and predicate queries |
| Latency target | `get_batch` ≤ 100 ms | Query latency seconds-to-minutes |
| Loss tolerance | Loss-tolerant (refill from producers) | Lossless |
| Horizon | Minutes | Weeks to months |
| Consumers | Live trainer(s) | Offline RL, distillation, audit, curation, evaluation |
| Wire form | TrainingSample / TrainingGroup | EpisodeRecord (canonical) and derived TrainingSample |
| Eviction by | `created_at_step` (staleness) | Never auto-evicts; retention policy is offline |
| HA story | Accept SPOF (current state) | Backup / replication is an offline storage problem |
| WAL story | A live-store recovery mechanism, optional, S6+ | Not WAL — separate product |

**Why archive cannot be "just live-store persistence":**
1. Different consumers want different things. The trainer wants
   ready-to-pack tensors. An auditor wants the original event stream and
   the verifier identity. A curator wants natural-language metadata.
2. Different access patterns. The trainer reads in FIFO order with
   pop-on-sample. An offline job wants `WHERE policy_version BETWEEN x
   AND y AND environment_id = "swe_agent" AND reward > 0.5`.
3. Different durability cost models. Live store can be RAM-only at ~$0
   storage cost. Archive at SkyRL-v0 scale is gigabytes per day; storage
   tiering matters.

**The first archive implementation can be simple.** Append-only Parquet
files in S3 partitioned by `(policy_id, date, environment_id)`, with a
small Postgres index for `query`. The minimum data unit is `EpisodeRecord`
(§6.1). `TrainingSample` derivation can run on demand or be pre-cached as
secondary Parquet. The planner picks the storage stack.

**Tee from the producer, not from the live store.** The producer writes
to both. Reasons:
- Decouples archive durability from live-store availability.
- Lets archive carry richer canonical records (full event stream) than
  the live store keeps.
- Avoids making the live store responsible for two consumer SLAs.

---

## 8. Component placement

Where each slot naturally runs in a deployed system:

| Component | Stateful? | Latency-sensitive? | Natural placement | Reason |
|---|---|---|---|---|
| InferenceBackend | Yes (model weights, LoRA cache, KV cache) | Yes (generation dominates wall-clock) | Dedicated inference GPU pool | Different resource profile from training. |
| EnvironmentProvider | Yes (sandboxes, sessions, task queues) | Medium (init/runtime can bottleneck) | High-CPU box with scratch disk; scale horizontally | Often needs CPU, containers, filesystem, network. |
| RolloutWorker | Mostly (dataloader cursor, inflight episodes) | Low–medium | Near env provider, or sharded across env clusters | Worker should be a coordinator, not a heavy state owner. |
| LiveStore | Yes (hot FIFO buffer) | Yes (`get_batch` is trainer hot path) | Colocate with trainer | Avoid moving large tensor batches over the network in the hot path. |
| ReplayArchive | Yes (long-term data) | No | S3/NFS/Iceberg/Postgres-style storage | Queryability and durability dominate. |
| PolicyRegistry / Coordination | Small (version/manifest registry) | Low | Anywhere reliable and reachable | Cold path except publish events. |
| TrainerAdapter | Yes (optimizer state, FSDP/Megatron shards) | Yes | Dedicated training GPU box/pool | Training and inference scale differently. |

Three implications:

1. **RolloutWorker is the easiest slot to fan out.** It is the right
   first place to introduce decentralization (S5).
2. **LiveStore and TrainerAdapter should usually be close** (same box, or
   high-bandwidth interconnect). The hot path is `get_batch`.
3. **InferenceBackend and TrainerAdapter should not be assumed to
   colocate.** At scale, inference is shared infrastructure across many
   trainers (S8).

---

## 9. Stage-wise migration plan

Stages S0–S8. Each stage is defined by:
- **Goal** — what the stage proves at the end.
- **Scope** — which slot(s) gain a network boundary or new behavior;
  which contracts get sealed.
- **Invariants preserved** — referenced from §3.
- **Validation** — the test that says the stage is done.
- **Reversibility** — can we roll back to the previous stage if it
  regresses.

Stages **S1–S4 are sequential** (each depends on the previous). Stages
**S5–S8 are parallel proofs of pluggability** — they exercise specific
slots independently and can be ordered by need, not by dependency.

**This document does not prescribe how to extract services**. The planner
chooses transports, sequencing within a stage, deployment topology, and
test strategy. What this section fixes: what each stage proves, what
contracts must be intact at the end of each stage, and what the rollback
path is.

### S0. Today (no changes) — reference baseline

**Goal.** Reference baseline. The full async loop runs in one process,
one container, one trainer host plus one EC2 vLLM box.

**Scope.** No changes. This is the starting point of the migration.

**Invariants preserved.** All of §3, all already.

**Validation.** Whatever currently certifies a healthy training run:
`val_before_train` validation pass, step-1 trainer step, step-20 wedge
stabilization (referenced in `s3_fullasync_docker.sh`), no spurious
abort, healthy WandB metrics.

**Reversibility.** Trivial — this is the current shape.

### S1. LiveStore behind a network boundary (same machine)

**Goal.** Prove the LiveStore slot is real. The in-process
`TrajectoryStore` is replaced by a same-machine LiveStore service
implementing the §5.4 interface. Trainer and producer both call the
service rather than the in-process object.

**Scope.**
- LiveStore becomes a separate process on the trainer box.
- TrainerAdapter calls `get_batch` over an out-of-process boundary
  (transport TBD by planner — same machine, so localhost gRPC, Ray, or
  shared memory are all plausible).
- RolloutWorker (still in trainer process) calls `push_group` over the
  same boundary.
- Re-padding still happens server-side.
- `get_batch`'s server-side blocking and no-progress detector replace
  the current `wait_until_with_progress` busy-loop.
- The wire schema is the §6.2 TrainingSample / TrainingGroup. **Sealing
  the live-path schema is the load-bearing artifact of S1.**

**What does not change.** RolloutWorker, EnvironmentProvider,
InferenceBackend, TrainerAdapter (modulo the swap of `self.trajectory_store`
for a client). Data ownership is still in the trainer process; that's S2.

**Invariants preserved.** All §3 invariants. In particular:
- 3.1 token-in/token-out: schema carries token IDs.
- 3.2 group integrity: the wire groups; pop-on-sample preserves it.
- 3.5 per-row `behavior_policy_version`: stamped at push.
- 3.6 pop-on-sample.
- 3.7 eager-push seam: still owned by RolloutWorker; the seam is now a
  network call rather than a function pointer.

**Validation.**
- Trainer step times within X% of S0 (the planner picks X based on a
  benchmark of `_pack` + serialize vs in-process `_pack`).
- WandB curves indistinguishable from S0 over a 20-step run.
- Live store can be killed and restarted; trainer recovers via the
  no-progress detector → checkpoint-and-wait pattern.

**Reversibility.** Trivial — keep S0 launcher available; the LiveStore
service is feature-flagged.

### S2. RolloutWorker as its own process — data ownership migration

**Goal.** Prove the RolloutWorker slot is real **and** that data ownership
moves from the trainer to the worker (invariant 3.8). After S2, the
trainer process does not import `openhands`, does not load any parquet,
and does not know task IDs.

**Scope.**
- RolloutWorker becomes a standalone process. It owns:
  - the SkyRL-v0-293 train and val parquet files,
  - the `StatefulDataLoader` and its checkpoint,
  - the `AsyncLLMServerManagerDAPO` (or successor) dispatch logic,
  - producer-side filters (zero-variance drop, length cuts),
  - the eager-push seam (§3.7) — now a network call to LiveStore.
- TrainerAdapter no longer instantiates `ContinuousRolloutProducer`,
  `AsyncLLMServerManagerDAPO`, the DAPO manager, or the `_push_fn`
  closure. It only calls `get_batch` and `request_validation`.
- The launcher splits: `s3_fullasync_docker.sh` becomes one launcher per
  service (planner picks names).
- Trainer dependencies removed: `openhands`, `aiohttp`, `fastapi`,
  `uvicorn`, `async_generator` — none are training-time concerns.
- Validation flow becomes RPC: trainer asks worker
  `request_validation(env_id, split=val)`, worker dispatches against the
  EnvironmentProvider, returns groups, trainer scores. **No trainer-local
  val parquet.**
- PolicyRegistry exists in minimal form: the trainer keeps direct pool
  publish (no change), and writes the current `policy_version` plus
  `adapter_uri` to a small registry the worker polls. The registry
  contract is sealed; the registry implementation is minimal.

**What does not change.** EnvironmentProvider, InferenceBackend,
LiveStore (still S1's same-machine service), TrainerAdapter math.

**Invariants preserved.** Critically 3.8 (data ownership) and 3.7
(eager-push seam). Also 3.5 (`behavior_policy_version` stamping moves to
worker, but stamping happens at push, exactly as today).

**Validation.**
- Trainer image (Docker) shrinks: `openhands` removed.
- Trainer config no longer references `data.train_files` or
  `data.val_files`.
- Validation results match S0/S1 within tolerance (val score variance is
  the planner's concern; specify a bound).
- Worker can be restarted independently; trainer no-progress detector
  fires correctly.
- Producer-side data ownership is verifiable: kill the trainer mid-run,
  the worker keeps generating episodes (which the live store will
  eventually evict, but the worker survives).

**Reversibility.** Higher cost than S1 — the trainer image shrinks and
the dataloader migrates. Roll back by reverting both image changes and
launcher changes; not a flag flip. The planner specifies a hold-period in
which both paths run side-by-side (one trainer reads from live store,
the other reads from local parquet) so a regression can fall back.

### S3. ReplayArchive — durable canonical record

**Goal.** Prove the ReplayArchive slot is real. Every episode the worker
produces is teed to the archive in canonical `EpisodeRecord` form (§6.1).
The archive is queryable.

**Scope.**
- RolloutWorker writes `EpisodeRecord` to the archive per completed
  episode (irrespective of producer-side filter — the archive sees
  everything; the live store sees only filter survivors).
- The archive supports `append_episodes` and `query`. The minimum query
  predicate set: by `policy_id`, `policy_version`, `environment_id`,
  `split`, and time range. `derive_training_samples` may be deferred.
- Trust and provenance fields (§6.1) are populated.

**What does not change.** Hot path. Trainer never reads the archive. Live
store unchanged.

**Invariants preserved.** All §3. Archive is a tee, not a bottleneck —
push to archive is async-fire-and-forget from the worker's perspective
(planner specifies durability semantics: at-least-once vs at-most-once
etc.).

**Validation.**
- Sample size of N episodes from S2 → archive has N records (within
  at-least-once semantics).
- Query by `(policy_id, version_range, env_id)` returns expected counts.
- One offline job (planner's choice — maybe a reward-distribution
  histogram) consumes the archive end-to-end.

**Reversibility.** Trivial — disable the tee.

### S4. PolicyRegistry / Coordination as the single source of truth

**Goal.** Prove the PolicyRegistry slot is real. The trainer publishes
versions to the registry; the registry fans out to InferenceBackend
(`/reload_lora`), LiveStore (notification), and RolloutWorker
(subscription) atomically. The trainer no longer talks directly to the
pool.

**Scope.**
- TrainerAdapter calls `publish_policy_version(version, adapter_uri,
  policy_id)` instead of POSTing `/reload_lora` directly.
- Adapter binaries move to durable storage (local FS, NFS, or S3 — planner
  picks; the registry stores the URI).
- The registry preserves the abort gate (invariant 3.3): if any pool
  child fails, the publish call returns failure and the trainer aborts.
- RolloutWorker subscribes to version updates instead of polling.

**What does not change.** Hot path between worker → live store → trainer.
The pool itself (vLLM child) is unchanged — it still serves
`/reload_lora` and `/v{N}/generate`.

**Invariants preserved.** 3.3 (abort gate), 3.4 (pinning), 3.5 (per-row
version stamping moves but is not changed semantically).

**Validation.**
- A simulated pool-child failure aborts the trainer (preserves §3.3).
- Worker version subscription latency below a target bound.
- Adapter URI manifest is queryable.

**Reversibility.** Roll back to S3 by re-enabling direct pool publish on
the trainer and disabling the registry subscription on the worker.
Higher-cost rollback than S2 because the publish path moves.

### S5. Multi-producer with heterogeneous EnvironmentProvider adapters

**Goal.** Prove the EnvironmentProvider slot is real and the
RolloutWorker slot is fan-out-able. Run two RolloutWorkers in parallel,
each pointed at a different EnvironmentProvider adapter (e.g. ProRL +
GEM, or ProRL + ORS, or ProRL + a partner). Both push to the same
LiveStore.

**Scope.**
- A second EnvironmentProvider adapter is implemented (planner picks the
  first concrete second adapter — most likely GEM or a stripped-down ORS
  test environment).
- A second RolloutWorker instance is launched against the second
  provider.
- Data sharding: workers either disjoint-shard the dataset by
  `task_id`, or each worker pulls from a distinct dataset entirely.
  Planner specifies sharding strategy.
- LiveStore receives concurrent `push_group` from multiple workers
  (already thread-safe in concept; planner verifies).
- TrainerAdapter samples mixed groups; advantage computation is
  group-relative (invariant 3.2 unchanged).

**What does not change.** TrainerAdapter math. LiveStore interface.
PolicyRegistry interface.

**Invariants preserved.** 3.2 (group integrity), 3.6 (pop-on-sample —
two workers don't collide because they push different groups), 3.8 (each
worker owns its own data).

**Validation.**
- Trainer step consumes groups from both providers in expected ratio.
- Heterogeneous-environment learning curve makes sense (planner specifies
  the metric — could be per-environment validation pass-rate).

**Reversibility.** Trivial — disable the second worker.

### S6. Alternate TrainerAdapter alongside VERL

**Goal.** Prove the TrainerAdapter slot is real. Plug ROLL or slime as a
second trainer adapter, consuming the same `TrainingGroup` shape. Run
A/B or distillation experiments where the same producer feeds both.

**Scope.**
- A second TrainerAdapter (planner picks ROLL or slime; ROLL is closer
  to VERL's DataProto so likely first) is implemented.
- The second trainer subscribes to the same LiveStore via `get_batch`.
- Pop-on-sample (invariant 3.6) ensures the two trainers see disjoint
  groups; if A/B is the goal, planner specifies dataset-level routing
  rather than store-level.
- Each trainer publishes to its own `policy_id` namespace in the
  PolicyRegistry; pool serves both via `--max-loras` (already supported,
  per `_vllm_child.py`).
- RolloutWorkers tag rollouts with the policy_id they dispatched
  against.

**What does not change.** EnvironmentProvider, InferenceBackend (modulo
multi-namespace policy serving — already supported), LiveStore.

**Invariants preserved.** 3.4 (pinning per policy_id), 3.5 (per-row
versioning), 3.6 (pop-on-sample), 3.8 (no trainer owns data).

**Validation.**
- ROLL/slime trainer consumes `TrainingGroup`s and produces a sane
  loss curve.
- Two-trainer concurrent run: both `policy_id` namespaces advance in
  the registry; pool serves both versions concurrently; LiveStore
  metrics show disjoint group draws.

**Reversibility.** Trivial — disable the second trainer adapter.

### S7. Alternate InferenceBackend alongside vLLM

**Goal.** Prove the InferenceBackend slot is real. Plug SGLang as a
second backend; verify pinning equivalent and per-token logprobs.

**Scope.**
- An SGLang adapter implements the §5.2 interface plus a pinning
  guarantee (invariant 3.4).
- A subset of RolloutWorkers route to SGLang; others stay on vLLM.
- TrainerAdapter is unchanged.

**Invariants preserved.** 3.1 (token-in/token-out — verify SGLang's
token interface matches), 3.4 (pinning equivalent — explicitly tested).

**Validation.**
- An end-to-end rollout from SGLang produces identical
  `behavior_log_probs` semantics (within numerical tolerance).
- Mixed-backend trainer step computes consistent advantages.

**Reversibility.** Trivial — disable the SGLang adapter.

### S8. Federation — multiple trainers, aggregation, multi-region

**Goal.** Prove the architecture composes at scale. Multiple
TrainerAdapters, each producing adapter deltas, with an aggregation
service merging them into a combined adapter that PolicyRegistry
publishes. Multi-region deployments are possible (each region has its
own LiveStore and one or more workers; archives may be region-local with
async cross-region replication).

**Scope.**
- Aggregation service (FedAvg or similar; this document does not pick the
  algorithm).
- Per-trainer adapter namespaces in the PolicyRegistry.
- Cross-region archive replication if multi-region is in scope.

**Invariants preserved.** All §3. The aggregation service's output is
itself a publish, so 3.3 (abort gate) and 3.4 (pinning) extend naturally.

**Validation.** Long-running. The criterion is qualitative: "the
architecture composes; adding a third trainer or a second region is a
bounded operational task, not a redesign."

**Reversibility.** Per-trainer; disable aggregation, fall back to
single-trainer publish.

### Stage map

```
                 S0 (today)
                    │
                    ▼
                 S1  LiveStore network boundary
                    │
                    ▼
                 S2  RolloutWorker is its own process
                     └── data ownership migrates here
                    │
                    ▼
                 S3  ReplayArchive exists
                    │
                    ▼
                 S4  PolicyRegistry is single source of truth
                    │
       ┌────────────┼────────────┬─────────────┐
       ▼            ▼            ▼             ▼
       S5           S6           S7            S8
   multi-          alt          alt        federation
  producer       trainer     inference     (multi-trainer
  (heterog       (ROLL or     (SGLang)      + aggregation
   envs)          slime)                    + multi-region)
```

S5–S8 ordering is by need, not by dependency. They are independent
proofs that the slot model is real.

---

## 10. What does NOT change

Concrete pieces of the current stack that the migration deliberately
preserves, recast in slot terms:

1. **vLLM pool implementation** — `_vllm_child.py`, `/v{N}/generate`,
   `/reload_lora`, pinning swap protocol, `--max-loras 8`. The pool is
   the current adapter for slot 5.2; it stays as-is. New backends are
   sibling adapters.

2. **ProRL FastAPI agent dispatcher** — `openhands/nvidia/async_server.py`,
   the three-stage pipeline (init → run → eval), the agent handler
   registry. ProRL is the current adapter for slot 5.1; its internal
   shape stays. The boundary is the EnvironmentProvider interface
   (§5.1, Appendix A); the planner exposes ProRL's existing operations
   as adapter methods, not the other way around.

3. **Token-in/token-out invariant** —
   `openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py`. Unchanged. Wire
   schemas store token IDs (§3.1).

4. **Chat template logic** — `chat_template_manager.py`. Unchanged.

5. **`filter_easy_hard_instance` (zero-variance drop)** — stays
   producer-side. The store never sees zero-variance groups (§3.7).

6. **GRPO/DAPO advantage computation** — stays trainer-side
   (per principle 4.4). Whole-group integrity (§3.2).

7. **Temporal IS correction** — `core_algos.py:664-690` stays trainer-side.
   `behavior_log_probs` come from the wire (§6.2); `old_log_prob`
   recomputed by the FSDP actor.

8. **Reward computation** — stays trainer-side via the reward manager;
   `reward_fn` is a Ray remote.

9. **`_save_checkpoint` and FSDP/Megatron actor weights** — stay
   trainer-side. The trainer adapter owns optimizer state.

10. **Producer-side filter location.** `filter_easy_hard_instance` is
    cheap (one boolean check per group) and saves ~30% of push bandwidth
    (typical zero-variance rate on SkyRL-v0). It stays in the
    RolloutWorker.

---

## 11. Non-goals

The following are intentionally **not** in scope of this design document.
Some are explicit non-goals (won't happen). Others are deferred to the
planner.

**Explicit non-goals:**
- Authentication, authorization, mTLS, per-producer API keys.
  In-cluster trust is assumed (same VPC / security group). External
  partner contributions are gated by `trust_level` routing (§6.3) but
  **not** by network-layer auth in this document.
- High availability for the LiveStore. The buffer is ephemeral; loss
  is acceptable.
- High availability for the PolicyRegistry. SPOF is acceptable in the
  initial deployment.
- Adapter-arithmetic / federated averaging math. S8 mentions
  aggregation; this document does not pick FedAvg vs trimmed-mean vs
  any other algorithm.

**Deferred to the planner (these will be decided, just not here):**
- Choice of transport (gRPC, HTTP, Ray, shared memory, Arrow Flight).
- Choice of archive storage stack (Parquet/Iceberg, Postgres, custom).
- Choice of coordination transport (gRPC streaming, HTTP polling,
  Redis pub/sub).
- Tensor serialization format (single large message vs streamed
  chunks).
- Backpressure model (advisory hint vs blocking; both are workable —
  see §12 question 4).
- Service extraction order *within* a stage. The stages here say what
  must be true at the end of each; the planner chooses how to get there.
- Deployment topology for S5–S8 (multi-region, K8s manifests, autoscale
  rules).
- Cost / capacity / throughput numbers — none have been measured against
  the new shape; the planner specifies a benchmark plan.
- Exact adapter mapping for ROCK / GEM / ORS environment integrations
  beyond the §5.1 interface — the planner specifies the first concrete
  second adapter and writes the integration plan.

---

## 12. Open questions for the planning agent

The planner is responsible for resolving each of these and choosing the
specific implementation path.

1. **LiveStore placement in S1.** Same-machine sidecar (lowest hot-path
   latency, simplest), or separate machine from trainer (lets the
   trainer host be cheaper), or shared-memory only (no transport)?
   The hot-path benchmark (`_pack` + serialize vs in-process) determines
   the answer. Recommendation: same-machine sidecar for S1; revisit at
   S5.

2. **Tensor serialization format.** Protobuf is not zero-copy for large
   byte fields. Options: single large message with raised gRPC limits
   (~200 MB), server-side streaming (chunked), shared memory for
   same-machine, Arrow IPC for cross-machine. Planner picks based on
   the S1 benchmark.

3. **`get_batch` blocking semantics.** Server-side blocking with
   `timeout_ms` (replaces current busy-loop) is the design here. Planner
   specifies the no-progress detector window and the trainer's
   exponential-backoff schedule.

4. **Backpressure model for `push_group`.** Four options: server-side
   blocking, push-always-evict, advisory hint, pre-query. The advisory
   hint with FIFO eviction is multi-producer-friendly; blocking is
   simpler for the single-producer case (S1–S4). Planner picks per stage;
   the `backpressure_hint` field is on the wire either way (it can
   always read 0).

5. **Validation flow placement.** Per invariant 3.8, the trainer asks
   the RolloutWorker for validation groups. Open question: does the
   worker need to **pause production** during validation (shared
   OpenHands session) or run validation on a **dedicated session**
   (separate ProRL port, no coordination)? Recommendation: dedicated
   val session for simplicity, especially since the 23-instance val
   set is small. Planner verifies and decides.

6. **Dataloader state under producer ownership (S2).** When the
   RolloutWorker owns the dataloader, it owns the
   `state_dict()`/`load_state_dict()` pair. On worker restart it loads
   the last checkpoint to avoid prompt repetition. Open question:
   where is the worker's checkpoint stored? Trainer-attached disk?
   Shared NFS? Object store? Recommendation: trainer-attached for S2
   (simplest); migrate when S5 makes the worker fan-out.

7. **External / partner trajectory routing.** Trust levels and routing
   matrix are sketched in §6.3, but the planner specifies the
   per-trainer-adapter routing rules (which trust levels each adapter
   admits, what filtering happens at LiveStore ingest, etc.).

8. **PolicyRegistry transport in S4.** Server-streaming gRPC for
   subscriptions, HTTP long-poll, or Redis pub/sub? The publish event
   rate is low (one per `save_freq` steps, ~1/min); any reasonable
   choice works.

9. **Adapter storage location in S4+.** Local filesystem (single-machine
   simplicity), NFS (multi-machine simplicity), S3 (multi-region).
   Planner picks per deployment.

10. **`TrainingSample` / `TrainingGroup` schema versioning.** §6.2
    states `schema_version`. The planner specifies the compatibility
    policy: forward-compatible field addition (adapters ignore unknown
    fields), strict version match, semver minor for additions only?

11. **First concrete second adapter for each slot.**
    - For slot 5.1 (env): GEM, ROCK, or stripped-down ORS test env?
    - For slot 5.2 (inference): SGLang? (likely yes — slime uses it)
    - For slot 5.6 (trainer): ROLL? (closer to VERL than slime)
    Pick one per slot for the S5–S7 demonstration runs.

12. **Schema migration in S2 vs S3.** When the wire schema changes from
    "VERL DataProto-shaped" (current) to "TrainingSample with
    `raw_reward`/`truncated`/`sample_indices` added", does the trainer
    adapter handle both shapes during the migration window, or is
    there a hard cutover at the S2/S3 boundary? Planner picks.

13. **Archive ingest semantics in S3.** At-least-once (worker retries
    on failure, dedup at archive) vs at-most-once (worker
    fire-and-forget; some data loss tolerated). Planner picks.

14. **Aggregation algorithm in S8.** Out of scope per §11, but the
    planner identifies the prerequisite work (adapter-delta arithmetic
    in the relevant trainer adapter; namespace conventions in the
    PolicyRegistry).

---

## Appendix A — Slot interfaces (vision-level signatures)

Pseudo-Python signatures. No types from VERL, ROLL, slime, or any
specific framework. Method bodies and return-type concretions are the
planner's job.

### A.1 EnvironmentProvider

```python
class EnvironmentProvider(Protocol):
    environment_id: str
    environment_version: str

    def list_tasks(self, split: str) -> list[str]: ...
    def create_episode(self, task_id: str, **opts) -> EpisodeHandle: ...
    def get_prompt(self, h: EpisodeHandle) -> list[ContentBlock]: ...
    def act(self, h: EpisodeHandle, tc: ToolCall) -> StepResult: ...
    def close(self, h: EpisodeHandle) -> None: ...

class StepResult(Protocol):
    observation: list[ContentBlock]
    reward: float
    done: bool
    info: dict
```

### A.2 InferenceBackend

```python
class InferenceBackend(Protocol):
    backend_id: str

    def generate(
        self,
        policy_ref: PolicyRef,
        tokenized_prompt: list[int],
        sampling_params: dict,
    ) -> GenerationResult: ...

    def reload_policy(
        self,
        policy_id: str,
        policy_version: int,
        adapter_uri_or_blob: str | bytes,
    ) -> ReloadResult: ...

    def health(self) -> HealthStatus: ...

class GenerationResult(Protocol):
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str
    metadata: dict  # includes the served policy_version
```

### A.3 RolloutWorker

```python
class RolloutWorker(Protocol):
    worker_id: str

    # Internal main loop — not RPC.
    def run(
        self,
        env: EnvironmentProvider,
        infer: InferenceBackend,
        policy_subscription: PolicyVersionStream,
        store: LiveStoreClient,
        archive: ReplayArchiveClient | None,
        task_source: TaskSource,
        gen_batch_size: int,
        filter_strategy: FilterStrategy,
    ) -> None: ...

    # RPC surface for the trainer.
    def pause_production(self) -> Ack: ...
    def resume_production(self) -> Ack: ...
    def run_validation(
        self,
        env_id: str,
        split: str,
        options: dict,
    ) -> list[TrainingGroup]: ...
    def get_dataloader_state(self) -> bytes: ...
    def load_dataloader_state(self, state: bytes) -> Ack: ...
```

### A.4 LiveStore

```python
class LiveStore(Protocol):
    def push_group(
        self,
        records: list[TrainingSample],
        group_uid: str,
        producer_id: str,
    ) -> PushResult: ...

    def get_batch(
        self,
        n_groups: int,
        current_step: int,
        staleness_cutoff_k: int,
        timeout_ms: int,
    ) -> BatchResult: ...

    def get_metrics(self, current_step: int) -> StoreMetrics: ...

    def notify_policy_version(
        self,
        version: int,
        adapter_uri: str,
    ) -> Ack: ...

class PushResult(Protocol):
    accepted: bool
    store_size: int
    backpressure_hint_ms: int
```

### A.5 ReplayArchive

```python
class ReplayArchive(Protocol):
    def append_episodes(
        self,
        records: list[EpisodeRecord],
    ) -> AppendResult: ...

    def query(
        self,
        filter_spec: FilterSpec,
    ) -> Iterable[EpisodeRecord | TrainingSample]: ...

    def derive_training_samples(
        self,
        episode_uids: list[str],
        schema_version: str,
    ) -> list[TrainingSample]: ...
```

### A.6 TrainerAdapter

```python
class TrainerAdapter(Protocol):
    trainer_id: str
    policy_id: str

    def request_batch(
        self,
        n_groups: int,
        step: int,
    ) -> TrainingGroup: ...

    def step(self, group: TrainingGroup) -> StepMetrics: ...

    def save_checkpoint(self, step: int, dir: str) -> str: ...

    def publish_policy_version(
        self,
        step: int,
        adapter_uri: str,
    ) -> PublishResult: ...

    def request_validation(
        self,
        env_id: str,
        split: str,
    ) -> list[TrainingGroup]: ...

    def score_validation(
        self,
        groups: list[TrainingGroup],
    ) -> ValidationMetrics: ...
```

### A.7 PolicyRegistry

```python
class PolicyRegistry(Protocol):
    def publish_policy_version(
        self,
        policy_id: str,
        version: int,
        adapter_uri: str,
        trainer_id: str,
    ) -> PublishResult: ...

    def get_latest_version(self, policy_id: str) -> VersionInfo: ...

    def subscribe_version_updates(
        self,
        policy_id: str,
    ) -> Iterable[VersionInfo]: ...

    def register_policy_namespace(
        self,
        policy_id: str,
        base_model_id: str,
        tokenizer_id: str,
        adapter_storage_root: str,
    ) -> Ack: ...

class PublishResult(Protocol):
    success: bool
    endpoints_ok: int
    endpoints_failed: int  # invariant: success implies this is 0
    latency_s: float
```

---

## Appendix B — Mapping current code to slots

For each current file/component: which slot it implements, what migrates
where, what stays.

| Current artifact | Slot | Migration |
|---|---|---|
| `openhands/nvidia/async_server.py` (FastAPI dispatcher) | 5.1 EnvProvider | Stays. Becomes the ProRL adapter for the env interface. |
| `openhands/nvidia/registry.py` + `AgentHandler`s | 5.1 EnvProvider | Internal to the ProRL adapter. |
| `openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py` | 5.1 (internal) | Internal to ProRL. Token-in/token-out invariant lives here. |
| `scripts/serving/_vllm_child.py` | 5.2 InferenceBackend | Stays. Becomes the vLLM-pinning adapter. |
| `scripts/serving/launch_remote_vllm_pool.sh` | 5.2 (orchestration) | Stays. Pool orchestration is internal to the vLLM adapter. |
| `trainer_integration/verl/verl_custom/replay/continuous_producer.py` | 5.3 RolloutWorker | Migrates to the worker process at S2. Daemon thread becomes a service main loop. |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` | 5.3 RolloutWorker | Migrates. The DAPO eager-push seam (§3.7) lives here. |
| `trainer_integration/verl/verl_custom/replay/trajectory_store.py` | 5.4 LiveStore | The data structure stays; the process boundary changes at S1. |
| (none today) | 5.5 ReplayArchive | New at S3. |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`, `ray_trainer_dapo.py` | 5.6 TrainerAdapter | Stays as the VERL adapter. Loses dataloader, val-loader, and direct rollout-mgr at S2; loses direct pool publish at S4. |
| `_publish_lora_adapter` in `ray_trainer.py` | 5.7 PolicyRegistry (today) | Migrates to PolicyRegistry at S4. |
| `s3_fullasync_docker.sh` launcher | (cross-cutting) | Splits across stages: into worker launcher, trainer launcher, store launcher, registry launcher per the planner's choice. |
| `data.train_files` Hydra config | 5.6 today, 5.3 after S2 | Moves from trainer config to worker config. |
| `data.val_files` Hydra config | same | Same. Validation flow becomes RPC after S2. |

---

## Appendix C — Mapping to external frameworks

For each external framework: which slot it would implement, what its
adapter would have to do.

| Framework | Slot(s) | Adapter notes |
|---|---|---|
| **ROCK** (Alibaba) | 5.1 EnvProvider | ROCK is sandbox provisioning + GEM-compatible env interface. Adapter wraps ROCK's `Sandbox` lifecycle and `rock.make()` / `reset()` / `step()` to the §5.1 five-method interface. State: ROCK's Admin/Worker/Rocklet topology is internal to the adapter; the fabric only sees `EnvironmentProvider`. |
| **GEM** (axon-rl) | 5.1 EnvProvider | GEM is the standard agentic-LLM Gymnasium. Adapter wraps `make/reset/step` (with text observations and tool-call actions) to the §5.1 interface. GEM's tool wrappers become the `act(ToolCall)` argument's input dict. |
| **ORS / OpenReward** | 5.1 EnvProvider | HTTP+SSE protocol. Adapter is essentially a thin HTTP client mapping the §5.1 methods to ORS endpoints (`/tasks` → `list_tasks`, `/create_session` → `create_episode`, `/{env}/prompt` → `get_prompt`, `/{env}/call` → `act`). Streamed reward updates from `/{env}/call` aggregate into `StepResult.reward`. |
| **vLLM** (current) | 5.2 InferenceBackend | Already implemented (`_vllm_child.py`). Pinning swap protocol is the reference for the §5.2 pinning constraint. |
| **SGLang** (and sgl-router) | 5.2 InferenceBackend | Adapter wraps SGLang's generation API plus router. Must verify per-token logprobs and a pinning-equivalent guarantee. slime uses SGLang as its inference module, so an SGLang adapter unlocks easier slime trainer plug-in. |
| **TGI / TRT-LLM / hosted APIs** | 5.2 InferenceBackend | Adapters as needed. Hosted APIs without per-token logprobs are usable for eval-only flows (matches `trust_level=external-eval-only` routing). |
| **VERL `RayPPOTrainerDAPO`** (current) | 5.6 TrainerAdapter | Already implemented. Loses dataloader at S2 per invariant 3.8. |
| **ROLL** (Alibaba) | 5.6 TrainerAdapter | Async controller + DeepSpeed/Megatron/FSDP2. Already first-class on `behavior_policy_version` and supports six off-policy IS variants. Adapter consumes `TrainingGroup` via `get_batch`, computes ROLL's flavor of advantages and IS correction, publishes via PolicyRegistry. ROLL's `SampleBuffer` is replaced by the fabric's LiveStore. |
| **slime** (THUDM) | 5.4 + 5.6 (paired) | slime separates training, rollout, and data buffer. Its data-buffer concept is closest to LiveStore; its training module (Megatron-based) is the trainer adapter; its rollout module is conceptually the RolloutWorker. Plug-in: replace slime's data buffer with a LiveStoreClient; replace its rollout module with a RolloutWorkerClient. The Megatron training module becomes the adapter. |
| **Aggregation services** (FedAvg etc.) | 5.7 (extension) | S8 only. Operate on adapter URIs from the PolicyRegistry. Out of scope for this document. |

---

## Sources

- Current launcher: `scripts/_internal/s3_fullasync_docker.sh`
- Current live store: `trainer_integration/verl/verl_custom/replay/trajectory_store.py`
- Current producer: `trainer_integration/verl/verl_custom/replay/continuous_producer.py`
- Current DAPO trainer: `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py`
- Pool child: `scripts/serving/_vllm_child.py`
- Token-level vLLM clients: `openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py`
- ROCK: https://github.com/alibaba/ROCK
- GEM: https://github.com/axon-rl/gem · https://arxiv.org/abs/2510.01051
- ORS / OpenReward: https://openrewardstandard.io/ · https://docs.openreward.ai/
- ROLL: https://github.com/alibaba/ROLL
- ROLL async rollout: https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/async_parallel_rollout/
- ROLL async training: https://alibaba.github.io/ROLL/docs/User%20Guides/Advanced%20Features/async_training/
- slime: https://github.com/THUDM/slime
- slime architecture blog: https://www.lmsys.org/blog/2025-07-09-slime/
- async RL training landscape: https://huggingface.co/blog/async-rl-training-landscape
