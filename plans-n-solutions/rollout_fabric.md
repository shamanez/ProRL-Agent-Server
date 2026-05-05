# Rollout Fabric Architecture

A contract-first design for evolving the current ProRL + OpenHands + vLLM +
VERL full-async stack into a decentralized agentic-RL fabric where
environments, inference backends, rollout managers, live stores, replay
archives, trainers, and policy coordination are independently replaceable
adapter slots.

**Status:** Design-only. No code changes implied by this document. Need a solid plan first.
**Author:** Architecture review, 2026-04-30.

**Reading order:**
0. Sec.0 Target operating model — the running system after S4: startup sequence,
   five service contracts, smoke-test harness. **Read this first** to understand
   the destination before following the migration path.
1. Sec.1 Vision — the one-paragraph thesis.
2. Sec.2 Current state — what runs in this repo right now.
3. Sec.3 Architectural invariants — what cannot change across the migration.
4. Sec.4 Pluggability principles — the four design rules.
5. Sec.5 The slot model — seven adapter slots with current and future
   implementations named.
6. Sec.6 Wire schemas — the two data contracts (episode and training sample).
7. Sec.7 Live store vs durable replay archive — why they are different
   systems from day one.
8. Sec.8 Component placement — where each slot naturally runs.
9. Sec.9 Stage-wise migration plan — S0–S4 (sequential core boundary cuts) and
   S5–S8 (independent pluggability proofs).
10. Sec.10 What does NOT change.
11. Sec.11 Non-goals.
12. Sec.12 Open questions for the planning agent.
13. Appendices — slot interfaces, code-to-slot mapping, external-framework
    mapping, recursive implementation loop, progress artifact format.

**What this document is for.** It is the input to a downstream planning
agent. The planner's job is to take the contracts and stages here and
produce an implementation plan: pick transports, pick storage, sequence
service extraction, write migration scripts, and define the test matrix. The
planner is explicitly authorized to choose between options when this
document leaves them open. The planner is explicitly *not* authorized to
soften or skip any of the invariants in Sec.3 or the principles in Sec.4.

---

## 0. Target operating model

Five independently launched services behind stable contracts. This is what the
system looks like after S4 is complete. The migration stages in Sec.9 are steps
toward this picture, not alternatives to it.

### 0.1 Services and startup sequence

Services start in strict order. Each step blocks until the service reports
healthy before the next begins. The scripts listed are S4-target names; S0–S4
replace them incrementally as the services are extracted.

```
Step 1 — Environment provider     (no upstream service dependencies)
  Script:   scripts/services/start_env_provider.sh   [S4 target; today: s0_prorl.sh]
  Health:   GET :8006/health → 200
  Contract: EnvironmentProvider (Sec.5.1)
  Role:     Task registry, sandbox lifecycle, reward computation.
            Owns training and validation task datasets (invariant 3.8).

Step 2 — Inference backend        (no upstream service dependencies)
  Script:   scripts/serving/launch_remote_vllm_pool.sh start   [unchanged across stages]
  Health:   GET :8100/health, :8101/health, :8102/health, :8103/health → 200 each
  Contract: InferenceBackend (Sec.5.2)
  Role:     Token-level generation with policy pinning. Owns model weights
            and LoRA cache. Does not own policy version semantics.

Step 3a — Live store              (no upstream service dependencies)
  Script:   scripts/services/start_live_store.sh     [S1 target; today: in-process]
  Health:   GET <host>:<port>/health → 200
  Contract: LiveStore (Sec.5.4)
  Role:     Bounded hot FIFO between rollout managers and trainer.
            Pop-on-sample. Staleness eviction by created_at_step.

Step 3b — Policy registry         (no upstream service dependencies)
  Script:   scripts/services/start_policy_registry.sh [S4 target; today: int in trainer]
  Health:   GET <host>:<port>/health → 200
  Contract: PolicyRegistry (Sec.5.7)
  Role:     Source of truth for policy versions and adapter URIs. Fans out
            reload_lora to inference backend on publish. Preserves the
            endpoints_failed > 0 abort gate (invariant 3.3).

Step 4 — Rollout worker(s)        (depends on steps 1, 2, 3a)
  Script:   scripts/services/start_rollout_manager.sh [S2 target; today: daemon thread]
  Health:   Worker registers with live store; live store reports producer_id active.
  Contract: RolloutManager (Sec.5.3)
  Role:     Executes agent loop against environment provider and inference
            backend. Owns task datasets. Pushes to live store and replay archive.

Step 5 — Trainer adapter          (depends on steps 3a, 3b, 4)
  Script:   scripts/_internal/s3_fullasync_docker.sh  [evolves across stages]
  Health:   Trainer calls get_batch without timeout; begins step 1.
  Contract: TrainerAdapter (Sec.5.6)
  Role:     Consumes training groups, computes algorithm-specific fields,
            publishes policy versions to policy registry.
```

Steps 3a and 3b can start in parallel. Steps 1 and 2 can start in parallel.
Step 4 requires steps 1, 2, and 3a. Step 5 requires steps 3a, 3b, and 4.

Stop in reverse order. The trainer must stop before the live store or the
policy registry; the rollout manager before the environment provider or the
inference backend.

### 0.2 Why this ordering

The ordering follows dependency, not latency. The environment provider and
inference backend have no upstream service dependencies — they start first.
The live store and policy registry are coordination infrastructure with no
upstream service dependencies. The rollout manager needs all three: a place to
dispatch episodes (environment provider), generate tokens (inference backend),
and push results (live store). The trainer is last: it depends on the live
store (get_batch) and the policy registry (publish_policy_version), and it
pre-flight-probes the rollout manager before beginning optimizer steps.

### 0.3 Contracts, not implementations

The startup sequence is defined in slot contracts (Sec.5), not in specific
software. VERL is one TrainerAdapter implementation. ProRL is one
EnvironmentProvider implementation. The vLLM child pool is one
InferenceBackend implementation. The in-process TrajectoryStore is one
LiveStore implementation. The trainer's Python int and direct `/reload_lora`
call are one PolicyRegistry implementation.

When a new trainer, environment, or inference backend is plugged in (S5–S8),
the startup sequence does not change — only the concrete script for that step
changes. If adding a new component requires changing the startup sequence or
the slot contracts, the component boundary is wrong.

The test of this claim is S5–S8. Each of those stages swaps one adapter into
a slot without the other four services needing changes.

### 0.4 Smoke-test harness

A minimal harness bootstrapped at S0 and updated incrementally through S4.
It starts all services in order, exercises each contract boundary, and exits
clean. All S0–S4 validation gates use this harness — not full training runs.

```
Smoke-test harness

SETUP
  1. Start all services in startup-sequence order.
     Health-probe each before proceeding to the next.

CONTRACT CHECKS (before trainer starts)
  2. Rollout worker: dispatch 1 episode batch.
        Assert: push_group returns accepted=True.
        Assert: live store reports ≥ 1 group.
        Assert: all token-id fields are list[int], not list[str].
  3. Live store: call get_batch(n_groups=1, current_step=0, staleness_cutoff_k=4).
        Assert: returns ≥ 1 TrainingGroup.
        Assert: behavior_policy_version is an int, not None.
        Assert: group_uid is the same across all n siblings in the group.

TRAINING SMOKE TEST
  4. Trainer: let the trainer run until the first policy publish checkpoint.
        Assert: loss at publish step is finite (not NaN, not inf).
        Assert: no worker thread or subprocess crashes during the run.

POLICY PUBLISH CHECK
  5. Trainer: first policy publish.
        Assert: endpoints_failed == 0 (invariant 3.3).
        Assert: policy registry records the new version and adapter URI.
  6. Rollout worker: subscribe or poll for new version.
        Assert: worker receives the new version within 10 s.

SHUTDOWN
  7. Stop in reverse order: trainer → worker → live store + registry → backend → env.
        Assert: all processes exit with code 0.
        Assert: no orphan processes remain (check with ps / pgrep).
```

**The smoke-test harness runs once — after S4 is structurally complete.**

Stages S1–S3 validate each extracted service in isolation using unit tests
and contract tests, not end-to-end training. Running training at each
intermediate stage is impractical: the system is partially extracted during
S1–S3 and cannot form a complete training loop. Instead, each stage writes
down what to verify when training eventually runs (the "training checklist",
Sec.D.4). After S4, when all five services are wired together, this harness
runs once and works through the full checklist.

Longer training runs (multi-step curves, WandB sweeps) start only after the
S4 harness is green.

---

## 1. Vision

The current ProRL + OpenHands + VERL stack runs end-to-end agentic RL with a
fully-async producer-consumer loop. It works in the form: one Python process,
one Docker container, one trainer machine, one EC2 vLLM pool. That is the
right shape for proving the algorithm. It is the wrong shape for everything
that comes next.

The vision is to turn this stack into a **rollout fabric** — a small set of
stable contracts between independently replaceable services. Many environment
providers generate verifiable, tool-rich episodes. Many rollout managers
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
pop-on-sample. The eager-push seam moves into the rollout-manager slot and
is preserved across the migration; the planner is not authorized to revert
to terminal-push under `filter_groups=True`.

### 3.8 Data ownership boundary — the trainer never holds the dataset

This is the invariant that prevents the trainer from sneaking back into
being the orchestrator after the migration. State this precisely:

> **Training task datasets** belong to the rollout manager. The trainer
> **does not sample raw tasks directly.** The trainer samples
> `TrainingGroup` records from the live store only. The trainer has no
> parquet files, no dataloader, and no knowledge of task IDs beyond the
> provenance fields stamped on each `TrainingGroup`.

Consequences and tests of this invariant:

- The trainer image after migration **does not import a dataloader.** The
  current `data.train_files` Hydra field becomes rollout-manager config.
- The trainer **does not know task IDs.** A `TrainingGroup` carries `task_id`
  and `environment_id`/`environment_version` as provenance, but the trainer
  never resolves a `task_id` back to a raw task description.
- A second trainer (S6 onward) plugs in **without bringing its own dataset
  loader.** It subscribes to the same live store.
- A new environment provider (S5: ROCK, GEM, ORS) brings its own task
  registry and split definitions. The trainer never knows.

**Validation (trainer-requested eval passes) is not part of this invariant
and is explicitly deferred** — see Sec.11. The `data.val_files` field is
removed from the trainer config at S2 and not replaced with an RPC. How
validation is handled in an async decoupled setup is an open problem left
for after S4.

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
different consumers (Sec.6 specifies them in full):

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
`EpisodeRecord`s. Sec.7 makes the property comparison explicit. The summary:
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
| 5.3 | RolloutManager | `ContinuousRolloutProducer` + `AsyncLLMServerManagerDAPO` | Multi-machine producer fleet, mixed-environment producers, partner producers | Reads tasks from EnvProvider; dispatches via InferenceBackend; emits `EpisodeRecord` and `TrainingGroup`; subscribes to PolicyRegistry |
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
trajectory (which lives at the RolloutManager / LiveStore / ReplayArchive).

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

### 5.3 RolloutManager

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
- `get_dataloader_state() / load_dataloader_state()` (for the worker's
  own resume after a crash or restart)

Validation (trainer-requested eval passes against the environment) is
**deferred** — see Sec.11. `pause_production`, `resume_production`, and
`run_validation` are not part of this interface in S0–S4.

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

**Role.** A bounded, low-latency, hot buffer between RolloutManager and
TrainerAdapter. Operates in groups (n siblings together), pops on sample,
evicts by staleness, returns batches sized for the trainer step.

**Minimum interface.**
- `push_group(records: list[TrainingSample], group_uid, producer_id) →
   {accepted, store_size, backpressure_hint}`
- `get_batch(n_groups, current_step, staleness_cutoff_k, timeout_ms) →
   {samples: list[TrainingSample] (unpadded),
    behavior_policy_versions, created_at_steps, sample_ages,
    metrics_pre, metrics_post}`
- `get_metrics(current_step) → store_metrics`
- `notify_policy_version(version, adapter_uri) → ack`

`get_batch` returns **unpadded** `TrainingSample` records (token-id
lists, masks, scalars). Padding, sequence-packing, and any other compute
shape transform are the **trainer adapter's** responsibility — see Sec.6.2's
padding stance. The store does not know tensor shapes.

`get_batch` blocks server-side up to `timeout_ms` if the store has fewer
than `n_groups` non-stale groups. The no-progress detector (invariant
analog from `continuous_producer.py:357-392`) lives inside the store: if
`total_pushes` does not increase for `no_progress_timeout_s`, the call
returns an error. The trainer is no longer responsible for the busy-loop.

**Current adapter.** In-process `TrajectoryStore`: `deque(maxlen=256)`,
single `threading.Lock`, K-staleness eviction. Today's `_pack` (re-pad
to sample-local max) lives inside the store; in the new design that
code migrates to the VERL trainer adapter (per Sec.6.2). The successor
LiveStore returns unpadded records.

**Future adapters.** Colocated gRPC server (Stage 1), Ray actor store,
shared-memory store for same-machine deployments. The hot path is
`get_batch`, which is called once per trainer step; minimizing this
latency is the placement constraint (Sec.8).

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
(Sec.6.1). External-trajectory routing (Sec.10.3 of the old draft, Sec.6.3 here)
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

`request_validation` and `score_validation` are **deferred** — see Sec.11.
Connecting trainer-triggered validation to an async decoupled worker is
deferred until after S4 is stable.

**Current adapter.** VERL `RayPPOTrainerDAPO`. Currently owns the
dataloader; in the migration that ownership moves to RolloutManager (S2),
leaving this adapter focused on optimizer math and policy publication.

**Future adapters.**
- **ROLL**: AsyncController + DeepSpeed/Megatron/FSDP2. Already tracks
  `behavior_policy_version` per sample, supports six off-policy IS
  variants. Likely path: a ROLL-side adapter that consumes `TrainingGroup`
  via the same `get_batch` protocol that VERL uses.
- **slime / Megatron + SGLang**: separates training, rollout, data buffer.
  Its data-buffer abstraction maps onto LiveStore; its rollout module
  maps onto RolloutManager. Plugging slime as a TrainerAdapter requires
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
| `behavior_log_probs` | list[float] OR null | Per-token log-probabilities from the inference backend at generation time. Required for async RL correction; `null` only if the backend cannot provide them (in which case `trust_level` and routing are restricted; see Sec.6.3). |
| `prompt_token_ids` | list[int] | Initial prompt tokens. |
| `response_token_ids` | list[int] | All model-emitted tokens across turns, concatenated. |
| `response_loss_mask` | list[int] (0/1) | 1 on assistant turns; 0 on tool/observation tokens. |
| `tool_calls` | list[ToolCall] | Structured representation of the tool calls made by the agent. |
| `provenance` | dict | `{worker_id, worker_version, dataset_uri, dispatch_time, ...}`. |
| `trust_level` | enum | `own-fabric`, `partner-validated`, `partner-untrusted`, `external-eval-only`. Routing per Sec.6.3 depends on this. |
| `schema_version` | str | EpisodeRecord schema version (semver). |

`events` is the load-bearing field: an EpisodeRecord can always be
reconstructed from its event stream. The flat fields above are summaries
and indexes for query.

**Tokenizer constraint.** The token-in/token-out invariant (Sec.3.1) requires
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
| `behavior_log_probs` | list[float] OR packed bytes | Per-token logprobs from the inference backend at generation time. Default-required (per Sec.4.4 and the algorithm matrix in Sec.6.3). |
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
- `advantage` — computed by trainer adapter (per Sec.4.4).
- `returns` — same.
- `KL penalty / ref_log_probs` — recomputed by FSDP/Megatron actor on
  sample.
- Any per-trainer normalization constants — those belong to the adapter.
- Raw task description / problem statement — that lives at the
  EnvironmentProvider behind `task_id` (invariant 3.8). Trainer never
  resolves `task_id` back to a problem statement.

**Padding stance — trainer-adapter local, not LiveStore:**

The wire schema is **unpadded everywhere** — both on push (S0 today) and
on `get_batch` return (S1+). Padding, sequence-packing, and any other
compute-shape transform are **trainer-adapter responsibilities**, not
LiveStore responsibilities. The store returns raw `TrainingSample`
records; each trainer adapter pads/packs to its own compute shape:

- **VERL / FSDP**: `(B, T_max)` padded tensors, sample-local max.
- **ROLL**: packed sequences with `cu_seqlens` (FlashAttention path).
- **slime / Megatron**: sequence-packed with the framework's expected
  layout.
- **SFT / distillation**: variable-length tokenizer-native, no padding
  required.

Today's `_pack` (`trainer_integration/verl/verl_custom/replay/trajectory_store.py:470-621`)
relocates to the VERL trainer adapter as a local helper (e.g.
`verl_adapter/pad.py`). The LiveStore stops knowing about tensor shapes
entirely. This is option B in the live-store padding decision: it makes
the LiveStore truly adapter-neutral (per principle 4.4) at the cost of
each adapter owning its own padding code.

For reference, the VERL adapter's compute shape **after its local pad**
— **not** what comes off the wire — is:

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
| `raw_reward` | `(B,)` | float32 (NEW, per Sec.6.2) |
| `truncated` | `(B,)` | bool (NEW, per Sec.6.2) |

Other adapters produce different shapes from the same unpadded wire; the
LiveStore guarantees only the wire field inventory and dtypes
(token-id lists, masks, scalars).

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

**Validation samples.** The `split` field (`train` / `val` / `test`) is
a provenance field on `TrainingSample` records. In S0–S4 all live-store
groups come from the `train` split. How val-split rollouts are triggered
and scored in a fully decoupled async setup is **deferred** — see Sec.11.

---

## 7. Live store vs durable replay archive

The two systems described in Sec.5.4 and Sec.5.5 are different products, with
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
(Sec.6.1). `TrainingSample` derivation can run on demand or be pre-cached as
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
| RolloutManager | Mostly (dataloader cursor, inflight episodes) | Low–medium | Near env provider, or sharded across env clusters | Worker should be a coordinator, not a heavy state owner. |
| LiveStore | Yes (hot FIFO buffer) | Yes (`get_batch` is trainer hot path) | Colocate with trainer | Avoid moving large tensor batches over the network in the hot path. |
| ReplayArchive | Yes (long-term data) | No | S3/NFS/Iceberg/Postgres-style storage | Queryability and durability dominate. |
| PolicyRegistry / Coordination | Small (version/manifest registry) | Low | Anywhere reliable and reachable | Cold path except publish events. |
| TrainerAdapter | Yes (optimizer state, FSDP/Megatron shards) | Yes | Dedicated training GPU box/pool | Training and inference scale differently. |

Three implications:

1. **RolloutManager is the easiest slot to fan out.** It is the right
   first place to introduce decentralization (S5).
2. **LiveStore and TrainerAdapter should usually be close** (same box, or
   high-bandwidth interconnect). The hot path is `get_batch`.
3. **InferenceBackend and TrainerAdapter should not be assumed to
   colocate.** At scale, inference is shared infrastructure across many
   trainers (S8).

---

## 9. Stage-wise migration plan

Nine stages. Each stage is defined by Goal, Scope, Invariants preserved,
Validation, and Reversibility.

**S0–S4 are sequential core boundary cuts.** Each depends on the previous.
After S4, the startup sequence from Sec.0 works end-to-end and the Sec.0
smoke-test harness passes clean. No stage runs partially in production while
the next begins.

**Training runs once — after S4, not at each intermediate stage.** During
S1–S3 the system is partially extracted and cannot form a complete training
loop; forcing training mid-migration produces failures that say nothing about
the architecture. S1–S3 validate each extracted service with unit tests and
contract tests in isolation. Each stage also appends items to a plain-text
training checklist (Sec.D.4) recording what to verify when training eventually
runs. The smoke-test harness (Sec.0.4) runs for the first time at S4, working
through the full accumulated checklist.

**S5–S8 are independent pluggability proofs.** They exercise specific slot
adapters without altering the core pipeline. They can be ordered by need,
not by dependency. They open only after S4 is green.

**Each stage is executed with the recursive loop in Sec.D.3.** Update the
progress artifact first, write the failing test first, implement, run the
smoke-test harness, record failures verbatim, fix, rerun until green.

**This document does not prescribe how to extract services.** The planner
chooses transports, sequencing within a stage, deployment topology, and
test strategy. What this section fixes: what each stage proves, what
contracts must be intact at the end, and what the rollback path is.

### S0. Today (no changes) — reference baseline

**Goal.** Reference baseline. The full async loop runs in one process,
one container, one trainer host plus one EC2 vLLM box. **The smoke-test
harness is created at this stage** and anchors all subsequent validation.

**Scope.** No code changes to the training path. One deliverable: the
smoke-test harness script (`tests/harness/smoke_test.sh` or equivalent)
that encodes the Sec.0.4 sequence for the current in-process topology. The
harness must run end-to-end cleanly before S1 begins.

**Stage harness at S0.** Restart ProRL (`:8006`) and the remote vLLM
pool, then start the Docker trainer container. The harness then:
1. Waits for all health probes to pass.
2. Asserts the in-process live store starts empty (`size == 0`, producer
   not yet running).
3. Waits for the producer to push ≥ 1 group (up to 120 s; fails if none).
4. Asserts the pushed group has `token_ids` as `list[int]`, not strings.
5. Asserts `behavior_policy_version` is an int, not None.
6. Asserts the first policy publish returns `endpoints_failed == 0`.
7. Shuts down in reverse order; asserts clean exits.

The harness is the source of truth for "the system is working." All later
stages update it to reflect the new process topology without changing the
contract assertions (steps 3–8 above).

**Invariants preserved.** All of Sec.3, all already.

**Validation.** Already done. The baseline was run against the current
in-process topology: ProRL and vLLM workers restarted, trainer started,
producer pushed groups, policy publish returned `endpoints_failed == 0`,
clean shutdown. Record the step-time and loss curve as the reference
values the S4 harness will be compared against.

**Reversibility.** Trivial — this is the current shape.

### S1. LiveStore behind a network boundary (same machine)

**Goal.** Prove the LiveStore slot is real. The in-process
`TrajectoryStore` is replaced by a same-machine LiveStore service
implementing the Sec.5.4 interface. Trainer and producer both call the
service rather than the in-process object.

**Scope.**
- LiveStore becomes a separate process on the trainer box.
- TrainerAdapter calls `get_batch` over an out-of-process boundary
  (transport TBD by planner — same machine, so localhost gRPC, Ray, or
  shared memory are all plausible).
- RolloutManager (still in trainer process) calls `push_group` over the
  same boundary.
- Re-padding still happens server-side.
- `get_batch`'s server-side blocking and no-progress detector replace
  the current `wait_until_with_progress` busy-loop.
- The wire schema is the Sec.6.2 TrainingSample / TrainingGroup. **Sealing
  the live-path schema is the load-bearing artifact of S1.**

**What does not change.** RolloutManager, EnvironmentProvider,
InferenceBackend, TrainerAdapter (modulo the swap of `self.trajectory_store`
for a client). Data ownership is still in the trainer process; that's S2.

**Invariants preserved.** All Sec.3 invariants. In particular:
- 3.1 token-in/token-out: schema carries token IDs.
- 3.2 group integrity: the wire groups; pop-on-sample preserves it.
- 3.5 per-row `behavior_policy_version`: stamped at push.
- 3.6 pop-on-sample.
- 3.7 eager-push seam: still owned by RolloutManager; the seam is now a
  network call rather than a function pointer.

**Validation — contract tests (no training at this stage).**

Run these tests against the extracted LiveStore service in isolation:
- Service starts and health-probe returns 200.
- `push_group` → `get_batch` round-trip: token IDs on the wire are
  `list[int]`, not strings (invariant 3.1).
- `get_batch` pops: a group pushed once cannot be retrieved twice
  (invariant 3.6).
- Staleness eviction: push a group with `created_at_step=0`, call
  `get_batch(current_step=5, staleness_cutoff_k=4)`; the group is
  evicted (not returned).
- No-progress detector: do not push for `no_progress_timeout_s`; assert
  `get_batch` returns an error, not a hang.
- Kill-restart: stop and restart the service; assert a fresh client
  reconnects and can push/get successfully.
- `behavior_policy_version` is stamped as an `int` on every returned
  `TrainingSample` (invariant 3.5).

**Training checklist items added at S1** (verified at S4):
- [ ] Trainer step time is within X% of S0 baseline with the network
      boundary in place. (Measure at S4; X determined by S0 baseline.)
- [ ] The trainer client reconnects cleanly after a LiveStore restart
      mid-training (kill-restart during an actual training run).
- [ ] No tensor-shape mismatches introduced by the wire serialization
      round-trip (compare output tensors at S4 vs S0).

**Reversibility.** Trivial — keep S0 launcher available; the LiveStore
service is feature-flagged.

### S2. RolloutManager as its own process — data ownership migration

**Goal.** Prove the RolloutManager slot is real **and** that data ownership
moves from the trainer to the worker (invariant 3.8). After S2, the
trainer process does not import `openhands`, does not load any parquet,
and does not know task IDs.

**Scope.**
- RolloutManager becomes a standalone process. It owns:
  - the SkyRL-v0-293 train parquet files (val split is not exercised — validation is deferred per Sec.11),
  - the `StatefulDataLoader` and its checkpoint,
  - the `AsyncLLMServerManagerDAPO` (or successor) dispatch logic,
  - producer-side filters (zero-variance drop, length cuts),
  - the eager-push seam (Sec.3.7) — now a network call to LiveStore.
- TrainerAdapter no longer instantiates `ContinuousRolloutProducer`,
  `AsyncLLMServerManagerDAPO`, the DAPO manager, or the `_push_fn`
  closure. It only calls `get_batch`.
- The launcher splits: `s3_fullasync_docker.sh` becomes one launcher per
  service (planner picks names).
- Trainer dependencies removed: `openhands`, `aiohttp`, `fastapi`,
  `uvicorn`, `async_generator` — none are training-time concerns.
- `data.train_files` and `data.val_files` Hydra fields are removed from
  the trainer config entirely. The trainer has no parquet files.
  **Validation is not replaced by an RPC at this stage — it is deferred
  (see Sec.11).**
- PolicyRegistry exists in minimal form: the trainer keeps direct pool
  publish (no change), and writes the current `policy_version` plus
  `adapter_uri` to a small registry the worker polls. The registry
  contract is sealed; the registry implementation is minimal.

**What does not change.** EnvironmentProvider, InferenceBackend,
LiveStore (still S1's same-machine service), TrainerAdapter math.

**Invariants preserved.** Critically 3.8 (data ownership) and 3.7
(eager-push seam). Also 3.5 (`behavior_policy_version` stamping moves to
worker, but stamping happens at push, exactly as today).

**Validation — contract tests (no training at this stage).**

Run these tests against the extracted RolloutManager process in isolation:
- Worker starts, health-probes, and registers with the LiveStore (S1
  service); live store metrics show the producer_id active.
- Worker pushes ≥ 1 group to LiveStore within 120 s of starting. Assert
  token IDs are `list[int]` (invariant 3.1) and `group_uid` is the same
  across all n siblings (invariant 3.2).
- `behavior_policy_version` on pushed groups is an `int` stamped by the
  worker, not None (invariant 3.5).
- `filter_easy_hard_instance` operates on the worker side: inject a
  synthetic zero-variance group; assert it does not appear in LiveStore
  (invariant 3.7).
- Image check: `docker inspect` the trainer image; assert `openhands`,
  `aiohttp`, `fastapi`, `uvicorn` are absent.
- Config check: trainer Hydra config has no `data.train_files` or
  `data.val_files` fields and imports no dataloader.
- Isolation check: start worker + LiveStore only (no trainer); verify
  the worker continues pushing; groups accumulate in the store.

**Training checklist items added at S2** (verified at S4):
- [ ] Trainer starts from `get_batch` on the network boundary, not from
      a local dataloader — confirm by checking no parquet I/O in trainer
      process during a training run.
- [ ] Worker independence: kill the trainer mid-run; the worker keeps
      generating (live store push-count increases); trainer restart picks
      up where it left off via the live store.
- [ ] `behavior_policy_version` on groups consumed by the trainer matches
      the version the worker stamped — no off-by-one across the network
      boundary (row-level invariant 3.5).

**Reversibility.** Higher cost than S1 — the trainer image shrinks and
the dataloader migrates. Roll back by reverting both image changes and
launcher changes; not a flag flip. The planner specifies a hold-period in
which both paths run side-by-side (one trainer reads from live store,
the other reads from local parquet) so a regression can fall back.

### S3. ReplayArchive — durable canonical record

**Goal.** Prove the ReplayArchive slot is real. Every episode the worker
produces is teed to the archive in canonical `EpisodeRecord` form (Sec.6.1).
The archive is queryable.

**Scope.**
- RolloutManager writes `EpisodeRecord` to the archive per completed
  episode (irrespective of producer-side filter — the archive sees
  everything; the live store sees only filter survivors).
- The archive supports `append_episodes` and `query`. The minimum query
  predicate set: by `policy_id`, `policy_version`, `environment_id`,
  `split`, and time range. `derive_training_samples` may be deferred.
- Trust and provenance fields (Sec.6.1) are populated.

**What does not change.** Hot path. Trainer never reads the archive. Live
store unchanged.

**Invariants preserved.** All Sec.3. Archive is a tee, not a bottleneck —
push to archive is async-fire-and-forget from the worker's perspective
(planner specifies durability semantics: at-least-once vs at-most-once
etc.).

**Validation — contract tests (no training at this stage).**

Run these tests against the extracted ReplayArchive in isolation, then
together with the RolloutManager from S2:
- Archive starts and health-probes.
- `append_episodes` → `query` round-trip: insert a synthetic
  `EpisodeRecord`; query by `(policy_id, environment_id)`; assert the
  record is returned.
- Schema check: the record retrieved has `episode_uid`, `token_ids` as
  `list[int]`, `trust_level`, and all required provenance fields from Sec.6.1.
- Tee check: start Worker + LiveStore + Archive together; let the worker
  produce 1 group; assert (a) the group appears in LiveStore and (b) the
  pre-filter EpisodeRecord (including episodes dropped by
  `filter_easy_hard_instance`) appears in the Archive. The live store must
  not contain the filtered episode.
- Offline consumer check: a standalone script queries the archive and
  computes a reward histogram; it does not touch the live store.

**Training checklist items added at S3** (verified at S4):
- [ ] After a 2-step training run, the archive contains ≥ 1 EpisodeRecord
      with all provenance fields populated (policy_id, version, env_id,
      token_ids as list[int]).
- [ ] Filtered episodes (zero-variance groups) appear in the archive but
      not in the live store — confirm via archive query count vs live
      store push-count metric after a real training run.

**Reversibility.** Trivial — disable the tee.

### S4. PolicyRegistry / Coordination as the single source of truth

**Goal.** Prove the PolicyRegistry slot is real. The trainer publishes
versions to the registry; the registry fans out to InferenceBackend
(`/reload_lora`), LiveStore (notification), and RolloutManager
(subscription) atomically. The trainer no longer talks directly to the
pool.

**Scope.**
- TrainerAdapter calls `publish_policy_version(version, adapter_uri,
  policy_id)` instead of POSTing `/reload_lora` directly.
- Adapter binaries move to durable storage (local FS, NFS, or S3 — planner
  picks; the registry stores the URI).
- The registry preserves the abort gate (invariant 3.3): if any pool
  child fails, the publish call returns failure and the trainer aborts.
- RolloutManager subscribes to version updates instead of polling.

**What does not change.** Hot path between worker → live store → trainer.
The pool itself (vLLM child) is unchanged — it still serves
`/reload_lora` and `/v{N}/generate`.

**Invariants preserved.** 3.3 (abort gate), 3.4 (pinning), 3.5 (per-row
version stamping moves but is not changed semantically).

**Validation — contract tests, then the first training run.**

Contract tests (run before training):
- Registry starts and health-probes.
- `publish_policy_version` → `get_latest_version` round-trip: published
  version is the one returned.
- Abort gate: kill one vLLM child process, then call
  `publish_policy_version`; assert `endpoints_failed > 0` and the
  trainer halts (invariant 3.3).
- Subscription latency: worker receives version update from the registry
  within the target bound (planner specifies; suggested 5 s locally).
- Adapter URI manifest: after publish, query the registry; assert the
  URI is a resolvable path.

**First training run — smoke-test harness (2 steps).** After contract
tests pass, run the Sec.0.4 harness for the first time against the full
five-service topology. Work through every item on the accumulated training
checklist from S1, S2, and S3:
- Loss at the first publish checkpoint is finite (not NaN, not inf).
- `endpoints_failed == 0` on publish (invariant 3.3).
- Token IDs across the full network path are `list[int]`, not strings
  (invariant 3.1).
- `behavior_policy_version` per row matches the version the worker stamped
  (invariant 3.5).
- Groups consumed by the trainer have all n siblings present (invariant
  3.2) — no split groups across the wire.
- Worker receives the published version within the latency target.
- Archive contains ≥ 1 EpisodeRecord with full provenance fields.
- All five services shut down cleanly (exit 0, no orphans).
- All S1, S2, S3 training checklist items checked off.

Step time is compared against the S0 baseline. The delta is the cost of
the network boundaries added in S1–S4. If it exceeds the planner's
threshold, investigate before opening S5.

Longer training runs (multi-step curves, WandB sweeps) start after this
harness is green.

**Reversibility.** Roll back to S3 by re-enabling direct pool publish on
the trainer and disabling the registry subscription on the worker.
Higher-cost rollback than S2 because the publish path moves.

---

### Pluggability proofs — S5 through S8

S5–S8 are independent. Each proves one slot is genuinely replaceable by
swapping in a second adapter without touching the other four services or the
core pipeline. They open only after S4 is green. They can run in any order.
Each still uses the smoke-test harness (updated to route a subset of traffic
to the new adapter) as the primary validation gate.

---

### S5. Multi-producer with heterogeneous EnvironmentProvider adapters

**Goal.** Prove the EnvironmentProvider slot is real and the
RolloutManager slot is fan-out-able. Run two RolloutWorkers in parallel,
each pointed at a different EnvironmentProvider adapter (e.g. ProRL +
GEM, or ProRL + ORS, or ProRL + a partner). Both push to the same
LiveStore.

**Scope.**
- A second EnvironmentProvider adapter is implemented (planner picks the
  first concrete second adapter — most likely GEM or a stripped-down ORS
  test environment).
- A second RolloutManager instance is launched against the second
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
- An SGLang adapter implements the Sec.5.2 interface plus a pinning
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

**Invariants preserved.** All Sec.3. The aggregation service's output is
itself a publish, so 3.3 (abort gate) and 3.4 (pinning) extend naturally.

**Validation.** Long-running. The criterion is qualitative: "the
architecture composes; adding a third trainer or a second region is a
bounded operational task, not a redesign."

**Reversibility.** Per-trainer; disable aggregation, fall back to
single-trainer publish.

### Stage map

```
═══════════════════════════════════════════════
  SEQUENTIAL CORE BOUNDARY CUTS (S0 → S4)
  Each depends on the previous.
  Smoke-test harness gates every stage.
═══════════════════════════════════════════════

             S0  reference baseline
                 └─ smoke-test harness created here
                    │
                    ▼
             S1  LiveStore extracted as a service
                 └─ live-path wire schema sealed
                    │
                    ▼
             S2  RolloutManager extracted as a service
                 └─ data ownership leaves the trainer
                    │
                    ▼
             S3  ReplayArchive wired as a tee
                 └─ every episode durable from here
                    │
                    ▼
             S4  PolicyRegistry is single source of truth
                 └─ Sec.0 startup sequence fully operational

═══════════════════════════════════════════════
  INDEPENDENT PLUGGABILITY PROOFS (S5 – S8)
  Any order. Open only after S4 is green.
  Prove each slot is replaceable.
═══════════════════════════════════════════════

     S5           S6           S7            S8
 multi-worker    alt-trainer  alt-backend  federation
 (heterog envs)  (ROLL/slime) (SGLang)     (multi-trainer
                                            + aggregation)
```

S5–S8 are independent proofs that the slot model is real. They do not need
to run in order. Each swaps one adapter into one slot and verifies the other
four services are unaffected.

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
   (Sec.5.1, Appendix A); the planner exposes ProRL's existing operations
   as adapter methods, not the other way around.

3. **Token-in/token-out invariant** —
   `openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py`. Unchanged. Wire
   schemas store token IDs (Sec.3.1).

4. **Chat template logic** — `chat_template_manager.py`. Unchanged.

5. **`filter_easy_hard_instance` (zero-variance drop)** — stays
   producer-side. The store never sees zero-variance groups (Sec.3.7).

6. **GRPO/DAPO advantage computation** — stays trainer-side
   (per principle 4.4). Whole-group integrity (Sec.3.2).

7. **Temporal IS correction** — `core_algos.py:664-690` stays trainer-side.
   `behavior_log_probs` come from the wire (Sec.6.2); `old_log_prob`
   recomputed by the FSDP actor.

8. **Reward computation** — stays trainer-side via the reward manager;
   `reward_fn` is a Ray remote.

9. **`_save_checkpoint` and FSDP/Megatron actor weights** — stay
   trainer-side. The trainer adapter owns optimizer state.

10. **Producer-side filter location.** `filter_easy_hard_instance` is
    cheap (one boolean check per group) and saves ~30% of push bandwidth
    (typical zero-variance rate on SkyRL-v0). It stays in the
    RolloutManager.

---

## 11. Non-goals

The following are intentionally **not** in scope of this design document.
Some are explicit non-goals (won't happen). Others are deferred to the
planner.

**Explicitly deferred — trainer-triggered validation:**

Connecting trainer-triggered evaluation passes (the trainer asking the
rollout manager to run a val split and return scored groups) to an async
decoupled setup is a hard scheduling problem: the trainer must pause or
interleave production rollouts, the worker must switch task splits, and
results must be routed back to the trainer in a form the FSDP actor can
consume without a separate forward pass across a network boundary. Getting
this wrong is a common source of silent failures and unexpected latency
spikes.

**Validation is therefore out of scope for S0–S4.** Specifically:
- `run_validation` is not on the RolloutManager RPC surface.
- `request_validation` and `score_validation` are not on the TrainerAdapter
  interface.
- `pause_production` and `resume_production` are not implemented.
- `data.val_files` is removed from trainer config at S2 and not replaced.
- The 23-instance val set (SkyRL-v0-293) is not exercised by any S0–S4
  validation gate.

The smoke-test harness (Sec.0.4) covers training correctness (finite loss,
group integrity, token IDs, publish abort gate) without needing val rollouts.

Validation design resumes after S4. At that point, the service boundaries
and the async scheduling model will be clear enough to reason about the
problem without speculative abstractions.

**Explicit non-goals (won't happen in this document):**
- Authentication, authorization, mTLS, per-producer API keys.
  In-cluster trust is assumed (same VPC / security group). External
  partner contributions are gated by `trust_level` routing (Sec.6.3) but
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
  see Sec.12 question 4).
- Service extraction order *within* a stage. The stages here say what
  must be true at the end of each; the planner chooses how to get there.
- Deployment topology for S5–S8 (multi-region, K8s manifests, autoscale
  rules).
- Cost / capacity / throughput numbers — none have been measured against
  the new shape; the planner specifies a benchmark plan.
- Exact adapter mapping for ROCK / GEM / ORS environment integrations
  beyond the Sec.5.1 interface — the planner specifies the first concrete
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
   the RolloutManager for validation groups. Open question: does the
   worker need to **pause production** during validation (shared
   OpenHands session) or run validation on a **dedicated session**
   (separate ProRL port, no coordination)? Recommendation: dedicated
   val session for simplicity, especially since the 23-instance val
   set is small. Planner verifies and decides.

6. **Dataloader state under producer ownership (S2).** When the
   RolloutManager owns the dataloader, it owns the
   `state_dict()`/`load_state_dict()` pair. On worker restart it loads
   the last checkpoint to avoid prompt repetition. Open question:
   where is the worker's checkpoint stored? Trainer-attached disk?
   Shared NFS? Object store? Recommendation: trainer-attached for S2
   (simplest); migrate when S5 makes the worker fan-out.

7. **External / partner trajectory routing.** Trust levels and routing
   matrix are sketched in Sec.6.3, but the planner specifies the
   per-trainer-adapter routing rules (which trust levels each adapter
   admits, what filtering happens at LiveStore ingest, etc.).

8. **PolicyRegistry transport in S4.** Server-streaming gRPC for
   subscriptions, HTTP long-poll, or Redis pub/sub? The publish event
   rate is low (one per `save_freq` steps, ~1/min); any reasonable
   choice works.

9. **Adapter storage location in S4+.** Local filesystem (single-machine
   simplicity), NFS (multi-machine simplicity), S3 (multi-region).
   Planner picks per deployment.

10. **`TrainingSample` / `TrainingGroup` schema versioning.** Sec.6.2
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

14. **Aggregation algorithm in S8.** Out of scope per Sec.11, but the
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

### A.3 RolloutManager

```python
class RolloutManager(Protocol):
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

    # RPC surface for the trainer (S0–S4 scope).
    # pause_production / resume_production / run_validation are deferred (Sec.11).
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

    # request_validation and score_validation are deferred (Sec.11).
    # Trainer-triggered async validation requires solving scheduling,
    # flow control, and result routing across decoupled services — left
    # for after S4 is stable.
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
| `trainer_integration/verl/verl_custom/replay/continuous_producer.py` | 5.3 RolloutManager | Migrates to the worker process at S2. Daemon thread becomes a service main loop. |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` | 5.3 RolloutManager | Migrates. The DAPO eager-push seam (Sec.3.7) lives here. |
| `trainer_integration/verl/verl_custom/replay/trajectory_store.py` | 5.4 LiveStore | The data structure stays; the process boundary changes at S1. |
| (none today) | 5.5 ReplayArchive | New at S3. |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`, `ray_trainer_dapo.py` | 5.6 TrainerAdapter | Stays as the VERL adapter. Loses dataloader and direct rollout-mgr at S2; loses direct pool publish at S4. Validation logic removed (deferred per Sec.11). |
| `_publish_lora_adapter` in `ray_trainer.py` | 5.7 PolicyRegistry (today) | Migrates to PolicyRegistry at S4. |
| `s3_fullasync_docker.sh` launcher | (cross-cutting) | Splits across stages: into worker launcher, trainer launcher, store launcher, registry launcher per the planner's choice. |
| `data.train_files` Hydra config | 5.6 today, 5.3 after S2 | Moves from trainer config to worker config. |
| `data.val_files` Hydra config | same | Removed from trainer config at S2. Not replaced by an RPC — validation is deferred (Sec.11). |

---

## Appendix C — Mapping to external frameworks

For each external framework: which slot it would implement, what its
adapter would have to do.

| Framework | Slot(s) | Adapter notes |
|---|---|---|
| **ROCK** (Alibaba) | 5.1 EnvProvider | ROCK is sandbox provisioning + GEM-compatible env interface. Adapter wraps ROCK's `Sandbox` lifecycle and `rock.make()` / `reset()` / `step()` to the Sec.5.1 five-method interface. State: ROCK's Admin/Worker/Rocklet topology is internal to the adapter; the fabric only sees `EnvironmentProvider`. |
| **GEM** (axon-rl) | 5.1 EnvProvider | GEM is the standard agentic-LLM Gymnasium. Adapter wraps `make/reset/step` (with text observations and tool-call actions) to the Sec.5.1 interface. GEM's tool wrappers become the `act(ToolCall)` argument's input dict. |
| **ORS / OpenReward** | 5.1 EnvProvider | HTTP+SSE protocol. Adapter is essentially a thin HTTP client mapping the Sec.5.1 methods to ORS endpoints (`/tasks` → `list_tasks`, `/create_session` → `create_episode`, `/{env}/prompt` → `get_prompt`, `/{env}/call` → `act`). Streamed reward updates from `/{env}/call` aggregate into `StepResult.reward`. |
| **vLLM** (current) | 5.2 InferenceBackend | Already implemented (`_vllm_child.py`). Pinning swap protocol is the reference for the Sec.5.2 pinning constraint. |
| **SGLang** (and sgl-router) | 5.2 InferenceBackend | Adapter wraps SGLang's generation API plus router. Must verify per-token logprobs and a pinning-equivalent guarantee. slime uses SGLang as its inference module, so an SGLang adapter unlocks easier slime trainer plug-in. |
| **TGI / TRT-LLM / hosted APIs** | 5.2 InferenceBackend | Adapters as needed. Hosted APIs without per-token logprobs are usable for eval-only flows (matches `trust_level=external-eval-only` routing). |
| **VERL `RayPPOTrainerDAPO`** (current) | 5.6 TrainerAdapter | Already implemented. Loses dataloader at S2 per invariant 3.8. |
| **ROLL** (Alibaba) | 5.6 TrainerAdapter | Async controller + DeepSpeed/Megatron/FSDP2. Already first-class on `behavior_policy_version` and supports six off-policy IS variants. Adapter consumes `TrainingGroup` via `get_batch`, computes ROLL's flavor of advantages and IS correction, publishes via PolicyRegistry. ROLL's `SampleBuffer` is replaced by the fabric's LiveStore. |
| **slime** (THUDM) | 5.4 + 5.6 (paired) | slime separates training, rollout, and data buffer. Its data-buffer concept is closest to LiveStore; its training module (Megatron-based) is the trainer adapter; its rollout module is conceptually the RolloutManager. Plug-in: replace slime's data buffer with a LiveStoreClient; replace its rollout module with a RolloutWorkerClient. The Megatron training module becomes the adapter. |
| **Aggregation services** (FedAvg etc.) | 5.7 (extension) | S8 only. Operate on adapter URIs from the PolicyRegistry. Out of scope for this document. |

---

## Appendix D — Execution discipline (skills, teams, progress)

This document is contract-first; the planner and the execution agent should
treat the migration as a sequenced set of cuts, not a single rewrite. This
appendix names the skills, the team shape, and the running progress
artifact that keep S0–S8 tractable.

### D.1 Most-needed skills

These Claude Code skills are available on this repo. Activate by name when
relevant — the table is a directory, not a checklist.

| Skill | Role in this work |
|---|---|
| `repo-architecture` | Orient before the first edit in any module; map slot ↔ files via Appendix B; avoid guessing in cross-cutting subsystems. |
| `karpathy-guidelines` | Surgical changes; surface assumptions; verifiable success criteria; no speculative abstractions. |
| `strategic-compact` | Compact at stage boundaries (e.g. S1 → S2) to keep long sessions tractable without losing invariant context. |
| `tdd-workflow` | Author Protocol contract tests (Appendix A) before implementing each slot adapter. |
| `python-testing` | Pytest fixtures, markers (`integration`/`slow`/`real_data`), mocking at adapter boundaries, coverage targets. |
| `python-patterns` | Type hints, dataclasses, asyncio idioms for new slot services. |
| `api-design` | Wire-schema versioning (`schema_version`), Protocol method shapes, query semantics for ReplayArchive (Sec.5.5). |
| `documentation-lookup` | Live API docs via Context7 for external adapters (ROCK, GEM, ORS, SGLang, ROLL, slime). |
| `eval-harness` | Codify each per-stage `Validation` gate as a pass/fail eval; gate stage close on it. |
| `verification-loop` | End-of-stage close-out: `make lint`, fast pytest loop, coverage report. |
| `security-review` | Trust-level routing review at S5+ when partner trajectories arrive (per Sec.6.3). |

The four context-management skills (`repo-architecture`,
`karpathy-guidelines`, `strategic-compact`, `documentation-lookup`) are the
ones most likely to determine whether a stage lands cleanly or sprawls.
They keep the working set small and bounded across the seven slots.

### D.2 Agent-team shape — track components, not turns

The slot model has **seven components** (Sec.5). The natural execution shape
mirrors that: a **team lead** coordinates the migration; **teammates** own
per-slot adapter work and challenge each other's slot interactions. This
is the [agent-teams](https://code.claude.com/docs/en/agent-teams#start-your-first-agent-team)
pattern, not subagent delegation — teammates have their own context
windows and message each other directly.

Why use a team here:

- **Per-slot ownership.** Keeping each slot's invariants resident in a
  dedicated teammate's working memory is the tightest fit to the
  architecture: invariants 3.1 (token-in/out) and 3.4 (pinning) live with
  EnvProvider/InferenceBackend teammates; 3.2 / 3.6 / 3.7 live with
  RolloutManager / LiveStore teammates; 3.3 / 3.5 live with TrainerAdapter
  / PolicyRegistry teammates; 3.8 (data ownership) is the lead's
  cross-cutting responsibility.
- **Parallel investigation.** Sec.12's open questions (transport, storage,
  schema migration timing, archive ingest semantics) benefit from
  independent exploration before convergence. Teammates with explicit
  adversarial roles surface failure modes a single session under-explores.
- **Bounded coordination cost.** Tokens scale linearly with active
  teammates. Use 3–5 teammates per stage; not all seven slots are active
  in any single stage. S1–S4 are sequential cuts; S5–S8 are parallel
  proofs, where teams are most valuable.

Bootstrap sequence:
1. **Planning team** (3 teammates: architect, reviewer, devil's-advocate)
   to resolve Sec.12 open questions per stage before implementation begins.
2. **Per-stage execution team** scoped to the slots that stage touches
   (e.g. S1 = LiveStore + TrainerAdapter teammates; S6 = TrainerAdapter +
   PolicyRegistry + RolloutManager teammates).
3. **Cleanup discipline.** Per the agent-teams contract, only the lead
   runs cleanup; teammates shut down on request before the lead cleans up.

### D.3 Recursive implementation loop — how to execute a stage

Each stage S in S0–S8 runs the same recursive loop. The agent does not move
to S(n+1) until this loop exits green for S(n).

```
STAGE LOOP for stage S:

1. OPEN the progress artifact (plans-n-solutions/rollout_fabric_progress.md).
   a. Mark stage S status → "in progress".
   b. Record the opening commit SHA in Notes.
   c. Copy the Goal verbatim from Sec.9.
   d. Expand the task list into implementation checkboxes.

2. WRITE TESTS FIRST.
   For each validation gate in Sec.9.S, author the test or harness assertion
   before writing production code.
   For each Sec.3.x invariant the stage touches, author or extend the
   corresponding invariant test in tests/invariants/.

3. IMPLEMENT.
   Work through tasks one checkpoint at a time. Mark each checkbox done
   when the test for that task passes. Do not expand scope beyond the stage.

4. RUN TESTS.
   For S1, S2, S3: run the stage's contract tests and unit tests in
   isolation (not a training loop — the system is partially extracted and
   cannot form a complete training loop at these stages).
   For S4: run the full smoke-test harness (Sec.0.4), including the
   accumulated training checklist from S1–S3. This is the first training
   run after S0.

   a. If tests pass:
        Record "tests: green @ <timestamp>" in Notes.
        Go to step 5.
   b. If tests fail:
        Record the failure verbatim in the Errors section of the progress
        artifact (include: timestamp, error message, stack trace if short).
        Fix the root cause.
        Go back to step 4.
   c. If looping ≥ 3 times without green tests:
        Stop. Describe the stuck point to the user before continuing.

5. RUN FAST PYTEST LOOP.
   pytest -m "not integration and not slow and not real_data" tests/ -q
   a. If all pass: go to step 6.
   b. If any fail: record failure in Errors, fix, go back to step 4.

6. CHECK COVERAGE.
   Run coverage on the new slot's module.
   Minimum: 80% overall; 100% for reward-scorer, message-format,
   serialization, and token-handling paths.
   If below threshold: add tests, go back to step 5.

7. MARK GREEN.
   a. Check all validation-gate checkboxes in the progress artifact.
   b. Check all invariant-test checkboxes.
   c. Set stage status → "green".
   d. Record the closing commit SHA in Notes.
   e. Update the artifact header: active stage → S(n+1), schema_version
      if changed, last harness run result.

8. VERIFY REVERSIBILITY.
   Start the S(n-1) launcher; confirm it runs cleanly.
   Record the rollback note in the progress artifact.
   Only then open stage S(n+1).
```

**Rules:**
- Never skip step 2 (write tests first). If time pressure is the reason,
  that is a scope problem — reduce the stage, do not skip the test.
- Never skip step 8 (reversibility). A stage that cannot be rolled back
  is not green.
- Never start S(n+1) while S(n) is "in progress". The progress artifact
  enforces this — there is no "partially green" status.
- Steps 4 and 5 iterate until both are simultaneously green. A green step 5
  after a red step 4 is not sufficient.

### D.4 Progress artifact — format and discipline

Maintain a running progress artifact at
`plans-n-solutions/rollout_fabric_progress.md`. It is the migration's
single source of truth for "where are we now." If it disagrees with chat
or PR descriptions, the artifact is correct by construction.

**Artifact header** (always visible at the top):

```markdown
# Rollout Fabric Migration — Progress

**Active stage:** S{n}
**Schema version:** v{x} (description of shape)
**Policy version anchor:** <commit-sha>
**Last training run:** S0 baseline | S4 harness — pass | fail | not yet run
**Open training checklist items:** {count} (from S1+S2+S3; resolved at S4)
```

**Per-stage entry format:**

```markdown
## S{n} — {Stage name}

**Status:** not started | in progress | green | rolled back
**Opening commit:** <sha>
**Closing commit:** <sha> (when green)

### Goal
{Copied verbatim from Sec.9.}

### Tasks
- [ ] {Task description}

### Validation gates
For S1–S3: contract and unit tests (no training).
For S4: smoke-test harness + full training checklist.
- [ ] {Gate from Sec.9.S{n} Validation}

### Invariant tests
- [ ] 3.{x} {invariant name} — {test file and line}

### Training checklist
<!-- Items added during this stage that must be verified when training
     runs for the first time (at S4). Plain text. Never delete entries.
     Format: "what to check" — "how to check it". -->
- {what to check} — {how to verify at S4}

### Errors
<!-- Record every test or harness failure verbatim during the recursive
     loop. Never delete. Format: [timestamp] description + resolution. -->

### Rollback notes
<!-- What state to restore, and how, if this stage must be rolled back.
     Filled in at step 8 of the recursive loop before opening S{n+1}. -->

### Notes
<!-- Commits, PRs, decisions on Sec.12 open questions, transport and storage
     choices, any other decisions made during this stage. -->
```

The **training checklist** accumulates across stages. Before opening the
S4 harness run, collect all checklist items from S1, S2, S3 into a single
checklist at the top of the S4 section. The harness run works through each
item and checks it off. An unchecked item at the end of the S4 harness run
is a blocking failure — the system is not green until every item is resolved.

**Discipline:**
- Update on every meaningful checkpoint, not only at end-of-stage. A
  merged PR, a failed test, a rollback decision, and a Sec.12 open-question
  resolution all warrant an update.
- Never delete entries; mark them done. The artifact doubles as the
  migration's audit log.
- A teammate joining mid-migration must be able to orient in under 60
  seconds from the artifact header alone.

**Skeleton — paste at the start of S0:**

```markdown
# Rollout Fabric Migration — Progress

**Active stage:** S1
**Schema version:** v0 (current in-process TrajectoryStore shape)
**Policy version anchor:** <starting-commit-sha>
**Last training run:** S0 baseline — pass
**Open training checklist items:** 0 (grows as S1–S3 run)

## S0 — Reference baseline

**Status:** green
**Opening commit:** <sha>
**Closing commit:** <sha>

### Goal
Reference baseline. Full async loop runs in one process, one container.
Smoke-test harness is created at this stage.

### Tasks
- [x] Harness script created at tests/harness/smoke_test.sh
- [x] Harness runs end-to-end cleanly (restart ProRL + vLLM workers, start trainer)
- [x] Harness asserts token IDs are list[int], not list[str]
- [x] Harness asserts behavior_policy_version is int
- [x] Harness asserts endpoints_failed == 0 on first publish
- [x] Harness shuts down cleanly (all exit 0)

### Validation gates
- [x] Harness runs cleanly against S0 in-process topology
- [x] S0 step time, loss, and endpoints_failed recorded as reference values

### Invariant tests
- [ ] 3.1 token-in/out — tests/invariants/test_token_ids.py
- [ ] 3.3 endpoints_failed abort — tests/invariants/test_publish_abort.py
- [ ] 3.5 behavior_policy_version per-row — tests/invariants/test_version_stamp.py
- [ ] 3.6 pop-on-sample — tests/invariants/test_pop_on_sample.py

### Errors
<!-- none yet -->

### Rollback notes
N/A — this is the baseline; no prior stage to roll back to.

### Notes
Starting commit: <sha>.
```

### D.5 Target file layout — modularity over legacy

The current repo layout is a historical artifact. Slot 5.1 internals
sit under `openhands/` and `openhands/nvidia/`; the live store, producer,
and trainer customizations sit under `trainer_integration/verl/verl_custom/`;
serving lives under `scripts/serving/`; tests are organized by today's
process topology. None of that layout is a contract. The slot model
(Sec.5) is.

The execution agent **is authorized** to define a target file layout
that aligns with the slot model, and to migrate code into it
stage-by-stage. The migration **is not** a single rename PR; each move
lands with its slot's stage cut and is verified by the same `Validation`
gate that proves the slot.

Suggested target shape (planner picks the names):

```
openhands_env_provider/   # slot 5.1 — ProRL adapter today
inference_backend/        # slot 5.2 — vLLM child + future SGLang/TGI/...
rollout_manager/           # slot 5.3 — was continuous_producer + async_server_dapo
live_store/               # slot 5.4 — was trajectory_store + client/server split
replay_archive/           # slot 5.5 — new at S3
trainer_adapters/
  verl/                   # slot 5.6 — was trainer_integration/verl/...
  roll/                   # slot 5.6 — added at S6
  slime/                  # slot 5.6 — added at S6
policy_registry/          # slot 5.7 — was _publish_lora_adapter, new home at S4
schemas/                  # Sec.6 wire schemas, single source of truth
tests/
  invariants/             # Sec.3.1–Sec.3.8 regression gates
  contracts/              # Sec.A.1–Sec.A.7 protocol contract tests
  slots/                  # per-slot internal tests
```

**Constraints — what does NOT move (per CLAUDE.md):**
1. **Token-in/out files** (`openhands/llm/nvidia/qwen3.py`,
   `qwen2_5_vl.py`) stay at their current path. They own invariant 3.1
   by location and are explicitly off-limits.
2. **Frozen siblings** (`scripts/_internal/s2_weightsync_docker.sh`,
   `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh`)
   stay where they are — they are a matched lock-step A/B baseline. Make
   new siblings; do not move or rename these.
3. **Upstream OpenHands tree** (`openhands/` excluding `openhands/nvidia/`)
   stays largely intact so upstream merges are tractable. Slot 5.1 is
   the ProRL **adapter** wrapping OpenHands, not a re-layout of
   OpenHands itself.

Outside those constraints, default to the slot-aligned layout. The doc
takes the position that **a clean target structure is worth the cost**
of moving files — it is part of what proves the slot model is real.

The `git mv`-vs-`git rm + git add` decision (preserve history vs clean
break) is the planner's call per move; both are acceptable. The
progress artifact (Sec.D.3) records every move as a Note on the relevant
stage so reviewers can trace history.

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
