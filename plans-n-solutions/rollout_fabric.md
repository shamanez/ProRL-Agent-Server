# Rollout Fabric — Design Rationale

This document explains **why** each design decision was made: the boundary conditions,
the slot contracts, the wire schemas, and the pluggability principles.

For **what** is running and **how** to operate it, see `CLAUDE.md` (startup sequence,
invariants) and `rollout_fabric_progress.md` (SIF build, rescue team).

---

## Boundary Conditions

Every BC maps to a testable invariant. Violations corrupt training silently or cause
crashes within 2 steps — neither is acceptable.

| BC | Boundary | Failure signature if violated |
|---|---|---|
| BC-0 | One `PolicyVersionSnapshot` per group dispatch | Siblings see different versions → invalid advantage computation → NaN loss |
| BC-1 | Token IDs as `int` on every wire (packed int32-LE in proto) | KL/entropy NaN within 2 training steps |
| BC-2 | `push_group` is atomic — 0 or N records, never partial | Partial group → wrong advantage denominator → NaN |
| BC-3 | `get_batch` pops under lock before sending (pop-on-sample) | Duplicate training on same rollout → policy collapse |
| BC-4 | `get_batch` blocking predicate is `num_fresh_groups`, not `num_groups` | Stale groups returned → `InsufficientTrajectoriesError` wedge |
| BC-5 | No-progress detector fires after `no_progress_timeout_s` (default 1800s) | Trainer hangs forever when RolloutManager is wedged |
| BC-6 | `eager_pushed_all` flag prevents double-push within the worker | Double-push corrupts `behavior_policy_version`/`created_at_step` |
| BC-7 | Worker polls policy version at ≤1s latency | IS correction uses wrong version → silent gradient bias |
| BC-8 | `created_at_step` comes from registry `trainer_step` field | Severely stale steps → all groups evicted immediately |
| BC-9 | `endpoints_failed > 0` = hard abort (not degraded mode) | Mixed-version pool → IS weights become lies |
| BC-10 | vLLM pinning protocol unchanged across adapter swaps | Mid-trajectory policy swap → `behavior_policy_version` meaningless |
| BC-11 | LiveStore wire is unpadded; trainer pads locally after `get_batch` | `torch.stack` shape mismatch → crash at first training step |
| BC-12 | Archive tee is pre-filter; LiveStore push is post-filter | Filtered groups lost from archive forever; offline RL is blind to them |
| BC-13 | RolloutManager imports zero VERL/OpenHands code | Framework coupling prevents swapping trainer or environment |
| BC-14 | RolloutManager owns the parquet dataloader | Trainer becomes hidden orchestrator; pluggability collapses |
| BC-15 | Trainer connects only to LiveStore and PolicyRegistry | Trainer sneaks back into orchestrator role on any other connection |
| BC-16 | Start trainer AFTER worker pushes ≥1 group (warm-up gate) | Trainer burns 1800s no-progress timer during buffer warm-up |

---

## Why each service is separate

### EnvironmentProvider (ProRL :8006)
Owns task registry, sandbox lifecycle (Singularity), reward computation, and the
token-in/token-out invariant (BC-1). Kept separate so ROCK, GEM, ORS/OpenReward,
or any other environment can be plugged in without touching the training loop.

### InferenceBackend (vLLM :8100-8103)
Owns model weights and the pinning swap protocol (BC-10). Every rollout trajectory
is pinned to the policy version active at dispatch via `/v{N}/generate`. Kept
separate so SGLang, TGI, or a hosted API can replace vLLM without changing the
RolloutManager or trainer.

### LiveStore (gRPC UDS)
Bounded FIFO between RolloutManager and trainer. Pop-on-sample (BC-3), server-side
blocking `get_batch` (BC-4, BC-5), staleness eviction (BC-8). Kept separate so
two independent trainers can draw disjoint groups from the same buffer, and so
the buffer's blocking logic does not live inside trainer code.

### PolicyRegistry (gRPC UDS)
Single source of truth for LoRA version and adapter URI. Fans out `/reload_lora`
synchronously and applies the `endpoints_failed > 0` abort gate (BC-9). Kept
separate so the RolloutManager can poll policy version without coupling to the
trainer, and so the trainer does not hold vLLM addresses (BC-15).

### RolloutManager (standalone Python process)
Owns the parquet dataloader (BC-14), reads one `PolicyVersionSnapshot` per group
(BC-0), dispatches N siblings all under the same snapshot, archives pre-filter and
pushes post-filter to LiveStore (BC-12). Zero VERL/OpenHands imports (BC-13): calls
ProRL via plain `httpx`. Kept separate so a multi-machine fleet of workers, a
partner worker, or a per-environment worker can replace it by implementing the same
slot contract.

### TrainerAdapter (VERL FSDP inside Docker)
Consumes `TrainingGroup` records, computes algorithm-specific fields locally
(advantages, IS ratios, KL penalties), runs FSDP forward/backward, publishes LoRA
via PolicyRegistry. Has no dataloader, no producer thread, no parquet files,
no ProRL address, no vLLM address (BC-14, BC-15). Kept separate so ROLL, slime,
DeepSpeed, or SFT pipelines can replace VERL by implementing the same slot contract.

---

## Wire schemas

### `TrainingSample` (hot path — LiveStore ↔ TrainerAdapter)

```
TrainingSample {
    group_uid:               str      # ties all N siblings together
    sample_uid:              str      # per-row identifier
    task_id:                 str      # provenance only; trainer never resolves this
    environment_id:          str
    behavior_policy_version: int      # version active when episode ran (BC-7)
    created_at_step:         int      # trainer global step at dispatch (BC-8)
    prompt_token_ids:        list[int] # BC-1: int, never str
    response_token_ids:      list[int]
    behavior_log_probs:      list[float]
    reward:                  float
    raw_reward:              float
    truncated:               bool
}
```

The wire is **unpadded** (BC-11). The trainer adapter calls `pack_unpadded_groups()`
from `trainer_adapters/verl/pad.py` after `get_batch()` returns.

### `EpisodeRecord` (durable archive — ReplayArchive)

```
EpisodeRecord {
    episode_uid:             str
    group_uid:               str
    task_id:                 str
    environment_id:          str
    environment_version:     str
    policy_id:               str
    behavior_policy_version: int
    created_at_step:         int
    trust_level:             TrustLevel  # SOLVED / PARTIAL / FAILED / TRUNCATED
    messages:                list[Message]  # full multi-turn with token_ids
    reward:                  float
    resolved:                bool
    metadata:                dict
}
```

`EpisodeRecord` is environment-agnostic and durable. `TrainingSample` is derived
from it and shaped for the trainer hot path. The two-tier contract is non-negotiable:
starting at `TrainingSample` alone precludes offline RL, distillation, and audit.

---

## Slot contracts (Protocol interfaces)

All seven Protocol classes live in `schemas/protocols/`. These are the only
boundaries that matter for pluggability — transport is an implementation detail.

```python
class EnvironmentProvider(Protocol):
    def list_tasks(self, split: str) -> list[str]: ...
    def create_episode(self, task_id: str, **opts) -> EpisodeHandle: ...
    def get_prompt(self, h: EpisodeHandle) -> list[ContentBlock]: ...
    def act(self, h: EpisodeHandle, tool_call: ToolCall) -> StepResult: ...
    def close(self, h: EpisodeHandle) -> None: ...

class InferenceBackend(Protocol):
    def generate(self, policy_ref: str, prompt_token_ids: list[int],
                 sampling_params: dict) -> GenerateResult: ...
    def reload_lora(self, version: int, adapter_uri: str) -> None: ...
    def health(self) -> bool: ...

class LiveStore(Protocol):
    def push_group(self, records: list[TrainingSample], group_uid: str,
                   producer_id: str) -> PushResult: ...
    def get_batch(self, n_groups: int, current_step: int,
                  timeout_ms: int) -> list[TrainingGroup]: ...
    def get_metrics(self) -> StoreMetrics: ...

class PolicyRegistry(Protocol):
    def publish_policy_version(self, version: int, adapter_uri: str,
                                policy_id: str, trainer_step: int) -> PublishResult: ...
    def get_latest_version(self, policy_id: str) -> PolicyInfo: ...
    def stream_version_updates(self, policy_id: str) -> Iterator[PolicyInfo]: ...

class RolloutManager(Protocol):
    def pause_production(self) -> None: ...
    def resume_production(self) -> None: ...
    def get_dataloader_state(self) -> dict: ...
    def load_dataloader_state(self, state: dict) -> None: ...

class ReplayArchive(Protocol):
    def append_episode(self, record: EpisodeRecord) -> None: ...
    def query(self, filter_spec: FilterSpec) -> Iterator[EpisodeRecord]: ...

class TrainerAdapter(Protocol):
    def train_step(self, batch: TrainingGroup) -> TrainStepResult: ...
    def publish_policy_version(self) -> PolicyInfo: ...
    def save_checkpoint(self, path: str) -> None: ...
```

---

## Pluggability principles

**P1 — Contract-first.** Wire schemas and method signatures are fixed before picking
transports. vLLM, ProRL, VERL, and the gRPC UDS transport are implementation details.
A new environment or trainer is an adapter, not a redesign.

**P2 — Two-tier data contract.** `EpisodeRecord` (durable, environment-agnostic) and
`TrainingSample` (compact, trainer-shaped) are separate products. Starting only from
`TrainingSample` precludes offline RL, distillation, and audit.

**P3 — Live store and durable archive are different systems.** LiveStore is bounded
RAM, FIFO, pop-on-sample, ephemeral, optimised for `get_batch` latency. ReplayArchive
is unbounded durable storage, append-only, queryable, weeks-to-months horizon. WAL
is a live-store recovery mechanism; the archive is a separate product surface.

**P4 — Trainer adapters compute algorithm-specific fields locally.** Wire schemas
carry rewards and `behavior_log_probs`. `ref_log_probs`, advantages, IS ratios, KL
penalties, value targets are computed by the trainer adapter. This keeps ROLL's
pre-computed advantages and slime's trainer-side computation compatible at the wire.
