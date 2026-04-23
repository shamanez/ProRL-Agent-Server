# Phase 2 — Fully-Async Decoupled Agentic RL (stage doc)

**Status:** DESIGN LOCKED, NOT YET IMPLEMENTED.
**Branch:** `full-async` (cut from `decoup-weight-sync` HEAD).
**Spec:** [`../handsoff.md`](../handsoff.md) (single source of truth for the phase).
**Intellectual reference:** [`../../docs/README.md`](../../docs/README.md) — Arnal et al. 2026, "Efficient RL Training for LLMs with Experience Replay".
**Approved plan:** `/home/ubuntu/.claude/plans/you-are-a-senior-fluffy-reef.md` (also tracks this doc as Cut 0).

This is the operator/implementer view. It resolves the design questions `handsoff.md §C` required to be resolved before code lands, names the cut order, the tests, the green signals, the failure modes, and the invariants. Read end-to-end before Cut 1.

---

## 1. Purpose

Phase 1 decoupled the *machines* — 8×A100 FSDP trainer and the 4-child vLLM pool on EC2 run on different boxes, and rank-16 LoRA adapters ship via `POST /reload_lora` after every `save_freq` steps. The *clocks* are still locked: `ray_trainer.py:fit()` calls `generate_sequences` and blocks on the slowest agentic rollout (tail up to `openhands_timeout=1000 s × max_iterations=30`). With `filter_groups.enable=True` the trainer additionally discards ~50 % of SWE-Gym groups (all-correct or all-wrong), wasting rollout wall-clock.

Phase 2 decouples the clocks: rollouts stream continuously into a **bounded in-process replay buffer** tagged with `behavior_policy_version`; the trainer samples from it on its own cadence with **clipped temporal importance-sampling correction** (extending the fork's existing per-step `tis_imp_ratio_cap`) and a **hard staleness cutoff K** (FIFO eviction). Both plain GRPO (`filter_groups=False`) and DAPO (`filter_groups=True`) must keep working.

Paper §5.2 / Fig 1: expected payoff ≈ 60 % of on-policy compute for matching accuracy; Fig 13: buffered runs gain a stability-regularization side effect (no post-peak crash).

---

## 2. Resolved known unknowns

Every row below is a decision the implementer does **not** re-litigate. Each is either approved by the user in plan-mode clarification or derives from the code reality mapped in the plan file.

| # | Question | Decision | Authority / rationale |
|---|---|---|---|
| 2.1 | Buffer location | **In-process single shard** in trainer driver (`RayPPOTrainer` / `RayPPOTrainerDAPO`) | User-approved. No new HTTP surface, no security-reviewer gate, no CPU-copy serialization, tensors can stay on-device. Buffer starts empty on resume — pre-resume entries would be maximally stale and evicted anyway. Paper Appendix D.4: sharding has "little impact". |
| 2.2 | `filter_groups` placement in an async world | **Ingest-time (Option A)** — the DAPO dispatcher in `async_server_dapo.py` keeps streaming until it has `train_batch_size` mixed groups, then the producer pushes the whole filtered batch to the buffer as one push-group-batch. | User-approved, overriding `docs/README.md §8`'s Option-B recommendation. Reason: Option B requires group-completion tracking inside the buffer and partial-group advantage recomputation — 500+ line rewrite of `async_server_dapo.py:147-523`. Option A: ~20-line diff at the producer seam; preserves all 692 lines of DAPO dispatcher. |
| 2.3 | Clock-separation mechanism | **Wrap, don't refactor** — new `ContinuousRolloutProducer` in a daemon `threading.Thread` calling the existing `generate_sequences` / `generate_sequences_dapo` unchanged | `generate_sequences` already uses `asyncio.run()` at `async_server.py:1548`, compatible with being called from a thread. Zero edits to either rollout manager. DAPO fits via a callable. |
| 2.4 | Off-policy correction form | **Extend the fork's existing per-step TIS across time.** `core_algos.py:586-590` already computes `exp(old_log_prob − rollout_log_probs) · clamp(max=tis_cap)`. Cut 2 makes `rollout_log_probs` the **stored-at-behavior-version** logprobs — no math change, same code, now temporal. | Fork already has the exact infrastructure. Upstream `/tmp/verl/verl/trainer/ppo/rollout_corr_helper.py` (token/sequence-level modes, ESS, IcePop) is also per-step; adopting it means rebasing 1058 lines of upstream into the fork for no Phase-2 gate benefit. V-trace / IMPALA would be net-new code; no precedent in tree. |
| 2.5 | Eviction policy | **FIFO circular** (`collections.deque(maxlen=N)` of groups) | Paper Appendix B.5 baseline (`BufferStructure`). Positive-bias sampling (δ-refinement, §9) deferred to Phase 2.5. |
| 2.6 | Staleness cutoff `K` | **Hard, K=4 steps** starting value. Samples with `created_at_step < current_step − K` dropped at `sample_mini_batch`, never passed to the loss. | `handsoff.md §12.3` suggestion. Hard eviction matches paper. Empirical tuning from first gate run's `is_weight/clip_fraction` (< 0.2 = healthy; ≥ 0.5 → shrink N/R before anything else). |
| 2.7 | Buffer size `N` | **128 rollouts** = `4 × train_batch_size × n = 4 × 4 × 8` | Four mini-batches of headroom. `handsoff.md §12` suggestion. Units = trajectories; Option A means `N / n = 16 groups`. |
| 2.8 | Sampling | **Uniform random with replacement across calls, without replacement within a single `sample_mini_batch`** | Paper Fig 18: without-replacement variants show no significant gain. Within-call without-replacement prevents trivial duplicate groups in one mini-batch. |
| 2.9 | Record storage shape | **Raw variable-length `tuple[int, ...]`, re-padded at `sample_mini_batch` time** | `DataProto.concat` → `torch.cat(dim=0)` at `/tmp/verl/verl/protocol.py:930` requires matching dim-1. Trajectories from different producer batches have different `max_len_prompt` / `max_len_response` because padding is batch-local (`async_server.py:1346-1367`). Raw storage + re-pad at sample time matches `_convert_results_to_dataproto_token`'s existing pattern. |
| 2.10 | Thread-safety model | **`threading.Lock`** guarding push/sample/evict. **Not `asyncio.Lock`** — producer runs in a `threading.Thread`, trainer is on the main thread. | Cut 4 puts the producer in a daemon `threading.Thread`. `asyncio.Lock` is useless across threads; `threading.Lock` is correct. |
| 2.11 | `policy_version` cross-thread read | Producer reads `self._rollout_manager.policy_version` **unlocked**. Trainer writes it under its own code path. Relies on CPython GIL atomicity of single-int load/store. | Documented reliance, not silent. Worst case: one producer batch stamped with the prior version across a publish boundary — benign, staleness bookkeeping catches it. Adding a lock here would serialize producer and trainer for no correctness benefit. |
| 2.12 | `wake_up()` / `sleep()` lifecycle | In continuous-producer mode: `wake_up()` called once in `ContinuousRolloutProducer.start()`, `sleep()` called once in `.stop()`. Not per-batch. | Currently called every `generate_sequences` round. In Cut 4 that round no longer bounds the producer's lifetime. |
| 2.13 | Interaction: `rollout/staleness_steps` vs `replay/sample_age_steps_p*` | **Complementary, both logged.** `rollout/staleness_steps` (`ray_trainer.py:1873`) = age of the pool adapter vs trainer steps. `replay/sample_age_steps_*` = age of *sampled* trajectories vs trainer steps. Former tells you when to publish; latter tells you the true off-policiness of the training batch. | Not duplicative. Preserve Phase 1 metric; add Phase 2 metric additively. |
| 2.14 | Checkpoint / resume behavior | **Buffer is ephemeral — not checkpointed.** On resume, buffer starts empty, re-warms from producer. Expected warm-up ≈ `N / producer_throughput` steps. | Serializing variable-shape DataProto tensors into `_save_checkpoint` adds fragile logic for no benefit: resumed trajectories would be `(resumed_step − checkpoint_step)` steps stale relative to the resumed model and mostly evicted by the `K=4` cutoff anyway. |
| 2.15 | Abort-on-`endpoints_failed>0` contract | **Preserved.** `ray_trainer.py:1348-1352` raises `RuntimeError` on partial publish. A warm buffer does **not** mask a broken pool; mixed-version state is a correctness bug, not a warning. | `handsoff.md §10.10` and Phase 1 commit `9191de66`. |
| 2.16 | `filter_groups=False` pass-through | **Zero cost.** When `filter_groups.enable=False`, the `DAPO` dispatcher isn't even selected (`main_ppo.py:232-236`); the plain producer pushes every trajectory. Buffer doesn't care. | Plan parity gate (§5.1 below). |
| 2.17 | Producer batch size | **`producer_batch_size` = `train_batch_size`** (= 4) as starting default, configurable via Hydra. | Smaller = more async but more overhead per batch; larger = more efficient but longer critical section. 4 matches current lock-step behavior exactly so Cut 2 has zero behavior change as a parity gate. |
| 2.18 | Positive-bias sampling + AsymRE loss | **Deferred to Phase 2.5.** `docs/README.md §9`. | Orthogonal to the data structure. Ship the minimal buffer first; add positive-bias only if the first gate run's diversity metrics demand it. |

---

## 3. Trajectory record shape

New package `trainer_integration/verl/verl_custom/replay/`. Dataclass:

```python
@dataclass(slots=True, frozen=True)
class TrajectoryRecord:
    prompt_ids: tuple[int, ...]          # exact tokens pool saw
    response_ids: tuple[int, ...]        # exact tokens pool emitted
    logprobs: tuple[float, ...]          # behavior-policy logprobs at generation, per response token
    loss_mask: tuple[int, ...]           # assistant-mask alignment
    advantage: float                     # stamped at generation time (docs/README.md §D.2)
    reward: float
    behavior_policy_version: int         # from async_server.py:1495
    created_at_step: int                 # trainer global_steps at push
    prompt_uid: str                      # for GRPO / DAPO advantage grouping
    group_uid: str                       # = prompt_uid in Option A (groups arrive whole)
    resolved: bool                       # filter_groups diagnostics
```

Derived at `sample_mini_batch`: pad to fresh per-sample `(max_prompt_len, max_response_len)`, assemble a `DataProto` whose tensor dict matches `_convert_results_to_dataproto_token` (`async_server.py:1425-1437`): `input_ids, responses, attention_mask, position_ids, loss_mask, rollout_log_probs, is_padded, error_mask`. Non-tensor batch: `success, error, instance, resolved, finish, uid` (the last = `prompt_uid` consumed by `compute_grpo_outcome_advantage` at `core_algos.py:168-221`).

**Hard rule (token-in/token-out invariant):** Never `decode(response_ids)` and re-tokenize. `openhands/llm/nvidia/qwen3.py` enforces exact token-ID round-trip across turns; a buffer that re-tokenizes silently shifts boundaries, makes stored `logprobs` misalign, and drives actor/reference apart (`docs/README.md §6`). Cut 1 ships a golden-file test to regression-guard this.

---

## 4. Cut order

Each cut = one commit. Per-cut loop: tests-first (red) → code (green) → `make lint` → fast-loop `pytest -m "not integration and not slow and not real_data" tests/ -q` → commit → `/codex:review` on the diff (background) → fold comments → proceed.

### Cut 0 — this stage doc

**Deliverable:** committed `plans-n-solutions/stages/full_async.md` (this file).
**Tests:** none (planning-only).
**Green signal:** `/codex:review` of the commit has no unresolved objections.
**Gate to Cut 1:** manual acceptance that Resolved Known Unknowns (§2) are final.

### Cut 1 — `TrajectoryStore` + sampler (pure Python, zero training-loop coupling)

**New files:**
- `trainer_integration/verl/verl_custom/replay/__init__.py`
- `trainer_integration/verl/verl_custom/replay/trajectory_store.py`

**Interface:**

```python
class TrajectoryStore:
    def __init__(self, max_size_groups: int, staleness_cutoff_k: int, group_size: int): ...

    def push_group(self, records: list[TrajectoryRecord]) -> None:
        """Option A: ingest a filtered group of `group_size` sibling records. FIFO evict when full."""

    def push_from_dataproto(
        self, batch: DataProto, behavior_policy_version: int, current_step: int,
    ) -> int:
        """Unpack a DataProto into groups (keyed by `uid`), push each group. Returns # groups pushed."""

    def sample_mini_batch(
        self, n_groups: int, current_step: int,
    ) -> tuple[DataProto, dict[str, float]]:
        """Evict stale groups first (< current_step − K); then uniform-random `n_groups` w/o replacement within call; re-pad to sample-local max; return DataProto + replay/* metrics dict."""

    def size(self) -> int: ...
    def metrics(self, current_step: int) -> dict[str, float]: ...
```

**Behavior guarantees:**
- FIFO eviction: `deque(maxlen=max_size_groups)` of groups, flat structure.
- Concurrency: one `threading.Lock` around push/sample/evict. No `asyncio.Lock`.
- Sample without replacement within a call (`random.sample`), with replacement across calls.
- Token-ID round trip: `push_from_dataproto → sample_mini_batch` produces bit-identical `prompt_ids` / `response_ids` / `rollout_log_probs` for any record that survives the sample.

**Tests (new file `tests/replay/test_trajectory_store.py`):**
1. `test_push_evicts_fifo_at_capacity` — push `max+3` groups; assert first 3 are gone, last `max` remain in order.
2. `test_sample_without_replacement_within_call` — sample `k` groups; all distinct `group_uid`.
3. `test_staleness_cutoff_drops_old_groups` — push groups at step 0, sample at step `K+1`, assert 0 survivors and metric `replay/dropped_by_staleness_per_step == initial_count`.
4. `test_sample_emits_valid_dataproto_with_padded_tensors` — 3 records of varying token lengths; sample; assert `DataProto.concat`-ability and dim-1 equality across tensor keys.
5. `test_concurrent_push_sample` — two `threading.Thread`s, one pushing, one sampling for 500ms; assert no exceptions, no torn reads (hash of sampled record sequence matches a pushed record).
6. `test_token_id_preservation_golden` — 3 fixed `TrajectoryRecord`s pickled as a golden fixture; push, sample all, assert bit-identical round trip. **This is the merge-blocking §6.5 gate.**

**Green signals:**
- `pytest tests/replay/ -m "not integration and not slow and not real_data" -q` green.
- `make lint` clean.

**Out of scope for this cut:** any import from `verl_custom.trainer` or `verl_custom.nvidia`. Buffer must be testable without a Ray cluster or a pool.

### Cut 2 — wire producer → store → trainer (clock still lock-step)

**Modified files:**
- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`
- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py`
- `trainer_integration/verl/verl_custom/trainer/main_ppo.py` (config passthrough only, no selector change)
- `trainer_integration/verl/verl_custom/trainer/config/ppo_trainer.yaml` (new `replay.*` section, defaults `enable: False`)

**Change at producer seam** (plain path, `ray_trainer.py:fit()` around line 1576–1577):

```python
raw_batch = self.async_rollout_manager.generate_sequences(gen_batch)
if self.config.replay.enable:
    self.trajectory_store.push_from_dataproto(
        raw_batch,
        behavior_policy_version=self.policy_version,
        current_step=self.global_steps,
    )
    gen_batch_output = self.trajectory_store.sample_mini_batch(
        n_groups=self.config.data.train_batch_size,
        current_step=self.global_steps,
    )
else:
    gen_batch_output = raw_batch
```

**Same pattern in DAPO** (`ray_trainer_dapo.py:fit()` around line 112). The returned DataProto from `generate_sequences_dapo` is already group-filtered — Option A; push as a whole.

**Trainer instantiation:** construct `TrajectoryStore` in `RayPPOTrainer.__init__` near line 394 (where `self.policy_version = 0` lives). Gate on `config.replay.enable`.

**Latent bug fix in scope (fix #18):** At `ray_trainer_dapo.py:69-74`, after `self._load_checkpoint()`, add the Phase-1 resume-sync block:

```python
if self.global_steps > 0:
    self.policy_version = self.global_steps
    if self.async_rollout_manager is not None:
        self.async_rollout_manager.policy_version = self.global_steps
```

This mirrors `ray_trainer.py:1510-1514`. Free win because this cut already touches the DAPO `fit()` entry.

**Config schema** added to `ppo_trainer.yaml`:

```yaml
replay:
  enable: False
  buffer_size: 128              # trajectories total; groups = buffer_size / rollout.n
  staleness_cutoff_k: 4
  producer_batch_size: 4        # = train_batch_size default
  use_temporal_is: False        # Cut 3 toggles on
  continuous_producer: False    # Cut 4 toggles on
```

Defaults preserve Phase-1 behavior exactly.

**Tests (new file `tests/trainer/test_trainer_buffer_integration.py`):**
1. `test_cut2_lockstep_parity` — construct two trainers, one with `replay.enable=False`, one with `replay.enable=True buffer_size=train_batch_size*n`. Drive both with the same seeds and a mock rollout manager; assert bit-identical `batch.batch[...]` after the producer→store→trainer round trip. **Behavior-preservation gate for Cut 2.**
2. `test_dapo_resume_policy_version_sync` — covers fix #18.

**Green signals:**
- Parity test green — the buffer is a no-op at the same size as one batch.
- `pytest tests/trainer/test_trainer_buffer_integration.py -q` green.
- `make lint` clean.

### Cut 3 — temporal importance-sampling correction

**Modified files:**
- `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py` (gate TIS on `replay.use_temporal_is`, or leave always-on if `tis_imp_ratio_cap > 0` — final form decided inline with code diff)
- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` (new metrics, `_measure_is_weights` helper)
- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py` (mirror metrics at `:377`)
- `trainer_integration/verl/verl_custom/workers/actor/dp_actor.py` — no change; line 344–345 already appends `rollout_log_probs` to select_keys when `tis_imp_ratio_cap > 0`.

**Math — no change.** The existing `tis_imp_ratio = torch.exp(old_log_prob - rollout_log_probs); clamp(max=cap); pg_losses *= tis_imp_ratio` at `core_algos.py:586-590` becomes temporal **by construction** once Cut 2 lands, because `rollout_log_probs` now comes from the *stored-at-behavior-version* field, not the current-batch rollout. `old_log_prob` is the actor's recomputed log-prob at training time — i.e. `π_θ_current`. Ratio `= exp(log π_θ_current − log π_θ_{behavior_pv}) = π_θ_current / π_θ_{behavior_pv}`. Exactly the temporal IS ratio.

**New metrics** (additive in the `metrics` dict, emitted at `ray_trainer.py:1878` / `ray_trainer_dapo.py:377`):

| Key | Paper anchor (`docs/README.md`) |
|---|---|
| `replay/store_size` | — |
| `replay/store_fill_ratio` | — |
| `replay/sample_age_steps_p50`, `_p95` | §3.2 Fig 2 left |
| `replay/replay_ratio_mean`, `_p99` | §3.2 Fig 2 middle |
| `replay/steps_since_last_use_p50` | §3.2 Fig 2 right |
| `replay/dropped_by_staleness_per_step` | — |
| `replay/dropped_by_filter_groups_per_step` | Option A: ≈ 0 (filter fires pre-ingest). Kept as dead canary for Option B/C migration. |
| `is_weight/mean`, `_p99`, `_clip_fraction` | §4.5 — primary health signal |

`is_weight/clip_fraction` = fraction of tokens where `tis_imp_ratio >= config.tis_imp_ratio_cap` after clamp. **Health gate: < 0.2. ≥ 0.5 → shrink N/R before anything else.**

Expose `is_weight` tensor out of `compute_policy_loss` via an additional return-dict key; read it in `_measure_is_weights` on the main process each step.

**Tests (new file `tests/trainer/test_temporal_is_correction.py`):**
1. `test_is_weight_unity_when_same_version` — record pushed with `behavior_policy_version == current`; sample; assert `tis_imp_ratio ≈ 1.0` within float tolerance.
2. `test_is_weight_increases_with_staleness` — synthetic `rollout_log_probs` set so that `old_log_prob − rollout_log_probs` is a known monotonic sequence in `staleness`; assert clamp fires on the oldest.
3. `test_clip_fraction_metric` — pathological stale sample; assert `is_weight/clip_fraction > 0`.

**Green signals:**
- Unit tests green.
- Short 10-step dry run with `replay.use_temporal_is=True, replay.enable=True, buffer_size=32`: `is_weight/mean ≈ 1.0`, `is_weight/clip_fraction < 0.2`, critic reward trending up.

### Cut 4 — clock separation (`ContinuousRolloutProducer`)

**New file:** `trainer_integration/verl/verl_custom/replay/continuous_producer.py`.

**Interface:**

```python
class ContinuousRolloutProducer:
    def __init__(
        self,
        rollout_manager,                 # AsyncLLMServerManager or …DAPO
        generate_fn: Callable[[DataProto | None], DataProto],
        store: TrajectoryStore,
        train_dataloader,                # plain-GRPO path only; DAPO pulls internally
        current_step_box: AtomicInt,     # lock-guarded int written by trainer each step
    ): ...

    def start(self) -> None:             # calls rollout_manager.wake_up(), spawns daemon thread
    def stop(self, timeout: float = 10.0) -> None:
```

- Plain GRPO: `generate_fn = partial(rollout_manager.generate_sequences)`; producer iterates `train_dataloader` and passes chunks of size `producer_batch_size`.
- DAPO: `generate_fn = lambda _: rollout_manager.generate_sequences_dapo()`; producer ignores its own dataloader because DAPO pulls from its internal `data_loader` (set at `ray_trainer.py:1186`).
- Each iteration: `batch = generate_fn(prompts); store.push_from_dataproto(batch, behavior_policy_version=rollout_manager.policy_version, current_step=current_step_box.get())`.

**Trainer loop change** (`ray_trainer.py:fit()` and `ray_trainer_dapo.py:fit()`): when `config.replay.enable and config.replay.continuous_producer`, don't call `generate_sequences` from `fit()`. Instead spawn producer, spin reading from the store:

```python
producer = ContinuousRolloutProducer(...); producer.start()
try:
    while training:
        while self.trajectory_store.size() < minimum_groups_for_step:
            time.sleep(0.05)
        mini_batch, replay_metrics = self.trajectory_store.sample_mini_batch(...)
        self.update_actor(mini_batch)
        ...
        if time_to_publish:
            self._publish_lora_adapter(...)
finally:
    producer.stop()
```

Keep `self.async_rollout_manager.wake_up()` / `sleep()` around the producer, not around each generate.

**Latent bug fix in scope (fix #16):** At `async_server_dapo.py:77-121` start of `generate_sequences_dapo`, when producer mode: explicitly reset `self.all_input_batch = None` and `self.last_data_index = 0`. Without this, leftover state from the prior call `DataProto.concat`s onto new-call data and corrupts the `instance_ids_before_filtering == after + output + filtered` assert at `:512-516`. Add an invariant assert after the reset.

**Tests (new file `tests/replay/test_continuous_producer.py`):**
1. `test_producer_start_stop_clean` — start, push some fake batches, stop within 10s, assert thread joined.
2. `test_producer_pushes_with_current_policy_version` — midway through, simulate a publish (bump `rollout_manager.policy_version`); next push's records carry the new version.
3. `test_trainer_sleep_wait_when_buffer_empty` — trainer-side consumer loop with empty store; asserts `time.sleep` path, doesn't error, eventually succeeds after a push.
4. `test_dapo_all_input_batch_reset_across_calls` — covers fix #16.

**Green signals:**
- Unit tests green.
- Run 50 steps of `s3_fullasync_docker.sh` with `+algorithm.filter_groups.enable=False`: WandB panel confirms **trainer completes ≥ 2 `update_actor` passes between consecutive publishes** when the store has capacity. `trainer_update_time_s` is not dominated by `rollout_wait_time_s`. **This is gate 5.2.**

### Cut 5 — sibling launcher + run script

**New files:**
- `scripts/_internal/s3_fullasync_docker.sh` — clone of `s2_weightsync_docker.sh` (134 lines). Container `s3-fullasync`. Log `/tmp/s3-fullasync.log`. Stage output `/workspace/outputs/ProAgent/fullasync`. Env knobs: `REPLAY_ENABLE`, `BUFFER_SIZE`, `STALENESS_CUTOFF_K`, `PRODUCER_BATCH_SIZE`.
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh` — clone of the weightsync sibling (145 lines). New Hydra overrides:

```
+replay.enable=True \
+replay.buffer_size=${BUFFER_SIZE:-128} \
+replay.staleness_cutoff_k=${STALENESS_CUTOFF_K:-4} \
+replay.producer_batch_size=${PRODUCER_BATCH_SIZE:-4} \
+replay.use_temporal_is=True \
+replay.continuous_producer=True \
actor_rollout_ref.actor.tis_imp_ratio_cap=2 \
```

**Frozen — do not edit:** `scripts/_internal/s2_weightsync_docker.sh`, `..._weightsync.sh` (Phase 1 baseline preservation).

**Green signal:** full 50-step run of `s3_fullasync_docker.sh` with `filter_groups.enable=True` hits all 6 merge-blocking gates in §5.

---

## 5. Green signals (Phase 2 merge-blocking)

Every gate has a WandB panel or a log grep — no verbal "looks green".

| # | Gate | Verification |
|---|---|---|
| 5.1 | **Filter-groups parity.** 50-step runs at `filter_groups.enable=True` AND `=False` both land cleanly on `full-async`; `critic/rewards/mean` trends up in both. | Two WandB runs compared. |
| 5.2 | **Clock separation.** `trainer_update_time_s` not dominated by `rollout_wait_time_s`. Trainer completes ≥ 2 `update_actor` passes between consecutive LoRA publishes when store has capacity. | WandB panel. |
| 5.3 | **Staleness bounded.** `replay/sample_age_steps_p95 ≤ K` (K = 4 starting). | WandB. |
| 5.4 | **IS sanity.** `is_weight/p99 < 10`, `is_weight/clip_fraction < 0.2`. `clip_fraction ≥ 0.5 → shrink N/R`. | WandB. |
| 5.5 | **Token-in/token-out preserved.** Cut-1 golden-file test green AND live dump of 10 random store entries round-trips bit-identical to pool output. | Test file + one-shot post-run script. |
| 5.6 | **Phase 1 baselines intact.** (a) ≥ 4 `/reload_lora` events per 20 steps at `save_freq=5`; (b) zero 5xx on `/generate` during publishes; (c) `weight_sync/endpoints_failed == 0`; (d) `grep -c 'EXTERNAL BYPASS ACTIVE' /tmp/s3-fullasync.log ≥ 1`. | Log grep + WandB. |

---

## 6. Failure modes

Named ahead-of-time so Phase E verification knows what to look for.

| Failure | Symptom | Mitigation / detection |
|---|---|---|
| **F1. Stale IS blow-up.** Staleness `K` too loose → `is_weight/clip_fraction ≥ 0.5` | Clip fraction panel climbs; training accuracy stalls. | Gate 5.4. Shrink `K`. |
| **F2. Buffer starvation.** Producer too slow or evictor too aggressive → `trajectory_store.size() < minimum_groups_for_step` most of the time. | Trainer sits in `time.sleep(0.05)` loop; WandB `replay/store_fill_ratio < 0.5` persistently. | Bump `buffer_size` or decrease trainer tempo. |
| **F3. Token-ID corruption.** A code path accidentally decode-then-retokenizes across the buffer. | Cut-1 golden test fails, OR live round-trip test at §5.5 fails, OR post-training KL/entropy goes NaN. | Golden test gates Cut 1. Do not add any `.decode(...)` call in the replay package. |
| **F4. DataProto shape mismatch at sample assembly.** Groups with different `max_len_*` passed to `DataProto.concat` without re-padding. | `RuntimeError: Sizes of tensors must match except in dimension 0` from `torch.cat`. | §2.9 (raw + re-pad) + Cut-1 test 4. |
| **F5. Race on `policy_version`.** Publish mid-producer-push stamps the wrong version on a batch of records in the buffer. | At worst, a few records tagged with version `v-1` instead of `v`. | §2.11 — documented benign. Staleness bookkeeping catches it. If somehow catastrophic, promote to `threading.Lock` around the int; trivial change. |
| **F6. DAPO `all_input_batch` leak.** Bug #16 unfixed → second `generate_sequences_dapo` call finds leftover state → invariant assert at `:512-516` blows up OR (worse) silently corrupts the filtered batch. | Assert fires; in the bad case, bit-pattern of pushed records differs from reality (hard to detect). | Cut 4 fix #16 + Cut 4 test 4. |
| **F7. Publish failure mid-run.** Partial `_publish_lora_adapter` → `endpoints_failed > 0` → trainer raises. Buffer still holds trajectories stamped with prior version. | `RuntimeError` from trainer; buffer intact but irrelevant (run aborts). | §2.15 (abort preserved). Operator stops the run, investigates pool, restarts from checkpoint (buffer re-warms empty). |
| **F8. Resume after crash uses pre-crash buffer entries.** If someone accidentally wires buffer-to-checkpoint, resumed runs would start with records `(resume_step − crash_step)` steps stale. | Severe IS weight clipping on step 0 of resumed run. | §2.14 — buffer is ephemeral. Unit test: after `_load_checkpoint`, `trajectory_store.size() == 0`. |

---

## 7. Invariants (must hold across every cut)

1. **Token-in/token-out** (`CLAUDE.md`, `openhands/llm/nvidia/qwen3.py`). `prompt_ids` and `response_ids` in the buffer are the exact integer tuples emitted by the pool; `logprobs` is the behavior-policy float tuple the pool returned. No decode across turns or across steps. **Merge-blocking via Cut-1 golden test.**
2. **Trainer-authoritative `policy_version`.** Only the trainer increments `self.policy_version`, and only after all endpoints ACK. `ray_trainer.py:1355` (plain) and `ray_trainer_dapo.py` (via inherited method) own this. Producer reads; never writes. `handsoff.md §10.6` preserved.
3. **Monotonic policy versions at pool children.** `_vllm_child.py:POST /reload_lora` rejects non-monotonic versions with 409. Phase 1 resume sync at `ray_trainer.py:1510-1514` (plain) + Cut 2 fix #18 (DAPO) protects against replay attacks from a restarted trainer.
4. **`EXTERNAL BYPASS ACTIVE` path stays live.** Phase 2 never co-locates vLLM on the trainer box; the remote pool is still the only inference backend. Preserved by not touching `async_server.py:399-430`.
5. **DAPO group completeness at ingest (Option A).** Every `push_group` receives exactly `rollout.n` siblings with identical `prompt_uid`. Advantages are computable per-group without cross-group dependencies.
6. **`filter_groups=False` is a zero-cost pass-through.** The DAPO dispatcher is not selected; the plain producer pushes every trajectory. No Phase 2 code path is gated on `filter_groups`.
7. **Buffer is ephemeral.** Never checkpointed. Starts empty on resume.
8. **Thread boundary.** Producer thread writes to store via `threading.Lock`. Trainer thread reads via the same lock. `self._rollout_manager.policy_version` is the only cross-thread unlocked read; relies on GIL atomicity of `int`; documented.
9. **Phase 1 abort contract.** `endpoints_failed > 0 → raise`. A warm buffer does not soften this.
10. **Frozen files stay frozen.** `s2_weightsync_docker.sh`, `..._weightsync.sh`, `openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py`, `dev_config/python/**`, `/tmp/verl/**`. Sibling files only.

---

## 8. Latent bug fixes in scope for Phase 2

Flagged during the Phase-A stress-test pass on the DAPO path. Captured here; `handsoff.md §10` will get matching gotcha entries during Phase F.

| # | File:line | Fix | Cut | Justification |
|---|---|---|---|---|
| 18 | `ray_trainer_dapo.py:69-74` | Add resume-time `policy_version` sync (mirror `ray_trainer.py:1510-1514`). | Cut 2 | Path already being modified; trivial; closes a real 409-on-resume bug for DAPO. |
| 16 | `async_server_dapo.py:77-121` | Reset `self.all_input_batch = None`, `self.last_data_index = 0` at start of `generate_sequences_dapo` (producer-mode). | Cut 4 | Required for continuous-producer correctness — leftover state corrupts the next call's filter invariant. |
| 17 | `async_server_dapo.py:92` | **Out of scope.** `_convert_results_to_dataproto` call-site signature mismatch (one arg vs two). Masked by `token_level_generation=True` config invariant in every known Hydra launcher. Documented only. | — | Fixing now introduces risk without a reproducer. Flag in `handsoff.md §10` during Phase F. |

---

## 9. Gotchas to fold into `handsoff.md §10` during Phase F

Numbered to continue from the existing list.

13. **Buffer is ephemeral.** Not checkpointed. Re-warms empty on resume.
14. **`/reload_lora` atomic stamping.** `_vllm_child.py:163-166` guarantees each `/generate` snapshots `active_lora` atomically under `_inflight_cond`. In-flight trajectories carry the prior version; new ones carry the new. The benign race is only between trainer-side publish commit and producer's *next* `generate_sequences` call (producer may stamp with the prior version for one more batch — staleness bookkeeping handles it).
15. **`endpoints_failed > 0` abort contract preserved.** Warm buffer does not mask a broken pool.
16. **DAPO `all_input_batch` state leak.** Fixed Cut 4.
17. **DAPO `_convert_results_to_dataproto` signature mismatch.** Pre-existing; masked by `token_level_generation=True`. Documented; out of scope.
18. **DAPO resume `policy_version` sync missing.** Fixed Cut 2.
19. **Variable tensor shapes across buffer entries.** Raw storage + re-pad at sample time. `DataProto.concat` requires matching dim-1 (`/tmp/verl/verl/protocol.py:930`).
20. **`policy_version` cross-thread read relies on CPython GIL atomicity** of single-int load/store. Not locked; documented benign.

---

## 10. Files created / modified by Phase 2

### Created
- `plans-n-solutions/stages/full_async.md` — **this doc (Cut 0)**
- `trainer_integration/verl/verl_custom/replay/__init__.py` (Cut 1)
- `trainer_integration/verl/verl_custom/replay/trajectory_store.py` (Cut 1)
- `trainer_integration/verl/verl_custom/replay/continuous_producer.py` (Cut 4)
- `tests/replay/__init__.py` (Cut 1)
- `tests/replay/test_trajectory_store.py` (Cut 1)
- `tests/replay/test_continuous_producer.py` (Cut 4)
- `tests/trainer/test_trainer_buffer_integration.py` (Cut 2)
- `tests/trainer/test_temporal_is_correction.py` (Cut 3)
- `scripts/_internal/s3_fullasync_docker.sh` (Cut 5)
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh` (Cut 5)

### Modified
- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` (Cuts 2, 3, 4)
- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py` (Cuts 2, 3, 4 + fix #18)
- `trainer_integration/verl/verl_custom/trainer/main_ppo.py` (Cut 2 — config passthrough)
- `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py` (Cut 3 — gate)
- `trainer_integration/verl/verl_custom/trainer/config/ppo_trainer.yaml` (Cut 2 — `replay.*`)
- `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` (Cut 4 — fix #16)
- `plans-n-solutions/handsoff.md` (Phase F — gotchas 13–20, ship summary)

### Frozen — DO NOT EDIT
- `scripts/_internal/s2_weightsync_docker.sh`
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh`
- `openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py`
- `dev_config/python/**`
- `/tmp/verl/**`

---

## 11. Out of scope (Phase 2.5 or later)

- Positive-bias sampling (δ-refinement) and AsymRE loss — `docs/README.md §9`.
- Sharded buffer (one per FSDP rank) — paper Appendix D.4 says little impact; promote only if central-lock profiles hot.
- Out-of-process buffer service — no multi-trainer use case today.
- Upstream `rollout_corr_helper.py` rebase (token/sequence-level IS modes, ESS metrics, IcePop) — fork's TIS + temporal extension covers Phase-2 gates.
- Fix for bug #17 (pre-existing DAPO signature mismatch).
- Option C hybrid (positive-bias + sample-time `filter_groups`). `TrajectoryStore.sample_mini_batch` leaves a clean extension point.

---

## 12. Execution order (session boundaries)

Per `handsoff.md §1`: one unit of work per session, commit at boundary, hand off to a fresh session.

| Session | Unit | Deliverable |
|---|---|---|
| (this) | **Phase A + Cut 0** | This stage doc, committed. |
| Next | **Phase B** (user-driven; live infra) | 2-step parity runs on `decoup-weight-sync` and `full-async`; short status note confirming gates green. |
| +1 | **Cut 1** (TDD) | `TrajectoryStore` + tests green, `/codex:review` on the diff, committed. |
| +2 | **Cut 2** | Producer→store wiring, parity gate green, `/codex:review`, committed. |
| +3 | **Cut 3** | Temporal IS gate green, committed. |
| +4 | **Cut 4** | Continuous producer, clock-separation gate green, committed. |
| +5 | **Cut 5** | Sibling launchers, full 50-step gate run, committed. |
| +6 | **Phase E + F** | All 6 gates green, handoff.md updated. |

Do not bundle units. Each session commits at its boundary.

---

*This doc was written as Cut 0 of Phase 2 (see `/home/ubuntu/.claude/plans/you-are-a-senior-fluffy-reef.md` for the approved plan). Subsequent cuts either implement the code described here or amend this doc (explicitly noted in the amending commit).*
