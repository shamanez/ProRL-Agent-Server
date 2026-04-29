# Replay dynamics — producer / store / trainer interaction

How a `/generate` rollout becomes a gradient tensor in the fully-async loop. Every claim here is grounded in code; file:line pointers given in-line.

Companion docs: [`latencies.md`](latencies.md) (per-component timings), [`how_to_run.md`](how_to_run.md) (env knobs), [`../handsoff.md`](../handsoff.md) §§4–5 (pointer table, gotchas).

## 1. The three moving parts

```
                                 push 1 surviving group
   ┌──────────────────────────┐  (eager, mid-call)
   │ Producer  (daemon thread)│ ─────────────────────┐
   │  generate_sequences_dapo │                      │
   │  + 32 OpenHands workers  │                      ▼
   │  + 4 vLLM children       │           ┌─────────────────────┐
   └──────────────────────────┘           │  TrajectoryStore    │
                                          │  FIFO, max 256 grps │
                                          │  K=4 staleness cap  │
                                          │  pop-on-sample      │
                                          └─────────────────────┘
   ┌──────────────────────────┐                      │
   │ Trainer   (main thread)  │                      │ pop train_batch_size
   │  wait_until_with_progress│ ◄────────────────────┘ groups, atomic
   │  sample_mini_batch       │
   │  compute_advantage       │
   │  update_actor (FSDP)     │   publish LoRA every save_freq
   │  publish_lora_adapter    │ ──► /reload_lora on 4 vLLM children
   └──────────────────────────┘
```

Two independent rates, glued by the store. **Producer fills the buffer; trainer drains it at its own cadence; they share nothing else.**

| Component | Code | Role |
|---|---|---|
| OpenHands env workers | `OPENHANDS_NUM_WORKERS=32` (launch script:49) | Concurrent agentic clients run by the producer; each opens a turn-by-turn dialog with a vLLM child. |
| vLLM pool | 4 children on ports 8100–8103 (`scripts/serving/launch_remote_vllm_pool.sh:48`) | Generation backend; serves all 32 OpenHands workers. LoRA adapters hot-reloaded via `POST /reload_lora`. |
| Producer thread | `verl_custom/replay/continuous_producer.py` `ContinuousRolloutProducer` | Single daemon thread driving `generate_sequences_dapo` in a loop. Pushes via `_push_fn`. |
| DAPO eager-push hook | `verl_custom/nvidia/rollout/async_server_dapo.py:512–548` | Each instance whose `n` siblings clear `filter_easy_hard_instance` is pushed mid-call. |
| Replay store | `verl_custom/replay/trajectory_store.py` | FIFO of GRPO **groups**. Single `threading.Lock` serializes push / evict / sample+pop. |
| Trainer sample seam | `verl_custom/trainer/ppo/ray_trainer_dapo.py:72–138` | `_acquire_training_batch_dapo`: wait → sample → compute_advantage. |

## 2. Lifecycle of one prompt

GRPO needs **multiple trajectories per prompt** (the group) to compute a within-group baseline. Default: `rollout.n=8`.

```
prompt P  ──┬──► /generate → traj_1 (response tokens, rollout_log_probs, reward_ext, ...)
            ├──► /generate → traj_2
            ├──► ...
            └──► /generate → traj_n          (run by 32 OH workers in parallel,
                                              spread over 4 vLLM children)

     all n done?
        │
        ▼
  filter_easy_hard_instance(all_responses)        ── async_server_dapo.py:763
        │
        ├── resolved == 0  → drop (all-fail, no GRPO signal)
        ├── resolved == n  → drop (all-pass, no GRPO signal)
        └── 0 < resolved < n → SURVIVOR
                │
                ▼
    _build_single_group_dataproto(...)            ── async_server_dapo.py:783
                │   (uid=uuid4(); single_row.repeat(n).union(response))
                ▼
    asyncio.to_thread(self._push_fn, single_group_dp)  ── async_server_dapo.py:534
                │
                ▼
    TrajectoryStore.push_from_dataproto(...)      ── trajectory_store.py:182
                │   stamps behavior_policy_version, created_at_step
                ▼
            store deque
```

**Eager-push is the live path.** A surviving group lands in the store the moment it clears the variance filter, not at the end of the producer call. The terminal `push_from_dataproto` is skipped via `meta_info['eager_pushed_all']` (`continuous_producer.py:241–253`). This means the trainer can begin sampling partway through a 30-min producer call.

## 3. What `TrajectoryRecord` actually stores

From `trajectory_store.py:43–74`:

| Field | Type | Set by | Purpose |
|---|---|---|---|
| `prompt_ids` | `tuple[int, ...]` | `push_from_dataproto:309` | Unpadded prompt token IDs (raw, no left-pad). |
| `response_ids` | `tuple[int, ...]` | `push_from_dataproto:310` | Unpadded response token IDs (the `responses` slice of the DataProto). |
| `response_loss_mask` | `tuple[int, ...]` | `push_from_dataproto:311` | 1 where the token contributes to the PPO loss (assistant tokens), 0 elsewhere. |
| `response_log_probs` | `tuple[float, ...]` | `push_from_dataproto:312` | **Behavior-policy** log-probs at generation time. Re-emitted as `rollout_log_probs` at sample time → consumed by temporal-IS. |
| `reward` | `float` | `push_from_dataproto:313` | 0.0 today (push happens pre-`compute_reward`); see "dead fields" below. |
| `advantage` | `float` | `push_from_dataproto:314` | 0.0 today (push happens pre-`compute_advantage`); see "dead fields". |
| `behavior_policy_version` | `int` | `continuous_producer.py:233` | Reads `rollout_manager.policy_version` at push time (single int → GIL-atomic, see handsoff §20). The temporal-IS denominator. |
| `created_at_step` | `int` | `continuous_producer.py:236` | Trainer step at push time (read from `StepCounter`, lock-guarded). The staleness clock. |
| `prompt_uid`, `group_uid` | `str` | `push_from_dataproto:317–318` | UUID stamped at the eager-push seam (`async_server_dapo.py:817`). Both are the same string today; `compute_advantage` groups by `uid`. |
| `resolved` | `bool` | `push_from_dataproto:319` | DAPO outcome flag. The buffer only ever holds `0 < resolved.sum() < n` — zero-variance groups never enter. |
| `success`, `finish` | `bool` | `push_from_dataproto:322–325` | OpenHands run flags. `finish=False` → trajectory hit max turns. |
| `is_padded` | `bool` | `push_from_dataproto:326` | Trajectory was synthetically padded to align the call-local batch (rare; flagged so loss can mask it). |
| `error` | `str \| None` | `push_from_dataproto:327` | Transport / runtime error string from the OpenHands client; `None` on healthy rollouts. |
| `instance` | `dict` | `push_from_dataproto:328–332` | The original task instance dict (`instance_id`, problem statement, gold patch, …). Reward managers depend on it. |
| `prompt_extras` | `dict` | `push_from_dataproto:299–306` | All other non-tensor keys from the dataloader (`data_source`, `ability`, `reward_model`, `extra_info`, `index`, …) preserved verbatim so the sampled DataProto is semantically equivalent to the pushed one. |

### Dead fields (today)

`reward` and `advantage` are **wired but always 0.0** in the live path: the eager-push seam fires before `compute_reward` / `compute_advantage` run. `push_from_dataproto:230–248` only populates them if `'token_level_rewards'` / `'advantages'` are already in the DataProto's tensors — they aren't, at the eager-push seam. The trainer recomputes both on the sampled mini-batch (`ray_trainer_dapo.py:361`), so this is correct, just wasteful: a deferred refactor could move advantage compute to the filter-clear seam (the `n` siblings are already grouped there) and let the buffer become a flat per-trajectory pool. Today the buffer must hand back **whole groups** so trainer-side `compute_advantage` has the `n` siblings to normalize over.

## 4. Push: `push_from_dataproto` step-by-step

`trajectory_store.py:182–361`. The eager-push seam calls this with a DataProto containing **exactly one group of `n` rows** (single-instance, repeated).

1. **Validate** `'uid'` exists in `non_tensor_batch` (`L208`). Without it the store can't bin records into groups.
2. **CPU-detach** every tensor (`L217–223`). The store never holds GPU memory.
3. **Fish out optional reward/advantage** if present (`L230–248`); else 0.0.
4. **Per-row construction** (`L287–339`):
   - `prompt_ids` = the prompt slice of `input_ids[i]`, masked by `attention_mask[:prompt_len]` (drops left-padding).
   - `response_ids` = `responses[i, :response_valid_len]` — `response_valid_len` is the sum of the response-side attention mask, which strips trailing pad.
   - `response_lp` / `response_lm` are sliced to the same valid length.
   - `prompt_extras` shallow-copies any `dict` values to avoid aliasing the caller's mutable state.
   - If `error_mask[i]` is set but the non-tensor `error` was `None`, stamp `'error_mask_set'`.
5. **Group by `uid`** into `groups: dict[str, list[TrajectoryRecord]]` (`L286, L339`).
6. **Atomic append** (`L357–360`): one lock acquire, all groups pushed under it. This matters because the producer runs in a daemon thread and the trainer samples from the main thread — without the single critical section, `sample_mini_batch` could observe a half-pushed batch.

## 5. Sample: `sample_mini_batch` step-by-step

`trajectory_store.py:391–437`. Called by `_acquire_training_batch_dapo` at `ray_trainer_dapo.py:130`.

1. **Acquire the lock.**
2. **Evict stale** (`_evict_stale_locked`): drop any group whose `current_step - created_at_step > staleness_cutoff_k`. Increments `_dropped_by_staleness_total` (a metric).
3. **Insufficient → raise.** If fewer than `n_groups` survive, raise `InsufficientTrajectoriesError`. The trainer never calls without first clearing the predicate via `wait_until_with_progress` (next section), so this is a defense-in-depth check.
4. **Random sample without replacement** (`rng.sample(range(len(self._groups)), n_groups)`).
5. **Pop chosen groups** out of the deque, replace the deque with the remaining groups (`L433–434`). **Consume-on-sample** — a group can never be sampled twice.
6. **Pack** the chosen records into a `SampledMiniBatch`:
   - Re-pad to either the configured cap (`prompt_length_cap` / `response_length_cap`) or a sample-local max (`L450–457`).
   - Right-pad responses, left-pad prompts (so left-pad-offset arithmetic is consistent with how vLLM emitted them).
   - Concatenate `prompt_ids` + `responses` to rebuild `input_ids`; recompute `position_ids` from the attention mask.
   - Re-emit every `prompt_extras` key as a non-tensor column.
   - `meta_info` carries `behavior_policy_versions`, `created_at_steps`, `sample_ages` — the inputs to temporal IS.
7. **Release the lock.** `_pack` runs **outside** the lock (only operates on the detached `records` list) so `DataProto` construction doesn't hold the producer thread.

The pop-on-sample contract is load-bearing: when the buffer holds 1 group, the trainer pops it once and then waits for the producer; nothing ever re-trains on the same group at the same trainer step.

## 6. Trainer wait-loop and the no-progress detector

`_acquire_training_batch_dapo` at `ray_trainer_dapo.py:72–138`:

```python
n_groups = int(self.config.data.train_batch_size)            # L94
no_progress_timeout_s = float(
    self.config.replay.get('no_progress_timeout_s', 1800.0)  # L102
)
filled = wait_until_with_progress(                           # L113
    lambda: self.trajectory_store.num_fresh_groups(...) >= n_groups,
    self.trajectory_store.total_pushes,
    no_progress_timeout=no_progress_timeout_s,
)
sampled = self.trajectory_store.sample_mini_batch(           # L130
    n_groups=n_groups, current_step=self.global_steps
)
```

Two predicates run on every poll (`continuous_producer.py:328–363`):

- **Readiness** — `num_fresh_groups(current_step) >= n_groups`. Excludes stale groups that `sample_mini_batch` would drop, avoiding the race that crashed an earlier run.
- **Progress** — `total_pushes()` (monotonic) is growing. The deadline resets every time a new group lands; aborts only when the producer pushes nothing new for `no_progress_timeout_s`. This replaced the old fixed 7200 s ceiling, which couldn't distinguish "wedged" from "healthy but slow as response_length climbs".

Once the predicate passes, the call to `sample_mini_batch` is the only place the deque shrinks.

## 7. Staleness, K=4

`replay.staleness_cutoff_k=4` (default in `s3_fullasync_docker.sh:52`). The clock unit is **trainer steps**, not wall-time:

- A group with `created_at_step=12` is fresh at `global_steps ∈ [12, 16]`, stale at 17+.
- `evict_stale` runs as the first thing inside `sample_mini_batch` (`L423`) and on demand.
- `num_fresh_groups` is what the trainer waits on, so a group about to be evicted never unblocks the wait predicate.

K=4 also bounds **how off-policy the temporal-IS correction has to fix up**. With `save_freq=1` (current setting in `s3_fullasync_docker.sh:64`) the policy version increments every step; a K=4 group has at most a 4-version lag. Diagnostics: `is_weight/p99` and `is_weight/clip_fraction` (`core_algos.py:707–711`); `clip_fraction > 0.5` is the signal to either shrink K or shrink `n`.

## 8. Temporal-IS correction at the trainer

`core_algos.py:665–711`. Activated when `actor.tis_imp_ratio_cap > 0` (set to **5** in `run_proagent_qwn3_4B_instruct_fullasync.sh:106`) **and** `rollout_log_probs` is non-None (it always is on the replay path):

```python
tis_imp_ratio_raw = torch.exp(old_log_prob - rollout_log_probs)        # current vs behavior
tis_imp_ratio     = torch.clamp(tis_imp_ratio_raw, max=tis_imp_ratio_cap)
pg_losses         = pg_losses * tis_imp_ratio
```

- `rollout_log_probs` = the stored `response_log_probs` (behavior policy at generation time).
- `old_log_prob` = the trainer's recompute of the same response under the **current** actor.
- The ratio re-weights the policy-gradient loss to correct for the lag introduced by the buffer / call-boundary delay. Clip cap=5 bounds variance from extreme ratios.

This is why we store `response_log_probs` at all: without it there's no temporal-IS, and any group older than 1 step would silently bias the gradient.

## 9. Per-step token & group accounting

Defaults from `s3_fullasync_docker.sh` and `run_proagent_qwn3_4B_instruct_fullasync.sh`:

| Knob | Default | Meaning |
|---|---|---|
| `data.train_batch_size` (`BATCH_SIZE`) | 4 | **Groups** drawn per trainer step (post Cut 5: groups, not prompts). |
| `data.gen_batch_size` (`GEN_BATCH_SIZE`) | `BATCH_SIZE * 4` = 16 | **Survivor target per producer call** — call returns once 16 groups have cleared the variance filter. |
| `actor_rollout_ref.rollout.n` (`NUM_TRAJ`) | 8 | Trajectories per prompt = group size. |
| `actor_rollout_ref.actor.tis_imp_ratio_cap` | 5 | Temporal-IS clip ceiling. |
| `replay.buffer_size` (`BUFFER_SIZE`) | 256 | Max groups in the FIFO. |
| `replay.staleness_cutoff_k` | 4 | Hard staleness cap, in trainer steps. |
| `replay.no_progress_timeout_s` | 1800 | Producer-progress watchdog. |
| `OPENHANDS_NUM_WORKERS` | 32 | Concurrent agentic clients driving the 4-child vLLM pool. |
| `trainer.save_freq` | 1 | LoRA publish cadence (steps). |

**Per trainer step (upper bound):**

- Trajectories: `BATCH_SIZE × n` = 4 × 8 = **32 trajectories**.
- Prompt tokens: 32 × `max_prompt_length` (≤ 4122) = ≤ 132K.
- Response tokens: 32 × `max_response_length` (≤ 1536) = ≤ 49K.
- **Gradient-bearing tokens** ≤ 49K (responses only).

## 10. How to fill the buffer faster

Throughput ≈ `vLLM TPS × concurrent agentic clients × (1 − filter_drop_rate)`. Levers, cheapest first:

1. **`GEN_BATCH_SIZE` up** — longer hot phase per producer call, fewer call boundaries per hour. No code change.
2. **More OpenHands workers** (`OPENHANDS_NUM_WORKERS`) — more concurrent in-flight agentic trajectories. The 4-child pool saturates at ~32 concurrent clients (handsoff §17); past that, scale the pool too.
3. **More vLLM children** — raises raw generation ceiling. Requires more EC2 GPUs.

Once buffer-fill rate ≥ trainer-drain rate, the trainer never waits and the loop runs near-on-policy.

## 11. Observability — the `replay/*` keys

`trajectory_store.py:600–632`:

| Key | Healthy | Red flag | Reads |
|---|---|---|---|
| `replay/store_size` | > 0 steady | 0 for >50% of steps | Producer slower than trainer. Scale producer (§10). |
| `replay/store_fill_ratio` | 0.02–0.5 | =0 steady, =1 steady | 0: producer-bound. 1: trainer-bound, FIFO evictions imminent. |
| `replay/store_age_p95` | ≤ K | > K | Buffer holding stale groups; about to be evicted. |
| `replay/sample_age_steps_p95` | ≤ K | > K | Trainer sampling stale groups; raise K only after checking `is_weight/clip_fraction`. |
| `replay/dropped_by_staleness_total` | 0 / few | ramping | Producer lagging publishes; investigate pool. |
| `is_weight/mean` | ≈ 1 | > 2 | Behavior-vs-current divergence is large. |
| `is_weight/p99` | < 5 | ≥ cap | Bumping the clip; shrink K or shrink `n`. |
| `is_weight/clip_fraction` | < 0.2 | > 0.5 | Too many clipped corrections; same fix. |
| `weight_sync/endpoints_failed` | 0 | > 0 | Pool-wide publish failure; abort contract fires. |

## 12. Gotchas relevant to this path

- **§14 (lock semantics)** — single `threading.Lock` serializes push / evict / sample+pop; `_pack` runs outside the lock on detached records.
- **§19 (cooperative producer stop)** — `stop()` event is checked at the top of the worker loop; mid-`generate_sequences_dapo` calls finish before exit. Daemon thread, so interpreter shutdown reclaims it on exit.
- **§20 (policy_version race)** — producer reads `rollout_manager.policy_version` unlocked (single int, GIL-atomic). Publish updates this between producer iterations; the next push carries the new version.
- **§21 (variable-length store entries)** — producer batches pad to call-local max, so two groups in the deque can have different `prompt_ids.shape`. Store uses raw tuples and re-pads at sample time so `DataProto.concat` sees matching dim-1.
- **§29 (groups sampled intact)** — `sample_mini_batch` never splits a group; load-bearing because trainer-side `compute_advantage` needs the `n` siblings. Deferred refactor: move advantage compute to producer side, then buffer becomes a flat per-trajectory pool.
- **§31 (no-progress detector)** — `wait_until_with_progress` replaced the fixed 7200 s ceiling. Distinguishes wedged from healthy-but-slow.
