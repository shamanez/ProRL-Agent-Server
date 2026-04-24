# Replay dynamics — producer / store / trainer interaction

Audience: anyone tuning the fully-async loop or reading metrics. Explains how a `/generate` rollout becomes a gradient tensor, what "replay" actually buys us at this scale, and where the slack lives. Baseline empirical numbers from a 50-step DAPO reference run (n=8, `filter_groups=True`, 9h04m wall-clock); n=16 deltas in `run9_n16_report.md`.

Companion docs: `latencies.md` (per-component timings), `current_bottlenecks_and_problems.md` (open problems), `handsoff.md` §§4–5 (pointer table, observability).

## 1. The three moving parts

```
          ┌────────────────────────────┐
          │ Producer (daemon thread)   │
          │  generate_sequences_dapo   │   push 4 groups
          │  streams until 4 surviving │──────────────────┐
          │  groups accumulate         │                  │
          └────────────────────────────┘                  ▼
                                              ┌─────────────────────┐
                                              │ TrajectoryStore     │
                                              │ FIFO deque, max 128 │
                                              │ staleness cap K=4   │
                                              │ pop-on-sample       │
                                              └─────────────────────┘
                                                       ▲ pop 1 group
                                                       │
          ┌────────────────────────────┐               │
          │ Trainer (main thread)      │               │
          │  wait_until fresh≥n_groups │───────────────┘
          │  update_actor on 1×n traj  │
          │  publish LoRA every K=5    │
          └────────────────────────────┘
```

**Three clocks**, loosely coupled by the store:

| Clock | Tick unit | Runs on | Current cadence |
|---|---|---|---|
| Producer | 1 producer call → 4 surviving groups pushed | Daemon thread in trainer driver | ~20–40 min / call (mostly DAPO filter waiting for mixed-sign batches) |
| Trainer  | 1 step → 1 group consumed                    | Main thread                    | ~1 min / step after the store fills |
| Publish  | 1 LoRA push per `save_freq=5` steps          | Trainer (inline after step)   | ~1 publish / 30 min at current pace |

## 2. How one rollout becomes a tensor (per-prompt lifecycle)

GRPO needs **multiple trajectories per prompt** (the group) to compute the within-group baseline advantage. Our settings: `n=8`.

```
 prompt P ─┬─ /generate → traj_1 (reward r_1, response tokens, logprobs)
           ├─ /generate → traj_2
           ├─ ...
           └─ /generate → traj_8

 ─► group (P, {traj_1..traj_8}) with:
        advantages = (r_i − mean(r_1..r_8)) / std(...)
        uid        = hash(P)  (stable identifier)
        policy_version = rollout_manager.policy_version at generation
        created_at_step = trainer.global_steps at push
```

Producer call (`generate_sequences_dapo`) keeps streaming prompts until **`train_batch_size=4` groups survive DAPO's filter** (`filter_groups=True`: drop any group where all 8 rewards share sign — all fail or all pass). Then it returns one DataProto containing the 4 surviving groups (32 trajectories).

The producer thread calls `store.push_from_dataproto(batch, …)`. Under a single lock (trajectory_store.py:351–353) the store unpacks the DataProto into per-group `TrajectoryRecord`s and appends each group to the internal deque. **Atomic**: either all 4 groups land or none — no half-observed state for the trainer.

## 3. What goes in the replay store

**Only filtered (surviving) groups.** This is correct:

- `filter_groups=True`: DAPO's internal filter runs inside `generate_sequences_dapo` **before** the producer calls `push_from_dataproto`. Groups with 0/n or n/n resolution never reach the store. In an earlier n=8 A/B run across 50 steps: 54 hard-filter events on 45 unique prompts (44× all-fail, 10× all-pass) — all dropped at producer-side, not at store-side.
- `filter_groups=False` (baseline path): every group pushes. Advantage variance then comes from replay temporal diversity rather than within-group spread.
- **Never** stored: the `dropped_by_filter_groups_per_step` key exists as a dead canary — it will be 0 under Option A (our current design) because filtering happens upstream of the store.

Each record includes `behavior_policy_version` stamped at generation time (from `rollout_manager.policy_version` — gotcha §20: single-int, GIL-atomic read). The trainer later uses this for the clipped temporal IS correction (§6).

## 4. "Can the store be idle?" — yes, and that's what we saw

Observed store-fill trace (earlier n=8 A/B):

```
replay/store_fill_ratio: min=0.000  max=0.023  mean≈0.007
```

0.023 = ~3 groups in a 128-group deque. Producer leads the trainer by about 3 groups at any moment; most of the time the store sits at 0 and the trainer is blocked in `wait_until`.

Why:

1. **Small model (~4B) × rank-16 LoRA** → most SWE-Gym prompts still get 0/n resolved. 81 % of the earlier n=8 A/B's filter events were all-fail groups.
2. DAPO must scan ~2–3× as many prompts to find 4 that yield mixed-sign rewards. That's the 20–40 min producer-call wait in logs.
3. Trainer drains a group in ~1 min (update_actor ≈18s + misc), so once a call lands the store empties quickly.

**Producer-bound regime** (current): throughput = producer throughput. Clock-separation gains a few percent (trainer runs while producer's next batch is in flight) but not the 60% gains the paper reports, because the paper's setup is **trainer-bound**, not producer-bound.

When replay *would* pay off at this scale:
- `filter_groups=False` (keeps all groups, buffer warms faster, variance comes from temporal diversity).
- Larger base model (fewer all-fail groups).
- Larger `n` (bigger groups → more likely mixed-sign in a single draw).
- Positive-bias sampling (upweights successful trajectories from the buffer instead of discarding hard-prompt batches — Arnal et al. §5, `docs/README.md §9`).

## 5. "Does the trainer reuse a group for 4 steps when the buffer has 1 group?" — **no**

The store uses **pop-on-sample** (consume-on-sample) semantics. From `trajectory_store.py:390-430`:

```python
chosen_idx = set(rng.sample(range(len(self._groups)), n_groups))
groups_list = list(self._groups)
chosen = [groups_list[i] for i in sorted(chosen_idx)]
remaining = [g for i, g in enumerate(groups_list) if i not in chosen_idx]
self._groups.clear()
self._groups.extend(remaining)
```

Docstring rationale (verbatim): "Consume-on-sample (queue semantics): each group is produced once and consumed once. Drawing removes the group from `self._groups` so subsequent calls cannot re-sample it. Rationale: when producer throughput < trainer throughput the buffer can shrink to ~1 group; with-replacement sampling would then re-train on the same batch K+1 times, which is overfitting, not replay."

Concrete scenario the user asked about — "store has 1 group, trainer wants a step":
1. Step N: `wait_until(fresh ≥ 1)` returns immediately. `sample_mini_batch(n_groups=1)` pops the group. Store → 0 groups.
2. Step N+1: `wait_until(fresh ≥ 1)` blocks on the condition variable. Trainer is idle. This is the "trainer stale time."
3. Producer eventually pushes 4 groups. `wait_until` unblocks. Steps N+1…N+4 consume them (one per step). Store → 0 again.
4. Back to blocking.

**Invariant** (user's concern addressed): nothing in the producer/store/trainer loop ever reuses a trajectory across gradient steps. There is no replay-with-replacement. The system is **strict queue + staleness cutoff**.

When we *want* replay reuse: flip `sample_mini_batch` to with-replacement and rely on the `sample_age_steps` cap + `is_weight` clip to bound drift. Not today — keep the simple queue contract until the producer-bound regime is relieved.

## 6. How much data does one gradient step consume?

Config knobs driving this (run_proagent_qwn3_4B_instruct_fullasync.sh):

| Knob | Value | Semantics |
|---|---|---|
| `data.train_batch_size` (BATCH_SIZE) | 4 | DAPO "survivors per producer call" (4 groups) |
| `actor_rollout_ref.rollout.n` (NUM_TRAJ) | 8 | trajectories per prompt (group size) |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 4 | prompts per PPO minibatch (mirrors train_batch_size) |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 1 | per-GPU micro-batch |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` (SP) | 2 | SP world size |
| `data.max_prompt_length` | 4122 | prompt token cap |
| `data.max_response_length` | 1536 | response token cap |

**Trainer per-step math** (ray_trainer_dapo.py:86–87):

```python
n = self.config.actor_rollout_ref.rollout.n                        # = 8
n_groups = max(1, self.config.data.train_batch_size // max(1, n))  # = max(1, 4 // 8) = 1
```

So **each trainer step consumes exactly 1 group = n = 8 trajectories**.

Token accounting per step (upper bound):
- Prompt tokens: 8 × 4122 = 32 976
- Response tokens: 8 × 1536 = 12 288
- **Total: ≤ 45 264 tokens per gradient step**

Response tokens are what drive the PPO loss (prompts are context only for logprob computation). **Effective gradient signal per step ≤ 12 288 response tokens**, typically less because responses don't always saturate.

Across 8 GPUs with SP=2 (DP=4), that's ~1 trajectory per DP rank per micro-batch — small even for LoRA.

**The formula at ray_trainer_dapo.py:87 is worth understanding** — it reads natural if you assume `train_batch_size` is "total prompts per step". With 4 prompts and n=8 trajectories/prompt you get 4 // 8 = 0 → `max(1, 0) = 1` group. The `max(1, …)` floor is load-bearing. Interpretations:

- Generous: BATCH_SIZE=4 is really "groups per producer call" (DAPO survivors target), not "prompts per trainer step". Producer pushes 4, trainer pops 1, cadence works out.
- Suspicious: if the intent was "4 prompts per trainer step", the formula should be `n_groups = train_batch_size` (= 4), which would make each step 4× larger. Empirically the earlier n=8 A/B matched the generous reading (4-step cadence, 1 group/step observed).

Decision for next run (§10): keep `train_batch_size=4` but treat it as "groups per step" going forward, and when we raise `n` to 16, re-examine whether we also want to raise `BATCH_SIZE` to 8 or 16 to get multiple groups per gradient step.

## 7. Trainer stale time — empirical

Earlier n=8 A/B reference, 50 steps, 9 h 04 min wall-clock:

| Component | Mean per step | % of step |
|---|---|---|
| Rollout wait (`wait_until` on store) | 623 s | 95.7% |
| `update_actor` (FSDP fwd+bwd+opt) | 18 s | 2.8% |
| `old_log_prob` recompute | 7.6 s | 1.2% |
| Weight sync `publish` (every 5 steps → amortized) | 6 s | 0.9% |
| **Step total** | **651 s** | **100%** |

Trainer-idle time = 95.7% of wall-clock. **The replay buffer removes almost none of this** in the current producer-bound regime. What it *does* remove — the blocking `generate_sequences` inside `fit()` — was already less than a step's rollout cost anyway.

**Where replay does help, even in this regime:**
1. **Smooths cadence.** Instead of "wait full rollout, step, wait, step", we get "producer bursts 4 groups, trainer runs 4 fast steps, wait". Publishes happen mid-burst, keeping pool versions fresher than they would under strict lock-step.
2. **Prevents wasted wake/sleep.** Producer keeps the pool warm between publishes; strict lock-step paid a wake/sleep round-trip per step.
3. **Substrate for positive-bias sampling.** The store is the data structure the next-wave change (upweight successful trajectories, AsymRE loss) plugs into — without it, those changes have nowhere to land.

## 8. Staleness — what K=4 actually caps

`sample_age_steps_p95` observed in run8: max=3, mean=1.47. **Well under K=4 cap**, zero drops by staleness.

K=4 means: "a group produced at step X is eligible until the trainer reaches step X+4; after that, `num_fresh_groups` excludes it and `evict_stale` drops it on the next sample call."

Interpretation: at current producer-slow regime, groups rarely sit in the store long enough to go stale — they're popped within 0–3 steps. K=4 is dormant at n=8; at n=16 it hits twice in 10 steps (Run9). If we move to a trainer-fast regime (larger producer batches, parallel producers, smaller model bottleneck) K will start biting more and we'll want to log `is_weight/clip_fraction`.

## 9. Tuning levers and their first-order effects

| Lever | Knob | Effect when raised | When to raise |
|---|---|---|---|
| Buffer size | `replay.buffer_size` (128) | More headroom for producer bursts | If producer produces in big bursts and we evict fresh groups — not observed in run8 |
| Staleness cap | `replay.staleness_cutoff_k` (4) | Tolerates older groups | If `is_weight/clip_fraction` stays low AND we want to extract more replay reuse |
| Surviving-groups target | `data.train_batch_size` (4) | Producer spends longer per call, trainer steps per call grow | If FSDP step time dominates (we're trainer-bound) |
| Trajectories per prompt | `actor_rollout_ref.rollout.n` (8) | Better baseline estimate, bigger groups, 2× rollout cost | If advantage noise dominates reward trend — run9 will try 16 |
| DAPO filter | `algorithm.filter_groups.enable` (True) | Turning OFF keeps all groups, buffer warms faster, replay carries variance | If producer bottleneck gets worse; plain-GRPO smoke-test first |
| Publish cadence | `SAVE_FREQ` (5) | Less publish overhead, more staleness | If `weight_sync/*` overhead > 10% of wall-clock |
| Producer parallelism | `OPENHANDS_NUM_WORKERS` (32) | More concurrent /generate calls | If pool CPU/GPU underutilized (check vLLM logs) |
| Wait budget | `replay.wait_timeout_s` (7200s) | Soft floor on "producer must be faster than this" | Only if producer genuinely hangs — don't raise to mask bugs |

## 10. What changed at n=16 (Run9)

Moving from `NUM_TRAJ=8` to `NUM_TRAJ=16` doubles per-prompt rollout cost but reduces all-fail/all-pass rate (DAPO's hard-filter target) because larger groups have higher probability of at least one success.

Observed at n=16 (Run9, 10 steps, `test_freq=10`, `val_before_train=True`):

| Metric | n=8 reference | n=16 observed (iters 1–4) | Comment |
|---|---|---|---|
| Producer wall per iter | ~30 min | **53, 53, 80, 58 min** | iter-3 regression unexplained (problem #8) |
| DAPO hard-filter drop rate | ~50 % | **18–40 %** | Confirmed paper §5 directional claim |
| Per-gradient-step group count | 1 group (8 traj) | **1 group (16 traj)** | `max(1, 4//n)` floor — see problem #1 |
| Staleness cap | K=4 dormant | K=4 hit **twice** (step 4, step 9) | Tighter — producer-bound worse at n=16 |
| Trainer utilisation | ~4 % | **1–4 %** of wall-clock | Producer-bound *worse* at n=16 |

Full report: `run9_n16_report.md`. Problem sheet: `current_bottlenecks_and_problems.md`.

## 11. Gotchas relevant to this doc

- **§19 (cooperative producer stop)**: when `fit()` exits mid-producer-call, the daemon thread is left alive and the interpreter reclaims it. Benign at shutdown; fires during `fit()` at validation boundaries when iter wall > 10 s (problem #7).
- **§20 (policy_version race)**: producer reads `rollout_manager.policy_version` unlocked (single int → GIL-atomic). Publish updates this between producer loop iterations, so the next push carries the new version. In-flight trajectories carry the pre-publish version — that's exactly the temporal IS correction's input.
- **DAPO bug #16 fix**: `all_input_batch = None; last_data_index = 0` reset (via `job_queue` rebuild) at start of each `generate_sequences_dapo` call under producer mode, otherwise leftover state from the previous producer call `DataProto.concat`s onto the current batch. Each producer call emits one "dropped N leftover jobs" marker confirming the fix fires.
- **Bug #18 fix (resume policy_version sync)**: DAPO path mirrors ray_trainer.py:1510–1514 — after `_load_checkpoint`, set `self.policy_version = self.global_steps` AND `self.async_rollout_manager.policy_version = self.global_steps`. Without this, the first post-resume `/reload_lora` is rejected as non-monotonic.

## 12. Quick metric cheatsheet (WandB keys → what to watch)

| Key | Healthy | Red flag | Interprets |
|---|---|---|---|
| `replay/store_size` | > 0 steady-state | = 0 for > 50% of steps | Producer slower than trainer (expected currently) |
| `replay/store_fill_ratio` | 0.02–0.5 | = 0 steady or = 1 steady | 0: producer-bound; 1: trainer-bound, FIFO evictions will start |
| `replay/sample_age_steps_p95` | ≤ K=4 | > K | Staleness drops will kick in; raise K only after checking `is_weight/clip_fraction` |
| `replay/dropped_by_staleness_total` | 0–few | ramping | Producer lagging publishes; investigate pool |
| `is_weight/p99` | < 5 | > 10 | Temporal drift is large; shrink K or N |
| `is_weight/clip_fraction` | < 0.2 | > 0.5 | Too many clipped corrections; shrink N/K |
| `weight_sync/endpoints_failed` | = 0 | > 0 | Pool-wide publish failure; abort contract fires |

## 13. What this doc informs

- **Why replay buys little at current scale** — producer-bound, modest cadence smoothing.
- **No-reuse contract** — pop-on-sample is explicit in code, not silent behavior.
- **Per-step data volume** — ~45K tokens, ≤ 12K response tokens for gradient.
- **Open problems attributable to this data path** — see `current_bottlenecks_and_problems.md` (problems #1, #2, #5, #6 all live here).
