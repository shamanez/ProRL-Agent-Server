# Current bottlenecks and problems

Branch: `full-async-optimization`. Basis: Run9 — the moment-of-truth run (n=16, DAPO `filter_groups=True`, K=4 staleness cap, 128-group FIFO buffer, LR=1e-6, rank-16 LoRA). Log: `/tmp/s3-fullasync-n16-baseline.log`. Monitor: `/tmp/replay-monitor.jsonl`. Full report: `run9_n16_report.md`.

**Scope.** Problems observed in Run9 plus the concrete operating-regime shifts needed for the next run to be a fast-improving DAPO run. Each problem includes the fix direction (not the full patch).

---

## 1. The trainer is treated as a batch-processor, when it should be a pure consumer of a big buffer

**Core insight.** In agentic RL, **not all rollouts make it through DAPO filtering — and DAPO filtering must stay**. The mixed-sign survivor criterion is what makes each group's GRPO baseline informative; dropping it would poison training. On SWE-Gym at n=16 we observe ~27–40 % survival. The problem is not the filter — it is that the trainer loop is coupled to "wait for exactly `train_batch_size` mixed-sign groups to arrive, then step", which makes the trainer's throughput hostage to the producer's survival rate and wall-clock.

**The right design.** The trainer doesn't need batches *pushed* to it — it only needs:

- `input_ids / responses` (tokens)
- `rollout_log_probs` (behavior-policy logprobs, stamped at generation)
- `advantages` (stamped at generation, per-group GRPO baseline)
- `loss_mask / attention_mask`

Given those four, `update_actor` is a pure optimisation step. The trainer's only other job is **weight syncing** — publishing LoRA to the pool on `save_freq`.

**What we actually have (Run9):**

```python
# ray_trainer_dapo.py:86-87
n        = config.actor_rollout_ref.rollout.n                    # 16
n_groups = max(1, config.data.train_batch_size // max(1, n))     # max(1, 4 // 16) = 1
```

- Trainer waits for producer to deliver `n_groups` surviving groups before stepping.
- `n_groups=1` with `n=16` means each optimizer step sees **1 prompt's worth** of 16 sibling trajectories. `ppo_mini_batch_size=4` → 4 mini-batches per step → 4 opt steps, all on the same prompt. No cross-prompt averaging within a step.
- GRPO advantage baseline is per-uid, so with one uid per step the baseline is already embedded — fine — but difficulty context still only averages over *time* (across consecutive steps), not within a step.

**The reframe.** Stop thinking of the trainer as "consume a batch → step". Instead:

1. **Producer side:** generate as many rollouts as possible, as fast as possible. **DAPO filter stays at ingest (Option A, unchanged)** — only surviving mixed-sign groups enter the buffer. But push each surviving group into the **big buffer** the moment it survives the filter; do **not** batch-hold them waiting for `train_batch_size` to fill on the trainer's behalf. Stamp `rollout_log_probs` and `advantages` at push. Grow the buffer (current 128-group FIFO → 256–512) so survivors accumulate across producer iters.
2. **Trainer side:** on every cadence tick, sample uniformly at random from the buffer (group-atomic to preserve GRPO baseline). `update_actor`. Publish LoRA on `save_freq`.

The producer and trainer become fully decoupled pipelines meeting only at the buffer — which is exactly the fully-async design intent. Run9's `n_groups=1` floor is a symptom of the trainer still being coupled to the producer's survival rate, not a problem with the filter itself.

**What to change.**

- **Do not remove DAPO filtering.** It stays exactly where it is (`generate_sequences_dapo` → Option A ingest filter). The mixed-sign survivor criterion is what makes each group's GRPO baseline informative.
- Drop the `n_groups = max(1, train_batch_size // n)` floor — read `train_batch_size` as "groups to sample per step" directly from the buffer.
- Increase buffer size (128 → 256 or larger) so the trainer has genuine randomness even when producer is bursty, and so surviving groups from slow iters accumulate instead of being overwritten.
- Trainer samples without blocking on producer — if the buffer is empty, poll. The fix is *capacity of surviving groups*, not *bypassing the filter*.
- Once the buffer always has ≥ `train_batch_size` surviving groups, per-step variance drops naturally — which also directly reduces the symptom in problem #3.

---

## 2. Producer-bound regime — trainer GPUs idle ~96 % of wall-clock

**Evidence (Run9, live `nvidia-smi`):**

| Side | GPUs | Utilisation | Memory | Power |
|---|---|---|---|---|
| vLLM pool (EC2 `vllm-instance`) | 4 × L4 | **100 %** | 20.7 / 23 GiB each | 71–72 W / 72 W cap |
| FSDP trainer (host, Docker) | 8 × A100-40GB | **0 %** | 2.1 / 40 GiB each | 80–95 W / 400 W |

Per-step timing (`run9_n16_report.md` §"Per-step wall-clock"): trainer active ~43 s, iter wall ~53 min. 12 steps × 43 s / 226 min = **~3.8 % trainer utilisation** (skews down over time — iter 3 spent 80 min).

**Throughput arithmetic.**

| Side | Rate |
|---|---|
| Producer | 4 surviving groups / 53 min = **~4.5 groups/hour** |
| Trainer | 1 group / 43 s = **~84 groups/hour** |
| Ratio | **18.5× mismatch** (trainer faster than producer) |

Even under maximal replay (every group sampled K=4 times before eviction), effective supply tops out at ~18 groups/hour — still 4.7× below trainer capacity.

**Mechanism.** Async-replay's "trainer always fires" promise holds only when `producer_rate ≥ trainer_rate / K`. We violate that by ~20×. The buffer is correctly sized and correctly evicting; there is simply nothing to sample between iters.

**Why agentic RL makes this specifically bad.** Multi-turn OpenHands on SWE-Gym with `max_iterations=30`, `openhands_timeout=1000s`, 4B model on 4× L4 → ~53 min per iter at n=16 (~30 min at n=8 in an earlier A/B).

**What to change.** The leverage here is all producer-side. DAPO filtering stays (see #1) — we are scaling *arrival rate of survivors*, not weakening the filter:

- **Parallel DAPO producers.** 2 concurrent producers → ~2× arrival rate of surviving groups → staleness holds ≤ K, trainer bursts overlap producer idle. Highest expected lift.
- **Push each survivor eagerly.** Instead of holding a producer iter until `train_batch_size` survivors accumulate, push each surviving group into the buffer the moment DAPO's filter clears it. Trainer can step as soon as the buffer holds enough survivors, regardless of iter boundary. Combined with problem #1's big-buffer design, this smooths supply across slow iters.
- **Per-prompt wall-time instrumentation** so we can see which prompts are 10× the median (iter 3's 80-min outlier is unexplained because we don't have per-prompt timing).

---

## 3. `is_weight/clip_fraction ~60 %` — raise the cap, not the alarm

**Evidence.** `log_ppl_diff` (log IS weight) across Run9 steps 1–10: 0.530, 0.697, 0.718, 0.785, 0.739, 0.614, 0.488, 0.747, 0.626, 0.726. Clip threshold = `log(tis_imp_ratio_cap) = log(2) ≈ 0.693`. Steps 2, 3, 4, 5, 8, 10 exceed it → mean clip fraction ~60 %. Paper target (§5.2, Fig 13): < 20 %.

**What this actually means.** The paper's 20 % target assumes a roughly on-policy MoE with fresh policy copies. We are **none of those things**:

- Not MoE — dense 4B Qwen3.
- Fully async — by construction the policy at generation time is ≥ 1 step behind the trainer's current weights. We cannot keep an up-to-the-step copy of the policy; that's the whole point of decoupling the clocks.
- Dominant term is **T-mismatch**, not real drift (from handsoff gotcha #27): rollout sampling at T=1.4 vs trainer `compute_log_prob` at T=1.0 accounts for ~0.35 of the ~0.55 mean log-ratio. Numerical floor from vLLM↔FSDP kernel divergence adds another ~0.20. Real policy drift is the smallest component.

So the current clamp is fighting a **known numerical bias**, not policy off-policy-ness.

**What to change.**

- **Raise `tis_imp_ratio_cap`** from 2.0 to something like 4.0 or 5.0. The clamp's job is to bound variance on *real* drift — not to paper over T-mismatch. With a higher cap the clip fraction drops below 20 % automatically.
- Experience replay (problem #1's big-buffer design) naturally helps here too: when the trainer has a large pool to sample from, per-batch variance of `log_ppl_diff` drops, and the fraction of tokens near the cap shrinks.
- **Do not** try to close the T-mismatch by re-sampling at T=1.0 — the T=1.4 is there for exploration and is part of the policy the reward signal was generated under. Changing it changes the thing being trained.

**Keep the metric.** `is_weight/clip_fraction` still matters as a sentinel for real drift — just expect ~10 % baseline from numerical noise and tune the cap so the metric has headroom to move.

---

## 4. `response_length` saturating at 1536 — bump to 4096

**Evidence.** Wandb panel shows `response_length/mean ≈ 1536` across most consecutive Run9 steps. `data.max_response_length=1536` is the hard truncation cap — this is systematic cap-hitting, not task-dependent variation.

**Mechanism.**

- OpenHands multi-turn rollout concatenates all assistant tokens across up to `max_iterations=30` turns into `response_ids`. Long thinking blocks (Qwen3 preserves `<think>…</think>` in `content`, per `openhands/llm/nvidia/qwen3.py` + CLAUDE.md token-in/token-out invariant), tool-call retry loops, and 30-turn accumulation easily exceed 1536.
- Truncation semantics: agent never reaches `finish_action` → reward scored as failure-ish regardless of whether the agent would have completed with more budget. **We are training the policy to associate "long task" with "low reward"** — but also with "truncated state", which is not what we want the signal to encode.
- Loss contamination: tokens right before the 1536 cut may be mid-`<tool_call>` JSON. Mask treats them as valid response tokens; PPO assigns gradient to "produce broken JSON" when local context looks coherent.

**What to change.**

- **Set `data.max_response_length=4096`.** Removes the systematic cap-hitting at 1536 and lets long-tail SWE-Gym trajectories complete without truncation-induced reward bias.
- Watch `response_length/mean` post-change — if it saturates near 4096 as well, we're still truncating and the knob needs another raise. Typical multi-turn SWE-Gym traces fit in 3–4k response tokens when not retrying tool calls.
- Add diagnostic `pct_capped = (response_length == max_response_length).float().mean()` so future saturation is visible from the WandB panel without having to eyeball the histogram.

---

## 5. Advantages computed twice — dead architectural work

**Evidence.** `trajectory_store.py:308` stamps `advantage` (scalar per trajectory) at push time. `ray_trainer_dapo.py:338-347` calls `compute_advantage` again on the sampled batch, overwriting.

**Why both passes yield the same number.** Groups are atomic in the store (`push_group` / `sample_mini_batch` never split). GRPO advantage = (reward − group_mean) / group_std over the 16 siblings of a uid. Identical sibling set at push and sample → identical mean/std → identical advantage. The second pass is compute-but-same-result.

**Cost.** ~20 ms/step on the driver process. Small. But it is the *only* step where the buffer's data model is coupled to the trainer's downstream expectations — removing it would let the trainer treat the buffer as a pure `(tokens, advantages, rollout_log_probs, masks)` source. **This is directly on the path to problem #1's reframe** — the trainer shouldn't need to recompute anything.

**Minimum fields the PPO loss actually needs per row** (from reading `dp_actor.py`):

```
input_ids / responses                   # forward pass
loss_mask, response_mask, attention_mask # zero-out padding
advantages (broadcast over response_mask)
rollout_log_probs                        # IS denominator
```

Everything else in the current sampled DataProto (`uid`, `behavior_policy_version`, `token_level_scores`, `token_level_rewards`, `reward`, `resolved`) is metrics/diagnostics only. The double-compute exists because the buffer emits a **scalar** advantage but the loss reads a **broadcast tensor**, and `compute_advantage` happens to do both jobs.

**What to change.** Move the scalar→broadcast conversion into `sample_mini_batch` (or a thin wrapper) and delete the second `compute_advantage` call. Trainer becomes a pure consumer as #1 intends.

---

## 6. Pool-adapter-age ≠ buffer-age when producer iters are long

**Evidence (Run9 step 9).**

```
replay/sample_age_steps_p50    = 0     # iter-3 groups are fresh in the buffer
rollout/staleness_steps        = 4     # pool is 4 steps behind current trainer
```

These are both "how stale is the data" metrics but measure different things:

| Metric | Measures | Goes up when |
|---|---|---|
| `replay/sample_age_steps_p*` | trainer_step − group.created_at_step | Group sits in buffer unused |
| `rollout/staleness_steps` | current_pv − pool_last_publish_pv | Publishes lag between iters |

**When they diverge.** Whenever a producer iter lasts longer than `save_freq × burst_duration`. Run9 has `save_freq=5` and burst ≈ 4 steps → if iter wall > ~5 × 43 s = 215 s it starts to matter. Our iters are 53 min → routinely divergent by K or more.

**Consequence.** Success gate 4 (`replay/sample_age_steps_p95 ≤ K=4`) **passes** in Run9 — but that is not enough evidence the IS weights are healthy, because clip ratio magnitude correlates with pool-adapter-age, not buffer-age. A second companion gate is missing from the runbook. Current run has hit K=4 on `rollout/staleness_steps` at steps 4 and 9 — one more step of producer delay would have evicted.

---

## 7. Validation–producer race — first fit()-time §19 cooperative skip

**Evidence (Run9 step 10, 2026-04-24T06:40:08):**

```
ContinuousRolloutProducer did not exit within 10.0s; leaving thread running
  (mid-generate_sequences); caller must skip contention with the producer ...
step=10 skipping _validate: producer stop timed out (still mid-generate_sequences)
```

**Sequence.** Step 10 triggers `save_checkpoint` + publish (pv=2) + `_validate()`. `_validate()` needs exclusive pool, calls `producer.stop(timeout=10.0)`. Producer is 3 min into a 53–80 min `generate_sequences_dapo` call. Cooperative-flag check only at top of the worker loop (between DAPO iters). Flag lands but worker is mid-async-wait → timeout expires → `_validate()` skips to avoid double-drain on pool.

**Blast radius.** No corruption — the cooperative skip path (commit `590f8281`) is the safety mechanism working as designed. But the first in-training `pass@k` datapoint (step 10 with `test_freq=10`) is lost. Next opportunity is step 20, which is ~7 h of wall-clock from step 10 at current producer rate. **Time-to-first-eval tripled** compared to planned.

**Contract.** "Zero fit()-time tracebacks / §19 skips" was previously upheld by shorter-producer-iter configs; Run9's 53+ min DAPO iters now routinely straddle scheduled validation boundaries, so this becomes a first-class problem.

---

## 8. Iter 3 wall-clock regression (80 min vs ~53 min steady state)

**Evidence.** Producer iter timing (Run9):

| iter | wall_s | drawn | survived | dropped | eff_tps |
|---|---|---|---|---|---|
| 1 | 3158 | 10 | 4 | 1 | 664.0 |
| 2 | 3201 | 10 | 4 | 2 | 655.1 |
| **3** | **4824** | **15** | 4 | **6** | **434.7** |
| 4 | 3498 | 11 | 4 | 2 | 599.6 |

Iter 3: +52 % wall, −34 % effective TPS, completion 40 % → 27 %.

**Two plausible causes, currently indistinguishable:**

- **Dataset difficulty drift** — the 10 prompts DAPO happened to draw for iter 3 were harder (more zero-variance groups → more filter drops → DAPO draws 5 extras → wall grows). Prompt-order effect, not systemic.
- **Pool degradation after publish #1** — pv=1 is the first trained policy (pv=0 is pristine SFT). If pv=1 regresses on some prompt types (plausible at LR=1e-6 for 4 steps), completion dips. Independent corroboration: publish #2 `transfer_latency_s` jumped 4.0 → 19.0 s (see #9).

Iter 4 recovered (3498 s, close to iters 1–2 mean 3180 s), making "dataset noise" the more likely explanation for now — but a single post-outlier recovery is not conclusive. **No per-prompt instrumentation exists to distinguish.**

---

## 9. Publish latency 1.84× on second publish

**Evidence.**

| # | Step | pv | publish_latency_s | transfer_s | vllm_load_s | endpoints_ok |
|---|---|---|---|---|---|---|
| 1 | 5 | 0 → 1 | 18.21 | 4.03 | 14.19 | 4 |
| 2 | 10 | 1 → 2 | **33.59** | **19.00** | 14.59 | 4 |

`vllm_load_latency_s` stable (14.2 / 14.6). The regression is in `transfer_latency_s` — **4.7× on publish #2**.

**Mechanism (unverified).** Publish #2 fired at step 10, which is also the exact moment iter 4 was starting in the producer (iter 3 had just emitted). Iter startup spawns a DAPO dispatcher rebuild + 32 OpenHands worker session re-inits — all hitting the pool inbound bandwidth (`/chat/completions` calls) at the moment the publisher wants to push the adapter tensor over the same network. Contention hypothesis; would need `iftop`-style inbound wire profile to confirm.

**Why it matters.** Publish is a single-event diagnostic. One data point is not a trend, but if publish #3 also exceeds 30 s (pending at time of writing), this is a systemic publish-vs-generate contention issue.

---

## Summary — which problems are independent and which are coupled

Grouping by root cause:

| Root | Problems |
|---|---|
| **Architectural reframe: trainer = pure consumer of a big buffer** | #1 (1-group floor), #5 (double advantage). Both dissolve when trainer stops waiting on batches and just samples from a large pool of pre-stamped `(logprobs, advantages)`. |
| **Producer-bound regime** | #2 (trainer idle), #6 (pool-vs-buffer-age divergence), #7 (validation race), #8 (iter 3 wall). All symptoms of "producer wall dominates trainer wall". Fix with parallel producers + keep-everything buffering. |
| **Config / numerics** | #3 (clip_fraction 60 % → raise cap to ~4–5), #4 (response_length cap → bump to 4096). Both are straightforward knob flips. |
| **Network contention** | #9 (publish #2 latency). Possibly #2-related (iter startup coincident with publish). |

---

## What the next run should look like

Direct knob deltas from Run9 baseline, aligned with the reframe above:

| Knob | Run9 | Next run | Why |
|---|---|---|---|
| `data.max_response_length` | 1536 | **4096** | Problem #4 — stop truncating multi-turn SWE-Gym trajectories mid-tool-call. |
| `actor_rollout_ref.actor.tis_imp_ratio_cap` | 2.0 | **5.0** | Problem #3 — T-mismatch floor puts most of the ~0.55 log-ratio beyond log(2); raise cap so clamp bounds real drift, not numerical noise. |
| `replay.buffer_size` | 128 | **256+** | Problem #1 — trainer as pure consumer needs a pool big enough for genuine random sampling. |
| Trainer batch coupling | `n_groups = max(1, tbs // n)` | **drop the floor**, read `tbs` as "groups to sample per step" | Problem #1 — trainer samples from buffer, does not wait on producer. |
| DAPO ingest filter | Option A (filter at push), batch-held until `train_batch_size` | **Option A stays — filter is not dropped**; push each survivor eagerly into the buffer instead of batch-holding | Problem #1 + #2 — survivors stream into buffer the moment they clear the filter, trainer samples whenever enough survivors exist. |
| Producer count | 1 | **2 concurrent** (if pool capacity allows) | Problem #2 — halve arrival latency, keep staleness ≤ K naturally. |
| Validation `producer.stop` timeout | 10 s | **300+ s** (or switch to pause/resume) | Problem #7 — stop losing validation datapoints to cooperative skips. |

Goal of the next run: **fast-improving DAPO** — reward trend clearly up across 20+ steps, `clip_fraction < 0.2` under the raised cap, ≥ 2 validation datapoints inside the first 20 steps, trainer utilisation meaningfully above 10 %.

---

## Cut 6 production validation + Cut 8 follow-up (2026-04-25/26)

The Cut 1–6 plan landed and ran a 25-step DAPO production training (`/tmp/s3-fullasync-cut6-prod.log`). Outcomes:

| Goal | Achieved? | Evidence |
|---|---|---|
| Trainer stall → 0 after warmup | **Partially** | Steps 11→12 ran in 10 min (warm buffer), but call-boundary gaps re-introduced ~26 min stalls between calls #3 and #4. |
| `is_weight/clip_fraction < 0.2` | Cut 2 (cap 2→5) shipped; observed clip held to ~0.06–0.44 range across steps 1–11, well below the 60 % seen in Run9. | `response_length/clip_ratio` 0.06–0.44; Cut 1 (1536→4096) holding. |
| ≥ 2 in-training pass@k datapoints | Not measured this run — `test_freq=-1` for the prod 25-step. | (Cut 8 prep-100 run sets `test_freq=10` to capture this.) |
| Trainer util > 10 % | Yes, when buffer is warm. Buffer-bound during call boundaries pulls the average down. | Mixed step pacing: 10 min ↔ 60 min depending on call phase. |

**The new finding (call-boundary dead gap).** Cut 5's eager-push smoothed pushes within a call, but DAPO's per-call lifecycle still has a hard ~26 min trough between calls (full description in handsoff §31). Two non-exclusive levers close it:

| Lever | Effect | Cost |
|---|---|---|
| **Cut 8 — bigger `gen_batch_size` (default 16 → 32)** | Longer hot phase, fewer call boundaries per hour, lower fraction of wall in the trough. | One env-var bump. Trivial; reversible. |
| **Cut 7 — multi-producer fan-out** | Producer B's hot phase covers producer A's trough; trainer never sees the gap. | Code change; OH-server `/start`/`/stop` race needs fix (per-producer OH server, or no-op the lifecycle). |

**Decision: validate Cut 8 first via prep-100 run, defer Cut 7 to a fresh session.** Run the 100-step prep with `GEN_BATCH_SIZE=32 TEST_FREQ=10` (env overrides; defaults in `s3_fullasync_docker.sh` stay at 4× and `-1` until the run completes cleanly). Only ship Cut 8 as the new default after the 100-step run lands without regression. Only revisit Cut 7 if Cut 8 still leaves the trainer stalling after warmup.

### Knob deltas (Cut 8 prep-100 run vs Cut 6 prod baseline)

| Knob | Cut 6 prod | Cut 8 prep-100 | Why |
|---|---|---|---|
| `GEN_BATCH_SIZE` | 16 | **32** | Halve call-boundary frequency. Default bumped in `s3_fullasync_docker.sh`. |
| `TOTAL_TRAINING_STEPS` | 25 | **100** | Real training-trend window; first run that can show pass@k movement across 10 datapoints. |
| `TEST_FREQ` | `-1` | **10** | Capture 10 in-training pass@k datapoints (steps 10/20/.../100). New env var; default stays `-1`. |
| `SAVE_FREQ` | 5 | 5 | Unchanged — same pool publish cadence. |

### Cut 8 prep-100 outcome (2026-04-26)

prep-100 ran 24 successful steps (resumed from step 20 → reached step 44) before crashing with `RuntimeError: Replay store did not reach 4 fresh groups within 7200.0s`. Root cause was *not* a wedged producer — it was a hard-coded 7200 s ceiling on the trainer's wait that didn't account for `response_length/mean` climbing 9 045 → 12 071 between step 43 and step 44 (clip_ratio 0.22 → 0.34). Producer per-iteration wall scales with response length; the threshold was sized for the original ~1500-token regime.

15 cooperative §19 validation skips fired during the run, so no in-training pass@k datapoints landed — but training itself was healthy (`reward_metrics/all` 0.5–0.53 around step 43–44, `actor/grad_norm` ~0.025).

Latest LoRA checkpoint preserved: `outputs/ProAgent/fullasync/global_step_40/actor/lora_adapter/` (253 MB adapter + adapter_config.json). Mid-run reference: `global_step_20/`. All other prep-100 checkpoints were deleted to free disk during session shutdown.

### Cut 9 — no-progress detector replaces 7200 s ceiling (shipped)

`wait_until_with_progress` in `verl_custom/replay/continuous_producer.py` now drives `_acquire_training_batch_dapo`. Resets the deadline whenever `trajectory_store.total_pushes()` (new monotonic accessor) grows; aborts only when no group lands for `replay.no_progress_timeout_s` seconds (default 1800 s, plumbed via `+replay.no_progress_timeout_s=1800` in `s3_fullasync_docker.sh`). Old `replay.wait_timeout_s` knob removed. Test coverage: `tests/replay/test_continuous_producer.py::TestWaitUntilWithProgress`.

This changes failure semantics:
- "Producer healthy but slow as model learns longer trajectories" → trainer waits, never aborts. **Desired.**
- "Producer wedged / pool dead / push thread stalled" → no `total_pushes` growth → trainer aborts after 1800 s with a clearer message.

### Next session — Cut 7 (multi-producer fan-out)

**This is the named next architectural step.** Land it only if the next prep-100 run still shows the trainer waiting on `_acquire_training_batch_dapo` after warmup (i.e. `gen_batch_size=32` did not fully cover the call-boundary trough).

Shape:
- Two `AsyncLLMServerManagerDAPO` + `ContinuousRolloutProducer` pairs sharing one `TrajectoryStore` (single `threading.Lock`, safe).
- Per-producer disjoint dataloader slice (offset by rank, stride by `num_producers`).
- **Per-producer OpenHands FastAPI** — `:8006` for producer A, `:8007` for producer B — to avoid the `/start`/`/stop` race on a shared server (handsoff.md §17 / §31). Or alternatively no-op `start_servers`/`stop_servers` when `num_producers > 1`.
- Pool-saturation guard: drop `OPENHANDS_NUM_WORKERS` to 16 per producer (still 32 concurrent on the pool).

Files that will change (when Cut 7 lands):
- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` — `_start_continuous_producer_if_needed` spawns `replay.num_producers` copies.
- `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` — `producer_index` / `num_producers` in ctor; per-producer dataloader slice.
- `scripts/_internal/s3_fullasync_docker.sh` — new `NUM_PRODUCERS` env var.
- ProRL launcher — second OH server on `:8007`.
- `plans-n-solutions/handsoff.md` — promote §31 deferred → shipped, document the OH-server-per-producer invariant.

```bash
# On trainer box (ProRL running at :8006, pool warm on vllm-instance:8100-8103)
TOTAL_TRAINING_STEPS=20 SAVE_FREQ=5 NUM_TRAJ=16 TEST_FREQ=10 VAL_BEFORE_TRAIN=True \
  bash scripts/_internal/s3_fullasync_docker.sh

# Monitor
python /tmp/replay_monitor.py &        # emits /tmp/replay-monitor.jsonl
tail -f /tmp/s3-fullasync-*.log
```

All nine problems surface within the first 20 steps.
