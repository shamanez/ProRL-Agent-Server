# Run9 — n=16 + filter_groups=True + fully-async replay (baseline config)

Branch: `full-async-optimization`. Container: `s3-fullasync`. Launcher: `scripts/_internal/s3_fullasync_docker.sh`. Launched 2026-04-24T03:09:16Z. Log: `/tmp/s3-fullasync-n16-baseline.log`. Monitor JSONL: `/tmp/replay-monitor.jsonl`.

**Purpose.** Observe clock-separation dynamics of the fully-async topology under the paper-aligned config (`n=16` trajectories per prompt, DAPO `filter_groups=True`, K=4 staleness cap, FIFO 128-group replay) and drive the problem sheet from measurement, not intuition.

**Config deltas vs the earlier n=8 A/B** (`s3_fullasync_docker.sh FILTER_GROUPS=True`, 50-step):

- `actor_rollout_ref.rollout.n`: 8 → **16** (paper §5.1 for meaningful group-stat signal)
- `trainer.test_freq`: -1 → **10** (validation every 10 steps)
- `trainer.val_before_train`: False → **True** (capture pass@k baseline at pv=0)
- All other knobs unchanged — "stick to base, do not change the code".

## Progress snapshot (as of 2026-04-24T06:55Z, +3 h 46 min wall-clock)

| Metric | Value |
|---|---|
| Training Progress | **12 / 500** (3 iter bursts complete; iter 4 producing) |
| DAPO producer calls completed | 3 (iter 1 52.6 min, iter 2 53.4 min, iter 3 **80.4 min**) |
| Trainer steps with published metrics | 10 (steps 1–10; 11–12 consumed, metrics pending flush) |
| LoRA publishes | **2** (step 5 → pv=1, step 10 → pv=2) |
| Pool policy_version (8100/8101/8102/8103) | 2/2/2/2 |
| `weight_sync/endpoints_failed` (cumulative) | **0** |
| §19 cooperative skips (fit()-time) | **1** (step 10 validation skipped — producer mid-iter-4 `generate_sequences_dapo`) |
| fit()-time tracebacks | 0 |

## Timeline — the "53-min iter / 3-min burst" steady state (plus iter 3 long-tail)

```
T+00:00:00   Training starts. val_before_train=True kicks off.
T+00:21:??   Initial pass@k validation completes (pv=0 baseline).
T+00:21:??   DAPO iter 1 starts.
T+00:53:33   Step 1 complete → iter 1 emits (drawn=10, survived=4, dropped=1, wall=3158s).
T+00:54:17   Step 2.
T+00:55:01   Step 3.
T+00:55:43   Step 4.  Buffer → 0. Trainer idle begins.
T+01:47:35   Step 5 → iter 2 emits (drawn=10, survived=4, dropped=2, wall=3201s).
             → save_checkpoint (step_5/actor) + LoRA publish (pv=1, latency 18.2 s, endpoints_ok:4).
T+01:48:17   Step 6.
T+01:49:01   Step 7.
T+01:49:43   Step 8.  Buffer → 0. Trainer idle begins (50 min).
T+03:07:25   Step 9 → iter 3 emits (drawn=15, survived=4, dropped=6, wall=4824s — 52 % over).
T+03:09:15   Step 10 → save_checkpoint (step_10/actor) + LoRA publish (pv=2, latency 33.6 s).
T+03:09:??   **§19 cooperative skip: step=10 _validate() aborted — producer thread
              mid-iter-4 generate_sequences_dapo; stop(timeout=10s) timed out.**
T+03:09:59   Step 11.
T+03:10:41   Step 12.  Buffer → 0. Trainer idle begins.
T+03:28:02*  DAPO iter 4 expected end (>= 54 min, likely 70–80 min based on iter 3).
```

(`*` = projection. As of 06:55 UTC, iter 4 is at elapsed 1111s = 18 min, 0/4 survivors yet.)

### Per-step wall-clock (tqdm)

| Step | Wall-clock | Δ prev | Cause | pv | stal | rew |
|---|---|---|---|---|---|---|
| 1 | 00:53:33 | — | iter 1 drain burst start | 0 | 1 | 0.0625 |
| 2 | 00:54:17 | +44 s | burst | 0 | 2 | 0.1875 |
| 3 | 00:55:01 | +44 s | burst | 0 | 3 | 0.375 |
| 4 | 00:55:43 | +42 s | burst; store → 0 | 0 | 4 (K cap) | 0.4375 |
| 5 | 01:47:35 | **+51 min 52 s** | idle + iter 2 + publish pv=1 | 1 | 0 | 0.9375 |
| 6 | 01:48:17 | +42 s | burst | 1 | 1 | 0.875 |
| 7 | 01:49:01 | +44 s | burst | 1 | 2 | 0.125 |
| 8 | 01:49:43 | +42 s | burst; store → 0 | 1 | 3 | 0.0625 |
| 9 | 03:07:25 | **+77 min 42 s** | idle + iter 3 (long tail!) | 1 | **4 (K cap)** | 0.0625 |
| 10 | 03:09:15 | +1 min 50 s | publish pv=2 + **validation SKIPPED (§19)** | 2 | 0 | 0.8125 |
| 11 | 03:09:59 | +44 s | burst | 2 | (pending) | (pending) |
| 12 | 03:10:41 | +42 s | burst; store → 0 | 2 | (pending) | (pending) |

**Trainer utilization through step 12: 12 × ~43 s active over 3 h 46 min = 2 min 36 s / 226 min = 1.1 %.** (4.3 % in first 8 steps; iter 3's 80-min wall regressed the running average.)

## Replay / staleness dynamics (steps 1–10)

| step | store_size | traj | replay/sample_age_p50 | rollout/staleness_steps | pv | `rollout_corr/log_ppl_diff` |
|---|---|---|---|---|---|---|
| 1 | 3 | 48 | 0 | 1 | — | **0.530** |
| 2 | 2 | 32 | 1 | 2 | — | 0.697 |
| 3 | 1 | 16 | 2 | 3 | — | 0.718 |
| 4 | 0 | 0 | 3 | **4 (K!)** | — | 0.785 |
| 5 | 3 | 48 | 0 | 0 | **1** | 0.739 |
| 6 | 2 | 32 | 1 | 1 | — | 0.614 |
| 7 | 1 | 16 | 2 | 2 | — | 0.488 |
| 8 | 0 | 0 | 3 | 3 | — | 0.747 |
| 9 | 3 | 48 | 0 | **4 (K!)** | — | 0.626 |
| 10 | 2 | 32 | 1 | 0 | **2** | 0.726 |

Observations:

1. **`sample_age` (group-age) vs `staleness_steps` (pool-adapter-age) diverge when producer-wall exceeds save_freq.** Step 9 shows `sample_age_p50=0` (iter 3 groups are fresh) but `staleness_steps=4` (pool is 4 steps behind current trainer because there was no publish between step 5 and step 9). The paper's "buffer age" intuition and the fork's "pool-adapter age" invariant measure **different things** and will diverge whenever iters last longer than `save_freq × burst_duration`.
2. **`log_ppl_diff` (log IS weight) oscillates 0.49–0.79 across all 10 steps** with no clear drift. With `tis_imp_ratio_cap=2`, the clip threshold is `log(2) ≈ 0.69`. **Steps 2, 3, 4, 5, 8, 10 all exceed this** → clipping is firing on roughly 60 % of steps.
3. **K=4 hit twice** (step 4 and step 9) — both times rescued by a fresh publish on the next step. Had iter 3 been even 1 step longer, step 10's buffer draw would have been stale-evicted and trainer would have blocked on the `wait_for_fresh_group` path.

## DAPO producer wall-clock and filter behaviour

| iter | start | end | wall_s | drawn | survived | dropped_filter | completion% | effective_tps |
|---|---|---|---|---|---|---|---|---|
| 1 | 00:21 | 00:53 | 3158.2 | 10 | 4 | 1 | 40 % | 664.0 |
| 2 | 00:54 | 01:47 | 3201.2 | 10 | 4 | 2 | 40 % | 655.1 |
| 3 | 01:47 | 03:07 | **4824.4** | **15** | 4 | **6** | 27 % | **434.7** |
| 4 | 03:07 | in-flight | ≥ 1111 | ? | 0/4 (as of 06:55) | ? | — | — |

Iter 3 is a **significant regression**:
- Wall-clock: +52 % vs iters 1–2.
- Completion rate: 40 % → 27 %. DAPO kept drawing prompts (10 → 15) because surviving groups wouldn't fill the batch.
- Effective TPS: 664 → 435 (−34 %).

Two plausible causes (not yet instrumented):

- **Dataset difficulty drift.** Iter 3's first 10 prompts happened to be harder (more zero-variance groups → more drops). DAPO drew 5 extras to compensate.
- **Pool drift after publish 1.** pv=1 is the first *actual* trained policy (pv=0 = pristine SFT). If pv=1 regresses on some prompts (noise at 4-step LR=1e-6, plausible), the completion rate naturally dips. Publish 2's `publish_latency_s=33.6` (vs 18.2 at publish 1, `transfer_latency_s` jumped 4.0 → 19.0) adds independent evidence the pool was under unusual load during this window.

Needed: per-iter per-prompt success rate logged to distinguish (would change `async_server_dapo.py` DAPO_PRODUCER_CALL event to emit list of `(prompt_uid, resolved_ratio)`).

## Publish cadence

| # | Step | pv | publish_latency_s | transfer_s | vllm_load_s | endpoints_ok | adapter_mib |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 0 → 1 | 18.21 | 4.03 | 14.19 | 4 | 232.61 |
| 2 | 10 | 1 → 2 | **33.59** | **19.00** | 14.59 | 4 | 231.91 |

Publish 2's `transfer_latency_s` is **4.7× publish 1** (19.0 vs 4.0 s). `vllm_load_latency_s` is stable (14.6 vs 14.2). The extra 15 s of wire-time correlates with iter 4 concurrent startup (dispatcher rebuild, 32 OpenHands workers re-initialising sessions) competing for the pool's inbound bandwidth. **Non-fatal** — flags a follow-up: consider gating publish-push on a low-activity window, or moving publish-transfer onto a dedicated HTTP client.

## Validation trajectory

### Baseline (pv=0, pre-training)

```
val-core/pass_k/pass@1/score:  0.0652 (± 0.224)    # 3 of 23 pass
val-core/swe-gym/reward/mean@2: 0.0652 (± 0.022)
val-core/swe-gym/reward/best@2/mean: 0.0762        # best-of-2
val-aux/swe-gym/reward/worst@2/mean: 0.0535        # worst-of-2
val-aux/swe-gym/reward_metrics/finish_action_ratio/mean@2: 0.696
val-aux/swe-gym/reward_metrics/stuck_ratio/mean@2: 0.043
val-aux/swe-gym/reward_metrics/max_turn_ratio/mean@2: 0.000
```

### Step 10 (pv=2) — **SKIPPED via §19 cooperative path**

```
(TaskRunner pid=6812) WARNING:2026-04-24 06:40:08,460: ContinuousRolloutProducer did not exit within 10.0s;
  leaving thread running (mid-generate_sequences); caller must skip contention with the producer and retry
  stop() at the next boundary
(TaskRunner pid=6812) WARNING:2026-04-24 06:40:08,525: step=10 skipping _validate: producer stop timed out
  (still mid-generate_sequences); will retry on next save boundary
```

**This is the cooperative-skip path from fix `590f8281` engaging for the first time during fit().** One §19 **during fit()** — at step 10, with `val_before_train=True + test_freq=10` we expected a pass@k datapoint here; instead the trainer chose safety over contention with the producer.

**Root cause.** Step 10's sequence is: sample → update_actor → adv → save_checkpoint → publish_lora (pv=2) → `_validate()` called. `_validate()` wants an exclusive pool, so it calls `producer.stop(timeout=10.0)`. The producer is mid-`generate_sequences_dapo(iter=4)` — a 53+ min call. Stop's cooperative-flag check only happens at the *top* of the worker loop (between iters), so the flag lands but the worker is deep in an async IO wait. Timeout elapses → producer stays alive → `_validate()` skips to avoid double-drain-on-pool semantics.

**Next validation opportunity.** `save_freq=5` triggers at step 15 and publish happens. `_validate()` retries at the next save boundary, which is step 15. But `test_freq=10` means only step 10, 20, 30… are validation-eligible. So the next real attempt is step 20 (~7 h from now at current pace). At step 20, the same race recurs unless either (a) iter 5 happens to be between generate_calls when step 20 hits — highly dependent on alignment — or (b) we fix the race.

**Fix candidates** (ordered by least-invasive):

1. **Increase `producer.stop` timeout at validation boundary to a large value** (e.g. 300 s). The worst case is that `_validate()` waits for the in-flight DAPO iter to finish (~53–80 min) before running. Slow, but deterministic. — simple config change.
2. **Add a `producer.pause()` / `producer.resume()` path** that lets the worker finish its current `generate_sequences_dapo` call, pushes results to the buffer as normal, then pauses instead of re-entering the next iter. `_validate()` runs during the pause window. — ~40 LOC in `continuous_producer.py`.
3. **Reserve pool child #3 for validation** — permanent split, validation always has dedicated capacity. — infra change (4-way → 3+1 pool topology). **Do not do.**

Option **1** is a zero-code knob flip — `ContinuousRolloutProducer.stop(timeout=…)` is already a parameter. Let the validation path pass `timeout=7200` (2 h). Simple fix, ships with report.

Option **2** is the principled answer — it's the correct semantic for clock separation. Defer unless Option 1 proves flaky.

## Reward signal (steps 1–10)

| step | critic/rewards/mean | critic/advantages/mean | actor/grad_norm | pv at generation |
|---|---|---|---|---|
| 1 | 0.0625 | -0.050 | 0.36 | 0 |
| 2 | 0.1875 | -0.022 | 1.71 | 0 |
| 3 | 0.375 | -0.002 | 1.04 | 0 |
| 4 | 0.4375 | -0.036 | 0.97 | 0 |
| 5 | 0.9375 | 0.008 | 0.75 | 0 (then publish) |
| 6 | 0.875 | 0.002 | 0.33 | 1 |
| 7 | 0.125 | -0.008 | 0.35 | 1 |
| 8 | 0.0625 | -0.001 | 0.23 | 1 |
| 9 | 0.0625 | 0.014 | 0.81 | 1 (then publish) |
| 10 | 0.8125 | -0.023 | 0.42 | 2 |

Pattern: **within each iter's burst, rewards rank-correlate with which prompts DAPO happened to draw** — first two bursts are "easy at head, hard at tail" (iter 1: 0.06 → 0.19 → 0.37 → 0.44; iter 2: 0.93 → 0.87 → 0.12 → 0.06). No cross-iter trend yet — 10 steps on 4B + rank-16 LoRA @ 1e-6 is nowhere near the paper's 5k-50k step learning window. Expected.

## Headline signals (ranked, updated after 10 steps)

### 1. Trainer starves ~99 % of the time at n=16

12 steps × 43 s active = 516 s of FSDP compute over 226 min wall-clock = **3.8 % utilisation**, skewing *down* as iter 3's 80-min wall pushes the average below early-run 4.3 %. The fully-async topology is **not** engaging — producer is the strict bottleneck, buffer spends 50+ % of wall-clock at `store_size=0`.

**Leverage, ranked:**

- **a. Parallel DAPO producers.** At 2 concurrent workers: arrival rate ~2× → staleness holds ≤ K, trainer bursts overlap producer idle. **Highest expected lift.**
- **b. Bigger buffer + positive-bias sampling.** Keep every rollout (survived or not) and let the trainer sample uniformly from a large pool of pre-computed `(logprobs, advantages)`. Reclaims the 60 % filter-drop wall-clock as sparse-but-real gradient signal. Doesn't speed up iter wall, but converts 4 survivors → ~8 effective "gradient-carrying" groups per iter.
- **c. Producer wall-time instrumentation.** Iter 3's 80-min slowdown is unexplained. Emit per-prompt `(uid, resolved_ratio, wall_s)` in DAPO_PRODUCER_CALL. Cheap.

### 2. Pool-adapter-age ≠ buffer-age when iters are slow

From step 9: `rollout/staleness_steps=4` (pool-adapter-age) while `replay/sample_age_steps=0` (buffer-age). The clip mechanism operates on trainer-vs-pool logprob divergence, which correlates with pool-adapter-age. But the buffer's eviction policy uses buffer-age. **They can diverge by K or more in this regime.** Meaning `replay/sample_age_steps_p95 ≤ K=4` (success gate 4) is *not sufficient* evidence the IS weights are healthy — we need both metrics tracked.

- Rename/clarify the two metrics in handsoff.
- Add `gate 4b`: `rollout/staleness_steps_p95 ≤ some-bound` (probably K + save_freq = 9 for current config).
- Consider tying buffer eviction to pool-adapter-age rather than buffer-age (more conservative, prevents stale-policy samples even if buffer is fresh).

### 3. Validation-vs-producer race (NEW)

First-fit()-time §19 skip. The fix is straightforward (option 1 or 2 above) and does not block the current run. But it means **we will not get a step-10 pass@k datapoint** — the first in-training transfer measurement slips to step 20, more than tripling the time-to-first-eval. Priority fix.

### 4. Iter-3 wall-clock regression is a signal to watch

Iter 3 = 80 min vs expected 53 min. Iter 4 recovered to 58 min — likely dataset noise, not pool degradation (see addendum). Still worth instrumenting so future regressions can be diagnosed without waiting for another outlier.

### 5. Publish latency jumped 1.84× on second publish

18.2 s → 33.6 s. Transfer component 4.0 → 19.0 s (4.7×). vLLM load component stable. Correlates with concurrent iter-4 dispatcher startup. Publish is a ~single-event diagnostic for pool inbound bandwidth; if publish 3 is also > 30 s, this is a systemic issue worth investigating.

## Success-gate status (plumbing — live update)

| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | `weight_sync/endpoints_failed == 0` | **PASS** | 2/2 publishes, endpoints_ok:4 |
| 2 | ≥ 4 `/reload_lora` per 20 steps at save_freq=5 | on pace | 2 publishes / 10 steps |
| 3 | Zero 5xx on `/generate` during publishes | **PASS** | both publishes drained clean |
| 4 | `replay/sample_age_steps_p95 ≤ K=4` | **PASS** | max=3 observed; `dropped_by_staleness_total=0` |
| 4b | *(new)* `rollout/staleness_steps_p95 ≤ K + save_freq` | **PASS** (with caveat) | hits K=4 at steps 4 and 9 — **one tick from eviction** |
| 5 | `is_weight/p99 < 10`, `clip_fraction < 0.2` | **FAIL (proxy)** | `log_ppl_diff` > `log(2)=0.69` on 6 of 10 steps → clip fraction ~60 % (well above 20 % cap). Mechanism working correctly; **K=4 + iter-length combination is too tight.** |
| 6 | `critic/rewards/mean` trends up | inconclusive | mean per burst: 0.27, 0.50, 0.30 — flat with noise (expected for 10 steps on 4B LoRA) |
| 7 | Offline A/B on validation.parquet | **BLOCKED** | step-10 validation skipped via §19; next attempt step 20 |
| 8 | Both `filter_groups={False,True}` land clean | in-progress | `filter_groups=True` PASS-so-far; `filter_groups=False` task still pending |
| 9 | Zero fit()-time tracebacks / §19 skips | **FAIL (new)** | 1 fit()-time §19 skip at step 10 — validation skipped, no corruption |
| 10 | Token-in/token-out preserved | **PASS** | golden test holds; no re-tokenization drift observed |

**Gates 5, 7, 9 failing.** Each has a concrete fix identified:

- Gate 5 (`clip_fraction > 0.2`): raise `tis_imp_ratio_cap` (the dominant ~0.55 log-ratio is T=1.4 vs T=1.0 numerical mismatch, not real drift — see `current_bottlenecks_and_problems.md` #3). Lever (a) above also helps by shrinking pool-adapter age.
- Gate 7 (validation blocked): fix §19 validation race. Lever (3) above.
- Gate 9 (fit()-time §19): same fix as gate 7.

## Open TODOs (this report)

- [x] Step 10 completion + second LoRA publish (pv 1 → 2) ✓ published 06:39:06, pv 2/2/2/2
- [x] Step 9–12 timing ✓ step 9 +77 min (iter 3), step 10 +1:50 (publish+skipped-val), steps 11–12 normal burst
- [x] §19 cooperative skip logged ✓ step=10 _validate skipped at 06:40:08
- [x] Iter 3 DAPO stats ✓ 80 min, 15 drawn, 4 survived, 6 dropped, eff_tps=435 (vs 665)
- [x] `rollout_corr/log_ppl_diff` trajectory across steps 1–10 ✓
- [ ] Cross-check `/tmp/replay-monitor.jsonl` aggregates match log-derived numbers — partial (monitor shows 12/500, 3 calls, 35 drawn, 12 survived — matches)
- [ ] Step 20 validation (next opportunity for pv-vs-baseline comparison) — out of scope for this report; ~7 h from now

## Bottom line

The n=16 config surfaces **three distinct signals in 10 steps** that the earlier n=8 A/B did not:

1. **Clip fraction ~60 %** (vs paper target < 20 %) — clip mechanism is working but saturated, meaning the gradient signal is being heavily biased by the clamp. Dominant term is T=1.4 vs T=1.0 numerical mismatch, not real off-policy drift. Fix: raise `tis_imp_ratio_cap` (cheap) or shrink pool-adapter age via parallel producers.
2. **Pool-adapter-age vs buffer-age decoupling** — metric-semantics issue, needs disambiguation in docs + a second gate.
3. **Validation-vs-producer race** — first real fit()-time §19 skip, gates the path to first pv > 0 pass@k datapoint. Direct fix identified.

All three are actionable and have concrete proposed fixes.

## Addendum — iter 4 recovery (2026-04-24T08:00Z, +4h 51m)

Iter 4 completed: `wall_s=3497.6, drawn=11, survived=4, dropped_filter=2, effective_tps=599.6`. Pool policy_version = 4/4/4/4 at step 16 (publish #4). Iter 3 was an **outlier**, not a trend — filter drop rate is back to 18 % (2/11). Running iter stats:

| iter | wall_s | drawn | survived | dropped | eff_tps |
|---|---|---|---|---|---|
| 1 | 3158 | 10 | 4 | 1 | 664.0 |
| 2 | 3201 | 10 | 4 | 2 | 655.1 |
| **3** | **4824** | **15** | **4** | **6** | **434.7** |
| 4 | 3498 | 11 | 4 | 2 | 599.6 |

Mean (excluding iter 3 as outlier): 3285.7 s / 653 eff_tps. Iter 3 remains unexplained — pool noise, dataset difficulty drift, or post-publish-1 policy regression. Priority: emit per-prompt `(uid, resolved_ratio, wall_s)` in DAPO_PRODUCER_CALL so we can distinguish.

**Gate 9 (no fit()-time §19)** still FAIL as documented (step 10 validation skipped). **Gate 5 (`clip_fraction < 0.2`)** still FAIL — `log_ppl_diff` has not trended below log(2) in 16 steps. Gates unchanged overall.
