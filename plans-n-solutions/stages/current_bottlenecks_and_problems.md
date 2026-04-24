# Current bottlenecks and problems — Run9 moment-of-truth

Branch: `full-async`. Basis: Run9 (n=16, DAPO `filter_groups=True`, K=4 staleness cap, 128-group FIFO buffer, LR=1e-6, rank-16 LoRA). Log: `/tmp/s3-fullasync-n16-baseline.log`. Monitor: `/tmp/replay-monitor.jsonl`. Full report: `run9_n16_report.md`.

**Scope.** Problems observed in Run9 only. No fixes proposed — see handsoff §16.2 and next session's deepdive work. This doc is the evidence sheet.

---

## 1. Each trainer step sees only 1 group → no cross-prompt gradient averaging

**Evidence.** `ray_trainer_dapo.py:86-87`:

```python
n        = config.actor_rollout_ref.rollout.n                    # Run9: 16
n_groups = max(1, config.data.train_batch_size // max(1, n))     # max(1, 4 // 16) = 1
```

With `train_batch_size=4, n=16`, the floor triggers → **1 group = 16 trajectories per trainer step**. Classic (Phase 1) mode would have produced 4 groups × 16 = 64 trajectories per step.

**Mechanism.**

- `ppo_mini_batch_size=4` rows → 16/4 = **4 mini-batches per trainer step = 4 optimizer steps**. Every mini-batch contains 4 sibling trajectories from the **same** prompt. No cross-prompt gradient averaging is possible within a step.
- GRPO advantage baseline is computed per uid (`core_algos.py:168-221`). With 1 group per step, every opt step uses the same single-prompt baseline. Difficulty context across tasks only averages over *time* (across 4 consecutive trainer steps per iter), not within a step.
- The 4 trainer steps per iter are sequential: `opt×4 on prompt A → opt×4 on prompt B → opt×4 on prompt C → opt×4 on prompt D`. Closer to sequential-task learning than batched SGD. Known catastrophic-forgetting regime, mitigated (not eliminated) by tiny LR=1e-6 + rank-16 LoRA.

**Why `max(1, ...)` hides it.** The formula silently collapses to 1 whenever `n ≥ train_batch_size`, which is every regime we have ever run. Upstream verl defines `data.train_batch_size` as *prompts per step* (dataloader `batch_size=`, `ray_trainer.py:358`); the fork reads it as "total trajectories" here and the floor then floors a nonsensical quotient.

**Observed consequence.** Contributes to the `is_weight/clip_fraction ~60%` symptom (problem #3) — the narrower the effective batch per opt step, the higher the variance of the IS ratio's per-step mean, the more clamping fires.

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

**Why agentic RL makes this specifically bad.** Multi-turn OpenHands on SWE-Gym with `max_iterations=30`, `openhands_timeout=1000s`, 4B model on 4× L4 → ~53 min per iter at n=16 (n=8 was ~30 min in Run8). The paper (Arnal et al. 2026) assumes producer and trainer rates within ~2× of each other — we are operating in a regime their paper does not cover.

---

## 3. `is_weight/clip_fraction ~60 %` — IS ratio saturated against the clamp

**Evidence.** `log_ppl_diff` (log IS weight) across Run9 steps 1–10: 0.530, 0.697, 0.718, 0.785, 0.739, 0.614, 0.488, 0.747, 0.626, 0.726. Clip threshold = `log(tis_imp_ratio_cap) = log(2) ≈ 0.693`. Steps 2, 3, 4, 5, 8, 10 exceed it → mean clip fraction ~60 %. Paper target (§5.2, Fig 13): < 20 %.

**What this means mechanically.** `core_algos.py:586-590`:

```python
tis_imp_ratio = torch.exp(old_log_prob - rollout_log_probs)
tis_imp_ratio = torch.clamp(tis_imp_ratio, max=tis_imp_ratio_cap)   # cap = 2.0
pg_losses    *= tis_imp_ratio
```

- The clamp **truncates** ratio weight, does not mask tokens. 60 % of tokens still contribute to the loss, just with gradient weight capped at 2.0 regardless of true ratio (may be 3, 10, 100).
- Asymmetric: `max=2` cap but no `min=0.5` floor. Under-weights tokens where policy moved *away*, symmetry broken.
- Tokens with largest true ratio are also the ones the policy is moving fastest on. Capping them shrinks effective LR exactly where gradient matters most.

**Decomposition of the ~0.55 mean log-ratio (from handsoff gotcha #27):**

| Source | Approx share |
|---|---|
| Rollout sampling T=1.4 vs trainer `compute_log_prob` T=1.0 | **~0.35** |
| vLLM ↔ FSDP numerical (BF16 fused attention, rotary precision, kernel path) | ~0.20 |
| LoRA-specific kernel divergence | ~0.05 |
| Actual policy drift during slow producer iter | ~0.15 |

The dominant term is a numerical artifact (T-mismatch), not real off-policy-ness. Current clamp is fighting a known bias, not real drift — the correction it is supposed to provide has been captured by noise floor.

**Interaction with problem #1.** Narrower per-step batch (1 group) → higher per-step variance of `log_ppl_diff` → more likely to hit the clamp. These are coupled, not independent symptoms.

---

## 4. `response_length` saturating at 1536 — reward signal is corrupted

**Evidence.** Wandb panel shows `response_length/mean ≈ 1536` across most consecutive Run9 steps. `data.max_response_length=1536` is the hard truncation cap — this is systematic cap-hitting, not task-dependent variation.

**Mechanism.**

- OpenHands multi-turn rollout concatenates all assistant tokens across up to `max_iterations=30` turns into `response_ids`. Long thinking blocks (Qwen3 preserves `<think>…</think>` in `content`, per `openhands/llm/nvidia/qwen3.py` + CLAUDE.md token-in/token-out invariant), tool-call retry loops, and 30-turn accumulation easily exceed 1536.
- Truncation semantics: agent never reaches `finish_action` → reward scored as failure-ish regardless of whether the agent would have completed with more budget. **We are training the policy to associate "long task" with "low reward"** — but also with "truncated state", which is not what we want the signal to encode.
- Loss contamination: tokens right before the 1536 cut may be mid-`<tool_call>` JSON. Mask treats them as valid response tokens; PPO assigns gradient to "produce broken JSON" when local context looks coherent.

**Why this is not independent of problem #3.** Trajectories ending with hundreds of cap-adjacent thinking/stuck-sampling tokens are exactly the tokens where high-temperature sampling (T=1.4) diverges most from the trainer's `compute_log_prob` at T=1.0. **The saturated tail is simultaneously generating the highest-ratio tokens that hit the IS clamp.** Problems #3 and #4 likely share a root cause.

**Required diagnostic (not in current metrics).** Add `pct_capped = (response_length == 1536).float().mean()` and cross-plot against `val-aux/swe-gym/reward_metrics/finish_action_ratio` — if they move together, direct evidence of corrupted reward.

---

## 5. Advantages computed twice — dead architectural work

**Evidence.** `trajectory_store.py:308` stamps `advantage` (scalar per trajectory) at push time. `ray_trainer_dapo.py:338-347` calls `compute_advantage` again on the sampled batch, overwriting.

**Why both passes yield the same number.** Groups are atomic in the store (`push_group` / `sample_mini_batch` never split). GRPO advantage = (reward − group_mean) / group_std over the 16 siblings of a uid. Identical sibling set at push and sample → identical mean/std → identical advantage. The second pass is compute-but-same-result.

**Cost.** ~20 ms/step on the driver process. Small. But it is the *only* step where the buffer's data model is coupled to the trainer's downstream expectations — removing it would let the trainer treat the buffer as a pure `(tokens, advantages, rollout_log_probs, masks)` source.

**Minimum fields the PPO loss actually needs per row** (from reading `dp_actor.py`):

```
input_ids / responses                   # forward pass
loss_mask, response_mask, attention_mask # zero-out padding
advantages (broadcast over response_mask)
rollout_log_probs                        # IS denominator
```

Everything else in the current sampled DataProto (`uid`, `behavior_policy_version`, `token_level_scores`, `token_level_rewards`, `reward`, `resolved`) is metrics/diagnostics only. The double-compute exists because the buffer emits a **scalar** advantage but the loss reads a **broadcast tensor**, and `compute_advantage` happens to do both jobs.

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

**Blast radius.** No corruption — the cooperative skip path (commit `590f8281`) is the Phase 2 safety mechanism working as designed. But the first in-training `pass@k` datapoint (step 10 with `test_freq=10`) is lost. Next opportunity is step 20, which is ~7 h of wall-clock from step 10 at current producer rate. **Time-to-first-eval tripled** compared to planned.

**Contract.** This is success gate 9 ("zero fit()-time tracebacks / §19 skips") turning FAIL for the first time across all Phase 2 runs (Run8 had zero). Run8 saw 1 §19 at shutdown only, which is benign.

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
| **Numerical / config mismatch** (shared surface) | #3 (clip_fraction 60 %), #4 (response_length cap). Likely same underlying T-mismatch + cap behavior. |
| **Producer-bound regime** (shared surface) | #2 (trainer idle), #6 (pool-vs-buffer-age divergence), #7 (validation race), #8 (iter 3 wall). All symptoms of "producer wall dominates trainer wall". |
| **Architectural / formula** | #1 (1 group per step), #5 (double advantage). Fork-specific formula/layout choices, not regime problems. |
| **Network contention** | #9 (publish #2 latency). Possibly #2-related (iter startup coincident with publish). |

**Independent decisions for next session.** Each group can be attacked on its own; within a group, symptoms share a fix. The ordering question (what to land first) is in handsoff.md §16.2.

---

## Reproducing

```bash
# On trainer box (ProRL running at :8006, pool warm on vllm-instance:8100-8103)
TOTAL_TRAINING_STEPS=20 SAVE_FREQ=5 NUM_TRAJ=16 TEST_FREQ=10 VAL_BEFORE_TRAIN=True \
  bash scripts/_internal/s3_fullasync_docker.sh

# Monitor
python /tmp/replay_monitor.py &        # emits /tmp/replay-monitor.jsonl
tail -f /tmp/s3-fullasync-*.log
```

All nine problems surface within the first 20 steps.
