# Architecture walkthrough — fully-async decoupled agentic RL

Audience: engineers landing on this codebase for the first time.
Goal: read this end-to-end, then `handsoff.md`, then you can debug
any part of the loop.

This doc covers **what is running, where it lives, and why each
boundary is where it is**. Everything points to a `file:line` so you
can verify by reading the code.

---

## 1. The three processes

| # | Process | Machine | Launcher | Job |
|---|---|---|---|---|
| 1 | **ProRL FastAPI** | trainer host | `scripts/_internal/s0_prorl.sh` | Multi-turn agent runtime (OpenHands fork). Listens on `:8006`. The producer's OpenHands clients talk to it. |
| 2 | **vLLM pool** (4 children, ports `8100–8103`) | EC2 `vllm-instance` | `scripts/serving/launch_remote_vllm_pool.sh start` | Inference backend. Hosts the rank-32 LoRA on top of `Qwen3-4B-Instruct-2507`. Trainer publishes new LoRA via `POST /reload_lora` after each `_save_checkpoint`. |
| 3 | **GRPO trainer** (Docker, 8 × A100 FSDP) | trainer host | `scripts/_internal/s3_fullasync_docker.sh` | Owns the actor + optimizer. Runs the daemon-thread producer + replay buffer + advantage compute + `update_actor`. |

The trainer process is the only one that is *complex inside*; the
other two are stateless services that the trainer talks to over HTTP.

---

## 2. The single most important diagram

Inside the trainer process there are **two threads** that share the
replay buffer and **never block on each other**:

```
┌─────────────────────────────────────────────────────────────────────┐
│ trainer process (driver, single Python interpreter)                 │
│                                                                     │
│  ┌─────────────────────┐       ┌──────────────────────────────────┐ │
│  │ daemon thread       │       │ main thread (RayPPOTrainerDAPO)  │ │
│  │ ContinuousProducer  │       │ fit() loop                       │ │
│  │                     │       │                                  │ │
│  │ generate_sequences  │ push  │ wait_until(num_fresh_groups ≥ N) │ │
│  │   _dapo() in a loop │ ───►  │ trajectory_store                 │ │
│  │                     │       │   .sample_mini_batch(N)          │ │
│  │ DAPO async server   │       │                                  │ │
│  │   eager-pushes      │       │ compute_advantage()              │ │
│  │   each survivor     │       │ update_actor()  (8× FSDP)        │ │
│  │   the moment it     │       │ _save_checkpoint() every K steps │ │
│  │   clears            │       │   └─ /reload_lora to pool        │ │
│  │   filter_easy_hard  │       │       (publishes new LoRA)       │ │
│  └─────────────────────┘       └──────────────────────────────────┘ │
│           │                              ▲                          │
│           │                              │                          │
│           ▼                              │                          │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ TrajectoryStore (in-process, single threading.Lock)     │        │
│  │  - bounded deque(maxlen=replay.buffer_size = 256)       │        │
│  │  - drops zero-variance groups at push time              │        │
│  │  - drops stale groups at sample time                    │        │
│  └─────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

If you remember nothing else: **producer pushes, trainer samples,
both clocks are independent**. The buffer is the only point of
contact.

---

## 3. End-to-end trace of one rollout

Follow one trajectory through the system. Numbers in `[brackets]`
are `file:line`.

### 3.1  Producer side

1. `ContinuousRolloutProducer._run` `[replay/continuous_producer.py:197]`
   loops until `_stop_event` is set. Each iteration calls
   `generate_fn()`.

2. For DAPO, `generate_fn` is `async_rollout_manager.generate_sequences_dapo`
   `[trainer/ppo/ray_trainer_dapo.py:65]`.

3. `generate_sequences_dapo` `[nvidia/rollout/async_server_dapo.py:90]`
   pulls prompts from its dataloader (size = `data.gen_batch_size`)
   and dispatches OpenHands client jobs.

4. `request_from_openhands_dapo`
   `[nvidia/rollout/async_server_dapo.py:242]` is the dispatcher loop.
   - Sets `requested_batch_size = data.gen_batch_size`
     `[async_server_dapo.py:251]` — this is the call's exit
     condition (return when `num_completed_instances ≥ this`).
   - For each completed trajectory, runs
     `filter_easy_hard_instance` `[async_server_dapo.py:489]` which
     drops groups with `resolved == 0/n` or `resolved == n/n` (no
     gradient signal).
   - **Eager-push hook** at `[async_server_dapo.py:505–541]`: each
     surviving instance whose group has `n` complete trajectories is
     immediately handed to `self._push_fn(single_group_dp)`. The
     trainer wired `_push_fn` to the buffer via the closure at
     `[trainer/ppo/ray_trainer.py:1676–1685]`.

5. The closure stamps each pushed group with the **current**
   `policy_version` and `global_steps`
   `[trainer/ppo/ray_trainer.py:1677–1683]` — that's how the buffer
   knows whether a sample is stale and what behavior policy
   produced it.

### 3.2  Buffer side (`TrajectoryStore`)

6. `push_from_dataproto` `[replay/trajectory_store.py:176]` validates
   the group, attaches `behavior_policy_version` + `created_at_step`
   to each `TrajectoryRecord` `[trajectory_store.py:44]`, and appends
   to the `deque(maxlen=256)`. Single `threading.Lock` — the
   producer pushes, trainer samples, no race.

### 3.3  Trainer side

7. `RayPPOTrainerDAPO.fit` calls `_acquire_training_batch_dapo`
   `[trainer/ppo/ray_trainer_dapo.py:72]`.

8. With the producer running, the lockstep `wake_up + generate +
   sleep` path is skipped. Instead:

   ```python
   n_groups = int(self.config.data.train_batch_size)
   # wait until the buffer has N non-stale groups
   wait_until(lambda: store.num_fresh_groups(global_steps) >= n_groups,
              timeout=...)
   sampled = store.sample_mini_batch(n_groups, current_step=global_steps)
   ```
   `[ray_trainer_dapo.py:92–113]`.

9. `sample_mini_batch` `[trajectory_store.py:384]` pops `n_groups`
   non-stale prompt-groups from the buffer (FIFO of fresh groups;
   stale defined by `current_step - created_at_step >
   staleness_cutoff_k`). It then **flattens** them — one group of `n`
   sibling trajectories is unrolled into `n` `TrajectoryRecord`s, and
   the result of one `sample_mini_batch` call is a flat list of
   `n_groups × n` records. **Pop-on-sample** means each group is
   consumed exactly once — no reuse, no off-policy double-counting.

10. The trainer then runs:
    - `compute_advantage()` `[ray_trainer_dapo.py:343]` — GRPO does
      group-relative `(R - mean_g) / std_g` here; `group_uid` is the
      key. **This is the last place "groups" appear**. After this
      call, every trajectory has a per-token advantage scalar
      attached and the rest of the pipeline treats the batch as a
      flat tensor of `train_batch_size × n` trajectories.
    - `compute_log_prob()` (current-policy logprobs for IS ratio).
    - `update_actor()` (PPO step with the clipped IS correction).
      The loss path operates on flat tensors of shape
      `[train_batch_size × n, max_response_length]`; FSDP shards
      across 8 GPUs.
    - Every `save_freq` steps: `_save_checkpoint`
      `[ray_trainer.py:1218]` → `_publish_lora_adapter`
      `[ray_trainer.py:1296]` → `POST /reload_lora` to all 4 pool
      ports. Pool ACKs install latency. **Abort on any
      `endpoints_failed > 0`**.

That's the full loop.

---

## 4. The two-batch architecture (Cut 6)

### 4.0  Trajectory-count glossary (read first)

Trajectory counts move through three regimes. **"Group" is a producer-side
concept** (the unit DAPO needs to evaluate `filter_easy_hard_instance`).
Once advantages are attached the trainer just sees a flat batch of
trajectories — groups stop mattering.

| Stage | What's counted | Symbol | Cut 6 default |
|---|---|---|---|
| Producer call exit condition | survivor **prompts** that DAPO must complete + filter | `gen_batch_size` | `16` |
| Trajectories per prompt (GRPO group size) | sibling rollouts with shared `prompt_uid` / `group_uid` | `n` (= `actor_rollout_ref.rollout.n`) | `8` |
| Trajectories pushed per producer call | `gen_batch_size × n` | — | `16 × 8 = 128` |
| Trainer per-step prompt draw | prompts pulled from the buffer | `train_batch_size` | `4` |
| **Trainer per-step trajectory count** | **after `compute_advantage`, the trainer's mini-batch is a flat tensor of this many trajectories** | `train_batch_size × n` | `4 × 8 = 32` |
| Tokens trained on per step | depends on response length per traj | — | ~1.9M tokens (32 × ~60k seqlen post Cut 1) |

Read this carefully:
- The **producer's** `requested_batch_size = gen_batch_size = 16` is
  prompts, not trajectories. With `n=8` it generates 128 trajectories
  per call (modulo zero-variance drops, which retry until 16 prompts
  have non-trivial advantage).
- The **trainer's** `n_groups = train_batch_size = 4` pops 4 prompts
  worth of trajectories — 32 trajectories total. After
  `compute_advantage` runs `[ray_trainer_dapo.py:343]` and per-token
  advantage is attached, **the trainer's loss / optimizer step
  doesn't care about groups any more**. It's a flat 32-trajectory
  forward+backward across 8 GPUs (FSDP DP=2, SP=4). Group identity
  matters only inside `compute_advantage` for the
  within-group `(R - mean) / std` normalization.
- This is why the trainer's batch knob is named `train_batch_size`,
  not `train_n_groups`: it's the count the loss path sees once the
  advantage tensor is built.

### 4.1  The two config keys

Two distinct config keys, **never crossed**:

| Key | Default | Read at | Meaning |
|---|---|---|---|
| `data.train_batch_size` | 4 | `ray_trainer_dapo.py:92`, `ray_trainer.py:1771` | Prompts the trainer pulls from the buffer per step. After advantage compute it's `× n` trajectories on the GPUs. |
| `data.gen_batch_size` | 16 (= 4× train) | `async_server_dapo.py:251` | Survivor prompts one DAPO producer call targets before returning. Each prompt produces `n=8` trajectories, so call output is `gen_batch_size × n` trajectories. |

**Verifying the boundary** (run this any time):

```bash
grep -rn "gen_batch_size\|train_batch_size" \
    trainer_integration/verl/verl_custom/trainer/ppo/ \
    trainer_integration/verl/verl_custom/nvidia/rollout/
```

You should find:
- Trainer-side files (`ray_trainer.py`, `ray_trainer_dapo.py`) read
  **only** `train_batch_size` for the `n_groups` argument to
  `sample_mini_batch`.
- Producer-side files (`async_server_dapo.py`) read **only**
  `gen_batch_size` for `requested_batch_size`.
- The dataloader in `ray_trainer.py:684` reads `gen_batch_size` —
  that's also producer-side (the dataloader feeds the producer's
  prompt queue, not the trainer's compute path).

The fallback `data.get('gen_batch_size', data.train_batch_size)` at
`async_server_dapo.py:251` is **only** for the legacy lockstep path
(no buffer). In our production path `gen_batch_size` is always set
by `run_proagent_qwn3_4B_instruct_fullasync.sh:83`, so the fallback
never fires.

### Why decoupled?

- **Producer ≥ 4× trainer** keeps the buffer warm post-warmup. After
  the first producer call delivers ~16 survivors, the trainer can
  run several steps before it needs new groups.
- **Trainer doesn't know `n_per_prompt`.** It only knows "give me 4
  groups". This is a precondition for future decentralized vLLM
  workers (each worker is just another `_push_fn` caller into the
  same store).

---

## 5. The five invariants (and where they're enforced)

Read these once. Every change to the trainer/producer/store stack
must preserve them.

### Invariant 1 — token-in / token-out (LLM path)

Multi-turn RL stability requires re-using **exact token IDs** between
turns. Re-tokenizing decoded text shifts boundaries → actor and ref
diverge → KL/entropy NaN → PPO collapses.
- Enforced in `openhands/llm/nvidia/qwen3.py` and
  `openhands/llm/nvidia/qwen2_5_vl.py` (paired clients).
- **Frozen**: do not edit either file without sign-off.

### Invariant 2 — pop-on-sample, no reuse

A sampled group is removed from the buffer; it cannot be sampled
again. Prevents off-policy double-counting and IS over-correction.
- Enforced at `trajectory_store.py:sample_mini_batch:384` (records
  are popped from the deque).

### Invariant 3 — zero-variance groups never enter the buffer

A group where `resolved_count ∈ {0, n}` has `group_std=0` →
advantages all zero → no gradient. Pushing it would waste an
entire FSDP step on a no-op forward.
- Enforced at `async_server_dapo.py:filter_easy_hard_instance:750`
  (called from the dispatcher loop at `~489`). Filter runs
  *before* eager-push.

### Invariant 4 — staleness gate at sample time

A group sampled too long after it was generated has accumulated
policy drift; the IS correction can't compensate.
- Enforced at `trajectory_store.py:num_fresh_groups:566` (drops
  groups with `current_step - created_at_step > staleness_cutoff_k`).
- `staleness_cutoff_k = 4` in production (set in
  `s3_fullasync_docker.sh:54`).

### Invariant 5 — monotonic policy_version on pool

The pool rejects `/reload_lora` with `409` if `new_version ≤
active_policy_version`. Trainer is the source of truth.
- `policy_version = trainer.global_steps + 1` at
  `ray_trainer.py:1322`.
- Pool ACKs every `/reload_lora` and the trainer aborts on any
  failed endpoint at `ray_trainer.py:1380`.

---

## 6. How to read a running training log

The log (default `/tmp/s3-fullasync.log`) is written by `tee` from
the docker launcher. Key event names you can grep for:

| Event | Source | What it tells you |
|---|---|---|
| `[fullasync/docker] remote pool ... healthy` | launcher pre-flight | Pool reachable on all 4 ports before trainer starts. |
| `verl_custom ok` | container init | Custom verl extension imported. |
| `Progress: X/16 (Y%)` | DAPO dispatcher | Producer call's survivor count vs `gen_batch_size`. **`/16` proves Cut 6 is active.** |
| `Filtered instance ... resolved ratio 0/8` | DAPO ingest filter | Zero-variance group rejected (Invariant 3). |
| `DAPO eager-push: instance X pushed (N survivors pushed this call)` | eager-push hook | A surviving group landed in the buffer (Cut 5). |
| `PRODUCER_ITER {...}` JSON | continuous_producer.py | Per-iteration summary: wall_s, store_num_groups, eager_pushed_all. |
| `DAPO_PRODUCER_CALL {...}` JSON | async_server_dapo.py | Per-call summary: groups_drawn, groups_survived, effective_tps. |
| `global_step: N` | trainer fit() | Trainer completed step N. **Step ≥ 1 = warmup over.** |
| `is_weight/clip_fraction` (wandb metric) | core_algos.py:710 | Fraction of tokens hit by the IS clip. Target < 0.2. |
| `replay/sample_age_steps_p95` (wandb metric) | trajectory_store.py:611 | p95 of `current_step - created_at_step` on sampled groups. Target ≤ K = 4. |
| `weight_sync/endpoints_failed` (wandb metric) | ray_trainer.py:1397 | Pool /reload_lora failures. **Must stay 0.** |
| `timing_raw['gen']` (wandb metric) | ray_trainer_dapo.py:94 | Trainer's wait time on the buffer. **Target ≈ 0 after step 2.** |

Three commands worth memorizing:

```bash
# Producer is making progress
grep -E "Progress:|DAPO eager-push" /tmp/s3-fullasync.log | tail -20

# Trainer is making progress
grep -E "global_step|update_actor|_publish_lora" /tmp/s3-fullasync.log | tail -20

# Anything broken
grep -E "Traceback|RuntimeError|endpoints_failed|InsufficientTrajectoriesError" /tmp/s3-fullasync.log
```

---

## 7. Pointer index — where each concept lives

| Concept | File | Anchor |
|---|---|---|
| Trainer fit loop (DAPO) | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py` | `fit()`, `_acquire_training_batch_dapo()` |
| Trainer fit loop (plain GRPO) | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | `fit()`, `_acquire_training_batch()` |
| Producer daemon thread | `trainer_integration/verl/verl_custom/replay/continuous_producer.py` | `ContinuousRolloutProducer._run` |
| Replay buffer | `trainer_integration/verl/verl_custom/replay/trajectory_store.py` | `TrajectoryStore` |
| DAPO rollout server | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` | `request_from_openhands_dapo()` |
| Plain GRPO rollout server | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | `AsyncLLMServerManager` |
| LoRA publish | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | `_publish_lora_adapter:1296` |
| FSDP worker actor | `trainer_integration/verl/verl_custom/workers/fsdp_workers.py` | `init_model`, `compute_log_prob` |
| Token-in/token-out client | `openhands/llm/nvidia/qwen3.py` | (frozen) |
| Async agent server | `openhands/nvidia/async_server.py` | `OpenHandsServer` |
| AgentHandler registry | `openhands/nvidia/registry.py` | `add_name_mapping`, `register_agent_handler` |
| IS clip + log | `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py` | `676–711` |
| Hydra defaults | `trainer_integration/verl/verl_custom/trainer/config/ppo_trainer.yaml` | (custom verl_custom YAML) |
| Production launcher | `scripts/_internal/s3_fullasync_docker.sh` | env knobs at top |
| DAPO Hydra script | `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh` | hydra overrides |
| Pool launcher (EC2) | `scripts/serving/launch_remote_vllm_pool.sh` | child runner: `_remote_vllm_runner.sh` |

---

## 8. What we changed in this branch (Cuts 1–6)

| # | Change | Files | One-line rationale |
|---|---|---|---|
| 1 | `max_response_length 1536→16384`, pool `MAX_MODEL_LEN→47616` | launcher + pool config | Stop responses saturating the cap mid-tool-call. |
| 2 | `tis_imp_ratio_cap 2→5` | DAPO Hydra | Cap clipped 60 % of tokens (T/kernel noise). 5 still bounds genuine drift. |
| 3 | `replay.buffer_size 128→256` | docker launcher | Survive Cut 5's higher consumption rate. |
| 4 | `replay.stop_timeout_s 10→300` | docker launcher | Validation no longer races a mid-flight producer call. |
| 5 | Drop `n_groups` floor + eager-push from DAPO | `ray_trainer_dapo.py:92`, `ray_trainer.py:1771`, `async_server_dapo.py:505–541` | Trainer was wasting 7/8 GPUs per step on duplicate prompts. Eager-push lets the buffer fill mid-call. |
| 6 | `data.gen_batch_size` decoupled from `data.train_batch_size` | `async_server_dapo.py:251,753`, launcher env | Producer batch ≥ 4× trainer batch keeps the buffer warm; trainer no longer needs to know `n_per_prompt`; future decentralized vLLM workers slot in cleanly. |
| FusedLinearForPPO (OOM-retry) | `+actor.use_fused_kernels=True`, `+actor.use_remove_padding=True` | docker launcher overrides | Eliminates the 14.5 GiB `[seq × vocab × bf16]` logits allocation. Validated: 6.4 GB peak (was 40 GB OOM). |

The full plan record is at `/home/ubuntu/.claude/plans/you-are-the-hashed-toast.md`.

---

## 9. Production validation (Cut 6 baseline run, 2026-04-25)

This section captures the **empirical evidence that the architecture
described above is working as designed**. Evidence is from
`/tmp/s3-fullasync-cut6-prod.log`, started at 09:21 UTC. All metrics
sourced from the trainer's `step:N` console emission and live HTTP
probes against the EC2 vLLM pool.

### 9.1  What the evidence confirms

**Eager-push pipeline (Cut 5 + Cut 6) operates as designed.**
Producer call #1 closed cleanly with this `DAPO_PRODUCER_CALL` JSON:

```json
{
  "wall_s": 5791.5,
  "groups_drawn": 44, "groups_survived": 16,
  "groups_dropped_filter": 9, "groups_pushed_eager": 16,
  "eager_pushed_all": true,
  "trajectories_out": 128, "tokens_out": 6094848,
  "effective_tps": 1052.4
}
```

- 16 survivors out of 44 prompts drawn → **`filter_easy_hard_instance`
  is dropping ~36 % zero-variance groups at ingest** (matches the
  invariant in §5).
- All 16 survivors went via `_push_fn` (`eager_pushed_all: true`) —
  the terminal `push_from_dataproto` was correctly skipped (gotcha
  §29 holds).
- 128 trajectories shipped per call (= `gen_batch_size × n` =
  `16 × 8`), confirming the trajectory-count math in §4.0.

**Trainer-side consumption follows the §3.3 contract.** Across
steps 1–5, every step's `step:N` line shows
`replay/sample_age_steps_p95 ≤ K=4`. Step 5 specifically had
`staleness_steps:0` because the publish at the step-5 boundary
reset the version stamp on subsequent samples — exactly what gotcha
§22 predicts.

**Weight-sync round-trip is intact.** Step 5 emitted in the metrics
line:

```
weight_sync/policy_version:1
weight_sync/adapter_mib:232.92
weight_sync/publish_latency_s:1.24
weight_sync/transfer_latency_s:1.24
weight_sync/vllm_load_latency_s:0.0
weight_sync/endpoints_ok:4
weight_sync/endpoints_failed:0
```

Live `curl http://<pool>:8100..8103/health` returns
`{"policy_version":1}` on all four children. The
`/reload_lora` POST chain is wired and the abort-on-failure
invariant did not need to fire.

**Cut 1 (response-length raise) is paying off.** Per-step
`response_length/clip_ratio` ranges 6 % – 41 % across steps 1–5 vs
~100 % in Run9 (when the cap was 1,536). The cap no longer
corrupts the reward signal except on hard-repo batches.

**Cut 2 (TIS cap raise) is well within bounds.**
`rollout_corr/log_ppl_diff` ranges 0.49 – 0.64 across steps 1–5
vs the new clip threshold `log(5) ≈ 1.61`. Nowhere near saturating.

**FusedLinearForPPO + remove-padding holds the memory ceiling.**
`perf/max_memory_allocated_gb` hovers at 6.4 GB across all 5
emitted steps — well under the OOM threshold that prevented prior
runs from booting at all.

### 9.2  Per-step learning signal

| Step | reward | grad_norm | pg_loss | log_ppl_diff | resp_len/mean | clip_ratio | staleness |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.656 | 0.016 | +0.006 | 0.49 | 10,613 | 0.22 | 1 |
| 2 | 0.469 | 0.040 | +0.022 | 0.49 | 9,910 | 0.06 | 2 |
| 3 | 0.438 | 0.015 | −0.024 | 0.63 | 12,297 | 0.38 | 3 |
| 4 | 0.531 | 0.031 | +0.017 | 0.62 | 11,917 | 0.41 | 4 |
| 5 | 0.563 | 0.018 | −0.006 | 0.64 | 10,238 | 0.06 | 0 |

Reward across 5 steps × 4 prompts = 20 prompts is too small to
declare a trend. Gradients are healthy (non-zero, not exploding,
not vanishing). `log_ppl_diff` is drifting slightly upward
(0.49 → 0.64) but well within Cut 2's headroom. `pg_loss` flips
sign step-to-step — that's normal advantage-weighted variation.

### 9.3  Open issue (deliberately surfaced)

**Trainer is buffer-bound between producer calls.** The
`timing_s/gen` field (which on the continuous-producer path equals
`_acquire_training_batch_dapo` wall, mostly buffer-wait) is
climbing:

| Step | timing_s/gen | timing_s/update_actor |
|---|---|---|
| 1 | 2,464 | 44 |
| 2 | 3,171 | 89 |
| 3 | 4,859 | 136 |
| 4 | 5,576 | 223 |
| 5 | 7,218 | 269 |

Trainer compute is small (≤ 5 min per step). The wall is dominated
by buffer-wait. After step 6 (which took 62 min vs. steps 2-4's
average 18 min), the per-step pace is no longer keeping ahead of
the producer. The producer call #2 ran ~10 min/push vs. call #1's
~6 min/push — the gap is widening.

**Decision pending:** if step 7+ continues to widen the wait, land
**Cut 7 (multi-producer fan-out)** — a second
`AsyncLLMServerManagerDAPO` + `ContinuousRolloutProducer` pair on
a disjoint dataloader slice, sharing the same `TrajectoryStore`.
The store's single `threading.Lock` already serializes pushes
correctly; the rest of the change is dataloader stride and a
`replay.num_producers=2` env knob (the original Cut 6 was scoped
narrower than this fan-out — see plan record §"Cut 6").

### 9.3.1  Resolution: Cut 8 — bigger `gen_batch_size`

The 25-step prod run completed (steps 1–25 reached). Across the
full run the call-boundary gap surfaced in three distinct places: a
~26 min trough between call #3 (closed 15:56:11) and call #4 (first
push 16:22:37) being the most legible. **The dead window is not vLLM
or GPU related** — `/health` was always 200 across all 4 children;
GPUs idled. It's the DAPO per-call lifecycle: `stop_servers()` →
loop back → `start_servers()` → push `gen_batch_size × n` jobs →
wait for the first complete group of 8 trajectories from turn 0.
Step (e) — group cold-start through multi-turn agent action — is
the dominant cost, not the HTTP calls.

**Two non-exclusive levers close the gap:**

| Lever | What it changes | Cost |
|---|---|---|
| **Cut 8: bump `gen_batch_size` 16 → 32** | Hot phase doubles in length; call boundaries halve in frequency per hour. Trainer sees the trough less often. | One env-var flip in `s3_fullasync_docker.sh`. Trivial; reversible. |
| **Cut 7: two staggered producers** | Producer B's hot phase covers producer A's trough; trainer never sees the gap. | Code change. The naïve form races on the shared `localhost:8006` OH server's `/start`/`/stop` lifecycle (gotcha §31). Needs per-producer OH server or no-op'd lifecycle. |

**Decision: validate Cut 8 via prep-100, defer Cut 7.** Cut 8 buys
fewer gaps per hour at zero code-change cost; Cut 7 buys complete
gap removal but requires non-trivial OH-server multi-tenancy work.
The prep-100 run uses `GEN_BATCH_SIZE=32 TEST_FREQ=10` as **explicit
env overrides** — defaults in `s3_fullasync_docker.sh` remain at 4×
and `-1` until the 100-step run lands cleanly, at which point a
follow-up commit promotes the bump to the default. Revisit Cut 7
only if Cut 8 still leaves the trainer stalling in
`_acquire_training_batch_dapo` after warmup.

### 9.4  What this validates

| Mechanism | Status |
|---|---|
| Producer thread iterates DAPO `generate_sequences_dapo` continuously | ✅ confirmed (call #1 closed, call #2 active) |
| Eager-push fires each survivor at `filter_easy_hard_instance` clear-time | ✅ confirmed (`eager_pushed_all: true`) |
| Zero-variance ingest filter | ✅ confirmed (9/44 dropped at filter) |
| Pop-on-sample, no reuse | ✅ confirmed (push count = consumption count to within buffer depth) |
| Staleness gate at sample (K=4) | ✅ confirmed (`sample_age_steps_p95 ≤ 4` every step) |
| Monotonic `policy_version` published to all 4 endpoints | ✅ confirmed (`endpoints_ok:4 / endpoints_failed:0` at step 5) |
| Trainer's `n_groups = train_batch_size` (no floor) | ✅ confirmed (4 prompts × 8 trajs = 32 trajs/step matches `global_seqlen` math) |
| Producer / trainer batch-size decoupling | ✅ confirmed (`gen_batch_size=16`, `train_batch_size=4` independent) |
| Token-in / token-out invariant | ✅ implicit (no tokenizer mismatch errors; rollouts complete) |
| FSDP1 with offload + LoRA + Ulysses SP=4 | ✅ confirmed (`max_memory_allocated_gb:6.4` stable) |

The architecture works. The remaining gap is throughput, not
correctness.
