# Experience Replay for Async LLM RL — Codebase Reference

**Paper:** Arnal, Cabannes, Cohen, Kempe, Munos — "Efficient RL Training for LLMs with Experience Replay," FAIR at Meta, April 2026 (arXiv:2604.08706v1). Local copy: `docs/Efficient RL Training for LLMs with Experience Replay.pdf`.

Read this **before** writing `plans-n-solutions/stages/full_async.md`. It is the load-bearing reference for the Phase 2 trajectory store — the summary distils the paper's core claims and maps them onto our topology, the Phase 1 weight-sync protocol, and the DAPO `filter_groups` path.

---

## 0. Why this paper is the Phase 2 compass

The paper validates the exact hypothesis behind the `full-async` branch: in LLM RL the `generate-then-discard` default that PPO/GRPO inherited from on-policy RL is **compute-suboptimal when generation is expensive**. A well-sized replay buffer saves up to 40% of the compute budget while matching — and sometimes surpassing — on-policy accuracy (Figure 1, §5.2).

Phase 1 already exhibits every pathology the paper calls out:

- **Lock-step generate → update.** `RayPPOTrainer.fit()` calls `generate_sequences()` and blocks on the slowest agentic rollout in the batch. Tail latency up to `openhands_timeout=1000 s × max_iterations=30`.
- **Discarded learning signal.** DAPO (`filter_groups.enable=True`) drops any GRPO group of `n` samples whose advantages all share a sign. On SWE-Gym, most early-training groups are all-failures — the trainer pays full rollout cost for zero gradient signal.
- **Observed staleness.** `rollout/staleness_steps` (`global_steps − policy_version`) confirms the trainer is **compute-starved, not off-policy-starved** — we are waiting on rollouts, not deciding to wait.

The paper's prescription — inference workers push to a bounded buffer, trainers sample it with clipped IS correction and a staleness horizon — is precisely the shape of Phase 2 in `plans-n-solutions/handsoff.md`.

---

## 1. One-page TL;DR

- **Central claim (§5.2, Fig. 1):** a simple FIFO-eviction replay buffer with uniform random sampling, plugged into an otherwise-standard async GRPO pipeline, reaches the baseline's peak accuracy using **≈ 60% of the compute**, and preserves pass@k (output diversity) better than on-policy.
- **Optimal-design theorem (§4.5):** given a fixed compute budget, the optimal staleness horizon `x_* = N/R` and replay ratio `y_* = B/R` minimize `σ̄²(x) · (1/√μ + √(ρ + 1/x))²`. As the rollout cost `μ` grows, the optimum moves **away** from on-policy. As `μ → 0`, the optimum collapses back to on-policy (`x_* → 0`).
- **Experimental setup (§5.1):** Qwen3-0.6B / Qwen2.5-7B, OpenR1-Math-220k, GRPO, `ε_low = ε_high = 0.2`, no KL, group `G = 16`, batch size `B = 60`. `(W, T) ∈ {(6,2), (5,3), (4,4)}`; `μ ≈ 5–7`.
- **Counter-intuitive finding:** the buffer also acts as a **stability regularizer**. Baseline on-policy runs peak then *crash* (Fig. 13); buffered runs stay stable and sometimes reach a higher peak. Attributed to increased training-distribution diversity.
- **Pseudo-code (Appendix B):** two-file diff. Replace a LIFO `QueueStructure` with a circular `BufferStructure` backed by `asyncio.Lock`. The producer/consumer loops are unchanged. See §5 below.
- **Refinements (§5.5):** *positive-bias sampling* (δ-refinement of the eviction rule — keep correct rollouts longer) and the *AsymRE* loss (no importance correction) push the Pareto frontier further. Both are orthogonal to the buffer structure and can ship after Cut 1.

---

## 2. The three-way trade-off (§3.2, §4.5)

Buffer design sits inside a three-axis box. Pick a corner; accept the others.

| Axis | What it measures | Levers | Phase 2 WandB key (suggested) |
|---|---|---|---|
| **Off-policiness / staleness** | Age, in gradient steps, of a rollout still in the buffer (`x = N/R`) | Buffer size `N`, ingestion rate `R` | `replay/sample_age_steps_p50`, `_p95` |
| **Sample diversity** | Global (replay ratio `y = B/R`) and local (steps-since-last-use) | `N`, `W/T`, sampling rule | `replay/replay_ratio_mean`, `replay/steps_since_last_use_p50` |
| **Compute ratio** `γ = (1 + W/T)/(1 + μ)` | Per-step compute cost *with* buffer vs. no-buffer baseline | `W/T`, `μ` is a given | derived from existing `weight_sync/*` + `update_actor_time_s` |

Two theorem-4.5 consequences the planner must internalize:

1. **As rollout cost `μ` grows, the optimal staleness horizon grows** (Fig. 6). Our agentic `μ` dwarfs the paper's `μ = 5.28` for Qwen2.5-7B math RL — so our `x_*` is almost certainly **larger**, not smaller. Don't reflexively set `K = 1`.
2. **When off-policy variance (`σ̄²`) or trajectory-to-iterate coupling (`ρ`) is high, `x_*` collapses to zero** — i.e., stay on-policy. On-policy RL wins exactly when its assumptions hold; async LLM RL is not that regime.

---

## 3. The `(W, T)` knobs mapped onto our topology

The paper splits a fixed pool of `W + T = 8` GPUs between `W` inference workers and `T` trainers. Not a clean analog for us: our inference lives on a **remote** 4-child vLLM pool on EC2; our training lives on a separate 8×A100 FSDP box. But the ratio `W/T` still controls the same quantity — how many times each rollout is reused.

Table 1 of the paper, reproduced for context (`μ ≈ 5.28`, Qwen2.5-7B):

| `(W, T)` | Compute ratio `γ` | Average replay ratio `y` |
|---|---|---|
| (7, 1) | 1.29 — **worse** than no buffer | ≈ 1 |
| (6, 2) | **0.65** — sweet spot at `N = 84` | 2.2 |
| (5, 3) | 0.43 | 5.6 |
| (4, 4) | 0.32 | 17.6 |

For our stack, reason in **effective compute ratio** rather than literal GPU count: `W/T ≈ pool_throughput / trainer_throughput`. Agentic `μ` (long-tail tool-using rollouts) is an order of magnitude larger than the paper's math-RL `μ`, which pushes our operating point toward **larger buffers, larger staleness horizons, and higher replay ratios** — closer to the paper's `(4,4)` regime than to `(6,2)`. Caveat: off-policy divergence (`ρ`, `σ̄²`) also grows with `μ`, so the horizon must be sanity-checked empirically via `is_weight/clip_fraction`.

Paper estimate of `μ` for their models (Table 2): Qwen3-0.6B `μ = 6.84`, Qwen2.5-7B `μ = 5.28`. Measure ours directly in Phase B parity runs — we expect ≥ 20.

---

## 4. Staleness horizon `x` and replay ratio `y` — how to pick them

Design levers in priority order:

1. **Staleness horizon `x = N/R`.** Pick first; bounded from above by what keeps `is_weight/clip_fraction < 0.2` (the paper's gate).
2. **Buffer size `N`.** Derived: `N = x · R` where `R` = rollouts added per trainer step. Our `R` is variable (agentic tail), so treat `N` as a soft wall-clock target rather than a hard step budget.
3. **Batch size `B`.** Inherited from Phase 1 (`ppo_mini_batch_size=4, n=8`) until there's a reason to re-tune. Paper uses `B = 60`, `G = 16`.

Closed-form optima (Theorem 4.5, power-law variance §C.3.2):

- `x_* = y²/(2α(μ + y))`  where `α` is the variance power-law exponent (`α < 1/2` assumed).
- `y_* = (−α + √(α² + μρ(1 − 2α))) / ρ`.

Don't solve this symbolically — pick `x`, measure `is_weight` distribution, adjust. The paper's Figure 6 shows `x_*` scaling roughly linearly in `μ` in the interesting regime.

**Starting point for our 50-step parity runs (handsoff.md §9):**

- `N` ≈ 4 × `ppo_mini_batch_size × n` = **128** rollouts (four mini-batches of headroom).
- Staleness horizon `K = 4` trainer steps (handsoff.md §12.3 suggestion). Older samples dropped or heavily down-weighted.
- Sampling: **uniform with replacement** (Fig. 18 — without-replacement variants showed no significant gain).

Re-tune after the first `is_weight/*` histograms.

---

## 5. How the authors built experience replay (Appendix B)

The paper's pseudo-code is ~20 lines and the diff from a standard LIFO queue is a **single data-structure swap**. The producer (`Sampler`) and consumer (`Trainer`) loops are unchanged.

### 5.1 Baseline (on-policy streaming)

```python
# Appendix B.1 — what Phase 1 semantically has
class QueueStructure:
    """Standard LIFO storage for on-policy streaming."""
    def __init__(self):
        self.queue = asyncio.LifoQueue()

    async def push(self, data):
        await self.queue.put(data)

    async def sample(self, batch_size):
        # Strictly consumes — items are removed once sampled
        return [await self.queue.get() for _ in range(batch_size)]
```

### 5.2 Replay buffer (Phase 2 target)

```python
# Appendix B.5 — the load-bearing cut
class BufferStructure:
    """Experience Replay buffer supporting random sampling."""
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size
        self.lock = asyncio.Lock()

    async def push(self, data):
        async with self.lock:
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)           # FIFO eviction (oldest out)
            self.buffer.append(data)

    async def sample(self, batch_size):
        async with self.lock:
            return random.sample(self.buffer, batch_size)   # no removal
```

The producer/consumer harness (Appendix B.2–B.4, unchanged between the two variants):

```python
class Sampler:
    # W copies run in parallel
    async def run(self, dataset):
        for data in dataset:
            await self.receive_weights()              # pull latest θ
            rollout = await self.generate_rollout(data)
            await self.dump_struct.push(rollout)
        await self.dump_struct.push("DONE")

class Trainer:
    # T copies run in parallel
    async def run(self, batch_size):
        while self.is_running:
            batch = await self.dump_struct.sample(batch_size)
            if "DONE" in batch:
                self.is_running = False; break
            await self.forward_backward(batch)        # GRPO / PPO loss
            await self.update_weights()
```

### 5.3 Sharding (Appendix D.4)

The paper's production implementation shards one `BufferStructure` per trainer GPU. Rollouts are ingested in a round-robin fashion, each trainer GPU samples only from its own buffer, and `N` in the paper denotes the **total** buffer size (sum over shards). Paper note: *"Our preliminary experiments suggest this design choice has little impact."*

For Phase 2, start with a **single in-process buffer on the trainer driver**. Promote to sharded (one per FSDP rank) only if the central lock profiles hot.

---

## 6. What our `BufferStructure` must hold — token-in / token-out invariant

`CLAUDE.md` and `openhands/llm/nvidia/qwen3.py` enforce an invariant the paper does not discuss: **token IDs, never decoded text across turns**. A replay buffer that decodes on write and re-tokenizes on read silently shifts token boundaries across steps, drives actor and reference apart, and collapses training. Do not store `messages: list[dict]`.

Record shape for Phase 2:

```python
@dataclass(slots=True, frozen=True)
class TrajectoryRecord:
    prompt_ids: tuple[int, ...]            # exact tokens the pool saw
    response_ids: tuple[int, ...]          # exact tokens the pool emitted
    logprobs: tuple[float, ...]            # behavior-policy logprobs at generation
    advantage: float                       # computed at generation time (§D.2)
    reward: float
    behavior_policy_version: int           # from async_server.py:~1495
    created_at_step: int                   # trainer global_steps at push
    prompt_uid: str                        # for DAPO group reconstruction
```

Paper justification for `advantage` being stamped at generation time, not sample time (Appendix D.2): *"the advantage is computed when the rollout is generated, and not when it is used to compose a gradient update"*. This matches the GRPO formulation `A_i = (r_i − mean(r_{group})) / std(r_{group})` and keeps the sampler stateless with respect to the trainer's current θ.

---

## 7. Interaction with Phase 1 `POST /reload_lora` and `policy_version`

Phase 1 already ships the instrumentation Phase 2 needs for IS correction:

- **Behavior-policy identity.** Every rollout is stamped with `policy_version` in `async_server.py:~1495`. That *is* the τ in the paper's `π_{θ_{t−τ}}` — the step at which the rollout was created (§3.2, "Degree of Off-Policiness").
- **Policy-version tempo.** `POST /reload_lora` on each pool child (`scripts/serving/_vllm_child.py`) is our equivalent of the paper's implicit "weight broadcast" event. Swaps are atomic under `_swap_lock`, so between publishes every rollout uses one deterministic policy version. This makes the per-trajectory IS ratio `π_θ(z|q) / π_{θ_{pv_i}}(z|q)` **tractable and exact** — computable from the stored `logprobs` field.
- **Resume path.** `ray_trainer.py:1506-1515` syncs `self.policy_version` to resumed `global_steps` on resume. Phase 2 must additionally **clear or freshness-check the replay buffer on resume** — pre-resume entries tagged with the old `policy_version` must not be silently reused.

The paper's GRPO loss (Appendix D.2) does not apply the joint-distribution IS correction — they admit this explicitly: *"the joint distribution over the current training batch is not corrected in expectation by the importance sampling factor."* They rely on the per-sample clipped ratio `clip(π_θ/π_{old}, 1−ε, 1+ε)` with `ε = 0.2`. That is what Phase 1's `tis_imp_ratio_cap` already does *per-step*; Phase 2 extends this *per trajectory, across time*, using the `logprobs` field stored at generation time.

---

## 8. Interaction with DAPO `filter_groups` — three options, pick one

The paper does not address group filtering. DAPO (`main_ppo.py:232-236`, `ray_trainer_dapo.py`) drops any `n`-sized group whose rewards share a sign (advantage ≡ 0). In async + replay, this forces a design choice:

| Option | When filtering applies | Pros | Cons |
|---|---|---|---|
| **A: ingest-time** | Compute group stats at `push()`; drop zero-advantage groups before they enter the buffer | Small buffer, every stored rollout is "useful" | Producer has to assemble `n = G` siblings synchronously — fights the fully-async ethos |
| **B: sample-time** (**recommended start**) | Store everything; at `sample()` reconstruct groups by `(prompt_uid, behavior_policy_version)`, drop zero-advantage groups | Fully-async producer, preserves paper's minimal-diff shape, degrades cleanly to `filter_groups=False` | Buffer holds rollouts that will never train |
| **C: positive-bias hybrid** | Store everything; sampler prefers groups with mixed-sign advantage | Closest in spirit to §5.5 positive-bias sampling, retains DAPO's intent | Extra bookkeeping; harder to reason about |

Start with **B**. Instrument `replay/dropped_by_filter_groups_per_step`. Promote to C only if buffer bloat materially hurts diversity. Both DAPO and plain GRPO must keep working after Phase 2 (handsoff.md §12.1); option B makes `filter_groups=False` a zero-cost pass-through.

---

## 9. Positive-bias sampling (§5.5) and AsymRE — Phase 2.5 extensions

Two refinements the paper explores, both **orthogonal to the minimal buffer cut** and both likely fits for our agentic workload:

- **Positive-bias sampling (δ-refinement).** Instead of keeping only the freshest `N` rollouts, keep the freshest `(1 − δ) · N` *plus* the freshest `δ · N` correct rollouts *not already in* that set. Intuition: correct rollouts degrade more slowly under off-policiness (their likelihood under the current policy stays high). With `N = 4608`, `(W,T) = (6,2)`, `δ = 0.5`, AsymRE loss, the paper reports its best accuracy in Fig. 5. **Especially natural for our regime**: on SWE-Gym, early-training groups are overwhelmingly all-failures; positive-bias sampling preserves the rare correct rollouts across more trainer steps, extending their learning value exactly when DAPO's `filter_groups=True` would otherwise have most groups crumble.
- **AsymRE loss (Arnal et al. 2025).** `J_AsymRE(θ) = E[(1/G) Σ_i (r(z_i, q) − (V̂ + δV)) log π_θ(z_i | q)]` with `δV = −0.1`. No IS ratio — no importance-weight variance to blow up under high off-policiness. Reported to outperform GRPO at aggressive replay ratios.

Defer both to after Cut 3 (off-policy correction lands). Neither changes the data structure.

---

## 10. What to measure (WandB keys — additive on Phase 1)

Phase 1 keys stay (handsoff.md §7). Phase 2 adds:

| Key | Paper anchor | Role |
|---|---|---|
| `replay/store_size`, `replay/store_fill_ratio` | — | buffer state |
| `replay/sample_age_steps_p50`, `_p95` | §3.2 Fig. 2 left | off-policiness distribution |
| `replay/replay_ratio_mean`, `_p99` | §3.2 Fig. 2 middle | global diversity |
| `replay/steps_since_last_use_p50` | §3.2 Fig. 2 right | local diversity |
| `replay/dropped_by_staleness_per_step` | — | staleness-horizon enforcement |
| `replay/dropped_by_filter_groups_per_step` | — | DAPO interaction (option B) |
| `is_weight/mean`, `_p99`, `_clip_fraction` | §4.5 | primary health signal |

**Gate:** `is_weight/clip_fraction < 0.2`. If ≥ 0.5, the staleness budget is too loose — shrink `N/R` before touching anything else.

---

## 11. What the paper does *not* answer (our contribution)

- **Agentic tail latency.** Paper's `R` (rollouts per step) is roughly constant because math rollouts are short. Ours is heavy-tailed — variance in `R` propagates to variance in staleness horizon. Phase 2 needs a wall-clock-based staleness metric, not just a step-count one.
- **LoRA publish tempo.** Paper broadcasts full weights every step. Our `POST /reload_lora` every `save_freq` steps creates coarser `policy_version` granularity. This is a feature, not a bug — it means the importance ratio over a trajectory is computable exactly against one reference, not against a moving target. Pick `save_freq` so that `publish_latency_s × publishes_per_epoch < step_time × save_freq` (handsoff.md §10.12).
- **`filter_groups` under async.** None of the 2025 buffer-for-LLM papers in §A.3 touch this. Document whichever of A/B/C you pick, with a failure-mode ablation, in `full_async.md`.
- **Sharded vs central buffer under FSDP.** Paper says little impact. We have FSDP + Docker; measure it under the actual trainer topology before deciding.

---

## 12. Where to go from here

1. Read §4 (math analysis) and §5 (experiments) in the paper directly — the bounds and ablation shapes are not summarised verbatim here.
2. Read Appendix B (pseudo-code) and Appendix D.4 (sharded-buffer implementation note).
3. Read the Phase 1 sites in `handsoff.md` §11: `async_server.py` (policy_version stamping), `ray_trainer.py` (publish hook, resume sync), `ray_trainer_dapo.py` (same hook, DAPO path), `_vllm_child.py` (`/reload_lora`).
4. Write `plans-n-solutions/stages/full_async.md` with concrete answers to: (a) buffer location (in-process / sharded / out-of-process), (b) staleness horizon `K`, (c) IS correction (clipped IS vs AsymRE), (d) `filter_groups` option (A/B/C), (e) starting point for `(N, R, B)`, (f) resume behaviour for buffer contents.
5. Cut 1 (TDD, handsoff.md §D): the `BufferStructure` itself. Mirror Appendix B.5, add the token-in/token-out tuple shape, ship unit tests for push/sample/eviction under concurrent `asyncio.Lock` access and for DAPO-group reconstruction by `(prompt_uid, behavior_policy_version)`.

---

## Appendix: paper glossary, quick reference

| Symbol | Meaning |
|---|---|
| `W` | Number of inference worker GPUs (our analog: pool throughput) |
| `T` | Number of trainer GPUs |
| `μ` | Rollout cost / trainer-step cost |
| `N` | Total replay buffer size (rollouts) |
| `R` | Rollouts inserted per trainer step |
| `B` | Trainer mini-batch size |
| `G` | GRPO group size (their `G=16`, ours `n=8`) |
| `x = N/R` | Staleness horizon (max age in steps of any rollout in buffer) |
| `y = B/R` | Average replay ratio (times a sample is used) |
| `γ = (1+W/T)/(1+μ)` | Compute cost with buffer / without buffer |
| `ρ` | Correlation between trajectory and iterate (algorithm-specific constant) |
| `σ̄²(H)` | Mean off-policy variance over a staleness horizon of `H` steps |
| `δ` | Positive-bias sampling coefficient (fraction of slots reserved for correct rollouts) |
