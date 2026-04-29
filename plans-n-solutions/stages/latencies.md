# Latency / TPS reference — DAPO, filter_groups=True, n=8

Reference numbers for `filter_groups=True, n=8`. Use this to decide where to invest tuning effort. Numbers are extracted from existing verl metric lines and the `publish_lora_adapter` JSON event stream.

## Mental model

Producer fills the buffer; trainer drains the buffer. Two independent rates. The trainer never waits once the buffer is warm; the producer never knows what step the trainer is on.

In **GRPO**, only prompts whose `n` samples are **neither all-wrong nor all-correct** produce gradient — a uniform group has zero variance and zero advantage. DAPO's `filter_easy_hard_instance` drops `resolved == 0` and `resolved == n` at push time, so the buffer only ever holds gradient-bearing groups. In the reference window, **81 %** of filtered groups were 0/8 (all-fail), **19 %** were 8/8 (all-pass) — that fraction of rollout wall-clock produces nothing the trainer can use.

## How long it takes to fill the buffer

A single `generate_sequences_dapo` call must complete `gen_batch_size × n` agentic trajectories AND survive the variance filter before the survivor target ships. Eager-push delivers each surviving group into the store the moment its `n` siblings clear `filter_easy_hard_instance`, so the trainer can sample partway through a call — but the **first** survivor of a fresh call still has to clear `n` trajectories from turn 0. That "group cold start" sets the worst-case time-to-first-fill after every call boundary.

Reference numbers from the n=8 window:

| `GEN_BATCH_SIZE` shape | Per-call producer wall-clock | When to use |
|---|---|---|
| `1 × train_batch_size = 4` (lock-step shape) | ~40–60 min/call (2500–3500 s observed) | **Don't.** Worst case for call-boundary dead time per hour. |
| `4 × train_batch_size = 16` (current default) | ~4× the above hot phase, 4× fewer call boundaries/hour | Default. |
| `8 × train_batch_size = 32` (`GEN_BATCH_SIZE=32`) | Longer hot phase, smallest fraction of wall in the call-boundary dead gap | Set when you want to push the call-boundary share lower. |

In steady state the trainer drains 4 steps back-to-back in ~25 s/step, then waits on the next producer release. **Rollout dominates ~96 % of wall-clock at this scale.**

## How to fill it faster

Buffer-fill rate ≈ `vLLM throughput × concurrent agentic clients × (1 − filter_drop_rate)`. Levers, in order of cheapness:

1. **`GEN_BATCH_SIZE` up** — longer hot phase per call, no code change, no extra capacity.
2. **More OpenHands workers** (`OPENHANDS_NUM_WORKERS`) — more concurrent agentic trajectories. The 4-child pool saturates at ~32 concurrent clients (handsoff §17); past that, scale the pool too.
3. **More vLLM children** (currently 4 on the EC2 pool) — raises the raw generation ceiling. Requires more EC2 GPUs.

Once buffer-fill rate ≥ trainer-drain rate, the trainer never waits and the loop converges on the near-on-policy regime.

## Where advantage is computed

GRPO advantage = `(reward − group_mean) / group_std` over the `n` siblings of one prompt. Today this runs at the **trainer** on the sampled mini-batch (`ray_trainer_dapo.py:361`, `ray_trainer.py:2073`), so the buffer must hand back **whole groups intact** — `sample_mini_batch` never splits a group. The `TrajectoryRecord.advantage` field exists in `trajectory_store.py` but is `0.0` in the live path (eager-push happens before `compute_advantage` runs). **Deferred refactor:** compute advantage at the producer's filter-clear seam (the `n` siblings are already grouped there), store the per-trajectory scalar, and the buffer becomes a flat per-trajectory pool — trainer can sample arbitrary cross-group subsets, no group-integrity requirement at sample time.

## Per-step component budget (n=8 reference)

| Component | Mean/step | p95 | Notes |
|---|---|---|---|
| Rollout wait (producer-gated) | **~620 s** | — | dominates wall-clock |
| FSDP update (fwd+bwd+opt) | 18 s | 21 s | ~340 tok/s, MFU ~3.9 % |
| `old_log_prob` recompute | 7.6 s | — | ~800 tok/s |
| Reward compute | 0.08 s | — | — |
| **Step wall-clock total** | **~650 s** | — | `timing_s/step` |

Inputs: `response_length/mean ≈ 1500 tok`, `global_seqlen/mean ≈ 6100 tok`, `perf/mfu/actor ≈ 0.039`.

## Weight sync is not a bottleneck

| Stat | `publish_latency_s` | `transfer_latency_s` (S3) | `vllm_load_latency_s` (4-child) |
|---|---|---|---|
| mean | ~30 | ~17 | ~13 |
| max | ~35 | ~19 | ~16 |

Adapter ~245 MB. **Budget: 30 s publish ÷ (5 steps × 650 s/step) ≈ 0.9 % of wall-clock.** `endpoints_ok:4` on every publish in the reference window.

## Staleness behaviour at this scale

- `replay/sample_age_steps_p95` max observed = **3** (K=4 hard cap not hit).
- `replay/dropped_by_staleness_total = 0`.
- `replay/store_fill_ratio` 0.000–0.023 → buffer near-empty; trainer consumes as fast as producer pushes.
- `rollout/staleness_steps` cycles 0→4 matching `save_freq=5` (expected).

Regime is **near-on-policy, NOT replay-reuse**. The buffer's role at this throughput ratio is smoothing trainer utilisation, not raw throughput.

## Decision table

| Signal threshold | Action |
|---|---|
| Raw vLLM TPS < 100 tok/s/GPU | Bump pool children or `tp_size`; reclaim vLLM-side waste |
| `groups_dropped_filter / groups_drawn > 0.5` | Either widen `gen_batch_size` so survivor yield/call goes up, or accept a longer hot phase |
| Trainer idle > 70 % of step time | Raise `GEN_BATCH_SIZE`, scale OpenHands workers, or scale the vLLM pool |
| `sample_age_steps_p95` approaches K=4 | Raise K or lower `save_freq` |
| `is_weight/clip_fraction > 0.25` | Investigate per-turn version stamping (handsoff §22) and T-mismatch (handsoff §27) |

## Related docs

- `plans-n-solutions/handsoff.md` — topology, pointer table, gotchas (§19 cooperative stop, §20 GIL atomicity, §27 IS-clip decomposition, §30 producer call-boundary gap)
- `plans-n-solutions/stages/replay_dynamics.md` — producer / store / trainer interaction reference
- `plans-n-solutions/stages/how_to_run.md` — env-knob matrix, monitoring, failure runbook
