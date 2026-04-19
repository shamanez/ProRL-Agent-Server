# Initial decentralized rollout timing — Qwen3-4B decoupled

First empirical timing record, 20-step GRPO run on the decoupled topology with Phase 1 LoRA weight-sync. Captured 2026-04-19.

- **Branch:** `decoup-weight-sync`
- **WandB:** `weight-sync-decup-prorl` / `w9nj4akn`
- **Wall clock:** 49:47 for 20 steps (~149 s/step average)

---

## Topology

| Component | Host | Hardware |
|---|---|---|
| Trainer (FSDP, 8-way) | trainer box, Docker `verlai/verl:vllm018.dev1` | 8 × A100-40GB (one node) |
| Remote vLLM pool (4 children) | EC2 `vllm-instance`, same region | 4 × 23 GiB GPUs, ports 8100–8103 |
| ProRL FastAPI | trainer box, host venv | `:8006` |

---

## Per-step configuration

- `data.train_batch_size=4`, `actor_rollout_ref.rollout.n=4` → 16 rollouts/step
- `tensor_model_parallel_size=2`, `ulysses_sequence_parallel_size=2`
- `max_prompt_length=8192`, `max_response_length=1536`
- `gpu_memory_utilization=0.45`
- `lora_rank=16`, `lora_alpha=32`, targets = `[q,k,v,o,gate,up,down]_proj`
- `save_freq=5`, `total_training_steps=20`

---

## Per-step wall clock

Averages across the 20 steps.

| Phase | Mean | Non-publish (step 19) | Publish (step 20) |
|---|---|---|---|
| `timing_s/gen` (rollout via ProRL+pool) | ~100 s | 156.3 s | 146.5 s |
| `timing_s/old_log_prob` | ~8 s | 7.6 s | 8.7 s |
| `timing_s/ref` | ~6.4 s | 6.4 s | 6.4 s |
| `timing_s/update_actor` | ~22 s | 22.8 s | 22.4 s |
| `timing_s/save_checkpoint` | — (publish only) | — | 10.9 s |
| `timing_s/publish_lora` | — (publish only) | — | 20.2 s |
| `timing_s/step` | ~150 s | 193.1 s | 215.0 s |

`gen` dominates at 60–70% of step wall clock. Trainer `perf/max_memory_allocated_gb = 18.9`, `perf/mfu/actor = 6.4–6.8%`.

---

## LoRA publish timing

Four publishes observed at steps 5, 10, 15, 20. All 4 endpoints 200 on every publish — zero `endpoints_failed`.

| Step | `policy_version` | `endpoints_ok` | Adapter (MiB) | `publish_latency_s` | `transfer_latency_s` | `vllm_load_latency_s` |
|---|---|---|---|---|---|---|
| 5 | 1 | 4/4 | 121.8 | 16.14 | 1.99 | 14.15 |
| 10 | 2 | 4/4 | 121.9 | 16.15 | 1.99 | 14.17 |
| 15 | 3 | 4/4 | 122.0 | 13.20 | 1.99 | 11.21 |
| 20 | 4 | 4/4 | 122.2 | 15.02 | 1.98 | 13.05 |

- `publish_latency_s` = trainer-side end-to-end wall, `max` across the 4-way parallel POST fanout.
- `transfer_latency_s` = `publish_latency_s − vllm_load_latency_s` (network-only).
- `vllm_load_latency_s` = pool-side `engine.add_lora(...)` GPU time, `max` across endpoints.

Observations:

- Network is steady at ~2.0 s for a ~122 MiB tarball on same-region EC2.
- `add_lora` GPU time dominates at 11–14 s per endpoint — the critical-path cost.
- Adapter size drifts ~0.4 MiB across publishes from gzip-compression variance; underlying tensor payload is constant (rank 16 × 7 modules × 36 layers).

At `save_freq=5` with 20 steps → 4 publishes → ~60 s total publish overhead → ~2% of wall clock.

---

## Drift metrics (the signal Phase 1 targets)

With in-step LoRA publish every 5 steps:

| Step | `rollout_corr/ppl_ratio` | `rollout_corr/kl` | `rollout/staleness_steps` |
|---|---|---|---|
| 18 | 1.72 | 0.52 | 3 |
| 19 | 1.60 | 0.45 | 4 |
| 20 (post-publish) | 1.63 | 0.47 | 0 |

`staleness_steps` resets to 0 immediately after each publish.

**Baseline comparison** (pre-Phase-1 run `wdqqu52k`, frozen pool): `ppl_ratio ≈ 1.6` and growing monotonically past 2.0 over 7 steps. Phase 1 keeps it in [1.6, 1.7] with resets — open-loop drift is capped.

---

## Takeaways

1. Rollout dominates step wall clock. Publish cost (~15 s every 5 steps) is ~2% overhead.
2. GPU `add_lora` is the inner bottleneck inside publish, not the wire.
3. `add_lora`/`remove_lora` interleaved with in-flight generates held up for 4 consecutive swaps with zero `/generate` 5xx.

---

## Caveats

- Single 20-step run; `add_lora` variance (11–14 s) is already wider than run-to-run variance would need to be to move conclusions. Treat as one data point.
- Qwen3-4B specifically; larger bases change every row non-linearly.
- Same-region EC2; cross-region would push `transfer_latency_s` by 10–100×.
- `token_level_generation=True` — retokenization cost is not in `timing_s/gen`.
