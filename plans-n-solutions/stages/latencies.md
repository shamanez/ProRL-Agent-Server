# Latency / TPS breakdown — run8 Phase E2 (filter_groups=True, 50-step DAPO)

Source log: `/tmp/s3-fullasync.log`. Branch `full-async`. Data window: steps 7-46 (n=40 per-step samples; step 6 excluded — cumulative-from-launch).

Purpose: decide where to invest tuning effort. Numbers below are extracted from existing verl metric lines and the `publish_lora_adapter` JSON event stream.

## Per-step component budget

| Component | Mean/step | Median | p95 | TPS / Bandwidth | Log key |
|---|---|---|---|---|---|
| Rollout wait (producer-gated) | **623.1 s** | — | — | — | `timing_s/gen` (diff) |
| FSDP update (fwd+bwd+opt) | 18.07 s | 17.57 s | 21.34 s | **338 tok/s** (seq/upd_s), MFU 3.9% | `timing_s/update_actor` |
| `old_log_prob` recompute | 7.59 s | 6.41 s | — | ~800 tok/s | `timing_s/old_log_prob` |
| Reward compute | 0.08 s | — | — | — | `timing_s/reward` |
| **Step wall-clock total** | **651 s** | — | — | — | `timing_s/step` |

Inputs: `response_length/mean = 1516 tok`, `global_seqlen/mean = 6098 tok` (8 trajectories × (~4100 prompt + ~1500 response)), `perf/mfu/actor = 0.039`.

## Weight sync (9 events, pv:6 → pv:14, save_freq=5)

| Stat | `publish_latency_s` | `transfer_latency_s` (S3 upload) | `vllm_load_latency_s` (4-child reload) |
|---|---|---|---|
| mean | 29.96 | 17.28 | 12.68 |
| median | 32.63 | 19.01 | 13.61 |
| min | 17.47 | 4.03 | 3.73 |
| max | 34.56 | 19.04 | 16.03 |

Adapter size: 244.5 MB mean. S3 upload bandwidth 14.2 MB/s; vLLM 4-child load bandwidth 19.3 MB/s (includes drain).

**Budget: 30 s publish ÷ (5 steps × 651 s/step) = 0.92% of wall-clock.** Weight sync is NOT a bottleneck. `endpoints_ok:4` on every publish. `endpoints_failed:0`.

## Staleness

- `replay/sample_age_steps_p95` max observed = **3 steps** (K=4 hard cap not hit).
- `replay/dropped_by_staleness_total = 0` end-to-end.
- `replay/store_fill_ratio` 0.000-0.023 → buffer near-empty; trainer consumes as fast as producer pushes. Regime is near-on-policy, NOT replay-reuse.
- `rollout/staleness_steps` cycles 0→4 matching `save_freq=5` (expected).

## DAPO pipeline

- 11 `generate_sequences_dapo` producer calls (bug #16 reset fires each call: "dropped N leftover jobs" × 11).
- 54 hard-filter events across 45 unique prompts (see `run8_findings.md`).
- 81% of filtered groups are 0/8 (all-fail), 19% are 8/8 (all-pass).
- Repeat filtering confirmed: `dask-8903` × 3, `dask-6809` × 3, `MONAI-4796` × 3.

## Bottleneck diagnosis

**Rollout dominates 96% of wall-clock.** Pattern per DAPO release:
1. DAPO accumulates until 4 surviving groups exist (~2500-3500 s wait).
2. Trainer consumes 4 steps back-to-back (~25 s each).
3. Store empties; trainer idle until next DAPO release.

Consequences:
- Replay buffer provides no speedup under current regime (on-policy-like). Its role is smoothing trainer utilization, not raw throughput.
- `filter_groups=True` amplifies rollout waste: each discarded 8-rollout group ≈ (1 prompt × 8 n × ~1500 tok) of wall-clock with zero gradient.

## Logging gaps

To inform the next tuning decision (bigger pool vs `filter_groups=False` + true replay reuse), the log is missing:

| Missing signal | Why it matters |
|---|---|
| Pure vLLM gen time inside each DAPO call | Separates vLLM throughput ceiling from DAPO filter overhead |
| `groups_drawn / groups_survived / groups_dropped` per DAPO call | Quantifies filter-induced rollout waste |
| Producer wall-clock per `generate_sequences_dapo` call | Confirms DAPO accumulation dominates vs pure generation |
| vLLM child-side output-tokens/sec | Raw rollout TPS ceiling — decides pool-size / tp sweeps |

## Proposed log additions

All additions are single JSON lines at existing log boundaries; no behavior change.

| # | File | Location | Event |
|---|---|---|---|
| 1 | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` | End of `generate_sequences_dapo` | `{"event":"dapo_producer_call","wall_s":…,"vllm_gen_s":…,"groups_drawn":…,"groups_survived":…,"groups_dropped_filter":…,"tokens_out":…,"effective_tps":…}` |
| 2 | `trainer_integration/verl/verl_custom/replay/continuous_producer.py` | `_run()` after each push | `{"event":"producer_iter","wall_s":…,"store_full_idles":…}` |
| 3 | `openhands/llm/nvidia/qwen3.py` + `qwen2_5_vl.py` | `/generate` completion | `{"event":"vllm_generate","tokens_out":…,"gen_s":…,"tps":…}` (DEBUG, gate-controlled) |

## Decision table (populate after logging lands)

| Signal threshold | Action |
|---|---|
| Raw vLLM TPS < 100 tok/s/GPU | Bump pool children or `tp_size`; reclaim vLLM-side waste |
| `groups_dropped_filter / groups_drawn > 0.5` | Flip to `filter_groups=False` + replay reuse (Phase 2 primary config) |
| Trainer idle > 70% of step time | Lower `train_batch_size` or raise `producer_batch_size` |
| `sample_age_steps_p95` approaches K=4 | Raise K or lower `save_freq` |
| `is_weight/clip_fraction > 0.2` | Shrink buffer_size or lower K; TIS clipping masking drift |

## Raw per-step table

Columns: step | step_s (wall-clock) | gen_s (rollout wait marginal) | upd_s (FSDP) | lp_s (old_log_prob) | rew_s | resp (tokens) | seq (global_seqlen) | trnTPS | MFU | sample_age_p95 | reward

```
step  step_s  gen_s  upd_s  lp_s  rew_s  resp   seq  trnTPS   mfu  agep95  rew
   6     0.0 2877.1  26.14 24.36  0.18  1536  5658     216  0.025   0.0  0.375
   7  2927.9    0.0  17.77  6.52  0.04  1536  5596     315  0.036   1.0  0.625
   8    24.4    0.0  17.24  6.16  0.02  1536  6277     364  0.042   2.0  0.875
   9    23.5    0.0  17.25  6.11  0.02  1536  6314     366  0.042   3.0  0.250
  10    23.4 3615.6  19.15 12.72  0.36  1536  6746     352  0.043   0.0  0.750
  11  3691.3    0.0  17.36  6.41  0.04  1536  6321     364  0.042   1.0  0.750
  12    23.8    0.0  17.18  6.16  0.02  1536  5996     349  0.040   2.0  0.875
  13    23.4    0.0  17.14  6.04  0.02  1536  5642     329  0.037   3.0  0.125
  14    23.2 3728.9  19.40 12.58  0.12  1536  6336     327  0.039   0.0  0.125
  15  3761.1    0.0  18.66  6.46  0.04  1536  5483     294  0.034   1.0  0.625
  16    84.7    0.0  17.23  6.41  0.03  1536  6073     353  0.041   2.0  0.125
  17    23.7    0.0  17.15  6.09  0.02  1536  5542     323  0.037   3.0  0.250
  18    23.3 2259.1  17.78  9.42  0.36  1536  6951     391  0.047   0.0  0.250
  19  2286.8    0.0  18.01  6.62  0.04  1536  6535     363  0.043   1.0  0.125
  20    24.7    0.0  17.30  6.17  0.04  1365  5460     316  0.036   2.0  0.750
  21    76.8    0.0  17.29  6.37  0.03  1536  6594     381  0.045   3.0  0.750
  22    23.7  606.1  19.40  9.77  0.35  1536  6442     332  0.039   0.0  0.750
  23   694.2    0.0  17.24  6.60  0.03  1536  5710     331  0.038   1.0  0.875
  24    23.9    0.0  17.21  6.17  0.02  1536  5485     319  0.036   2.0  0.125
  25    23.4    0.0  17.10  6.09  0.02  1536  5585     327  0.037   3.0  0.625
  26    78.5 2786.1  19.55 11.65  0.35  1536  5529     283  0.033   0.0  0.500
  27  2817.8    0.0  18.29  6.61  0.04  1536  6978     382  0.046   1.0  0.875
  28    25.0    0.0  17.46  6.38  0.03  1536  5492     315  0.036   2.0  0.125
  29    23.9    0.0  17.29  6.08  0.02  1392  6085     352  0.040   3.0  0.125
  30    23.4 2261.2  22.58 13.09  0.41  1536  6399     283  0.033   0.0  0.125
  31  2357.1    0.0  18.50  6.42  0.02  1536  5518     298  0.034   1.0  0.250
  32    25.0    0.0  17.24  6.22  0.02  1536  6206     360  0.042   2.0  0.375
  33    23.5    0.0  17.32  6.17  0.02  1536  6073     351  0.041   3.0  0.250
  34    23.5 1121.7  18.57 11.27  0.33  1536  6525     351  0.042   0.0  0.500
  35  1152.0    0.0  18.29  6.45  0.04  1536  6221     340  0.040   1.0  0.125
  36    77.4    0.0  18.29  6.39  0.02  1373  6372     348  0.040   2.0  0.125
  37    24.7    0.0  17.39  6.16  0.03  1536  5887     338  0.039   3.0  0.750
  38    23.6 1948.7  17.69 10.04  0.30  1536  5700     322  0.037   0.0  0.625
  39  1976.8    0.0  17.84  6.55  0.03  1536  5562     312  0.036   1.0  0.500
  40    24.5    0.0  17.30  6.27  0.02  1536  5517     319  0.036   2.0  0.875
  41    67.7    0.0  17.21  6.37  0.02  1536  5760     335  0.038   3.0  0.375
  42    23.6 3304.7  21.34 12.00  0.29  1211  6067     284  0.034   0.0  0.125
  43  3338.5    0.0  18.12  6.56  0.04  1536  7142     394  0.047   1.0  0.500
  44    24.8    0.0  17.55  6.31  0.02  1536  7205     411  0.049   2.0  0.125
  45    23.9    0.0  17.60  6.13  0.02  1536  6544     372  0.044   3.0  0.125
  46    77.7 3291.5  20.71 11.58  0.14  1536  6059     293  0.034   0.0  0.750
  47    24.8    0.0  17.33  6.60  0.04  1536  5720     330  0.038   1.0  0.125
  48    23.5    0.0  17.24  6.11  0.02  1536  6193     359  0.042   2.0  0.750
  49    23.5    0.0  17.55  6.20  0.03  1536  5849     333  0.038   3.0  0.625
  50  3348.9 3284.4  20.98 11.73  0.27  1536  7105     339  0.039   0.0  0.875
```

**Final aggregate (steps 7-50, n=44):** mean upd_actor=18.1s, mean step=672s, gen_wait=96% of wall-clock, trainer MFU 3.9%, sample_age_p95 max=3. Step 50 reward 0.875 (new high).

## Publish events (raw)

```
pv: 6  publish:17.47 xfer: 4.03 load:13.44
pv: 7  publish:33.61 xfer:18.99 load:14.62
pv: 8  publish:31.24 xfer:19.02 load:12.22
pv: 9  publish:33.83 xfer:19.04 load:14.78
pv:10  publish:33.48 xfer:19.02 load:14.46
pv:11  publish:34.57 xfer:18.53 load:16.03
pv:12  publish:30.25 xfer:19.01 load:11.24
pv:13  publish:22.58 xfer:18.84 load: 3.73
pv:14  publish:32.63 xfer:19.02 load:13.61
pv:15  publish:35.38 xfer:19.06 load:16.32
```

**Final publish aggregate (10 events, pv:6→15):** mean publish 30.50s, mean xfer 17.46s, mean load 12.99s. All `endpoints_ok:4, endpoints_failed:0`. Adapter mean 244.4 MB.

## Related docs

- `plans-n-solutions/stages/run8_findings.md` — plumbing gates + DAPO filter pathology
- `plans-n-solutions/stages/full_async.md` — Phase 2 design
- `plans-n-solutions/handsoff.md` — Phase 2 spec + gotchas (incl. §19 cooperative stop, §20 GIL atomicity)
