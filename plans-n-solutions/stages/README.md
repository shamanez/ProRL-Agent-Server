# `plans-n-solutions/stages/` — measurements + problem sheet

The system runs fully-async decoupled agentic RL (replay store + temporal IS + continuous producer + LoRA weight-sync). The **moment-of-truth run** is Run9 (n=16, DAPO `filter_groups=True`, 128-group FIFO buffer, K=4 staleness, LR=1e-6, rank-16 LoRA). The plumbing is correct; the operating regime is not yet. This folder holds the evidence and the problem sheet.

## Reading order

| # | Doc | Purpose |
|---|---|---|
| 1 | `../handsoff.md` | How the current system works — topology, launch, credentials, observability, pointer table, gotchas, what not to do. Read this first. |
| 2 | `current_bottlenecks_and_problems.md` | **The work queue.** Nine problems with evidence, mechanism, and coupling. No fixes proposed. Pick one per branch. |
| 3 | `run9_n16_report.md` | Primary evidence: per-step wall-clock, replay dynamics, publish cadence, IS weight trajectory. Read as the *data* behind the problem sheet. |
| 4 | `replay_dynamics.md` | Mechanical reference for the producer → store → trainer topology: what `train_batch_size`/`n_groups`/`ppo_mini_batch_size` mean, how groups flow, what K=4 caps and what it does not. |
| 5 | `how_to_run.md` | Full runbook: env-knob matrix, smoke test, restart/resume, failure runbook. |
| 6 | `latencies.md` | Per-component latency / TPS breakdown (publish-path, FSDP update, old_log_prob recompute, raw per-step table). |

## Run9 at a glance

**Config:** `n=16` (paper §5.1 minimum for meaningful group statistics), `val_before_train=True`, `test_freq=10`, DAPO `filter_groups=True`, baseline replay config.

**Plumbing PASS:**

- Weight-sync closed-loop: 4 publishes at steps 5/10/15/20; `endpoints_failed=0`; pool pv monotonic.
- Replay buffer correctness: 128-group FIFO, K=4 staleness eviction, token-in/token-out preserved.
- Cooperative-skip safety: zero fit()-time tracebacks, §19 skips engage correctly (see problem #7).

**Operating-regime FAIL (the nine problems):**

1. Each trainer step sees only 1 group → no cross-prompt gradient averaging.
2. Trainer GPUs idle ~96 %: vLLM pool at 100 % util, A100s at 0 %.
3. `is_weight/clip_fraction ~60 %` (target < 20 %) — dominant term is T-mismatch, not real drift.
4. `response_length` saturating at `max_response_length=1536` — reward signal corrupted.
5. Advantage computed twice (once at push, once at sample).
6. Pool-adapter-age and buffer-age diverge when iters are long.
7. Validation–producer race: first fit()-time §19 skip seen at step 10.
8. Iter 3 wall-clock regression (80 min vs 53 min) — cause TBD, recovered by iter 4.
9. Publish #2 transfer latency 4.7× publish #1.

Evidence: `current_bottlenecks_and_problems.md`.

## Working on problems

- Read `current_bottlenecks_and_problems.md` fully before picking a lever. The problem sheet groups by root cause (numerical/config mismatch, producer-bound, architectural, network contention).
- One lever per branch. Keep the monitoring infra (`/tmp/replay_monitor.py`, `/tmp/replay-monitor.jsonl`, Run9 log at `/tmp/s3-fullasync-n16-baseline.log`) so new runs are comparable to Run9 baseline numbers.
- Re-read gotchas #19, #20, #22, #25–28 in `handsoff.md` before modifying the producer or validation paths.

## Frozen — do not edit

- `scripts/_internal/s2_weightsync_docker.sh`
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh`

Both preserved for matched-`global_steps` A/B comparison against the current fully-async path.
