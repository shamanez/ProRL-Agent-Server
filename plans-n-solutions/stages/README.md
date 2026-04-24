# `plans-n-solutions/stages/` — Run9 moment-of-truth docs

Branch: `full-async`. Phase 2 (fully-async decoupled agentic RL) is **shipped** on commit `55e94122`. Run9 is the evaluation basis that hardened the design and surfaced the Phase 2.5 problem list.

## Why Run9 is the moment of truth

Phase 2 was designed on paper-derived reasoning (Arnal et al. 2026) but tuned in a different regime than ours. Run8 (n=8, `filter_groups=True`, 50 steps) validated the plumbing: weight-sync, replay buffer, temporal IS clip, cooperative-skip safety, §19 shutdown path. All 10 merge-blocking gates passed.

Run9 is the **first paper-aligned config** — `rollout.n=16` (paper §5.1 minimum for meaningful group statistics), `val_before_train=True + test_freq=10` (continuous pass@k signal), DAPO `filter_groups=True`. That config revealed that the plumbing is sound but the **operating regime is not the paper's regime**. The measurements in Run9 — not Run8 — are what drive Phase 2.5 priorities.

Do not read Run8 findings as current. Run9 supersedes.

## The docs

Read in this order on a fresh session:

| # | Doc | Purpose |
|---|---|---|
| 1 | `../handsoff.md` | Single source of truth for the current phase. §15 = ship summary. §16 = Phase 2.5 kickoff (branching point, tiered roadmap). |
| 2 | `run9_n16_report.md` | Primary Run9 evidence: per-step wall-clock, replay dynamics, publish cadence, IS weight trajectory, success-gate status. Read this as the *data*. |
| 3 | `current_bottlenecks_and_problems.md` | Nine problems Run9 surfaces, each with evidence, mechanism, and coupling to other problems. No fixes proposed — this is the problem sheet for Phase 2.5 planning. |
| 4 | `replay_dynamics.md` | Mechanical reference for the fully-async topology: what `train_batch_size`/`n_groups`/`ppo_mini_batch_size` mean, how groups flow from producer → store → trainer, what K=4 caps and what it does not. |
| 5 | `full_async.md` | Phase 2 stage doc (Cut 0): pre-implementation plan. Historical — kept for traceability. |
| 6 | `latencies.md` | Publish-path latency accounting (Phase 1 carryover). |
| 7 | `run8_findings.md` | Run8 (n=8) summary. Superseded by Run9 for current priorities; kept as baseline A/B reference. |
| 8 | `how_to_run.md` | Launch procedures for the three-process topology. |

## Run9 at a glance

**Config deltas vs Run8:** `rollout.n = 8 → 16`, `test_freq = -1 → 10`, `val_before_train = False → True`. Everything else unchanged.

**What was confirmed (plumbing PASS):**

- Weight-sync closed-loop: 4 publishes at steps 5, 10, 15, 20; `endpoints_failed = 0`; pool pv monotonic.
- Replay buffer correctness: 128-group FIFO, K=4 staleness eviction, token-in/token-out preserved across push→sample.
- Cooperative-skip safety (commit `590f8281`): zero fit()-time tracebacks, §19 skips engage correctly (step 10 validation skipped cleanly — see problem #7 in `current_bottlenecks_and_problems.md`).

**What Run9 surfaced (operating regime FAIL):**

1. Each trainer step sees only 1 group → no cross-prompt gradient averaging.
2. Trainer GPUs idle ~96 %: vLLM pool at 100 % util, A100 trainer GPUs at 0 %.
3. `is_weight/clip_fraction ~60 %` (paper target < 20 %) — dominant term is T-mismatch, not real drift.
4. `response_length` saturating at `max_response_length=1536` — reward signal corrupted.
5. Advantage computed twice (once at push, once at sample — redundant).
6. Pool-adapter-age and buffer-age diverge when iters are long.
7. Validation–producer race: first fit()-time §19 skip seen at step 10.
8. Iter 3 wall-clock regression (80 min vs 53 min) — cause TBD, recovered by iter 4.
9. Publish #2 transfer latency 4.7× publish #1.

Full detail, mechanism, and evidence: `current_bottlenecks_and_problems.md`.

## For the next session

1. Branch off commit `55e94122` on `full-async`:

   ```bash
   git checkout -b phase2.5-<topic> 55e94122
   ```

2. Read `current_bottlenecks_and_problems.md` **before** reading the fix roadmap. The problem sheet is regime-measurements; the roadmap is opinion on ordering. Decide your own ordering after reading evidence.

3. Handsoff §16.2 proposes a tiered T1/T2/T3 ranking (cheap fixes → medium changes → architectural moves). That ordering was based on measurement-at-10-steps + conversation extensions through 2026-04-24 and includes two additional signals not yet folded into the tier list:
   - `response_length` saturation (problem #4 here) — promotes to T1.a; invalidates every IS/clip/advantage argument until fixed.
   - "1 group per trainer step" formula (problem #1 here) — stability/generalization issue, not just efficiency.

4. One lever per branch. Keep the monitoring infra (`/tmp/replay_monitor.py`, `/tmp/replay-monitor.jsonl`, Run9 log at `/tmp/s3-fullasync-n16-baseline.log`). Re-read gotchas #19, #20, #22, #25–28 in `handsoff.md` before modifying the producer or validation paths.

## Frozen — do not edit

- `scripts/_internal/s2_weightsync_docker.sh`
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh`

Both preserved for Phase 1 A/B comparison at matched `global_steps`.
