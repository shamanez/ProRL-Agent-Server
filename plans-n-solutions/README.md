# Decoupled Rollouts — Staged Implementation

Decoupling vLLM inference from the GRPO trainer in `ProRL-Agent-Server`, one stage at a time. Each stage is gated by a short training run on `8 × A100-SXM4-40GB`.

Three stages total, numbered `0 → 1 → 2`. Stage 1 shipped in three cuts (A, B, C) but is one milestone.

Setup guide: [`docs/SETUP.md`](../docs/SETUP.md). Architecture brief: [`docs/decoupling-walkthrough.html`](../docs/decoupling-walkthrough.html).

## Status (2026-04-18)

| Stage | Status | Evidence |
|---|---|---|
| **0 — Baseline sanity (v0.8 + vLLM 0.18)** | **DONE** | Run #13, 20/20 steps, 5 gates green. Commit `13f95697`. |
| **1 — Decoupling milestone (external vLLM + trainer bypass, stale weights)** | **DONE** | Three cuts: Cut A local smoke (7/7 gates, `849314ff`); Cut B local decoupled trainer (5/5 decoupling proofs, `53949b72`, WandB `bgbvlqslo`); Cut C remote HTTP pool (8/8 gates, WandB `wdqqu52k`). |
| **2 — Weight sync + replay buffer** | **NEXT** | Closes the staleness gap opened by Stage 1 Cut C. Plan: [`stages/stage2_weight_sync_and_replay.md`](./stages/stage2_weight_sync_and_replay.md). Fresh-session kickoff (plan-mode first): `/continue-weight-sync`. |

## Stage map

| # | Stage | vLLM location | Weight sync |
|---|---|---|---|
| 0 | Baseline sanity | colocated (Ray actors) | implicit (same tensors) |
| 1 | Decoupling milestone (Cuts A / B / C) | Cut A+B: local GPUs 4–7 · Cut C: remote EC2 (public HTTP) | none (stale on purpose) |
| 2 | Weight sync + replay buffer | remote EC2 host | periodic publish from trainer + bounded replay buffer |

## Per-stage docs

- [`stages/stage0.md`](./stages/stage0.md) — Stage 0 baseline (DONE)
- [`stages/stage1.md`](./stages/stage1.md) — Stage 1 umbrella: Cuts A + B (DONE)
- [`stages/stage1_remote_pool.md`](./stages/stage1_remote_pool.md) — Stage 1 Cut C remote HTTP pool (DONE)
- [`stages/stage1_playbook.md`](./stages/stage1_playbook.md) — Stage 1 historical execution runbook (reference only)
- [`stages/stage2_weight_sync_and_replay.md`](./stages/stage2_weight_sync_and_replay.md) — Stage 2 weight sync + replay buffer (NEXT)

## Gating standard

A stage is done when a short GRPO run satisfies:
- `training/global_step` reaches the target (7+ for a Stage 1 Cut-C smoke; 20+ for Stage 2)
- `actor/grad_norm` finite every step (> 0 on steps with reward variance)
- `critic/rewards/mean` non-zero on at least one step
- `actor/kl_loss` finite every step
- Advantage variance > 0
- Stage 2+: at least one weight-publish event observed and `rollout/staleness` bounded

## How to advance to the next stage

1. **Always start a fresh session in Claude Code plan mode.** The continuation command says so explicitly; `ExitPlanMode` is the gate for writing any code.
2. Invoke the matching slash command at the start of the session:
   - `/continue-weight-sync` — Stage 2 (weight sync + replay buffer). Current frontier.
   - `/continue-decoupling` — deprecated; redirects to the above.
3. The slash command prints the reading order, runs the environment checks, and emits the `STATUS` preamble. Only after the preamble is printed should planning begin.
4. The execution rules in each continuation command are binding — sibling launchers (no edits to frozen files), never `--no-verify`, never `git push` without explicit approval.

## Hard constraints

- Single-box trainer: `8 × A100-SXM4-40GB`, no Slurm.
- Trainer runs in Docker: `verlai/verl:vllm018.dev1` (vLLM 0.18, PyTorch 2.6+).
- verl source: `/tmp/verl`, `shamanez/verl` main at commit `910ba344` (v0.8.0.dev).
- Stage-specific pool locations:
  - Stage 1 Cuts A + B: local GPUs 4–7 via `scripts/serving/launch_external_vllm_pool.sh`.
  - Stage 1 Cut C and Stage 2+: remote EC2 via `scripts/serving/launch_remote_vllm_pool.sh`.
- Frozen files (never modify — baseline reproduction depends on them):
  - `scripts/_internal/s0_baseline_docker.sh`
  - `scripts/_internal/s0_prorl.sh`
  - `scripts/_internal/s1_remote_docker.sh` (frozen from Stage 1 Cut C)
  - `scripts/_internal/s2_decoupled_docker.sh` (frozen from Stage 1 Cut B)
  - `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh`
  - `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_decoupled.sh`
  - `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_remote_decoupled.sh`
- Token-level invariant: never modify `openhands/llm/nvidia/qwen3.py` or `qwen2_5_vl.py`.
- Config / pins: never modify `dev_config/python/**`; don't widen `pyproject.toml` pins without reading the pin comment.
