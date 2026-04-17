# Decoupled Rollouts — Staged Implementation

Decoupling vLLM inference from the GRPO trainer in `ProRL-Agent-Server`, one stage at a time. Each stage is gated by a 20-step training run on `8 x A100-SXM4-40GB`.

Setup guide: [`docs/SETUP.md`](../docs/SETUP.md). Architecture brief: [`docs/decoupling-walkthrough.html`](../docs/decoupling-walkthrough.html).

**The "decoupled rollouts" milestone is a single stage** ([`stages/stage1.md`](./stages/stage1.md)) with two parts: Part A hosts vLLM standalone; Part B cuts the cord by making the trainer bypass its in-Ray vLLM. Shipping only one of them proves nothing. See [`stages/stage1_playbook.md`](./stages/stage1_playbook.md) — the operator playbook — and the reusable launchers (`scripts/_internal/s0_prorl.sh` + the new sibling `scripts/_internal/s2_decoupled_docker.sh`).

## Status (2026-04-17)

| Stage | Status | Evidence |
|---|---|---|
| **0 - Baseline sanity (v0.8 + vLLM 0.18)** | **DONE** | Run #13, 20/20 steps, all 5 gates green. Commit `13f95697`. |
| **1 - Decoupling milestone (external vLLM + trainer bypass, stale weights)** | **ENGINEERING PROVEN** | 7/7 Part-A smoke gates green (commit `849314ff`). 5/5 Part-B decoupling proofs green on trainer PID 6769 (see `stages/stage1.md` §"Decoupling proofs"): `EXTERNAL BYPASS ACTIVE` marker logged, 0 Ray vLLM actor spawns, 0 `init_engine` calls, external endpoints registered as `server_addresses`, and ≈1.5 k `POST /generate` hits on each of the 4 pool children. The trainer's rollout logprobs are sourced from the external pool, not an in-Ray vLLM worker. |
| 2 - Iterative off-policy publish | NOT STARTED | Plan in `stages/stage3.md` (number kept to match the doc on disk). |
| 3 - Policy Registry + blue-green | NOT STARTED | |
| 4 - Trajectory store + replay | NOT STARTED | |

## Stage map

| # | Stage | vLLM location | Weight sync |
|---|---|---|---|
| 0 | Baseline sanity | colocated (Ray actors) | implicit (same tensors) |
| 1 | Decoupling milestone (external vLLM + trainer bypass, stale weights) | standalone (Part A: GPUs 0-1; Part B: GPUs 4-7) | none (stale on purpose in Part B) |
| 3 | Iterative off-policy publish | standalone | HF checkpoint round-trip |
| 4 | Policy Registry + blue-green | two standalone pools | warm + cutover |
| 5 | Trajectory store + replay | two pools, continuous | periodic publish |

## Per-stage docs

Each stage's plan and solution (post-mortem) are in a single file under [`stages/`](./stages/):

- [`stages/stage0.md`](./stages/stage0.md) - Baseline sanity on v0.8 + vLLM 0.18 (DONE)
- [`stages/stage1_playbook.md`](./stages/stage1_playbook.md) - **Operator playbook for the decoupling milestone**
- [`stages/stage1.md`](./stages/stage1.md) - Decoupling milestone: external vLLM + trainer bypass (DONE — engineering proven)
- [`stages/stage3.md`](./stages/stage3.md) - Iterative off-policy publish
- [`stages/stage4.md`](./stages/stage4.md) - Policy Registry + blue-green
- [`stages/stage5.md`](./stages/stage5.md) - Trajectory store + replay

## Gating standard (every stage)

A stage is done when a 20-step GRPO run satisfies:
- `step >= 20` in wandb
- `actor/grad_norm` finite (> 0, < 1e6)
- `critic/rewards/mean` not identically zero
- Advantage variance > 0
- `actor/kl` finite
- For stages >= 3: at least one weight publish event observed

## Hard constraints

- Single-box `8 x A100-SXM4-40GB`. No Slurm.
- Trainer runs in Docker: `verlai/verl:vllm018.dev1` (vLLM 0.18)
- verl pinned to commit `910ba344` v0.8.0.dev at `/tmp/verl`
- **Reuse the existing launchers.** Server = `bash scripts/_internal/s0_prorl.sh` (poetry). Stage-2 trainer = a **new sibling** `scripts/_internal/s2_decoupled_docker.sh` that mirrors `s0_baseline_docker.sh` — not an edit of it.
- Never modify `scripts/_internal/s0_baseline_docker.sh`, `scripts/_internal/s0_prorl.sh`, or `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` — those reproduce the Stage 0 baseline and must stay frozen.
- Never modify `dev_config/python/**` or widen `pyproject.toml` pins
- Never modify `openhands/llm/nvidia/qwen3.py` (token-level invariant)
