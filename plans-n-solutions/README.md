# Decoupled Rollouts — Staged Implementation

Decoupling vLLM inference from the GRPO trainer in `ProRL-Agent-Server`, one stage at a time. Each stage is gated by a 20-step training run on `8 x A100-SXM4-40GB`.

Setup guide: [`docs/SETUP.md`](../docs/SETUP.md). Architecture brief: [`docs/decoupling-walkthrough.html`](../docs/decoupling-walkthrough.html).

## Status (2026-04-16)

| Stage | Status | Evidence |
|---|---|---|
| **0 - Baseline sanity** | **DONE** | [wandb xncnwaie](https://wandb.ai/shamanework-pl/ProAgent/runs/xncnwaie) - 20/20 steps, rewards 0.375-0.500, grad_norm finite |
| 1 - External vLLM standalone | NOT STARTED | |
| 2 - Trainer bypass (stale weights) | NOT STARTED | |
| 3 - Iterative off-policy publish | NOT STARTED | |
| 4 - Policy Registry + blue-green | NOT STARTED | |
| 5 - Trajectory store + replay | NOT STARTED | |

## Stage map

| # | Stage | vLLM location | Weight sync |
|---|---|---|---|
| 0 | Baseline sanity | colocated (Ray actors) | implicit (same tensors) |
| 1 | External vLLM standalone | standalone (GPUs 0-1) | n/a (no training) |
| 2 | Trainer bypass, stale weights | standalone (GPUs 4-7) | none (stale on purpose) |
| 3 | Iterative off-policy publish | standalone | HF checkpoint round-trip |
| 4 | Policy Registry + blue-green | two standalone pools | warm + cutover |
| 5 | Trajectory store + replay | two pools, continuous | periodic publish |

## Per-stage docs

Each stage's plan and solution (post-mortem) are in a single file under [`stages/`](./stages/):

- [`stages/stage0.md`](./stages/stage0.md) - Baseline sanity (DONE)
- [`stages/stage1.md`](./stages/stage1.md) - External vLLM standalone
- [`stages/stage2.md`](./stages/stage2.md) - Trainer bypass
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
- Trainer runs in Docker: `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2`
- verl pinned to commit `60138ebd` (cloned at `/tmp/verl`)
- Never modify `dev_config/python/**` or widen `pyproject.toml` pins
- Never modify `openhands/llm/nvidia/qwen3.py` (token-level invariant)
