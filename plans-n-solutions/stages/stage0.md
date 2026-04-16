# Stage 0 - Baseline sanity

**Status: DONE** | [wandb xncnwaie](https://wandb.ai/shamanework-pl/ProAgent/runs/xncnwaie)

## Plan

Reproduce the working GRPO run for 20 steps. Lock a metrics baseline that every later stage is compared against. No code changes - observation only.

**GPU plan:** Colocated. FSDP + vLLM share all 8 GPUs via Ray actors.

**Run commands:** See [`docs/SETUP.md`](../../docs/SETUP.md) Step 6.

## Solution (what broke and how we fixed it)

**Run:** 2026-04-15 to 2026-04-16 | 2h08m wall clock | 20/20 steps

### Problems encountered (16 total)

| # | Problem | Fix |
|---|---|---|
| 1 | `/opt/pytorch` vs Poetry env confusion | Use DLAMI python for runtime, Poetry for lint only |
| 2 | Missing `httpx` in DLAMI | `pip install httpx` |
| 3 | `pull_swe_images.py` used wrong Python | Added `PATH` override in build script |
| 4 | WandB key boundary ambiguity | Validated via GraphQL API |
| 5 | Hydra `+key` collision | Use `++key=value` (add-or-override) |
| 6 | verl needs vLLM 0.8.x, host has 0.19 | Pivoted to Docker image |
| 7 | `logprobs_mode` kwarg rejected by vLLM 0.8.5 | Signature-gated kwarg passing |
| 8 | Short-lived commands exit before tmux registers | Run without tmux |
| 9 | 64 init workers thundering-herd Docker builds | Reduced to 4 for cold start |
| 10 | Hydra bracket syntax looks like shell glob | Documented - not a bug |
| 11 | `prorl.pth` pointed at stale sibling repo | Repointed `.pth` file |
| 12 | Apptainer `su root` PAM failure | `sandbox_config.run_as_fakeroot = True` |
| 13 | `+trainer.max_steps=20` silently ignored | Use `++trainer.total_training_steps=20` |
| 14 | 40GB VRAM OOM during FSDP backward | `gpu_memory_utilization=0.6` + `ulysses_sp=2` |
| 15 | Disk full at step 8 (63GB per checkpoint) | `trainer.save_freq=10` |
| 16 | Checkpoint save included in step timer | Observation only (not a bug) |

Full details of each problem, root cause, and fix are preserved in git history at `plans-n-solutions/solutions/stage0_baseline.md`.

### Validation

| Gate | Value | Pass |
|---|---|---|
| 20 steps completed | 20/20 | Yes |
| `actor/grad_norm` finite | 1.110 (range 1.110-1.629) | Yes |
| `critic/rewards/mean` moves | 0.375 to 0.500 | Yes |
| `actor/kl_loss` finite | 0.003 (range 0.001-0.003) | Yes |
| Checkpoints saved | step 10, step 20 (126 GB total) | Yes |

### Key metrics (step 20)

| Metric | Value |
|---|---|
| `critic/rewards/mean` | 0.500 |
| `actor/grad_norm` | 1.110 |
| `actor/kl_loss` | 0.003 |
| `perf/max_memory_allocated_gb` | 40.322 |
| `response_length/mean` | 8730 tokens |
| `prompt_length/mean` | 4533 tokens |
| Wall clock (20 steps) | 2h08m |
