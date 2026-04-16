---
description: Resume the decoupled-rollouts staged implementation from where it left off.
---

# Continue Decoupled Rollouts Implementation

You are continuing a multi-stage project to decouple vLLM inference from the GRPO trainer.

## Step 1: Read the plan and current status

Read these files in order:

1. `plans-n-solutions/README.md` - status table, stage map, gating criteria
2. `docs/SETUP.md` - infrastructure setup, 40GB config, Docker image, verl version
3. The stage file for the most recently completed stage (e.g., `plans-n-solutions/stages/stage0.md`) - what broke and how it was fixed
4. The stage file for the next incomplete stage (e.g., `plans-n-solutions/stages/stage1.md`) - implementation plan

## Step 2: Check the environment

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
df -h /home
source ~/.prorl_creds.env
for v in HF_TOKEN WANDB_API_KEY OH_RUNTIME_SINGULARITY_IMAGE_REPO; do
  test -n "${!v}" && echo "$v OK" || echo "$v MISSING"
done
```

## Step 3: Report status and confirm

Before doing any work, report:

```
DECOUPLED ROLLOUTS - STATUS
============================
Last completed stage: [N] - [name] ([wandb URL])
Next stage: [N+1] - [name]
Environment: [OK / issues found]
Plan file: plans-n-solutions/stages/stage{N+1}.md
============================
```

Then ask: "Ready to start Stage [N+1]? Say 'go' to proceed or tell me what to adjust."

## Step 4: Execute the next stage

Follow the plan in the stage file. Key rules:

- **Surgical changes only.** Only touch files listed in the plan.
- **Verify assumptions** before writing code (check that referenced files/functions still exist).
- **Test criteria are in the plan.** Don't declare done until every test passes.
- **Document what breaks** in the Solution section of the stage file.

## Key constraints

- Trainer runs in Docker: `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2`
- verl pinned to commit `60138ebd` at `/tmp/verl`
- Never modify `dev_config/python/**` or widen `pyproject.toml` pins
- Never modify `openhands/llm/nvidia/qwen3.py` (token-level invariant)
- 40GB tuning: `gpu_memory_utilization=0.6`, `ulysses_sequence_parallel_size=2`, `max_prompt_length=16384`, `save_freq=10`
