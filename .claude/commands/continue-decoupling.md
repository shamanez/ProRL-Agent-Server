---
description: Resume the decoupled-rollouts staged implementation from where it left off.
---

# Continue Decoupled Rollouts Implementation

You are continuing a multi-stage project to decouple vLLM inference from the GRPO trainer. The session is fresh — assume no prior context.

## Step 1: Read the plan and current status

Read these files in order:

1. `plans-n-solutions/README.md` — status table, stage map, gating criteria
2. `docs/SETUP.md` — infrastructure setup, 40 GB config, Docker image, verl version
3. The stage file for the most recently completed stage (e.g., `plans-n-solutions/stages/stage0_1.md`) — what broke and how it was fixed
4. The stage file for the next incomplete stage (e.g., `plans-n-solutions/stages/stage1.md`) — implementation plan and test criteria

## Step 2: Check the environment

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
df -h /home
source ~/.prorl_creds.env
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/singularity_images
for v in HF_TOKEN WANDB_API_KEY OH_RUNTIME_SINGULARITY_IMAGE_REPO; do
  test -n "${!v}" && echo "$v OK" || echo "$v MISSING"
done

# Upgraded-stack sanity
docker image inspect verlai/verl:vllm018.dev1 >/dev/null 2>&1 \
  && echo "docker image OK" || echo "docker image MISSING — pull verlai/verl:vllm018.dev1"
test -d /tmp/verl && echo "/tmp/verl OK" || echo "/tmp/verl MISSING — clone shamanez/verl main there"
```

## Step 3: Report status and confirm

Before doing any work, report:

```
DECOUPLED ROLLOUTS - STATUS
============================
Last completed stage: [N] - [name] ([wandb URL])
Next stage:           [N+1] - [name]
Environment:          [OK / issues found]
Plan file:            plans-n-solutions/stages/stage<N+1>.md
============================
```

Then ask: "Ready to start Stage [N+1]? Say 'go' to proceed or tell me what to adjust."

## Step 4: Execute the next stage

Follow the plan in the stage file. Key rules:

- **Surgical changes only.** Only touch files listed in the plan.
- **Verify assumptions** before writing code (check that referenced files/functions still exist in the upgraded stack — v0.4 paths may be gone).
- **Test criteria are in the plan.** Don't declare done until every test passes.
- **Document what breaks** in the Solution section of the stage file, using the same table format Stage 0.1 uses for its Problem log.

## Key constraints (upgraded stack — post Stage 0.1)

- **Trainer runs in Docker:** `verlai/verl:vllm018.dev1` (vLLM 0.18, PyTorch 2.6+).
- **verl source:** `/tmp/verl` holds `shamanez/verl` main (v0.8.0.dev, commit `910ba344`). Installed in editable mode inside the container via `pip install --no-deps -e /opt/verl` (the script mounts `/tmp/verl` → `/opt/verl`).
- **verl_custom:** installed via `pip install --no-deps -e /workspace/trainer_integration/verl` inside the container. Custom worker subclass + config monkey-patches live here; upstream `/tmp/verl` stays read-only.
- **ProRL runs on the host**, not in the Docker container. Trainer container joins host networking to reach `http://localhost:8006`.
- **40 GB tuning (A100-40GB × 8):**
  - `gpu_memory_utilization=0.45` (lowered from 0.6 after backward-pass OOM on step 5)
  - `ulysses_sequence_parallel_size=2`
  - `max_prompt_length=16384`, `max_response_length=1536`
  - `save_freq=10`, `total_training_steps=20` for validation runs
  - `+actor_rollout_ref.actor.calculate_entropy=false` (required — custom compute_log_prob override relies on this)
- **Env vars inside the container:**
  - `PYTORCH_ALLOC_CONF=expandable_segments:True` — required to avoid activation-memory fragmentation OOM on 40 GB cards
  - `PYTHONPATH=/workspace`
  - `WANDB_API_KEY`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` forwarded from the host
- **Never modify:**
  - `dev_config/python/**` (linter/formatter/type-checker configs — ask user first)
  - `pyproject.toml` pins (read the comment line before widening anything)
  - `openhands/llm/nvidia/qwen3.py` (token-level invariant — rollouts pass token IDs verbatim)
- **Pre-commit autoflake will strip module-level imports it thinks are unused.** If you add an import that's only referenced inside a decorator, late-bound method, or newly-generated string, wrap it in a closure or use an inline import at the call site with `# noqa: PLC0415`. See `verl_custom/workers/fsdp_workers.py` (`_build_compute_log_prob`) and `verl_custom/trainer/ppo/ray_trainer.py:1242` (inline `local_mkdir_safe`) for precedent.
- **Never commit with `--no-verify`.** If a hook fails, fix it.
- **Do not `git push`** without explicit user approval.

## Stage 0.1 artifacts you can reuse

These are already in place — do NOT rebuild:

- 49 Singularity `.sif` images at `singularity_images/`
- Training parquet at `/home/ubuntu/data/SkyRL-v0-293/{train,validation}.filtered.parquet`
- Docker image `verlai/verl:vllm018.dev1` (pulled)
- verl checkout at `/tmp/verl`
- Qwen3-4B weights at `/home/ubuntu/.cache/huggingface/`

## If you need to rerun Stage 0.1 validation

Follow `plans-n-solutions/stages/stage0_1.md` Steps 1–6. The whole chain (clean state → ProRL host launch → Docker trainer launch → monitor → gating metrics → commit) is self-contained there.
