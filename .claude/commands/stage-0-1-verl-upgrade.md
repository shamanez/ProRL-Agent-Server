---
description: Plan and execute Stage 0.1 — upgrade verl + verl_custom to latest, get a working 20-step training run.
---

# Stage 0.1 — Upgrade verl to latest and port verl_custom

## Context (read this first — you have no prior conversation)

This repo (`ProRL-Agent-Server`) is a scalable RL training system for software engineering agents. It has two main parts:

1. **ProRL Server** (host, Poetry venv) — drives coding agents inside Singularity sandboxes, talks to vLLM over HTTP with token IDs
2. **Trainer** (Docker container) — runs GRPO training via `verl` with FSDP, colocated vLLM

The trainer depends on **verl** (an RL training framework) plus a **patch package** called `verl_custom` that extends verl with custom rollout workers, reward managers, and training scripts. Today this works with:

- verl pinned to commit `60138ebd` (v0.4-dev, from the official repo)
- Docker image `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2` (vLLM 0.8.5, torch 2.6)
- `verl_custom` at `trainer_integration/verl/verl_custom/` (75 .py files, 50+ imports from verl internals)

**Stage 0** (baseline) is DONE — verified with a 20-step training run, rewards 0.375→0.500. See `plans-n-solutions/stages/stage0.md`.

The larger project goal is to **fully decouple vLLM inference from the trainer** (Stages 1-5 in `plans-n-solutions/README.md`). But building that decoupling against an old verl API means the work gets redone when upgrading later. So **Stage 0.1 upgrades verl first**, then the decoupling work builds on a modern foundation.

## What Stage 0.1 must accomplish

**Primary goal**: Get one end-to-end 20-step training run working with the LATEST verl (from the user's fork) and a compatible Docker image. Loss must decrease. Only after this succeeds should anything else be updated.

**Secondary goal** (only after primary succeeds): Update `docs/SETUP.md` to reflect the new versions.

## Step-by-step instructions

### Phase 1: Understand the current coupling (READ ONLY — no edits yet)

1. Read `plans-n-solutions/README.md` and `plans-n-solutions/stages/stage0.md` for current state.
2. Read `docs/SETUP.md` for how the current system is set up.
3. Read the critical verl_custom files that import verl internals. Focus on these (they have the most imports and are the most likely to break):
   - `trainer_integration/verl/verl_custom/trainer/main_ppo.py` (~15 verl imports)
   - `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` (~15 verl imports)
   - `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` (~5 verl imports)
   - `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py` (~1 verl import)
   - `trainer_integration/verl/verl_custom/workers/actor/dp_actor.py` (~12 verl imports)
   - `trainer_integration/verl/verl_custom/utils/dataset/rl_dataset.py` (~5 verl imports)
4. Read the training script: `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh`
5. Read the Docker launch script: `scripts/_internal/s0_baseline_docker.sh`

### Phase 2: Analyze the target verl version

1. Clone/fetch the user's fork: `https://github.com/shamanez/verl/tree/main`
2. Compare the API surface between the old verl (commit `60138ebd`) and the new one. Specifically check:
   - Do the 50+ import paths in verl_custom still exist? (`verl.protocol`, `verl.single_controller.ray`, `verl.workers.fsdp_workers`, `verl.workers.rollout.async_server`, `verl.utils.*`, etc.)
   - What new async/decoupled rollout features does the latest verl have?
   - What Docker images are available for the latest verl version?
3. Check available Docker images: `docker search verlai/verl` or check DockerHub for tags matching the new verl version.

### Phase 3: Write the plan BEFORE any code changes

Create `plans-n-solutions/stages/stage0_1.md` with:

1. **Target architecture** — what the end state looks like (new verl version, new Docker image, what changes in verl_custom)
2. **Import compatibility matrix** — for each of the 50+ verl imports, does it still exist in the new version? What's the replacement?
3. **Decision: patch vs. minimize** — for each verl_custom file, explicitly decide:
   - Keep and port (update imports/API calls to match new verl)
   - Remove (if the new verl already provides this functionality)
   - Keep as-is (if imports haven't changed)
4. **Docker image choice** — which verlai/verl image to use, and why
5. **Risk assessment** — what's most likely to break

Present this plan to the user and ask for approval before proceeding.

### Phase 4: Implement (only after plan approval)

1. Create branch `prorl-verl` in BOTH repos:
   - `https://github.com/shamanez/verl/` — branch `prorl-verl` from main
   - `https://github.com/shamanez/ProRL-Agent-Server` — branch `prorl-verl` from current branch (`de-coupled`)
2. Port verl_custom to the new verl API. Update imports, fix breaking changes.
3. Update `scripts/_internal/s0_baseline_docker.sh` with the new Docker image (for the latest and stable - https://hub.docker.com/r/verlai/verl/tags) and any changed mount/install steps.
4. Update the verl clone step (new repo URL, new commit/branch).

### Phase 5: Validate (THE MOST IMPORTANT STEP)

Run a 20-step training using the same gating criteria as Stage 0:

| Metric | Pass |
|---|---|
| `step` | >= 20 |
| `actor/grad_norm` | finite, > 0, < 1e6 |
| `critic/rewards/mean` | not identically zero |
| Advantage variance | > 0 |
| `actor/kl` | finite |
| Loss/rewards | Must show improvement trend (not flat) |

If validation fails, diagnose and fix. Do NOT move on until a clean 20-step run completes.

### Phase 6: Update docs and push (only after Phase 5 succeeds)

1. Update `docs/SETUP.md` with:
   - New Docker image name and version
   - New verl clone URL and commit/branch
   - Any changed Hydra overrides
   - Updated version reference table and compatibility matrix
2. Update `plans-n-solutions/stages/stage0_1.md` Solution section with:
   - What broke and how it was fixed
   - Import mapping (old → new)
   - Wandb run URL as evidence
3. Update `plans-n-solutions/README.md` status table
4. Push to both repos:
   - `git push -u origin prorl-verl` on `shamanez/verl`
   - `git push -u origin prorl-verl` on `shamanez/ProRL-Agent-Server`

## Key constraints

- **Machine**: 8 x A100-SXM4-40GB, single box
- **40GB Hydra overrides** (must be preserved): `gpu_memory_utilization=0.6`, `ulysses_sequence_parallel_size=2`, `max_prompt_length=16384`, `save_freq=10`
- **NEVER modify** `openhands/llm/nvidia/qwen3.py` (token-level invariant)
- **NEVER modify** `dev_config/python/**` or widen `pyproject.toml` pins
- **ProRL server side is unchanged** — only the trainer side (Docker + verl + verl_custom) changes
- The user's verl fork: `https://github.com/shamanez/verl/tree/main`
- Both the base verl repo AND the patch (verl_custom) may need updating — they are tightly coupled

## Critical verl imports to check (full list)

These are all `from verl.*` imports in verl_custom. Every one must be verified against the new verl:

```
verl.protocol.DataProto, pad_dataproto_to_divisor, unpad_dataproto
verl.single_controller.ray.RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
verl.single_controller.ray.base.create_colocated_worker_cls
verl.single_controller.base.Worker
verl.workers.fsdp_workers.ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
verl.workers.rollout.async_server.async_server_class, AsyncLLMServerManager
verl.workers.megatron_workers.ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
verl.workers.reward_manager.NaiveRewardManager, DAPORewardManager, get_reward_manager_cls
verl.workers.actor.BasePPOActor
verl.utils.torch_functional (verl_F), masked_mean, logprobs_from_logits
verl.utils.model.compute_position_id_with_mask
verl.utils.fs.copy_to_local, copy_local_path_from_hdfs
verl.utils.checkpoint.checkpoint_manager.BaseCheckpointManager, find_latest_ckpt_path
verl.utils.debug.performance._timer
verl.utils.debug.GPUMemoryLogger
verl.utils.metric.reduce_metrics
verl.utils.seqlen_balancing.get_seqlen_balanced_partitions, log_seqlen_unbalance, get_reverse_idx, rearrange_micro_batches
verl.utils.tracking.Tracking, ValidationGenerationsLogger
verl.utils.reward_score._default_compute_score, default_compute_score
verl.utils.import_utils.load_extern_type, deprecated
verl.utils.vllm_utils.is_version_ge
verl.utils.hf_tokenizer, hf_processor
verl.utils.fsdp_utils.FSDPModule, fsdp2_clip_grad_norm_
verl.utils.py_functional.append_to_dict
verl.utils.ulysses.gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
verl.utils.device.get_device_name, get_torch_device, is_cuda_available, is_npu_available
verl.utils.dataset.vision_utils.process_image, process_video
verl.models.transformers.qwen2_vl.get_rope_index
verl.single_controller.ray.megatron.NVMegatronRayWorkerGroup
```

## What success looks like

1. A `plans-n-solutions/stages/stage0_1.md` file with the full plan and (after execution) solution
2. A 20-step wandb run with decreasing loss on the new verl
3. Updated `docs/SETUP.md` reflecting the new versions
4. Changes pushed to `prorl-verl` branch on both repos
5. The foundation is modern enough that Stages 1-5 decoupling work won't need another verl upgrade
