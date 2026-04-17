# Stage 0.1 — Upgrade verl v0.4 → v0.8.0.dev + vLLM 0.8.5 → 0.18

**Status:** Upgrade code-complete. 26 compatibility issues resolved. Validation run in progress (see Validation section at bottom). From this point forward, **all subsequent stages run against the upgraded stack** — do not fall back to v0.4 or vLLM 0.8.5.

---

## What this stage does

Upgrades the RL trainer and rollout engine so the decoupling work in Stages 1-5 builds on modern verl APIs.

| Component | Old | New |
|---|---|---|
| verl | v0.4 (commit `60138ebd`) | v0.8.0.dev (`shamanez/verl` main, commit `910ba344`) |
| vLLM | 0.8.5 | 0.18 |
| PyTorch | 2.4 (image) | 2.6+ (image) |
| Docker image | `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2` | `verlai/verl:vllm018.dev1` |
| PyTorch alloc | default | `expandable_segments:True` (fragmentation control on 40 GB) |
| `gpu_memory_utilization` override | `0.6` | `0.45` (leaves more VRAM for backward activations on A100-40GB) |

The trainer + vLLM run **inside Docker**; ProRL runs **on the host**; the trainer container joins host networking to reach `localhost:8006`.

---

## How to run Stage 0.1 validation (self-contained)

All commands run from the project root `/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server` (referred to below as `$REPO`).

### Prerequisites

On the current box, everything below is already in place — verify with the gate commands in the right column before doing anything else. If a gate fails, run the bootstrap in the left column. **Total cold-start cost: ~3–5 hours (SIF builds dominate).**

| Prereq | Bootstrap (cold-start only) | Gate (already-in-place check) |
|---|---|---|
| Python env + Poetry deps | `make build` (runs `poetry install --with dev,test,runtime,evaluation` + installs pre-commit hooks) | `poetry env info -p` prints a venv path |
| SWE-Gym / R2E-Gym git deps (not on PyPI) | `pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git git+https://github.com/R2E-Gym/R2E-Gym.git` | `python -c "import swebench, r2e_gym"` exits 0 |
| Credentials file | Create `/home/ubuntu/.prorl_creds.env` with 6 exports: `WANDB_API_KEY`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `SINGULARITY_DOCKER_USERNAME`, `SINGULARITY_DOCKER_PASSWORD`, `OH_RUNTIME_SINGULARITY_IMAGE_REPO=$REPO/singularity_images` | `source ~/.prorl_creds.env && echo $WANDB_API_KEY \| head -c 5` prints something |
| Docker image | `docker pull verlai/verl:vllm018.dev1` | `docker image inspect verlai/verl:vllm018.dev1` exits 0 |
| Apptainer / Singularity binary (host) | Install Apptainer 1.3+ per vendor instructions | `apptainer --version` exits 0 |
| verl v0.8.0.dev source at `/tmp/verl` | `git clone https://github.com/shamanez/verl.git /tmp/verl && cd /tmp/verl && git checkout 910ba344` | `cd /tmp/verl && git log --oneline -1` shows `910ba344` |
| Qwen3-4B-Instruct-2507 weights | `huggingface-cli download Qwen/Qwen3-4B-Instruct-2507` (7.6 GiB) | `ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/config.json` finds one |
| Training parquets (`SkyRL-v0-293`) | Stage the 4 parquets manually to `/home/ubuntu/data/SkyRL-v0-293/`: `train.parquet`, `train.filtered.parquet`, `validation.parquet`, `validation.filtered.parquet`. The `.filtered` variants are a subset pre-filtered for SWE-Bench-resolvable instances (source: internal pipeline — if missing, ask the maintainer for the artefact) | `ls /home/ubuntu/data/SkyRL-v0-293/train.filtered.parquet` exits 0 |
| 49 Singularity `.sif` images | `SINGULARITY_IMAGES_DIR=$REPO/singularity_images python scripts/pull_swe_images.py --parquet /home/ubuntu/data/SkyRL-v0-293/train.filtered.parquet` (takes hours — converts each Docker image referenced in the parquet to `.sif`, idempotent) | `ls $REPO/singularity_images/*.sif \| wc -l` ≥ 49 |
| Pre-commit hooks | `make build` (installs them) | `poetry run pre-commit --version` exits 0 |

### Step 1 — Clean any stale state

```bash
docker rm -f s0-baseline 2>/dev/null || true
pkill -9 -f 'start_server.py'     2>/dev/null || true
pkill -9 -f 'action_execution'    2>/dev/null || true
pkill -9 -f 'singularity run'     2>/dev/null || true
pkill -9 -f 'ray::'               2>/dev/null || true
ray stop --force                  2>/dev/null || true
sleep 3

# Sanity gates
ss -tlnp 2>/dev/null | grep ':8006 ' && { echo "FATAL: :8006 still busy"; exit 1; } || echo "port 8006 free"
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # expect ~0 MiB across all 8 GPUs
```

### Step 2 — Start ProRL on the host

```bash
source /home/ubuntu/.prorl_creds.env
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/singularity_images

nohup poetry run python scripts/start_server.py \
  --host 0.0.0.0 --port 8006 \
  --max-init-workers 64 --max-run-workers 64 \
  --timeout 1000 \
  > /tmp/s0-prorl.log 2>&1 &
echo $! > /tmp/s0-prorl.pid

# Wait until the server listens
until ss -tlnp 2>/dev/null | grep -q ':8006 '; do sleep 1; done
echo "ProRL listening on :8006"
```

### Step 3 — Launch the trainer Docker container

```bash
bash scripts/_internal/s0_baseline_docker.sh 2>&1 | tee /tmp/s0-baseline.log
```

`s0_baseline_docker.sh` handles the full launch chain:
- Mounts `/tmp/verl` (verl v0.8 upstream) and the repo at `/workspace`
- Mounts `/home/ubuntu/data` (read-only) and the HF cache
- Exports `PYTORCH_ALLOC_CONF=expandable_segments:True` inside the container
- Installs `verl` and `verl_custom` in editable mode (`--no-deps`)
- Runs the smoke import test, then `run_proagent_qwn3_4B_instruct.sh`
- Overrides the 40 GB knobs: `gpu_memory_utilization=0.45`, `ulysses_sequence_parallel_size=2`, `max_prompt_length=16384`, `save_freq=10`, `total_training_steps=20`, `+actor_rollout_ref.actor.calculate_entropy=false`

The run writes checkpoints to `/workspace/outputs/ProAgent/...` inside the container (bind-mounted to the repo). Rollout data goes to `/workspace/outputs/rollout_data`.

### Step 4 — Monitor

```bash
# Step progress
grep -oE 'training/global_step:[0-9]+'       /tmp/s0-baseline.log | tail
grep -oE 'Training Progress:[^|]*\|[ 0-9/]+' /tmp/s0-baseline.log | tail -5

# Errors (non-empty output = problem)
grep -cE 'OutOfMemoryError|CUDA out of memory|AttributeError|NameError|ImportError|Missing key|ConfigKey|AssertionError|Traceback' /tmp/s0-baseline.log

# ProRL pipeline state
curl -s http://localhost:8006/status | python3 -m json.tool
```

Expected cadence: ~350–400 s per step on A100-40GB × 8 (8-way FSDP, ulysses SP=2). 20 steps ≈ 2 hours.

### Step 5 — Validate gating criteria (in WandB)

Pull the run URL printed near the top of `/tmp/s0-baseline.log` (format: `https://wandb.ai/shamanework-pl/ProAgent/runs/<id>`). Confirm all five:

| # | Metric | Pass criterion |
|---|---|---|
| 1 | `training/global_step` | ≥ 20 |
| 2 | `actor/grad_norm` | finite, > 0, < 1e6 every step |
| 3 | `critic/rewards/mean` | not identically zero across 20 steps |
| 4 | advantage variance (`critic/advantages/max` - `critic/advantages/min`) | > 0 |
| 5 | `actor/kl` (or `actor/kl_loss`) | finite every step |

### Step 6 — Commit

After gates pass, run `make lint` from the repo root. Then a single commit on `de-coupled`:

```bash
git add -A
git commit -m "Stage 0.1: upgrade verl v0.4→v0.8.0.dev + vLLM 0.8.5→0.18

- All 26 compatibility issues resolved (see plans-n-solutions/stages/stage0_1.md)
- Custom AsyncActorRolloutRefWorker in verl_custom/workers/fsdp_workers.py
  overrides compute_log_prob to force calculate_entropy=False (OOM fix)
- YAML schema: added policy_loss, global_batch_info, loss_scale_factor
- Docker env: PYTORCH_ALLOC_CONF=expandable_segments:True
- 40 GB knobs: gpu_memory_utilization=0.45 (down from 0.6)

WandB: <url>"
```

No `--no-verify`. Do not push without explicit user approval.

---

## Problem log (26 issues, grouped by phase)

### Phase 1 — verl v0.8 config and import changes (P1–P8)

| # | Symptom | Fix |
|---|---|---|
| 1 | `ConfigAttributeError: Missing key 'mode'` | Removed `_target_` from rollout section; added `mode`, `multi_turn`, `engine_kwargs`, etc. to `ppo_trainer.yaml` |
| 2 | `ConfigAttributeError: Missing key 'optimizer'` | Added `optimizer: adamw`, `optimizer_impl: torch`, etc. |
| 3 | `omega_conf_to_dataclass` rejects custom fields (`policy_loss_type`, `tis_imp_ratio_cap`) | `init_model` monkey-patches `omega_conf_to_dataclass` to return raw OmegaConf when `_target_` absent; `OmegaConf.set_struct(self.config, False)` |
| 4 | `ImportError: verl.utils.debug.performance` | Moved to `verl.utils.profiler.performance`; updated imports in `ray_trainer*.py` |
| 5 | `ImportError: verl.utils.vllm_utils` | Moved to `verl.utils.vllm`; updated import in `main_ppo.py` |
| 6 | `AsyncActorRolloutRefWorker` missing `execute_method`, `chat_completion`, `wake_up`, `sleep` | Custom subclass re-adds the 4 `@register` methods |
| 7 | `_build_rollout` upstream uses `ServerAdapter` (no `execute_method`) | Overrode `_build_rollout` to install `VLLMAsyncRolloutCompat` directly |
| 8 | `NVMegatronRayWorkerGroup` import fails | Entire `verl.single_controller.ray.megatron` removed in v0.8 — guarded with try/except in `main_ppo.py` |

### Phase 2 — vLLM 0.18 runtime API changes (P9–P17)

| # | Symptom | Fix |
|---|---|---|
| 9 | `Failed to look up actor 'register_center'` | `WorkerGroupRegisterCenter` removed — replaced with direct `__ray_call__` queries |
| 10 | `ModuleNotFoundError: vllm.entrypoints.openai.protocol` | Updated 5 import paths in `vllm_async_server.py` |
| 11 | `AsyncEngineArgs got unexpected 'disable_mm_preprocessor_cache'` | Removed the kwarg |
| 12 | `WorkerWrapperBase got unexpected 'vllm_config'` | Constructor changed to `(rpc_rank=0)` |
| 13 | `AssertionError: local_world_size > visible devices` | vLLM 0.18 asserts unless backend is `"ray"` or `"external_launcher"` — temporarily patch backend to `"ray"` during `init_device` |
| 14 | `WorkerWrapperBase has no attribute 'execute_method'` | `getattr(self.inference_engine, method)()` + cloudpickle |
| 15 | `CachedQwen2Tokenizer has no attribute 'tokenizer'` | vLLM 0.18 returns tokenizer directly — added `hasattr` guard |
| 16 | `OutputProcessor.abort_requests() missing 'internal'` | Added `internal=False` |
| 17 | `'generator' object does not support context manager protocol` | `_timer` lost `@contextmanager`; switched to `simple_timer as _timer` |

### Phase 3 — Operational bugs exposed after runtime came up (P18–P26)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 18 | 0/20 steps after 35 min; empty rollouts | Two ProRL servers bound to :8006 via `SO_REUSEADDR`; stale process from prior session returned 503 | `pkill` all stale `start_server.py` + `action_execution_server` before launch (Step 1 above) |
| 19 | Every `/generate` raised `TypeError: Unexpected keyword 'stream'` | vLLM 0.18 `SamplingParams(**kwargs)` rejects unknown args; client sent `stream`, `tools`, `tool_choice` | Filter kwargs in `vllm_async_server.py:generate()` to `inspect.signature(SamplingParams).parameters` |
| 20 | SIF `action_execution_server` hung in bash init; SIGTERM at T+3:42; 488/493 SIFs failed | Singularity `--pid` + `EfficientBashSession` PTY loses output across PID namespaces | `run_as_fakeroot=True` on launch path (see `openhands/nvidia/swe_agent/utils.py`) |
| 21 | `collective_rpc` got unexpected kwarg `non_block` | vLLM 0.18 added `non_block` param; our dispatcher forwarded it to actors that don't accept it | Added `non_block=False` handling in `ExternalRayDistributedExecutor.collective_rpc` — wraps refs in `FutureWrapper` when `non_block=True` |
| 22 | OOM at `entropy_from_logits` during `compute_log_prob` (10 GiB alloc) | Upstream `fsdp_workers.py:1145` hardcodes `calculate_entropy = not is_lora`; our config override ignored | Custom `compute_log_prob` override in `verl_custom/workers/fsdp_workers.py` forces `calculate_entropy=False` and injects `tensors['entropys'] = torch.zeros_like(outputs['log_probs'])` for `ray_trainer.py:1499` consumer. Uses closure pattern to survive autoflake stripping imports |
| 23 | `ConfigAttributeError: Missing key 'policy_loss'` at `dp_actor.py:609` | verl v0.8 expects `actor.policy_loss` dict, not legacy `policy_loss_type: ppo` string | Added YAML block: `policy_loss: {loss_mode: vanilla, clip_cov_ratio: 0.0002, clip_cov_lb: 1.0, clip_cov_ub: 5.0, kl_cov_ratio: 0.0002}` |
| 24 | `ConfigAttributeError: Missing key 'global_batch_info'` at `core_algos.py:1361` | verl v0.8 unpacks `**config.global_batch_info` in the ppo loss path | Added YAML: `global_batch_info: {}`, `loss_scale_factor: null` — empty dict preserves old token-mean agg behavior |
| 25 | CUDA OOM at `loss.backward()` on step 5 (8.47 GiB alloc, 4.78 GiB reserved-but-unallocated) | Activation memory growth as response length climbs across steps; 40 GB is tight | `PYTORCH_ALLOC_CONF=expandable_segments:True` in Docker env + lowered `gpu_memory_utilization` from 0.6 → 0.45 |
| 26 | `AttributeError: BaseCheckpointManager has no attribute 'local_mkdir'` at step-10 checkpoint save | verl v0.8 renamed to `local_mkdir_safe` in `verl.utils.fs` | Inline import at call site in `ray_trainer.py:_save_checkpoint` (inline to survive autoflake) |

---

## Files modified

| File | Change |
|---|---|
| `trainer_integration/verl/verl_custom/workers/fsdp_workers.py` | **NEW** — `VLLMAsyncRolloutCompat` + `AsyncActorRolloutRefWorker` with restored dispatch methods, `init_model` monkey-patch, `_build_rollout` override, `compute_log_prob` override (P22) |
| `trainer_integration/verl/verl_custom/trainer/main_ppo.py` | Import fixes: `vllm_utils`→`vllm`, megatron guard, `AsyncActorRolloutRefWorker` from `verl_custom` |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | `simple_timer as _timer`; inline `local_mkdir_safe` import (P26) |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py` | `simple_timer as _timer` |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | `register_center` → direct `__ray_call__`; tokenizer `hasattr` guard |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` | Same as `async_server.py` |
| `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py` | vLLM 0.18 imports; kwargs filter (P19); `abort_requests(internal=False)`; `collective_rpc(non_block=...)` (P21) |
| `trainer_integration/verl/verl_custom/nvidia/eval/gen_utils.py` | vLLM 0.18 API fixes |
| `trainer_integration/verl/verl_custom/trainer/config/ppo_trainer.yaml` | Removed `_target_`; added v0.8 dataclass fields; `policy_loss` block (P23); `global_batch_info` + `loss_scale_factor` (P24); `entropy_checkpointing: false` |
| `openhands/nvidia/swe_agent/utils.py` | `run_as_fakeroot=True` (P20) |
| `scripts/_internal/s0_baseline_docker.sh` | New Docker image; verl v0.8 mount; `PYTORCH_ALLOC_CONF=expandable_segments:True`; `gpu_memory_utilization=0.45` |
| `tests/nvidia/test_verl_import_compat.py` | **NEW** — smoke test for critical verl v0.8 imports |

---

## Key design decisions

1. **Custom worker subclass over fork-patching.** `verl_custom/workers/fsdp_workers.py` subclasses upstream. Keeps `/tmp/verl` pristine.
2. **Closure / inline-import pattern.** Autoflake (in pre-commit) strips module-level imports it thinks are unused, even when they're used inside decorators or late-bound methods. Worked around by wrapping `compute_log_prob` in a `_build_compute_log_prob` closure with imports inside, and by using `from verl.utils.fs import local_mkdir_safe  # noqa: PLC0415` at the call site for `local_mkdir_safe`.
3. **`VLLMAsyncRolloutCompat` thin wrapper.** Minimal stand-in for the old `vLLMAsyncRollout`; wraps vLLM's `WorkerWrapperBase` directly. Real engine created lazily when `ExternalRayDistributedExecutor` calls `init_worker`.
4. **`distributed_executor_backend='ray'` trick.** vLLM 0.18 asserts `local_world_size <= visible_device_count` unless backend is `"ray"` or `"external_launcher"`. Our `ExternalRayDistributedExecutor` IS Ray-based but is a class, not string `"ray"`. Temporarily patching to `"ray"` during `init_device` is semantically correct.
5. **`PYTORCH_ALLOC_CONF=expandable_segments:True`.** The single most effective memory knob on 40 GB cards with growing sequence lengths. PyTorch's own OOM hint recommends it. Applied at Docker-env level so every child process inherits it.

---

## Architecture change (v0.4 → v0.8)

- **v0.4:** Workers are hybrid actors holding both FSDP training modules and vLLM inference engines. `AsyncActorRolloutRefWorker` has dispatch methods for both. vLLM engine lifecycle tied to training loop.
- **v0.8:** Workers split into training actors and `RolloutReplica` inference actors. `AsyncActorRolloutRefWorker` only has training methods + `update_weights`. Inference handled by separate `vLLMHttpServer` actors via HTTP. The old dispatch pattern still exists in the codebase but no upstream worker uses it.

`verl_custom` relies on the v0.4 hybrid-engine pattern: colocated training + inference workers with `ExternalRayDistributedExecutor` doing Ray RPC dispatch. The compatibility layer in `verl_custom/workers/fsdp_workers.py` bridges v0.4's dispatch pattern with v0.8's updated internals.

---

## Validation

| Gate | Status |
|---|---|
| Import smoke test (`tests/nvidia/test_verl_import_compat.py`) | Pass |
| vLLM engine init | Pass (run #13) |
| FSDP model init | Pass (run #13) |
| `global_step ≥ 20` | **In progress** — run #13 |
| `actor/grad_norm` finite every step | Confirmed for steps 1–10 (run #12) |
| `critic/rewards/mean` non-zero | Confirmed (run #12: 0.375 → 0.5625 across steps 1–4) |
| Advantage variance > 0 | Confirmed (run #12) |
| `actor/kl` finite | Confirmed (run #12) |

Run history (most recent first):

| Run | WandB | Outcome |
|---|---|---|
| #13 | (in progress) | Run with P25 + P26 fixes |
| #12 | a1ghlj8p | Reached step 10, crashed at checkpoint save (P26) |
| #11 | a1ghlj8p | Reached step 4, OOM at backward (P25) |
| #10 | — | P24 — `global_batch_info` missing |
| #9 | — | P23 — `policy_loss` missing |
| #8 | — | P22 — entropy OOM |

---

## Reference

- Runbook: this file + `scripts/_internal/s0_baseline_docker.sh` + `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh`
- Stage 0 baseline (old stack): `plans-n-solutions/stages/stage0.md`
- Next stage: `plans-n-solutions/stages/stage1.md`
- Continue command: `.claude/commands/continue-decoupling.md`
