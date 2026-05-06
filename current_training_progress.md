# Training Progress — 2026-05-05 (fabric-rl-cleanup branch)

## What Was Fixed This Session

### 1. RolloutManager — parallel episode dispatch
**File:** `core/rollout_fabric/rollout_manager/loop.py`
- Episodes within a group were sequential. With 32 ProRL workers available, switched to `threading.Thread` per episode so all 4 run in parallel (group latency: 10 min → group latency ≈ single episode).
- `_dl_lock` protects `next(self._dataloader)` across concurrent group workers.
- Added `--num-parallel-groups` (default 1, set to 8 in production) so 8 groups × 4 episodes = 32 concurrent `/process` calls, fully saturating ProRL's worker pool.

### 2. DAPO trainer — sample_mini_batch method missing
**File:** `trainers/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py`
- `_acquire_training_batch_dapo` called `self.trajectory_store.sample_mini_batch(...)` — method does not exist on `LiveStoreClient`. Fixed to call the standalone `sample_mini_batch(client, n_groups, step)` from `fabric_adapter/live_store_batch.py`.

### 3. VERL v0.8 compatibility — `omega_conf_to_dataclass` API change
**File:** `trainers/verl/verl_custom/workers/fsdp_workers.py`

VERL v0.8 requires `_target_:` in every YAML config section for `omega_conf_to_dataclass`. Our YAML predates this. Fixes:

| Issue | Fix |
|---|---|
| `verl.workers.fsdp_workers` module removed | Fall back to `engine_workers` |
| `actor_config.engine` missing | `_inject_engine_fields()` synthesises `FSDPEngineConfig` from `fsdp_config` |
| `HFModelConfig.hf_config` (Qwen3Config) rejected by OmegaConf AnyNode | `_ConfigProxy`: stores Python objects directly, bypassing OmegaConf wrapping |
| `actor_config.model_config = HFModelConfig_instance` would wrap & reject Qwen3Config | `_ConfigProxy.__setattr__` uses `object.__setattr__` |
| `**config.global_batch_info` called `keys()` → `None()` | Added `keys()`/`values()`/`items()`/`__iter__` to `_ConfigProxy` |
| New VERL fields absent from YAML (`data_parallel_size`, `pipeline_model_parallel_size`, `expert_parallel_size`, `checkpoint_engine.backend`, `optim.clip_grad`) | Added to `ppo_trainer.yaml` with correct defaults |
| FSDPEngineWithLMHead only accepts nested (no-padding) tensors | `compute_log_prob` and `update_actor` convert padded→nested before calling engine, nested→padded on output |
| `compute_log_prob` called non-existent `tokenizer` | Removed; `infer_batch` uses `attention_mask` directly |
| `compute_log_prob` called PPO loss (reads `global_batch_size`) | `compute_loss=False` for log-prob pass |
| `update_actor` needed `temperature`, `global_batch_size`, `mini_batch_size`, etc. | Injected via `assign_non_tensor` in `update_actor` override |
| `save_checkpoint` didn't save LoRA adapter | New override: calls `get_per_tensor_param()`, deduplicates shared tensors (tied embeddings), JSON-serialises sets in peft_config |
| LoRA adapter path was `actor/actor/lora_adapter` instead of `actor/lora_adapter` | Fixed: `local_path` already ends with `actor`, removed duplicate |
| `train_dataloader.state_dict()` crashed in LiveStore mode (no DataLoader) | Guarded with `if self.train_dataloader is not None` |
| `balance_batch=True` crashed when batch size not divisible by world_size (partial groups) | Set `balance_batch: False` — not needed for LiveStore batch sizes |

### 4. LoRA adapter config serialisation
**File:** `trainers/verl/verl_custom/workers/fsdp_workers.py`
- `peft_config.to_dict()` contains Python `set` objects (target_modules). Added `default=_json_safe` converter to JSON dump.
- Tied embedding tensors (`lm_head.weight ≡ model.embed_tokens.weight`) share memory — deduplicated by `data_ptr()` before `save_file`.

---

## Current Status

| Component | Status |
|---|---|
| EnvironmentProvider (:8006) | ✓ Running |
| LiveStore (UDS) | ✓ Running |
| PolicyRegistry (UDS) | ✓ Running |
| RolloutManager (8 parallel groups) | ✓ Running — ~3000 episodes completed |
| LiveStore buffer | ~200 groups buffered |
| VERL Trainer (Docker, 8×A100) | ⟳ Restarting (Ray keepalive timeout) |
| W&B run | https://wandb.ai/shamanework-pl/ProAgent/runs/jwijscn8 |
| Checkpoint | global_step_1 saved; global_step_2+ in progress |

## Remaining Watch Point
The trainer resumes from `global_step_1` on each restart. Once the LoRA adapter path fix lands in the running container, step 2 will complete `_publish_lora_adapter` → PolicyRegistry publish → vLLM reload → policy_version bumps to 1 → RolloutManager starts using the updated model.
