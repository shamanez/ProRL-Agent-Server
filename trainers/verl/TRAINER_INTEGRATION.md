# trainer_integration/verl/ — VERL TrainerAdapter Integration

## Role

This directory is a `pip`-installable patch package (`verl_custom`) installed
**inside** the `verlai/verl` Docker container:

```bash
pip install --no-deps -e /workspace/trainer_integration/verl
```

It does NOT run on the host machine. The host poetry env never sees it.

## What it adds to VERL

Two integration seams:

1. **`_acquire_training_batch_dapo()`** in `verl_custom/trainer/ppo/ray_trainer_dapo.py`  
   Replaces VERL's internal rollout with `LiveStoreClient.get_batch()` + `pack_unpadded_groups()`.

2. **`_publish_lora_adapter()`** in the same trainer  
   Replaced with `PolicyRegistryClient.publish_policy_version(version, adapter_uri)`.

## What it does NOT own

- Training data (no parquet path) — BC-14
- ProRL URL — BC-15
- vLLM addresses — BC-15
- Reward computation for non-SWE-Bench tasks (naive/prime reward managers disabled)

## Active reward manager

Only `verl_custom/nvidia/reward_manager/swebench.py` is active in production.
The `naive.py` and `prime.py` reward managers are disabled; restore from git if needed.

## BC-14 note: data.train_files

`scripts/adapters/start_trainer_verl.sh` passes `data.train_files` to satisfy VERL's
Hydra config schema requirement. The trainer does **NOT** use this for rollout generation.
Rollout data comes exclusively from `LiveStoreClient.get_batch()`.

TODO: replace with a `LiveStoreOnlyDataset` dummy config to remove the parquet mount entirely.

## How to add slime or ROLL

1. Create `trainer_integration/slime/` with its own `pyproject.toml` + integration code
2. Create `trainer_adapters/slime/pad.py` implementing `pack_unpadded_groups()`
3. Create `scripts/adapters/start_trainer_slime.sh` using the slime Docker image

See `../../trainer_adapters/README.md` for the step-by-step onboarding guide.
