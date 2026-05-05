# How to Swap Any Service

Each service sits behind a typed `Protocol` in this directory. Swapping a service
means implementing its Protocol contract — transport is an implementation detail.

See `../../docs/service-envs.md` for the dependency footprint per service.

## Protocol contracts

### EnvironmentProvider

**Implement:** `POST /process` HTTP endpoint  
**BC rules:** BC-1 (token IDs as int), BC-15 (must NOT call LiveStore/PolicyRegistry)  
**Dep footprint:** any language/runtime that can serve HTTP  
**Reference implementation:** `openhands/nvidia/` (ProRL SWE-Bench)  
**Guide:** `../../environment_providers/README.md`

Wire contract (from `rollout_manager/prorl_client.py`):
```
POST /process
  Request:  { "instance": { <task>, "policy_version": N }, "sampling_params": {...} }
  Response: { "messages": [{role, content, token_ids:[int], logprobs:[float]}],
              "resolved": bool, "reward": float, "raw_reward": float }
```

### TrainerAdapter

**Implement:** consume `LiveStoreClient.get_batch()` + call `PolicyRegistryClient.publish_policy_version()`  
**BC rules:** BC-14 (no parquet path), BC-15 (no ProRL URL, no vLLM URL), BC-11 (pad locally after `get_batch`)  
**Dep footprint:** your Docker image + `pip install -e trainer_integration/{trainer}/`  
**Reference implementation:** `trainer_integration/verl/` (VERL FSDP)  
**Guide:** `../../trainer_adapters/README.md`

```python
# Minimal integration seam:
batch = live_store_client.get_batch(n_groups=4, current_step=step)
tensors = pack_unpadded_groups(batch)       # from trainer_adapters/{trainer}/pad.py
loss = trainer.step(tensors)
registry_client.publish_policy_version(version=step, adapter_uri=checkpoint_uri)
```

### InferenceBackend

**Implement:** `POST /vN/generate` + `POST /reload_lora`  
**BC rules:** BC-1 (accept/return token IDs only), BC-10 (version pinning via /vN/)  
**Dep footprint:** any language/runtime; must NOT decode token IDs to text  
**Reference implementation:** `scripts/inference/_vllm_child.py` (vLLM, frozen)

### LiveStore

**Implement:** `push_group()` and `get_batch()` with pop-on-sample semantics  
**BC rules:** BC-2 (atomic push), BC-3 (pop-on-sample), BC-4 (blocking predicate on fresh groups), BC-5 (no-progress timeout)  
**Reference implementation:** `live_store/` (gRPC UDS)

### PolicyRegistry

**Implement:** `publish_policy_version()` + fanout + manifest write  
**BC rules:** BC-9 (hard abort if `endpoints_failed > 0`)  
**Reference implementation:** `policy_registry/` (gRPC UDS + SQLite)

### RolloutManager

**Implement:** owns parquet dataloader, dispatches episodes, pushes groups to LiveStore  
**BC rules:** BC-0 (one snapshot per group), BC-13 (zero VERL/OpenHands imports), BC-14 (owns dataloader)  
**Reference implementation:** `rollout_manager/`

### ReplayArchive

**Implement:** `append_episode()` + `query()`  
**BC rules:** BC-12 (archive tee is pre-filter; LiveStore push is post-filter)  
**Reference implementation:** `replay_archive/`

## Import boundary rules

| Service | May import | MUST NOT import |
|---|---|---|
| RolloutManager | httpx, grpcio, pyarrow | openhands, VERL, torch |
| TrainerAdapter | torch, VERL/slime/ROLL, grpcio | openhands, parquet loaders |
| EnvironmentProvider | openhands, litellm, fastapi | live_store, policy_registry |
| InferenceBackend | vllm, fastapi | openhands, torch training code |
