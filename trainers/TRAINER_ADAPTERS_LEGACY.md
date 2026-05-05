# Trainer Adapters

Code that runs **inside** each trainer's Docker container. Translates `TrainingSample`
records from LiveStore into the trainer's native tensor format.

## Convention

Each subdirectory corresponds to one trainer and exports:

```python
def pack_unpadded_groups(groups: list[TrainingGroup]) -> TrainerBatch:
    ...
```

The trainer calls this immediately after `LiveStoreClient.get_batch()`. The LiveStore
wire is **unpadded** (BC-11); the trainer pads locally.

## Current implementations

| Dir | Trainer | Docker image |
|-----|---------|-------------|
| `verl/` | VERL FSDP (DAPO/GRPO) | `verlai/verl:vllm018.dev1` |

## Stub documentation (not yet implemented)

| Dir | Trainer | Upstream |
|-----|---------|----------|
| `slime/` | slime | https://github.com/THUDM/slime |
| `roll/` | ROLL | https://github.com/alibaba/ROLL |

## How to add a new trainer

```
Step 1. Create trainer_adapters/{trainer}/README.md
        Document the upstream repo and the tensor format the trainer expects.

Step 2. Create trainer_adapters/{trainer}/pad.py
        Implement pack_unpadded_groups(groups) -> NativeBatch.
        Reference: trainer_adapters/verl/pad.py

Step 3. Create trainer_integration/{trainer}/ with pyproject.toml
        This pip-installable package is installed INSIDE the trainer's Docker:
          pip install -e /workspace/trainer_integration/{trainer}
        It must:
          - Call LiveStoreClient.get_batch() + your pack_unpadded_groups()
          - Call PolicyRegistryClient.publish_policy_version() after each checkpoint
          - NEVER receive: parquet path, ProRL URL, vLLM URL (BC-14/15)

Step 4. Create scripts/adapters/start_trainer_{trainer}.sh
        docker run {trainer_image} \
          -v $(pwd):/workspace \
          -e LIVE_STORE_SOCKET=/tmp/prorl_live_store.sock \
          -e REGISTRY_SOCKET=/tmp/prorl_policy_registry.sock \
          bash -c "pip install -e /workspace/trainer_integration/{trainer} && \
                   python -m {trainer}.train ..."

Step 5. Update scripts/services/start_all.sh to call your launcher.
No changes to fabric services needed.
```

Each `trainer_adapters/{trainer}/` that becomes a `pip install -e` target inside Docker
needs its own `pyproject.toml`. Stub dirs are documentation-only until a real integration
is implemented.
