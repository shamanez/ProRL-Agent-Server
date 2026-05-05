# scripts/adapters/ — Adapter Launchers

Each script in this directory launches one concrete implementation of a pluggable
Protocol slot. These are NOT internal implementation details — they are the canonical
launchers for swappable components.

## Environment scripts

| Script | Protocol slot | Python env | Notes |
|--------|--------------|------------|-------|
| `start_env_prorl.sh` | EnvironmentProvider | `PRORL_OPENHANDS_PYTHON` (full poetry env — needs openhands + Singularity) | Runs on host |

## Trainer scripts

| Script | Protocol slot | Python env | Docker image |
|--------|--------------|------------|-------------|
| `start_trainer_verl.sh` | TrainerAdapter | **Docker container — NOT the host env** | `verlai/verl:vllm018.dev1` |
| `start_trainer_slime.sh` | TrainerAdapter | **Docker container** | `{slime image}` (stub) |
| `start_trainer_roll.sh` | TrainerAdapter | **Docker container** | `{ROLL image}` (stub) |

## Naming convention

`start_{component_type}_{implementation_name}.sh`

- `component_type`: `env` (EnvironmentProvider) or `trainer` (TrainerAdapter)
- `implementation_name`: `prorl`, `verl`, `slime`, `roll`

## Adding a new adapter

1. Create `start_env_{name}.sh` or `start_trainer_{name}.sh` here
2. Add a row to this README
3. Update `scripts/services/start_env_provider.sh` or `start_all.sh` to call it
4. Add a subdirectory under `environment_providers/{name}/` or `trainer_adapters/{name}/`
