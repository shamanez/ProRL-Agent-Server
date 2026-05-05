# Environment Providers

This directory is the navigation index for EnvironmentProvider implementations. Each
subdirectory documents one concrete implementation of the `EnvironmentProvider` Protocol
defined in `schemas/protocols/environment_provider.py`.

## Current implementations

| Dir | Description | Source |
|-----|-------------|--------|
| `prorl_openhands/` | SWE-Bench via ProRL + OpenHands + Singularity | `../../openhands/` |

## Planned / stub implementations

| Dir | Description | Upstream |
|-----|-------------|----------|
| `rock/` | ROLL-compatible environment manager | https://github.com/alibaba/ROCK |
| `openreward/` | OpenReward environment | https://openreward.ai |

## How to add a new EnvironmentProvider

**Step 1.** Implement a server that accepts `POST /process`.

The current compatibility contract (from `rollout_manager/prorl_client.py`):
```
Request body:
  { "instance": { <task fields>, "policy_version": N },
    "sampling_params": { "temperature": ..., "max_tokens": ... } }

Response body:
  { "messages": [
      { "role": "user"|"assistant"|"tool",
        "content": "...",
        "token_ids": [int, ...],         # BC-1: MUST be int, never str
        "logprobs": [float, ...] }       # per-token log probabilities
    ],
    "resolved": bool,
    "reward": float,
    "raw_reward": float }
```

Token IDs **MUST** be `int` (BC-1). Never return decoded text strings — BC-1 violations
cause KL/entropy NaN within 2 training steps.

The EnvironmentProvider **MUST NOT** call LiveStore or PolicyRegistry.

**Step 2.** Create `environment_providers/{your_env}/README.md` documenting:
- Which Python env or Docker image your server needs
- How to start it (`scripts/adapters/start_env_{your_env}.sh`)
- Any BC rules specific to your environment

**Step 3.** Create `scripts/adapters/start_env_{your_env}.sh` that starts your server.

**Step 4.** Update `scripts/services/start_env_provider.sh` to call your script.

**Step 5.** Pass `--prorl-url http://your-host:PORT` to `rollout_manager.main`.

No fabric service changes needed (LiveStore, PolicyRegistry, RolloutManager are
unaware of which EnvironmentProvider is behind `--prorl-url`).
