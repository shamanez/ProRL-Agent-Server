# OpenReward EnvironmentProvider (stub)

**Upstream:** https://openreward.ai

This is a stub for a future OpenReward-based EnvironmentProvider implementation.

## Integration steps

1. OpenReward must serve `POST /process` satisfying the contract in `../README.md`.
2. Create `scripts/adapters/start_env_openreward.sh` to launch the server.
3. Set `--prorl-url http://openreward-host:PORT` on `rollout_manager.main`.

See `schemas/protocols/PLUGGING_IN.md` for the full Protocol contract.
