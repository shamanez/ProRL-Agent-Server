# ROCK EnvironmentProvider (stub)

**Upstream:** https://github.com/alibaba/ROCK

This is a stub for a future ROCK-based EnvironmentProvider implementation.

## Integration steps

1. ROCK must serve `POST /process` satisfying the contract in `../README.md`.
2. Create `scripts/adapters/start_env_rock.sh` to launch the ROCK server.
3. Set `--prorl-url http://rock-host:PORT` on `rollout_manager.main`.
4. Add your Python env or Docker image requirements here.

See `schemas/protocols/PLUGGING_IN.md` for the full Protocol contract.
