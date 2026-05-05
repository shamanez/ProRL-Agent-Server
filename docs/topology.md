# Deployment Topology

## Current topology (single trainer host)

```
Trainer host (8×A100)
  ├── EnvironmentProvider :8006   [PRORL_OPENHANDS_PYTHON — openhands + Singularity]
  ├── LiveStore UDS               [ROLLOUT_FABRIC_PYTHON — fabric-core env]
  │     /tmp/prorl_live_store.sock
  ├── PolicyRegistry UDS          [ROLLOUT_FABRIC_PYTHON — fabric-core env]
  │     /tmp/prorl_policy_registry.sock
  ├── ReplayArchive (optional)    [ROLLOUT_FABRIC_PYTHON — fabric-core env]
  ├── RolloutManager (no port)    [ROLLOUT_FABRIC_PYTHON — fabric-core env]
  └── TrainerAdapter              [verlai/verl Docker image — NOT the host env]
        Docker --network host

vLLM host (EC2)
  └── vLLM :8100-8103             [scripts/inference/requirements-remote.txt]
```

See `docs/service-envs.md` for the per-service dependency footprint.

## Co-location requirements

| Service pair | Co-location required? | If separated |
|---|---|---|
| LiveStore + TrainerAdapter | Yes (UDS socket in `/tmp/`) | Change socket path to `grpc://host:port` in server.py |
| PolicyRegistry + TrainerAdapter | Yes (UDS socket in `/tmp/`) | Same upgrade path |
| EnvironmentProvider + RolloutManager | No | Set `--prorl-url http://remote-host:8006` |
| RolloutManager + LiveStore | No | LiveStore gRPC already accepts remote connections via TCP |
| vLLM pool | No | Set `REMOTE_DNS` to any reachable host |

## Remote-ready paths

**Multiple RolloutManagers:** each independently pushes to the same LiveStore over
gRPC. Works today — just start multiple instances with different `--data-files` slices.

**Remote EnvironmentProvider:** RolloutManager calls ProRL via `httpx` with no
knowledge of Singularity. Already works: `--prorl-url http://remote-host:8006`.

**Remote LiveStore or PolicyRegistry:** Change the UDS `socket_path` argument in
`scripts/services/start_live_store.sh` and `start_policy_registry.sh` from
`unix:/tmp/prorl_*.sock` to `host:port`. The gRPC server supports both transports.

## Future topologies

The design supports these topologies without code changes to fabric services:
- Per-environment worker fleet: N RolloutManagers, one LiveStore
- Partner worker: a separate machine runs its own RolloutManager against the same LiveStore
- Remote sandbox execution: EnvironmentProvider on a dedicated high-memory host running many Singularity containers

## Startup order (enforced by `scripts/services/start_all.sh`)

1. InferenceBackend (vLLM pool) — remote EC2
2. EnvironmentProvider (ProRL :8006)
3. LiveStore (UDS) + PolicyRegistry (UDS) — parallel
4. RolloutManager
5. TrainerAdapter (Docker) — starts ONLY after ≥1 group in LiveStore (BC-16 warm-up gate)
