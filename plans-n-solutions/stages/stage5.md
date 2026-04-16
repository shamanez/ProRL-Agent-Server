# Stage 5 - Trajectory store + bounded replay buffer (Mode A)

**Status: NOT STARTED**

## Plan

Decouple rollout generation from training consumption. Rollouts stream into a disk-backed trajectory store; the trainer samples from a bounded replay buffer with a freshness rule; inference workers run continuously.

### New files

- `openhands/nvidia/trajectory_store.py` - JSONL writer per episode, rotation, gzip
- `trainer_integration/verl/verl_custom/nvidia/rollout/replay_buffer.py` - bounded sampler with freshness window

### Files modified

- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` - `rollout.mode=replay` branch in `fit()`, background rollout pump
- `openhands/nvidia/async_server.py` - write to trajectory store after eval

### New wandb metrics

`replay/buffer_size`, `replay/staleness_mean`, `replay/fresh_ratio`, `serving/w_over_t`

### GPU plan

Same as Stage 4 (trainer 0-3, two pools 4-7). Inference now runs continuously.

### Success criteria

- 20 steps with replay enabled
- Buffer size stays <= `max_buffer_size`
- Staleness bounded by `max_staleness`
- `serving/w_over_t` >= 1 over last 10 steps
- All Stage 4 invariants still hold

### Risks

- Low `fresh_ratio` triggers PPO importance-ratio caps, wasting compute
- JSONL I/O contention under many eval workers (mitigated by per-worker sharding)
- Trusted-logprob rescoring deferred to follow-up

## Solution

*Not started yet.*
