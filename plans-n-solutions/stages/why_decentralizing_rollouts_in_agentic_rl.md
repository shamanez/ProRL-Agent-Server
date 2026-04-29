# Why Decentralizing Rollouts in Agentic RL

Agentic RL is often rollout-bound, not trainer-bound. A single trajectory can take a long time because the agent must interact with an environment, call tools, execute code, and wait for verification. If the trainer waits synchronously for this process, expensive training GPUs can sit idle.

The core idea of decentralized rollouts is to separate trajectory production from gradient computation:

- vLLM workers generate trajectories.
- ProRL/OpenHands workers execute environment interactions.
- A producer continuously fills replay with completed DAPO/GRPO groups.
- The trainer samples fresh groups from replay and updates the model.

This only works if producer throughput is higher than trainer consumption.

With `NUM_TRAJ=8`, one usable replay group costs 8 completed trajectories from the same prompt. The trainer consumes:

```text
groups_per_step = BATCH_SIZE
trajectories_per_step = BATCH_SIZE * NUM_TRAJ
```

So if:

```text
BATCH_SIZE = 16
NUM_TRAJ = 8
```

then each trainer step consumes:

```text
16 * 8 = 128 trajectories
```

That means the producer must create at least 16 fresh surviving DAPO groups before the trainer needs the next batch. In trajectory terms, it must complete at least 128 usable trajectories per trainer step, plus extra rollout work lost to DAPO filtering.

The claim is simple:

```text
trainer stays busy only if:
producer_survivor_groups_per_second > trainer_groups_consumed_per_second
```

Increasing `BATCH_SIZE` can make gradients less noisy, but it also increases replay demand. Increasing `NUM_TRAJ` can improve group-level advantage estimates, but it multiplies rollout cost per usable group. Scaling vLLM/OpenHands workers only helps if the rollout system is actually capacity-bound and not already queueing on the same GPUs or environment bottlenecks.

We should treat this as an empirical claim. A follow-up experiment should measure:

- producer survivor groups per minute
- raw trajectories per minute
- raw generated tokens per second
- accepted/surviving tokens per second after DAPO filtering
- trainer tokens per second during old-logprob recompute and actor update
- DAPO filter drop rate
- trainer groups consumed per minute
- replay store size and sample age
- trainer idle time waiting for replay
- vLLM GPU utilization and queueing

Tokens matter because two runs can produce the same number of trajectories with very different costs. Long agent trajectories can keep replay warm in group count while still making vLLM decode, FSDP logprob recompute, or actor backward the actual bottleneck. We should separate at least:

```text
rollout_tokens_per_second
survivor_tokens_per_second
trainer_tokens_per_second
```

Group throughput answers "does replay stay full?" Token throughput answers "which compute path is limiting scale?"

For `N` rollout machines with `N` vLLM servers, the useful estimate is aggregate survivor production, not just raw decode capacity:

```text
raw_rollout_capacity = sum(raw_trajectories_per_second_per_server)
survivor_group_rate = raw_rollout_capacity * dapo_survival_rate / NUM_TRAJ
trainer_group_rate = BATCH_SIZE / trainer_step_seconds
```

To keep the trainer busy:

```text
survivor_group_rate > trainer_group_rate
```

Equivalently:

```text
sum(raw_trajectories_per_second_per_server) * dapo_survival_rate / NUM_TRAJ
  > BATCH_SIZE / trainer_step_seconds
```

Token-level scaling should use separate token definitions:

```text
rollout_decode_tokens_per_second = sum(vLLM generated tokens/sec)
trainer_forward_tokens_per_second = prompt + response tokens/sec through FSDP
```

Rollout decode mostly pays for generated response tokens. The trainer pays for full `prompt + response` sequences when recomputing logprobs and doing actor backward. In agentic RL, environment/tool latency also matters, so more vLLM servers only helps while vLLM decode is the bottleneck. If OpenHands, environment execution, verification, or replay filtering is the bottleneck, adding vLLM servers will not linearly increase survivor groups.

Only after measuring those numbers should we decide whether to scale `BATCH_SIZE`, `GEN_BATCH_SIZE`, `NUM_TRAJ`, OpenHands worker count, or the number of vLLM endpoints.
