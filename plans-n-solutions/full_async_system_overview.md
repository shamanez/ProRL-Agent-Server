# Full-Async Agentic RL System Overview

This note explains the main runtime path for the fully-async OpenHands + vLLM
rollout system: what the trainer consumes, how generated trajectories are
packed and trained on, how OpenHands fits in, how the producer/replay store
handles staleness, how the 4-child vLLM pool is used, and how LoRA weight sync
works.

## 1. System Sketch

```text
┌──────────────────────── trainer box ────────────────────────┐
│                                                              │
│  VERL / RayPPOTrainer                                        │
│    - samples training groups                                 │
│    - drains replay store                                     │
│    - computes reward / logprobs / advantages                 │
│    - runs actor backward/update                              │
│    - saves LoRA checkpoint                                   │
│    - publishes LoRA to vLLM pool                             │
│                                                              │
│  ContinuousRolloutProducer                                   │
│    - runs in daemon thread                                   │
│    - calls OpenHands for multi-turn rollouts                 │
│    - pushes completed groups into TrajectoryStore            │
│                                                              │
│  OpenHands / ProRL server :8006                              │
│    - owns sandbox/runtime                                    │
│    - runs CodeAct tool loop                                  │
│    - calls vLLM every assistant turn                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP /v{policy_version}/generate
                              │ HTTP /reload_lora
                              ▼
┌────────────────────── remote vLLM instance ──────────────────┐
│                                                              │
│  child :8100      child :8101      child :8102      child :8103
│  GPU 0            GPU 1            GPU 2            GPU 3
│                                                              │
│  Qwen base model + LoRA adapters                             │
│  --enable-lora --max-loras 8 --max-cpu-loras 16              │
│  /v{N}/generate pins one request to policy version N          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

Mental model:

```text
producer fills the buffer
trainer drains the buffer
OpenHands creates trajectories
vLLM only serves token generation
LoRA publish advances the policy version
```

Start with `plans-n-solutions/handsoff.md` for the operational runbook.

## 2. Trainer: What It Gets

The trainer receives prompt batches from the dataloader. In classic mode, the
current dataloader batch is used immediately. In continuous-producer mode, the
main trainer loop mostly ignores the current `batch_dict`; a background producer
has its own dataloader iterator and fills the replay store.

Classic path:

```text
batch_dict
  -> DataProto.from_single_dict
  -> pop prompt-side fields
       input_ids, attention_mask, position_ids, instance/raw_prompt_ids
  -> generate_sequences(...)
  -> stamp uid
  -> repeat prompt rows n times
  -> union generated responses
  -> optional replay push/sample
  -> reward/logprob/advantage/update
```

Continuous-producer path:

```text
producer thread:
  dataloader batch -> generate_sequences -> TrajectoryStore.push_from_dataproto

trainer thread:
  wait until store has enough fresh groups
  -> sample_mini_batch
  -> train on sampled groups
```

Code paths:

- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`
  - `fit()` main loop around line 2070.
  - `_acquire_training_batch()` around line 1887.
  - classic generation branch around line 1951.
  - continuous-producer sampling branch around line 1896.
- `trainer_integration/verl/verl_custom/replay/continuous_producer.py`
  - `ContinuousRolloutProducer`.

## 3. Padding And Tensor Packing

OpenHands returns a full multi-turn message history. The rollout manager turns
that into training tensors.

Packing sketch:

```text
OpenHands messages
  -> keep messages ending at assistant turn
  -> split at first assistant message

prompt:
  all tokens before first assistant message
  -> left-pad to max_starting_message_length

response:
  all assistant/tool-loop continuation tokens
  -> right-pad to total_len

final tensors:
  input_ids = left_padded_prompt || right_padded_response
  responses = right_padded_response
  attention_mask = prompt_mask || response_mask
  position_ids = cumsum(attention_mask)
  loss_mask = assistant-token mask
  rollout_log_probs = vLLM behavior-policy logprobs
  is_padded = copied fallback trajectory marker
  error_mask = failed trajectory marker
```

Important details:

- Prompt is left-padded because the train-side model expects prompt tokens in a
  fixed seed slot.
- Response is right-padded because loss is over response positions.
- `loss_mask` marks only assistant-generated tokens. Tool responses are context,
  not policy-loss tokens.
- `rollout_log_probs` are stored behavior-policy logprobs from vLLM.

Code paths:

- `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`
  - `_convert_results_to_dataproto_token()` around line 1260.
  - final tensor dict around line 1449.
- `trainer_integration/verl/verl_custom/nvidia/rollout/utils.py`
  - `convert_right_padding_to_left()`.
  - `pad_to_max_length_right()`.

## 4. Forward, Logprobs, Backward

After the trainer has a sampled batch:

```text
batch
  -> compute_response_mask
  -> reward_fn(batch)
  -> actor.compute_log_prob(batch)
       gives old_log_probs under current actor snapshot
  -> reference.compute_ref_log_prob(batch), if enabled
  -> compute_advantage(...)
  -> actor.update_actor(batch)
       forward current actor -> log_prob
       PPO loss uses exp(log_prob - old_log_probs)
       optional TIS uses exp(old_log_probs - rollout_log_probs)
       backward
       grad clip
       optimizer step
```

Meaning of the logprob tensors:

```text
rollout_log_probs:
  behavior policy logprobs from the vLLM rollout time

old_log_probs:
  trainer actor logprobs recomputed before the PPO update

log_prob:
  trainer actor logprobs during update_policy forward

ref_log_prob:
  reference policy logprobs for KL
```

The actor forward path can remove padding before the model forward, then pad
logprobs back to `[batch, response_len]`. Temperature is applied before
log-softmax in both logprob recompute and actor update paths.

Code paths:

- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`
  - reward/logprob/ref/adv/update around lines 2100-2243.
- `trainer_integration/verl/verl_custom/workers/fsdp_workers.py`
  - `compute_log_prob()` wrapper around line 330.
- `trainer_integration/verl/verl_custom/workers/actor/dp_actor.py`
  - `_forward_micro_batch()` around line 109.
  - `compute_log_prob()` around line 329.
  - `update_policy()` around line 409.
- `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py`
  - `compute_policy_loss()` around line 591.

## 5. OpenHands Role

OpenHands is the multi-turn agent executor. The trainer does not directly run
tools. It sends an instance to OpenHands `/process`; OpenHands creates a sandbox,
runs the CodeAct agent, calls vLLM for model turns, executes tools, appends tool
observations, and returns the trajectory.

```text
POST /process
  instance + sampling_params
    -> choose handler by instance["data_source"]
    -> create sandbox/runtime
    -> create CodeActAgent
    -> initial user task
    -> model call to vLLM
    -> parse tool call
    -> execute tool in sandbox
    -> append tool response
    -> repeat until finish/max_iterations
    -> return messages, tools, git_patch, resolved, finish, error
```

Default SWE-style tools here:

```text
execute_bash
str_replace_editor
finish
```

`browser` is only enabled if `RUN_WITH_BROWSING=true`. `execute_ipython_cell`
and `think` are disabled in the SWE config used here.

Code paths:

- `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`
  - `_send_single_message_to_openhands()` around line 2201.
- `openhands/nvidia/async_server_process.py`
  - `OpenHandsServer.process()` around line 830.
  - path-version rewrite to `/v{N}` around line 853.
- `openhands/nvidia/swe_agent/utils.py`
  - sandbox/runtime creation around line 143 and line 288.
  - agent run around line 307.
- `openhands/agenthub/codeact_agent/codeact_agent.py`
  - tool selection around line 111.

## 6. Per-Group And Per-Trajectory Policy Pinning

For GRPO/DAPO, all `n` sibling trajectories for one prompt must come from the
same behavior policy. The code handles this by reading `policy_version` once
when expanding the prompt batch, then stamping every sibling with that version.

```text
prompt P expanded with n = 8

P/sample_0 -> policy_version 12 -> /v12/generate on every turn
P/sample_1 -> policy_version 12 -> /v12/generate on every turn
...
P/sample_7 -> policy_version 12 -> /v12/generate on every turn

trainer may publish policy 13 while these jobs run
but these jobs continue using /v12/generate
```

This prevents a group where some samples come from policy 12 and others from
policy 13.

Code paths:

- `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`
  - `DataProto2Messages()` policy stamp around line 1508.
- `openhands/nvidia/async_server_process.py`
  - rewrites vLLM base URL to `/v{policy_version}` around line 853.
- `scripts/serving/_vllm_child.py`
  - `/v{lora_int_id}/generate` around line 268.

## 7. Producer And Trajectory Store

The trajectory store is a bounded FIFO buffer of groups. A group is the full set
of sibling trajectories for one prompt. Groups are never split.

```text
push_from_dataproto
  -> group rows by uid
  -> each uid becomes one group
  -> store raw unpadded token tuples
  -> remember behavior_policy_version
  -> remember created_at_step

sample_mini_batch
  -> evict stale groups
  -> randomly choose n groups
  -> REMOVE chosen groups from store
  -> re-pad sampled records
  -> return SampledMiniBatch
```

Important: sampled trajectories are not kept for reuse. This is queue-like
replay, not with-replacement replay. Once `sample_mini_batch()` chooses a group,
that group is removed from the store and cannot be sampled again.

Why store unpadded records:

```text
producer batch A may have response width 12000
producer batch B may have response width 9000

DataProto.concat requires matching dim-1
so store raw token tuples and re-pad at sample time
```

Code paths:

- `trainer_integration/verl/verl_custom/replay/trajectory_store.py`
  - record schema around line 43.
  - push/grouping around line 194.
  - consume-on-sample around line 422.
  - re-padding around line 470.
- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`
  - `_push_and_sample_replay()` around line 1643.
  - producer eager-push wiring around line 1771.

## 8. Staleness In The Store

Each trajectory record carries:

```text
behavior_policy_version:
  LoRA policy version that actually generated the trajectory

created_at_step:
  trainer step when the trajectory was pushed

rollout_log_probs:
  behavior-policy token logprobs from rollout time
```

Staleness cutoff:

```text
age = current_step - created_at_step

if age > staleness_cutoff_k:
    drop group
else:
    group is fresh enough to sample
```

This cutoff mainly handles trajectories that sit unused during validation
pauses, producer/trainer imbalance, or buffer buildup. Since sampled groups are
removed, there is no repeated training on the same old group.

Off-policy correction:

```text
rollout_log_probs = log pi_behavior(a | s)
old_log_probs     = log pi_current_before_update(a | s)

TIS weight = exp(old_log_probs - rollout_log_probs)
TIS weight = min(TIS weight, tis_imp_ratio_cap)

PPO token loss is multiplied by the clipped TIS weight
```

So staleness is controlled in two places:

- Hard drop by `staleness_cutoff_k`.
- Soft correction by clipped temporal importance sampling.

Code paths:

- `trainer_integration/verl/verl_custom/replay/trajectory_store.py`
  - `behavior_policy_version` and `created_at_step` fields around line 52.
  - row policy-version selection around line 330.
  - stale eviction around line 396.
  - sample metadata around line 614.
- `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py`
  - TIS weighting around line 664.

## 9. Four vLLM Workers

The external vLLM pool has four child servers:

```text
GPU 0 -> http://host:8100
GPU 1 -> http://host:8101
GPU 2 -> http://host:8102
GPU 3 -> http://host:8103
```

The trainer config lists these as `external_llm_endpoints`. OpenHands workers
load-balance requests across them. Each child is a FastAPI wrapper over vLLM
with token-level `/generate`, `/v{N}/generate`, `/reload_lora`, and `/health`.

Each child starts with:

```text
--enable-lora
--max-loras 8
--max-lora-rank 32
--max-cpu-loras 16
--swap-protocol pinning
```

Meaning:

- `max_loras=8`: GPU-side LoRA capacity.
- `max_cpu_loras=16`: CPU-side LoRA cache capacity.
- Pinning mode keeps old policy versions addressable through `/v{N}/generate`.
- vLLM internal LRU may move adapters between GPU/CPU/disk-backed paths, but
  the route still names the requested policy version.

Code paths:

- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh`
  - `external_llm_endpoints` around line 137.
- `scripts/serving/launch_remote_vllm_pool.sh`
  - GPU/port mapping around line 45.
- `scripts/serving/_remote_vllm_runner.sh`
  - child launch flags around line 61.
- `scripts/serving/_vllm_child.py`
  - LoRA state and pinning comments around line 50.

## 10. LoRA Weight Sync

Weight sync publishes only the LoRA adapter, not the full base model.

```text
trainer update
  -> save checkpoint
  -> read actor/lora_adapter
       adapter_model.safetensors
       adapter_config.json
  -> new_version = policy_version + 1
  -> wait for vLLM headroom
  -> POST adapter tarball to every vLLM child /reload_lora
  -> if every child ACKs, commit policy_version = new_version
  -> async_rollout_manager.policy_version = new_version
  -> future OpenHands jobs stamp new_version
```

The trainer commits the new `policy_version` only after every endpoint returns
success. If any endpoint fails, the publish aborts; this avoids a mixed pool
where different vLLM children serve different latest policies.

Pre-publish wait:

```text
for each vLLM child:
  GET /health
  read inflight_versions_count

if all children <= 6 active in-flight versions:
  publish now
else:
  wait briefly, up to 30s
  then publish anyway and log metrics
```

Reason: each child has `--max-loras 8`. Publishing while 7 or 8 old versions are
actively serving can cause unnecessary LRU churn and latency spikes. Waiting for
the active distinct version count to drop keeps headroom.

vLLM child reload:

```text
/reload_lora(policy_version=N, adapter.tgz)
  -> reject if N <= active_policy_version
  -> extract to /tmp/lora_adapters/pvN
  -> create LoRARequest(lora_name="pvN", lora_int_id=N)
  -> engine.add_lora(...)
  -> active_policy_version = N
  -> _resident_loras[N] = request
  -> in pinning mode: do NOT remove old versions
```

Pinned generate:

```text
/v12/generate
  -> find _resident_loras[12]
  -> if missing, try /tmp/lora_adapters/pv12
  -> if not recoverable, return HTTP 410
  -> otherwise generate with LoRA id 12
```

Code paths:

- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`
  - `_wait_for_publish_headroom()` around line 1316.
  - `_publish_lora_adapter()` around line 1413.
  - commit `policy_version` around line 1508.
  - publish after checkpoint save around line 2284.
- `scripts/serving/_vllm_child.py`
  - `/health` around line 142.
  - `/v{lora_int_id}/generate` around line 268.
  - `/reload_lora` around line 397.
  - pinning-mode no-remove behavior around line 554.

## 11. Reading Map

Use this order when debugging or onboarding:

1. `plans-n-solutions/handsoff.md`
   - operational overview, launch order, gotchas.

2. `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`
   - main trainer loop, replay acquisition, logprob/reward/update, LoRA publish.

3. `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`
   - prompt expansion, policy-version stamping, OpenHands dispatch, DataProto
     packing.

4. `openhands/nvidia/async_server_process.py`
   - OpenHands `/process`, per-job routing to `/v{policy_version}`.

5. `openhands/nvidia/swe_agent/utils.py`
   - sandbox/runtime creation and CodeAct run path.

6. `trainer_integration/verl/verl_custom/replay/continuous_producer.py`
   - producer thread lifecycle.

7. `trainer_integration/verl/verl_custom/replay/trajectory_store.py`
   - group storage, staleness, consume-on-sample, re-padding.

8. `trainer_integration/verl/verl_custom/workers/actor/dp_actor.py`
   - actor forward, PPO loss input, backward.

9. `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py`
   - PPO and TIS math.

10. `scripts/serving/_vllm_child.py`
    - vLLM token endpoint, LoRA pinning, reload, health.

11. `scripts/serving/_remote_vllm_runner.sh`
    - per-child vLLM launch flags.

