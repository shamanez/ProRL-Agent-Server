# Decoupling Walkthrough — Trainer ↔ vLLM

> **Purpose on this branch.** This is the "big picture" doc: it explains why colocated vLLM training works (tensor aliasing via `wake_up()`) and what breaks the moment vLLM moves to a different node. Read it to understand **why** we decoupled and **what** weight sync has to put back. For what the `decoup-weight-sync` branch actually runs today, see [`../plans-n-solutions/stages/baseline.md`](../plans-n-solutions/stages/baseline.md); for the LoRA-first closure plan, see [`../plans-n-solutions/stages/weight_sync_lora.md`](../plans-n-solutions/stages/weight_sync_lora.md).
>
> **Historical refs.** This walkthrough was written against the colocated baseline. Source-file citations (ray_trainer, vllm_async_server, async_server, qwen3) still resolve. Citations to the pre-decoupling launch script `run_proagent_qwn3_4B_instruct.sh` are **historical** — on this branch the live launcher is `..._remote_decoupled.sh` and the `s1_remote_docker.sh` Docker wrapper.

> *Read-only code walkthrough. Every claim is backed by a `path/to/file.py:<line>` citation with a verbatim code quote.*

## Contents

1. [TL;DR](#1-tldr)
2. [Hardware layout for this investigation](#2-hardware-layout-for-this-investigation)
3. [Boot sequence](#3-boot-sequence)
4. [One rollout step (end-to-end trace)](#4-one-rollout-step-end-to-end-trace)
5. [One training step (after rollout)](#5-one-training-step-after-rollout)
6. [The weight handoff — the crucial section](#6-the-weight-handoff--the-crucial-section)
7. [What would break if vLLM moved to a different node](#7-what-would-break-if-vllm-moved-to-a-different-node)
8. [Experimental check you can run locally](#8-experimental-check-you-can-run-locally)
9. [File & line index](#9-file--line-index)

---

## 1. TL;DR

- **Colocated:** FSDP actor + reference model **and** vLLM engines run inside the **same Ray named actors** (`{wg_prefix}WorkerDict_{pg_idx}:{local_rank}`). This is not "same node" — it is "same Python process on the same CUDA device." See `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:43-85`.
- **Not colocated:** the **ProRL Agent Server** (FastAPI on port `8006`) is a *separate* host-level Python process. It orchestrates agent loops and forwards token IDs to vLLM over HTTP — see `openhands/nvidia/async_server.py:59-121` and `scripts/start_server.py:604-691`.
- **Weight reuse is implicit via GPU memory, not a transfer.** After FSDP's optimizer step mutates the model tensors in place, the vLLM engine — which was initialized with `enable_sleep_mode=True` and loaded its weights *by RPCing the same FSDP actors* — simply `wake_up()`s on those same tensors. No `state_dict()`, no NCCL broadcast, no file round-trip. See `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:169-189` and `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:938-942`.
- **The trainer drives sleep/wake.** Every GRPO iteration wakes vLLM before rollout and sleeps it before the optimizer step — see `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:1076-1082`.
- **Token-in/token-out invariant.** ProRL sends *prompt token IDs* to vLLM and stores the returned `response_ids` verbatim for the next turn — no decode/re-encode — which prevents KL drift in GRPO. See `openhands/llm/nvidia/qwen3.py:188-257`.
- **Load-balancing across vLLM endpoints** is a weighted min-heap keyed on per-address in-flight count. See `openhands/nvidia/async_server.py:105-164` (init + `create_llm_config`).
- **Load-bearing files to read first:** `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`, `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py`, `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`, `openhands/nvidia/async_server.py`, `openhands/llm/nvidia/qwen3.py`, and the launch script `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh`.

---

## 2. Hardware layout for this investigation

The launch script `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` encodes the 8-GPU (H200) assumption at lines 22-27:

```bash
# run_proagent_qwn3_4B_instruct.sh:22-27
GPU_MEM_UTIL=0.8
TP_SIZE=2
# Assumes a h200 node
# For 2xH100: change tp size -> 2, sequence parallel size -> 2, nnodes -> 2
NNODES=1
SP_SIZE=1
```

Trainer GPU allocation at line 98:

```bash
# run_proagent_qwn3_4B_instruct.sh:98
trainer.n_gpus_per_node=8 \
```

vLLM tensor-parallel size at line 58 (each vLLM "data-parallel rank" uses `TP_SIZE=2` GPUs):

```bash
# run_proagent_qwn3_4B_instruct.sh:58
actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
```

FSDP offload toggles at lines 51-52 and 86 — keeping **params on GPU** (critical for shared-tensor semantics) while offloading optimizer state:

```bash
# run_proagent_qwn3_4B_instruct.sh:51-52
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
```

```bash
# run_proagent_qwn3_4B_instruct.sh:86
actor_rollout_ref.ref.fsdp_config.param_offload=True \
```

> `param_offload=True` means FSDP will *shard* and *offload to CPU* when idle, but loads back onto GPU for compute. In the wake/sleep cycle, vLLM uses the GPU during rollout while FSDP parameters are temporarily offloaded; during training, vLLM is asleep (its VRAM freed) and FSDP reclaims the GPU. This is why sharing works despite the two systems "both" living in the same actor.

Memory-saver (vLLM sleep-mode) is enabled at line 68:

```bash
# run_proagent_qwn3_4B_instruct.sh:68
+actor_rollout_ref.rollout.enable_memory_saver=True \
```

vLLM's GPU memory utilization at line 62:

```bash
# run_proagent_qwn3_4B_instruct.sh:62
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
```

### Text diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Single physical node  (example: 8 × H200 / A100-40GB)              │
│                                                                     │
│  ┌──────────────────────────────┐   ┌──────────────────────────┐    │
│  │ Host-level Python process    │   │ Ray cluster (num_cpus=…)  │   │
│  │                              │   │                          │    │
│  │  ProRL Agent Server          │   │  8 × Ray named actors    │    │
│  │  (scripts/start_server.py)   │   │  "{prefix}WorkerDict_*"  │    │
│  │  FastAPI :8006               │   │                          │    │
│  │   /process                   │   │  Each actor holds:       │    │
│  │   /add_llm_server            │   │   • FSDP-sharded policy  │    │
│  │   /status                    │   │   • FSDP-sharded ref     │    │
│  │                              │   │   • vLLM worker stub     │    │
│  │  Maintains weighted-heap     │   │     (loaded on same      │    │
│  │  of vLLM /generate URLs      │   │      GPU memory)         │    │
│  └────────────┬─────────────────┘   └──────────┬───────────────┘    │
│               │  POST /generate                │ collective_rpc     │
│               │  {prompt_ids, top_p,...}       │ (init_worker,      │
│               │                                │  init_device,      │
│               │                                │  load_model)       │
│               │                                │                    │
│               └────────────────────────────────┘                    │
│                                                                     │
│  GPUs:                                                              │
│   • FSDP actor + ref  → all 8 GPUs (n_gpus_per_node=8)              │
│   • vLLM engines      → 4 data-parallel replicas × TP=2            │
│                         colocated on the same 8 GPUs                │
│   • Weight sharing    → shared CUDA memory within each actor        │
└─────────────────────────────────────────────────────────────────────┘
```

`NNODES=1`, `TP_SIZE=2`, `n_gpus_per_node=8` ⇒ 4 vLLM replicas (8/2) colocated with 8 FSDP ranks. Because both live inside the same Ray actor process for their respective GPU rank, they see the same CUDA context and the same GPU tensors.

---

## 3. Boot sequence

### Step 3.1 — Ray cluster init

`trainer_integration/verl/verl_custom/trainer/main_ppo.py:27-48`:

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", "VLLM_USE_V1": "1"}},
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
```

### Step 3.2 — Worker class selection (async vs sync)

Still in `trainer_integration/verl/verl_custom/trainer/main_ppo.py:90-97`:

```python
# Define worker classes based on the actor strategy.
if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
    assert config.critic.strategy in ["fsdp", "fsdp2"]
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

    actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
    ray_worker_group_cls = RayWorkerGroup
```

Because `run_proagent_qwn3_4B_instruct.sh:60` sets `actor_rollout_ref.rollout.mode=async`, the hybrid `AsyncActorRolloutRefWorker` is picked — a Ray actor that holds both FSDP-wrapped tensors and a vLLM engine. (`AsyncActorRolloutRefWorker` itself is defined upstream in the `verl` package.)

### Step 3.3 — FSDP worker group creation & `init_model()`

`trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:880-882`:

```python
# we should create rollout at the end so that vllm can have a better estimation of kv cache memory
self.actor_rollout_wg = all_wg["actor_rollout"]
self.actor_rollout_wg.init_model()
```

`init_model()` (upstream in `verl.workers.fsdp_workers`) constructs the FSDP-wrapped policy *on the same Ray actors that will later host the vLLM engine*.

### Step 3.4 — `AsyncLLMServerManager` instantiation

`trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:884-902`:

```python
# create async rollout manager and request scheduler
self.async_rollout_mode = False
if self.config.actor_rollout_ref.rollout.mode == "async":
    if self.config.actor_rollout_ref.rollout.get("async_manager", "")=="openhands":
        if self.config.algorithm.get("filter_groups", {}).get("enable", False):
            from verl_custom.nvidia.rollout.async_server_dapo import AsyncLLMServerManagerDAPO as AsyncLLMServerManager
        else:
            from verl_custom.nvidia.rollout.async_server import AsyncLLMServerManager
    else:
        from verl.workers.rollout.async_server import AsyncLLMServerManager

    self.async_rollout_mode = True
    self.async_rollout_manager = AsyncLLMServerManager(
        config=self.config,
        worker_group=self.actor_rollout_wg,
    )
```

Because `run_proagent_qwn3_4B_instruct.sh:66` sets `+actor_rollout_ref.rollout.async_manager=openhands`, the manager is the OpenHands-integrated one at `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` — i.e. it will forward rollout requests to the ProRL FastAPI server rather than streaming via verl's built-in scheduler.

Note the constructor passes `worker_group=self.actor_rollout_wg` — the manager holds a reference to the *same* actors FSDP is using.

### Step 3.5 — vLLM async engines spawned via `ExternalRayDistributedExecutor`

`trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:43-85`:

```python
class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

        fields = self.vllm_config.instance_id.split(":")
        assert len(fields) == 4, f"instance_id: {self.vllm_config.instance_id} must be in the format of <namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
        namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

        # Make sure subprocess in same namespace as parent actor.
        # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
        ray.init(address="auto", namespace=namespace)
        actor_names = [actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")]

        vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        assert len(actor_names) == vllm_dp_size * vllm_tp_size, f"instance_id: {self.vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: {vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."

        def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
            fields = actor_name.split(":")
            assert len(fields) == 2, f"invalid actor name: {actor_name}"
            pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
            return pg_index, local_rank

        # sort actor names by pg_index and local_rank
        actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
        actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
        self.workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
        print(f"instance_id: {self.vllm_config.instance_id} initializes with external actors: {actor_names}")

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} initializes finished.")
```

This is the crux of the colocation contract. Three facts:

1. **No new actor is spawned for vLLM.** Line 72 is `self.workers = [ray.get_actor(actor_name) for actor_name in actor_names]` — vLLM *adopts* existing Ray actors that were created for FSDP.
2. **Actors are picked by name-prefix lookup** (line 58: `...if actor_name.startswith(f"{wg_prefix}WorkerDict")`). That prefix is set on the same Ray cluster; it implicitly assumes local namespace (line 57: `ray.init(address="auto", namespace=namespace)`).
3. **vLLM runs `init_worker`, `init_device`, and `load_model` inside the FSDP actors** (lines 82-84). So the model weights that vLLM's `forward` will later use live on the same GPUs that FSDP manages.

### Step 3.6 — vLLM engine config with `enable_sleep_mode=True`

`trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:169-189`:

```python
engine_args = AsyncEngineArgs(
    model=local_path,
    enable_sleep_mode=True,
    override_generation_config=kwargs,
    tensor_parallel_size=tensor_parallel_size,
    distributed_executor_backend=ExternalRayDistributedExecutor,
    dtype=config.dtype,
    enforce_eager=config.enforce_eager,
    gpu_memory_utilization=config.gpu_memory_utilization,
    disable_custom_all_reduce=True,
    disable_mm_preprocessor_cache=True,
    skip_tokenizer_init=False,
    max_model_len=max_model_len,
    load_format="auto",
    disable_log_stats=config.disable_log_stats,
    max_num_batched_tokens=max_num_batched_tokens,
    enable_chunked_prefill=config.enable_chunked_prefill,
    enable_prefix_caching=True,
    trust_remote_code=trust_remote_code,
    seed=self.vllm_dp_rank,
    logprobs_mode=config.get("logprobs_mode", "processed_logprobs"),
```

Two settings matter for this walkthrough:

- `enable_sleep_mode=True` wires in vLLM's `memory_saver` subsystem so that `engine.sleep()` / `engine.wake_up()` are callable.
- `distributed_executor_backend=ExternalRayDistributedExecutor` is the custom backend from step 3.5 that adopts FSDP's actors.

### Step 3.7 — ProRL Agent Server started separately

This is a separate Python process (not a Ray actor). `scripts/start_server.py` constructs a FastAPI app at module load and exposes the endpoints described in step 3.8. The server is started out-of-band (typically `python scripts/start_server.py --host 0.0.0.0 --port 8006 ...` per `CLAUDE.md`). The training loop expects it to be already reachable at the URL configured by `run_proagent_qwn3_4B_instruct.sh:71`:

```bash
# run_proagent_qwn3_4B_instruct.sh:71
+actor_rollout_ref.rollout.openhands_base_url=http://localhost:8006 \
```

The server's `OpenHandsServer` class maintains the three job queues and the weighted-address list:

```python
# openhands/nvidia/async_server.py:88-106
self.init_queue: queue.Queue[str] = queue.Queue()
self.run_queue: queue.Queue[str] = queue.Queue()
self.evaluate_queue: queue.Queue[str] = queue.Queue()
...
self.weighted_addresses = [[0, address] for address in llm_server_addresses]
heapq.heapify(self.weighted_addresses)
```

### Step 3.8 — `POST /add_llm_server` registers each vLLM endpoint

`scripts/start_server.py:550-575`:

```python
@app.post('/add_llm_server')
async def add_llm_server(request: LLMServerRequest):
    try:
        address = request.address
        # Always keep buffer in sync
        with config_lock:
            if address not in llm_server_addresses_buffer:
                llm_server_addresses_buffer.append(address)
        if not _is_server_running():
            return {'status': f'Buffered LLM server address: {address}'}

        req_id = str(uuid.uuid4())
        if request_queue is not None:
            request_queue.put({'type': 'add_llm_server', 'address': address, 'request_id': req_id})

        if control_response_queue is not None:
            try:
                ack = control_response_queue.get(timeout=5)
                if not ack.get('ok', False):
                    raise HTTPException(status_code=500, detail=f"Failed to add LLM server: {ack.get('error', 'unknown')}")
            except Exception:
                pass
        return {'status': f'Added LLM server address: {address}'}
    except Exception as e:
        logger.error(f'Failed to add LLM server: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to add LLM server: {str(e)}')
```

The trainer (through the rollout manager) calls this endpoint *once per vLLM instance* during startup, supplying the `http://host:port` base-URL. That URL is where the vLLM FastAPI from step 3.5 listens — not the ProRL server.

At this point the system is fully booted:

| Component | Where it lives | How many |
|----|----|----|
| FSDP policy + reference | Ray named actors `{prefix}WorkerDict_*` | 8 (n_gpus_per_node) |
| vLLM engines (same actors) | Same Ray actors | 4 replicas × TP=2 = 8 |
| ProRL Agent Server | Host OS process, port 8006 | 1 |
| Trainer driver | `RayPPOTrainer` on Ray head | 1 |

---

## 4. One rollout step (end-to-end trace)

### Step 4.1 — Trainer wakes vLLM

Top of the iteration inside `RayPPOTrainer.fit()`. `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:1074-1084`:

```python
with _timer("step", timing_raw):
    # generate a batch
    with _timer("gen", timing_raw):
        if not self.async_rollout_mode:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            self.async_rollout_manager.wake_up()
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
            self.async_rollout_manager.sleep()
        timing_raw.update(gen_batch_output.meta_info["timing"])
        gen_batch_output.meta_info.pop("timing", None)
```

`wake_up()` on line 1080 blocks until every vLLM instance has re-acquired GPU memory and its engine loop is running.

### Step 4.2 — Trainer issues `generate_sequences`

`generate_sequences(gen_batch)` on line 1081 dispatches through `AsyncLLMServerManager`. For `async_manager=openhands` the manager POSTs each instance to the ProRL FastAPI server at `openhands_base_url` (the `http://localhost:8006` from `run_proagent_qwn3_4B_instruct.sh:71`).

### Step 4.3 — ProRL Agent Server enqueues the job

`openhands/nvidia/async_server.py:263-308` (the `OpenHandsServer.process` method that the FastAPI handler forwards each request into):

```python
def process(self, instance, sampling_params, job_id=None, timeout: float = 300.0):
    if not self._server_running:
        raise RuntimeError('Server is not running')

    with self._address_lock:
        if len(self.weighted_addresses) == 0:
            raise ValueError('No LLM server addresses added')

    is_reasoning_task = sampling_params.pop('is_reasoning_task', False)
    dataset_type = instance.get('data_source', 'swebench')
    if not is_registered_handler(dataset_type, reasoning=is_reasoning_task):
        raise FunctionNotRegisteredError(
            f'Dataset type {dataset_type} is not registered'
        )

    # Create job details
    if job_id is None:
        job_id = self.get_unique_id(instance)
    job_details = JobDetails()
    job_details.job_id = job_id
    job_details.instance = instance
    job_details.is_reasoning_task = is_reasoning_task
    for agent_config_key in job_details.agent_config:
        if agent_config_key in sampling_params:
            job_details.agent_config[agent_config_key] = sampling_params.pop(
                agent_config_key
            )
    llm_config = self.create_llm_config(sampling_params)
    job_details.llm_config = llm_config
    job_details.event = threading.Event()

    # Initialize timer - only tracks init/run/eval phases
    # All other time is automatically counted as "others" (not counted toward timeout)
    job_details.timer = PausableTimer(timeout=timeout)
    job_details.timer.start()

    with self._job_details_lock:
        self._job_details[job_id] = job_details
    logger.info(f'Job {job_id} added to job details')

    # Add job to init queue
    self.init_queue.put(job_id)
    logger.info(f'Job {job_id} added to init queue')

    # Wait for job to be finished
    job_details.event.wait()
```

Key behaviours:

- Line 272 — handler selection by `instance['data_source']` (e.g. `'swebench'`, `'math'`).
- Line 290 — **`create_llm_config` picks a vLLM endpoint** (see step 4.4).
- Line 304 — job moves into the `init` queue. It will then flow `init → run → eval` through dedicated worker pools.

### Step 4.4 — Load-balanced vLLM selection

`openhands/nvidia/async_server.py:133-164` (init / `add_llm_server_address` + `create_llm_config`). The heap is `[[weight, address], ...]`:

```python
# openhands/nvidia/async_server.py:105-106
self.weighted_addresses = [[0, address] for address in llm_server_addresses]
heapq.heapify(self.weighted_addresses)
```

Each new job pops the lowest-weight address and increments that weight. This is how requests spread across the 4 vLLM replicas without any explicit coordinator.

### Step 4.5 — Agent loop hits the Qwen3 client

Inside the `run` stage, the agent may make multiple LLM calls per job. Each call goes through `openhands/llm/nvidia/qwen3.py:188-228`:

```python
def request_response_tokens(
    tokenizer: AutoTokenizer,
    base_url: str | None,
    timeout: int | None,
    top_p: float,
    seed: int | None,
    max_model_len: int,
    **kwargs,
) -> ModelResponse:

    input_ids = []
    messages = kwargs['messages']
    pending_messages = []
    for message in reversed(messages):
        if message['output_ids'] is not None:
            if message['input_ids'] is not None:
                input_ids.extend(message['input_ids'])
            input_ids.extend(message['output_ids'])
            # we break here because all messages before this message have been processed
            break
        else:
            new_message = dict(message)
            new_message.pop('input_ids', None)
            new_message.pop('output_ids', None)
            pending_messages.append(new_message)
```

This is the **token-in/token-out invariant** in action:

- Lines 204-210 walk messages *newest-first* and reuse cached `output_ids` verbatim — no re-tokenization of prior turns.
- Only *new* messages (since the last model turn) are re-tokenized via `convert_messages_to_tokens` (lines 219-226).
- Concatenation at line 228 produces the final `input_ids` that will be POSTed to vLLM.

### Step 4.6 — The actual HTTP call

`openhands/llm/nvidia/qwen3.py:248-260`:

```python
resp = httpx.post(
    url=f'{base_url}/generate',
    json={
        'prompt_ids': input_ids,
        'top_p': top_p,
        'seed': seed,
        **kwargs,
    },
    timeout=timeout,
)
# Raise error if the request is not successful
resp.raise_for_status()
resp = resp.json()
```

- **`prompt_ids`** is the concatenated token-ID list from step 4.5 — integers, not strings.
- **No chat template is re-rendered here**; vLLM receives tokens it will not re-tokenize.

### Step 4.7 — vLLM FastAPI decodes tokens to tokens

On the vLLM side, the `AsyncvLLMServer.generate` endpoint (in `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:198-237`) receives the request:

```python
async def generate(self, raw_request: Request) -> list[int]:
    request_dict = await raw_request.json()
    # Set logprobs to 0 to only return selected tokens
    request_dict['logprobs'] = 0

    prompt_ids = request_dict.pop("prompt_ids")
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    self.request_ids.add(request_id)
    # Create prompt from token IDs
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

    # Get final response
    final_res: Optional[RequestOutput] = None
    try:
        async for output in generator:
            final_res = output
    except asyncio.CancelledError:
        return Response(status_code=499)
    finally:
        self.request_ids.remove(request_id)

    assert final_res is not None

    def obtain_logprobs(logprobs):
        if logprobs is None:
            return None
        log_probs = []
        for d in logprobs:
            cur_logprobs = list(d.values())
            assert len(cur_logprobs) == 1, f"Expected 1 logprob per token when logprobs=0, but got {len(cur_logprobs)}"
            log_probs.append(cur_logprobs[0].logprob)
        return log_probs

    ret = {
        "response_ids": final_res.outputs[0].token_ids,
        "logprobs": obtain_logprobs(final_res.outputs[0].logprobs),
    }
    return JSONResponse(ret)
```

Key invariants:

- Line 208 — `TokensPrompt(prompt_token_ids=prompt_ids)` means vLLM **accepts pre-tokenized IDs**; no `tokenizer.encode` is called on the server.
- Line 234 — returns `response_ids` (token IDs) and per-token `logprobs` (line 235).

### Step 4.8 — Tokens flow back; agent persists the IDs

The Qwen3 client then stores `response_ids` on the message — lines 262-287 of `openhands/llm/nvidia/qwen3.py` (not quoted here; behaviour summarised).

On the *next* turn, step 4.5 will walk messages and reuse those `response_ids` verbatim. The actor and reference models see byte-identical token sequences across turns, which is what keeps KL stable for GRPO.

### Step 4.9 — Rollout batch returns to the trainer

The FastAPI response flows back through the `AsyncLLMServerManager` (`trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`), which collects results across the batch and returns a `DataProto` to `generate_sequences` (line 1081 of `ray_trainer.py`).

### Step 4.10 — Trainer sleeps vLLM

`trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:1082` (still inside the same `with _timer("gen", ...)` block):

```python
self.async_rollout_manager.sleep()
```

This is the handshake that lets FSDP reclaim GPU memory for training. See section 6 for the mechanics.

---

## 5. One training step (after rollout)

All cites are `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`. Steps follow the structure of `RayPPOTrainer.fit()` for a single iteration after step 4.10.

### Step 5.1 — Reward computation

The reward manager is a Ray remote actor (`SWEBenchRewardManager` from `trainer_integration/verl/verl_custom/nvidia/reward_manager/swebench.py`). Its `__call__` takes a `DataProto` (the rollout output) and returns a per-token reward tensor, which the trainer `ray.get(...)`s later (synchronisation point for the GRPO advantage computation).

### Step 5.2 — Old log-prob & reference log-prob

After rewards, the trainer computes `old_log_prob` (from the policy at time of rollout) and — if `use_reference_policy` — `ref_log_prob` (from the frozen reference). Both call the FSDP actor via `self.actor_rollout_wg.compute_log_prob(batch)` and the mirror on the reference actor.

### Step 5.3 — Advantage (GRPO)

The advantage estimator is chosen from `config.algorithm.adv_estimator` (`run_proagent_qwn3_4B_instruct.sh:32` sets `algorithm.adv_estimator=grpo`). The `compute_advantage` helper writes `batch.batch["advantages"]`.

### Step 5.4 — Actor update via FSDP

`trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:1226-1233`:

```python
# implement critic warmup
if self.config.trainer.critic_warmup <= self.global_steps:
    # update actor
    with _timer("update_actor", timing_raw):
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        actor_output = self.actor_rollout_wg.update_actor(batch)
    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    metrics.update(actor_output_metrics)
```

`update_actor(batch)` runs the forward + backward + optimizer step **inside the FSDP actor**. The FSDP parameters held on GPU are mutated in place. There is no `state_dict` export, no `load_state_dict`, no NCCL broadcast to any remote worker — this is one Ray actor updating its own tensors.

This is where the next rollout's weight *content* is produced; the next section explains how those tensors are observed by vLLM.

---

## 6. The weight handoff — the crucial section

The system is designed around a very specific (and fragile) contract: **the weights vLLM reads during generation are literally the tensors FSDP has been updating**. No transfer layer exists between them because both live in the same CUDA context inside the same Ray actor.

### 6.1 vLLM was configured with `enable_sleep_mode=True`

`trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:169-171`:

```python
engine_args = AsyncEngineArgs(
    model=local_path,
    enable_sleep_mode=True,
```

This turns on vLLM's `memory_saver` subsystem (upstream in the `vllm` package). Without this, `engine.sleep()` / `engine.wake_up()` would be unavailable.

### 6.2 vLLM adopts the FSDP actors

Already quoted at step 3.5 — `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:43-85`. The takeaways for this section:

```python
# vllm_async_server.py:56-58
# Make sure subprocess in same namespace as parent actor.
# actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
ray.init(address="auto", namespace=namespace)
actor_names = [actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")]
```

```python
# vllm_async_server.py:72-73
self.workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
print(f"instance_id: {self.vllm_config.instance_id} initializes with external actors: {actor_names}")
```

```python
# vllm_async_server.py:82-84
self.collective_rpc("init_worker", args=([kwargs],))
self.collective_rpc("init_device")
self.collective_rpc("load_model")
```

`collective_rpc("load_model")` runs vLLM's model-loading code inside the FSDP actor. **vLLM's parameters end up on the same CUDA device, in GPU memory managed by the same process that also owns the FSDP-wrapped model.**

### 6.3 `sleep()` aborts requests, resets prefix cache, suspends the engine

`trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:239-247`:

```python
async def wake_up(self):
    await self.engine.wake_up()

async def sleep(self):
    # Abort all unfinished generation requests
    await self._abort_requests()
    # TODO: https://github.com/vllm-project/vllm/issues/17103
    await self.engine.reset_prefix_cache()
    await self.engine.sleep()
```

After `sleep()`:

- any in-flight generations are aborted (`_abort_requests`),
- KV-cache state is cleared (`reset_prefix_cache`),
- `engine.sleep()` (upstream vLLM, via `memory_saver`) **releases vLLM's GPU memory — weights and KV cache — so FSDP can freely use the GPU for backprop**.

### 6.4 `AsyncLLMServerManager` fans sleep/wake across all replicas

`trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py:575-600`:

```python
def wake_up(self):
    """
    Wake up all vLLM instances from sleep mode.

    This method activates all LLM servers that may have been put into sleep mode
    for resource conservation. It's useful for resuming operations after
    periods of inactivity.

    The wake_up operation is performed synchronously across all servers to ensure
    they are ready before returning control to the caller.
    """
    ray.get([server.wake_up.remote() for server in self.async_llm_servers])

def sleep(self):
    """
    Put all vLLM instances into sleep mode.

    This method puts all LLM servers into a low-power state to conserve resources
    when they are not actively processing requests. This is useful for:
    - Reducing GPU memory usage during idle periods
    - Conserving power in multi-tenant environments
    - Allowing other processes to use GPU resources temporarily

    The sleep operation is performed synchronously across all servers.
    """
    ray.get([server.sleep.remote() for server in self.async_llm_servers])
```

`ray.get([... for server in self.async_llm_servers])` blocks until every vLLM replica has executed the state transition.

### 6.5 FSDP runs the optimizer step in place

See section 5.4 — `update_actor(batch)` at `ray_trainer.py:1231`. Weights mutate in place at the same CUDA addresses that vLLM populated at step 3.5.

### 6.6 `wake_up()` re-activates vLLM — no payload

At the top of the *next* iteration (`ray_trainer.py:1080`):

```python
self.async_rollout_manager.wake_up()
```

propagates into `vllm_async_server.py:239-240`:

```python
async def wake_up(self):
    await self.engine.wake_up()
```

**There is no tensor payload.** No `state_dict`, no `torch.distributed.broadcast`, no file write. vLLM's upstream `engine.wake_up()` re-allocates KV-cache space and resumes using *whatever is in the model tensors right now* — which is exactly what FSDP just finished writing.

### 6.7 The comment in `_load_checkpoint` confirms this design

`trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:938-942`:

```python
def _load_checkpoint(self):
    # Sleep and wake up the async rollout manager: https://github.com/volcengine/verl/issues/2613
    # This syncs weights to vllm server. Also release GPU memory.
    if self.async_rollout_mode:
        self.async_rollout_manager.sleep()
```

The phrase **"This syncs weights to vllm server. Also release GPU memory."** is the codebase's own acknowledgement of the design: the sleep-then-wake dance is the sync mechanism. There is no other mechanism.

### 6.8 `param_offload=True` makes the sharing cheap

`run_proagent_qwn3_4B_instruct.sh:51-52` (repeated from section 2):

```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
```

FSDP's `param_offload=True` allows the policy parameters and optimizer state to be held on CPU when not actively needed. During rollout, they can be offloaded so vLLM has headroom for KV cache; during training, FSDP pulls them back onto GPU. Because *vLLM is asleep* while FSDP trains, the two do not fight over VRAM.

### 6.9 Summary of the handoff (no claim without a citation)

| Claim | Citation |
|---|---|
| vLLM runs in the same Ray actor as FSDP | `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:43-85` |
| vLLM calls `load_model` on that actor via `collective_rpc` | `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:82-84` |
| `enable_sleep_mode=True` is set on the engine | `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:169-171` |
| Trainer wakes vLLM before rollout, sleeps it after | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:1076-1082` |
| `sleep()` aborts + resets + suspends | `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:239-247` |
| The codebase itself calls this "sync" | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:938-942` |
| FSDP `update_actor(batch)` mutates tensors in place | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:1229-1231` |
| `param_offload=True` + `optimizer_offload=True` in launch | `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh:51-52` |

No code in the repository implements a `state_dict()`/`load_state_dict` transfer between a "trainer" and a separate "inference server". The assumption is colocation.

---

## 7. What would break if vLLM moved to a different node

Stay within what the code implies; do not speculate on fixes.

- **`ExternalRayDistributedExecutor` assumes local Ray actor discovery.** `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:57-58` calls `ray.init(address="auto", namespace=namespace)` and then `ray.util.list_named_actors()` with a name-prefix filter. A vLLM process launched on a different node/cluster would see a different actor set — the assertion at line 61 (`len(actor_names) == vllm_dp_size * vllm_tp_size`) would fail during bring-up.
- **`collective_rpc("load_model")` is not a network transfer.** It is an RPC that runs vLLM's loader *inside* the FSDP actor (`trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:82-84`). Run remotely, it would load from whatever local checkpoint the remote process can see — not from FSDP's live tensors.
- **`wake_up()` has no tensor argument.** `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:239-240` is `await self.engine.wake_up()` — zero state transferred. On a remote node, the weights visible to the remote engine after `wake_up` would be whatever was loaded at bring-up, not whatever the trainer just optimized.
- **`sleep()` drops VRAM; it does not upload deltas.** `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py:242-247` aborts requests, resets KV cache, suspends the engine. There is no "push updated weights to the other node" step.
- **ProRL ↔ vLLM HTTP (`/generate`) is already cross-machine-friendly.** `openhands/llm/nvidia/qwen3.py:248-260` is a normal `httpx.post` over the network. Only the *weight* side is coupled; the *inference* side would work fine against a remote vLLM.
- **The implication:** moving vLLM off-node requires adding an explicit weight-materialization channel (e.g. `state_dict` shipping, torch-distributed broadcast on a wider process group, or a checkpoint round-trip). No such code exists today in this repo; the current design would ship stale weights.

---

## 8. Experimental check you can run locally

All commands below are read-only. None start servers or mutate state.

### 8.1 Prove vLLM adopts FSDP's Ray actors

```bash
grep -n "WorkerDict" trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py
```

Expected: two hits, both at lines 56–58 showing the name-prefix lookup. Use a second grep for completeness:

```bash
grep -n "collective_rpc" trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py
```

Expect hits at lines 82-84, proving `init_worker`/`init_device`/`load_model` run on those adopted actors.

### 8.2 Prove vLLM is booted in sleep-mode

```bash
grep -n "enable_sleep_mode" trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py
```

Expect line 171 to be the only hit.

### 8.3 Prove the training loop drives wake_up/sleep each iteration

```bash
grep -n "async_rollout_manager\.\(wake_up\|sleep\)" \
     trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py
```

Expect hits at (≈) lines 942, 1080, 1082 — the `_load_checkpoint` sleep plus the per-iteration wake/sleep.

### 8.4 Prove the "sync by sleep" comment exists

```bash
grep -n "syncs weights" trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py
```

Expect a single hit inside `_load_checkpoint` around line 940.

### 8.5 Prove ProRL → vLLM is token-in / token-out

```bash
grep -n "prompt_ids" openhands/llm/nvidia/qwen3.py
grep -n "prompt_ids" trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py
```

Expect hits in both: the client sends `prompt_ids`, the server reads the same key via `request_dict.pop("prompt_ids")`.

### 8.6 Prove ProRL load-balances with a weighted heap

```bash
grep -n "weighted_addresses" openhands/nvidia/async_server.py
```

Expect several hits: initialisation at ≈ lines 105-106, the lock at ≈ line 111, and reads inside `create_llm_config` used by `process`.

### 8.7 Prove FSDP `param_offload=True`

```bash
grep -n "param_offload" trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh
```

Expect two hits (actor + ref).

### 8.8 Confirm no `state_dict` transfer exists

```bash
grep -rn "state_dict" trainer_integration/verl/verl_custom/ openhands/nvidia/ openhands/llm/nvidia/
```

Any hits should be upstream references or serialization for checkpointing, *not* inter-process weight transfer for inference. Verify that none of the hits shows the trainer sending a `state_dict` to a rollout server; if a future PR adds such a bridge, this grep is where it would land.

---

## 9. File & line index

| File | Lines | Purpose |
|---|---|---|
| `CLAUDE.md` | all | Repo orientation and invariants |
| `trainer_integration/verl/verl_custom/trainer/main_ppo.py` | 27-48 | `@hydra.main`, `ray.init`, `TaskRunner` launch |
| `trainer_integration/verl/verl_custom/trainer/main_ppo.py` | 90-97 | Selects `AsyncActorRolloutRefWorker` when `rollout.mode=async` |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | 880-882 | `actor_rollout_wg.init_model()` |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | 884-902 | `AsyncLLMServerManager` instantiation |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | 938-942 | `_load_checkpoint` — "syncs weights to vllm server" comment |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | 1074-1084 | Per-iteration wake/generate/sleep |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | 1226-1233 | `update_actor(batch)` — FSDP in-place update |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | 105-111 | `weighted_addresses` init + locks (OpenHands-side rollout manager has a mirror; this one is upstream) |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | 575-600 | `AsyncLLMServerManager.wake_up` / `sleep` fan-out |
| `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py` | 43-85 | `ExternalRayDistributedExecutor._init_executor` — colocation |
| `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py` | 169-189 | `AsyncEngineArgs(enable_sleep_mode=True, ...)` |
| `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py` | 198-237 | `generate` endpoint — token-in / token-out |
| `trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py` | 239-247 | `wake_up` / `sleep` on the engine |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 22-27 | Hardware config (`TP_SIZE=2`, `NNODES=1`, etc.) |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 51-52 | `fsdp_config.param_offload=True` + `optimizer_offload=True` |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 58 | `rollout.tensor_model_parallel_size=$TP_SIZE` |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 62 | `rollout.gpu_memory_utilization=$GPU_MEM_UTIL` |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 66 | `rollout.async_manager=openhands` |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 68 | `rollout.enable_memory_saver=True` |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 71 | `rollout.openhands_base_url=http://localhost:8006` |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 86 | `ref.fsdp_config.param_offload=True` |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | 98 | `trainer.n_gpus_per_node=8` |
| `openhands/nvidia/async_server.py` | 59-121 | `OpenHandsServer.__init__` — queues + weighted heap |
| `openhands/nvidia/async_server.py` | 133-164 | `add_llm_server_address` + `create_llm_config` selection |
| `openhands/nvidia/async_server.py` | 263-308 | `process` — job enqueue + wait |
| `openhands/nvidia/registry.py` | 127-154 | Dual registries (`_registries`, `_registries_reasoning`) |
| `openhands/nvidia/registry.py` | 164-187 | `get_registered_functions` / `is_registered_handler` |
| `openhands/nvidia/registry.py` | 172-173 / 185-186 | Hardcoded `deepcoder` prefix special case |
| `openhands/nvidia/registry.py` | 190-205 | `register_agent_handler` |
| `openhands/llm/nvidia/qwen3.py` | 188-228 | `request_response_tokens` — token-in assembly |
| `openhands/llm/nvidia/qwen3.py` | 248-260 | `httpx.post(f'{base_url}/generate', ...)` |
| `scripts/start_server.py` | 550-575 | `POST /add_llm_server` |
| `scripts/start_server.py` | 604-691 | `POST /process` |

### Upstream references (not in this repo)

- `verl.workers.fsdp_workers.AsyncActorRolloutRefWorker` — imported at `main_ppo.py:94`, defines the hybrid FSDP+vLLM worker.
- `vllm.AsyncLLM` — `from_vllm_config` is called at `vllm_async_server.py:196`.
- `vllm` `engine.sleep` / `engine.wake_up` / `memory_saver` — invoked at `vllm_async_server.py:240` and `247`.

Tracking these upstream symbols is outside the scope of a read-only investigation of *this* repo. If you need their behaviour, inspect the respective packages in your Poetry venv with `poetry run python -c "import vllm; print(vllm.__file__)"`.
