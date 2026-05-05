# Plugging In a New Trainer or Environment

This guide explains how to add a new trainer (e.g. Slime or ROLL) or a new environment
(e.g. ROCK with a Hermes agent) to the rollout fabric without changing any of the
existing services. An agent reading this should be able to complete the integration
without asking questions.

The fabric is designed so that swapping a trainer or environment requires only:
1. Implementing the service's typed Protocol contract
2. Wiring the correct env vars and Docker mounts
3. Verifying against the smoke test gates

Shared invariants that must never be violated: `plans-n-solutions/rollout_fabric.md`
Protocol definitions: `core/rollout_fabric/schemas/protocols/`

---

## Part 1: Adding a New Trainer

### 1.1 What the fabric expects from any trainer

The fabric makes four non-negotiable demands on every trainer. Violating any one of
them silently corrupts training.

**Connect to LiveStore to receive training data (BC-14, BC-15):**

```python
from rollout_fabric.live_store.client import LiveStoreClient
client = LiveStoreClient(
    socket_path=os.environ['LIVE_STORE_SOCKET'],   # UDS path
    policy_id='qwen3-4b-skyrl',
    environment_id='swe_agent',
)
samples = client.get_batch(n_groups=N, current_step=step, timeout_ms=5_400_000)
```

`get_batch` blocks server-side until N groups are available. The 90-minute timeout
matches the no-progress timeout on the LiveStore. Never use a short timeout here.

**Connect to PolicyRegistry to publish checkpoints (BC-9):**

```python
from rollout_fabric.policy_registry.client import PolicyRegistryClient
registry = PolicyRegistryClient(
    socket_path=os.environ['POLICY_REGISTRY_SOCKET'],  # UDS path
)
registry.publish_policy_version(
    policy_id='qwen3-4b-skyrl',
    version=step,
    adapter_uri='/workspace/outputs/lora_adapters/step_{step}',
)
```

`publish_policy_version` fans out `reload_lora` to all vLLM pool endpoints
synchronously and raises `PublishFailedError` if any endpoint fails (BC-9).
The trainer must abort on this error — never catch and continue.

**Do NOT receive or use: parquet paths, ProRL URL, vLLM URLs (BC-14, BC-15):**

The trainer has no `data.train_files`, no producer thread, no HTTP connection to
ProRL or to vLLM. If you find yourself passing these to a new trainer, stop.

**Pad locally after `get_batch` (BC-11):**

The LiveStore wire is unpadded. Every trainer must call its own pad function after
`get_batch` returns. The store returns `list[TrainingSample]` with variable-length
token arrays.

**Token IDs are `list[int]`, never `list[str]` (BC-1):**

Do not decode token IDs to text and re-tokenize. Any re-tokenization shifts token
boundaries, causing actor vs. reference log-prob divergence and NaN loss within 2
steps.

### 1.2 Directory structure to create

```
trainers/{name}/
├── pyproject.toml              <- installable package (pip install -e inside Docker)
├── {name}_custom/
│   └── fabric_adapter/
│       ├── __init__.py
│       ├── pad.py              <- pack list[TrainingSample] → trainer's tensor format
│       └── live_store_batch.py <- call get_batch() + pad
├── scripts/
│   └── start.sh               <- docker run with correct env vars
└── README.md
```

### 1.3 The `pad.py` contract

`pad.py` must implement a function that converts `list[TrainingSample]` to the
trainer's native batch format. The VERL reference implementation is:
`trainers/verl/verl_custom/fabric_adapter/pad.py`

Minimum required fields from each `TrainingSample`:

```python
from rollout_fabric.schemas.training_sample import TrainingSample

@dataclass
class TrainingSample:
    group_uid:               str        # ties N siblings together (BC-0)
    sample_uid:              str        # per-row identifier
    prompt_token_ids:        list[int]  # BC-1: int only, variable length
    response_token_ids:      list[int]  # BC-1: int only, variable length
    response_loss_mask:      list[int]  # 1 where loss computed, 0 on pad/tool tokens
    behavior_log_probs:      list[float] # rollout log probs (for KL / IS correction)
    reward:                  float
    raw_reward:              float
    behavior_policy_version: int        # version at dispatch time (BC-7, BC-8)
    created_at_step:         int        # trainer step when dispatched (BC-8)
    truncated:               bool
    error:                   str | None
    instance:                dict       # task metadata
    is_padded:               bool       # True if this is a padding row
```

The wire is unpadded (BC-11). Your `pad.py` must:
1. Determine the max prompt and response lengths across the batch (or use a fixed cap)
2. Left-pad prompt IDs (so the last token aligns at position `max_prompt - 1`)
3. Right-pad response IDs with `pad_token_id`
4. Build attention masks and position IDs from padding positions
5. Raise `RuntimeError` if any sequence exceeds the cap — never silently truncate

Reference signature from VERL:

```python
def pack_unpadded_groups(
    samples: Sequence[TrainingSample],
    *,
    pad_token_id: int,
    prompt_length_cap: int | None = None,
    response_length_cap: int | None = None,
    current_step: int = 0,
) -> SampledMiniBatch:
    ...
```

The `SampledMiniBatch` returned by VERL's implementation contains:
- `tensors`: `input_ids`, `responses`, `attention_mask`, `position_ids`, `loss_mask`,
  `rollout_log_probs`, `is_padded`, `error_mask`, `reward`, `raw_reward`, `truncated`
- `non_tensors`: `uid`, `success`, `error`, `resolved`, `finish`, `instance`, `ability`
- `meta_info`: `behavior_policy_versions`, `created_at_steps`, `sample_ages`

Adapt the tensor keys and shapes to match your trainer's expected format.

### 1.4 The `live_store_batch.py` contract

`live_store_batch.py` is the single call site that bridges `LiveStoreClient` and
your `pad.py`. The VERL reference is:
`trainers/verl/verl_custom/fabric_adapter/live_store_batch.py`

```python
# trainers/{name}/{name}_custom/fabric_adapter/live_store_batch.py
from rollout_fabric.live_store.client import LiveStoreClient
from {name}_custom.fabric_adapter.pad import pack_unpadded_groups

def sample_mini_batch(client: LiveStoreClient, n_groups: int, current_step: int):
    """Call get_batch then pack into trainer's batch format."""
    samples = client.get_batch(
        n_groups=n_groups,
        current_step=current_step,
        timeout_ms=5_400_000,   # 90 min — matches no_progress_timeout_s=5400
    )
    return pack_unpadded_groups(
        samples,
        pad_token_id=client._pad_token_id,
        prompt_length_cap=client._prompt_cap,
        response_length_cap=client._response_cap,
        current_step=current_step,
    )
```

Call this function from your training loop wherever you currently read from a
dataset or replay buffer.

### 1.5 The Docker start script

The start script must set the env vars the trainer needs to find the UDS sockets.
Failure to set `LIVE_STORE_SOCKET` causes `NoneType is not callable` on the first
`get_batch`.

Minimum required in `docker run`:

```bash
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  -v /workspace:/workspace \
  -v /tmp:/tmp \                          # CRITICAL: UDS sockets live in /tmp on the host
  -e LIVE_STORE_SOCKET=/tmp/prorl_live_store.sock \
  -e POLICY_REGISTRY_SOCKET=/tmp/prorl_policy_registry.sock \
  -e PYTHONPATH=/workspace/core \         # makes rollout_fabric importable
  -e WANDB_API_KEY \
  -e HF_TOKEN \
  {your_image} \
  bash -c '
    pip install --no-deps -e /workspace/trainers/{name}
    # your training launch command here
  '
```

The `-v /tmp:/tmp` mount is load-bearing: the UDS sockets are created by the host
LiveStore and PolicyRegistry services under `/tmp/`. Without this mount, the Docker
container cannot reach them.

### 1.6 Publishing policy versions

After each checkpoint save, call `publish_policy_version`. This fans out `reload_lora`
to all vLLM endpoints via the PolicyRegistry and updates the manifest that the
RolloutManager reads for the next group dispatch:

```python
import os
from rollout_fabric.policy_registry.client import PolicyRegistryClient, PublishFailedError

registry = PolicyRegistryClient(os.environ['POLICY_REGISTRY_SOCKET'])

# After saving checkpoint at step N:
lora_path = f'/workspace/outputs/lora_adapters/step_{step}'
try:
    metrics = registry.publish_policy_version(
        policy_id=os.environ.get('POLICY_ID', 'qwen3-4b-skyrl'),
        version=step,
        adapter_uri=lora_path,
    )
    # metrics contains: weight_sync/policy_version, weight_sync/publish_latency_s,
    #                   weight_sync/endpoints_ok, weight_sync/endpoints_failed
except PublishFailedError as exc:
    # BC-9: hard abort — endpoints_failed > 0
    logger.error('ABORT: %s', exc)
    raise SystemExit(1)
```

Never catch `PublishFailedError` and continue. A warm replay buffer must not mask a
broken vLLM pool (BC-9).

---

## Part 2: Specific Notes for Slime

### What Slime is

Slime (THUDM/slime) is an LLM post-training framework that connects Megatron with
SGLang. It uses a Ray-based actor model with a `RolloutManager` that generates data
into a `DataBuffer`. The training loop in `train_async.py` is:

```
rollout_manager.generate() → data_buffer → actor_model.async_train()
```

Slime also exposes a `slime_plugins/rollout_buffer/` FastAPI server that provides an
external HTTP buffer endpoint (`POST /get_rollout_data`). This is the correct injection
point for the fabric.

### Where to inject the LiveStore call in Slime

Slime supports custom rollout functions via `--rollout-function-path`. In the external
buffer plugin (`slime_plugins/rollout_buffer/rollout_buffer_example.py`), the function
`generate_rollout()` is called by the Slime rollout server to produce training samples.

**Option A (recommended): replace the rollout function**

Create a custom rollout function that calls `LiveStoreClient.get_batch()` instead of
generating rollouts inline:

```python
# trainers/slime/slime_custom/fabric_adapter/slime_rollout_fn.py
import os
from rollout_fabric.live_store.client import LiveStoreClient
from slime_custom.fabric_adapter.pad import pack_for_slime

_client = None

def get_live_store_client():
    global _client
    if _client is None:
        _client = LiveStoreClient(
            os.environ['LIVE_STORE_SOCKET'],
            policy_id=os.environ.get('POLICY_ID', 'qwen3-4b-skyrl'),
            environment_id=os.environ.get('ENVIRONMENT_ID', 'swe_agent'),
        )
    return _client

def generate_rollout(args, server_group, batch_data, **kwargs):
    """Custom rollout function — reads from LiveStore instead of generating inline."""
    client = get_live_store_client()
    n_groups = args.rollout_batch_size // args.group_size
    current_step = kwargs.get('rollout_id', 0)
    samples = client.get_batch(
        n_groups=n_groups,
        current_step=current_step,
        timeout_ms=5_400_000,
    )
    return pack_for_slime(samples)  # convert to Slime's Sample list format
```

Pass this as: `--rollout-function-path slime_custom.fabric_adapter.slime_rollout_fn.generate_rollout`

**What Slime's `Sample` format expects:**

Slime's `Sample` type (in `slime/utils/types.py`) is a dict-like with keys including
`input_ids`, `output_ids`, `rewards`, `loss_mask`. The `pack_for_slime` function must
map `TrainingSample.prompt_token_ids + response_token_ids` to these keys.

**Policy version publish after Slime saves checkpoint:**

Slime calls `actor_model.save_model(rollout_id)` after each save interval. Hook into
the post-save callback:

```python
# In your custom actor model wrapper:
def save_model(self, rollout_id, **kwargs):
    super().save_model(rollout_id, **kwargs)
    lora_path = os.path.join(self.args.save, f'step_{rollout_id}')
    registry = PolicyRegistryClient(os.environ['POLICY_REGISTRY_SOCKET'])
    registry.publish_policy_version(
        policy_id=os.environ.get('POLICY_ID', 'qwen3-4b-skyrl'),
        version=rollout_id,
        adapter_uri=lora_path,
    )
```

### Slime training launch

Slime launches via `python train_async.py` (Ray-based) or `python train.py` (synchronous).
The Docker entry point for a Slime trainer should be:

```bash
bash -c '
  pip install --no-deps -e /workspace/trainers/slime
  torchrun --nproc_per_node=8 /workspace/trainers/slime/train_entry.py \
    --rollout-function-path slime_custom.fabric_adapter.slime_rollout_fn.generate_rollout \
    --hf-checkpoint /root/.cache/huggingface/Qwen3-4B \
    ...other Megatron/slime args...
'
```

Slime is Megatron-based, so the Docker image must contain Megatron-LM and SGLang.
The `PYTHONPATH=/workspace/core` must be set so `rollout_fabric` is importable.

---

## Part 3: Specific Notes for ROLL

### What ROLL is

ROLL (alibaba/ROLL) is an RL library that uses Ray with Megatron-Core or DeepSpeed
as the training backend and vLLM/SGLang for inference. Its training loop is
pipeline-based:

```
RLVRPipeline.run() → DynamicSamplingScheduler.get_batch() → actor_train.update()
```

ROLL's `DataProto` is its tensor-batch protocol (similar to VERL's DataProto).

### Where to inject the LiveStore call in ROLL

ROLL's rollout data flows through `DynamicSamplingScheduler.get_batch()`. The
injection point is the pipeline's `run()` loop, where `generate_output = scheduler.get_batch.remote(...)`.

**Recommended approach:** Create a custom pipeline subclass that overrides data
acquisition:

```python
# trainers/roll/roll_custom/fabric_adapter/live_store_pipeline.py
import os
import ray
from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
from roll.distributed.scheduler.protocol import DataProto
from rollout_fabric.live_store.client import LiveStoreClient
from roll_custom.fabric_adapter.pad import pack_for_roll

class LiveStorePipeline(RLVRPipeline):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self._live_store = LiveStoreClient(
            os.environ['LIVE_STORE_SOCKET'],
            policy_id=os.environ.get('POLICY_ID', 'qwen3-4b-skyrl'),
            environment_id=os.environ.get('ENVIRONMENT_ID', 'swe_agent'),
        )

    def _get_rollout_batch(self, global_step: int, batch_size: int) -> DataProto:
        """Override ROLL's scheduler.get_batch with LiveStore.get_batch."""
        n_groups = batch_size // self.pipeline_config.group_size
        samples = self._live_store.get_batch(
            n_groups=n_groups,
            current_step=global_step,
            timeout_ms=5_400_000,
        )
        return pack_for_roll(samples, self.tokenizer)
```

**What ROLL's `DataProto` format expects:**

ROLL's `DataProto` holds `batch` (a `TensorDict`) and `non_tensor_batch` (a dict of
numpy arrays). At minimum, the training worker needs `input_ids`, `attention_mask`,
`position_ids`, `responses`, `loss_mask`, `rollout_log_probs`, and reward fields.
Map `TrainingSample` fields to these keys in `pack_for_roll`.

**Policy version publish after ROLL saves checkpoint:**

ROLL calls `do_checkpoint()` in `BasePipeline`. Override it:

```python
def do_checkpoint(self, global_step, is_last_step=None):
    super().do_checkpoint(global_step, is_last_step)
    lora_path = os.path.join(self.pipeline_config.output_dir, f'checkpoint-{global_step}', 'actor')
    from rollout_fabric.policy_registry.client import PolicyRegistryClient, PublishFailedError
    registry = PolicyRegistryClient(os.environ['POLICY_REGISTRY_SOCKET'])
    try:
        registry.publish_policy_version(
            policy_id=os.environ.get('POLICY_ID', 'qwen3-4b-skyrl'),
            version=global_step,
            adapter_uri=lora_path,
        )
    except PublishFailedError as exc:
        raise SystemExit(1) from exc
```

### ROLL training launch

ROLL is launched via `examples/start_rlvr_rollout_pipeline.py` with a Hydra config
YAML. The Docker entry point for a ROLL trainer:

```bash
bash -c '
  pip install --no-deps -e /workspace/trainers/roll
  python /workspace/trainers/roll/examples/start_live_store_pipeline.py \
    --config_path config \
    --config_name your_config
'
```

The ROLL Docker image must contain Megatron-Core or DeepSpeed and the ROLL package.
Set `PYTHONPATH=/workspace/core` so `rollout_fabric` is importable.

---

## Part 4: Adding a New Environment

### 4.1 What the fabric expects from any environment

The environment must expose three HTTP endpoints, run independently of the fabric's
internal services, and never call LiveStore or PolicyRegistry.

**Required endpoints:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/process` | POST | Run one episode; returns messages, reward, resolved |
| `/start` | POST | Initialize or reset the server |
| `/status` | GET | Health check; must return JSON with `status` field |

**BC-15 boundary:** The environment must not import or call `LiveStoreClient` or
`PolicyRegistryClient`. It receives task instances from the RolloutManager and
returns episodes to the RolloutManager. No other connections.

**BC-1 requirement:** All token arrays in the response must be `list[int]`.
Never return decoded text strings in `token_ids` fields.

### 4.2 Directory structure to create

```
environments/{name}/
├── pyproject.toml            <- installable package
├── {name}/                   <- environment Python package
│   ├── __init__.py
│   ├── server.py             <- FastAPI app with /process, /start, /status
│   ├── agent_handler.py      <- implements the agent that runs episodes
│   └── reward.py             <- reward computation
├── scripts/
│   ├── start.sh              <- launcher script
│   └── start_server.py      <- Python entry point called by start.sh
└── README.md
```

### 4.3 The `/process` API contract

The RolloutManager calls `POST /process` (in `core/rollout_fabric/rollout_manager/prorl_client.py`).

Request body (JSON):

```json
{
  "instance": {
    "task_id": "django__django-11099",
    "policy_version": 42,
    "repo": "...",
    "... other task fields ...": "..."
  },
  "sampling_params": {
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

Response body (JSON):

```json
{
  "messages": [
    {
      "role": "user",
      "content": "...",
      "token_ids": [1234, 5678, 9012],
      "logprobs": null
    },
    {
      "role": "assistant",
      "content": "...",
      "token_ids": [3456, 7890],
      "logprobs": [-0.12, -0.34]
    }
  ],
  "resolved": true,
  "reward": 1.0,
  "raw_reward": 1.0,
  "error": null,
  "truncated": false
}
```

Critical: `token_ids` must be `list[int]`. `logprobs` must be `list[float]` for
assistant turns (used for IS correction). The `resolved` field is the binary task
success signal.

The environment must call vLLM for token generation. The vLLM endpoints are passed
at startup (via `--llm-server-address` flags in the ProRL reference implementation
or via env var). Do not hardcode them.

### 4.4 Token-in / token-out invariant (BC-1 — DO NOT VIOLATE)

The vLLM endpoints in the pool accept and return token IDs. The environment must
pass token IDs directly to vLLM — never decode to text and re-tokenize between turns.
Re-tokenization shifts token boundaries across turns, causing actor vs. reference
log-prob divergence. Loss NaN appears within 2 training steps.

Reference implementation (frozen — never edit these files):
- `environments/prorl_openhands/openhands/llm/nvidia/qwen3.py`
- `environments/prorl_openhands/openhands/llm/nvidia/qwen2_5_vl.py`

If you need a new model, create a sibling file (e.g. `qwen4.py`) in the same
directory. Never edit the frozen files.

### 4.5 The `/status` and `/start` endpoints

```python
# Minimum implementation:
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class StatusResponse(BaseModel):
    status: str   # "running" when ready

@app.get("/status")
def status() -> StatusResponse:
    return StatusResponse(status="running" if _is_running else "starting")

@app.post("/start")
def start():
    # Initialize workers, connect to vLLM, etc.
    global _is_running
    _is_running = True
    return {"ok": True}
```

The RolloutManager waits for `GET /status` to return `{"status": "running"}` before
sending any `POST /process` requests.

### 4.6 The start script

```bash
#!/bin/bash
# environments/{name}/scripts/start.sh
set -eo pipefail
source /home/ubuntu/.prorl_creds.env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Build vLLM endpoint args from REMOTE_DNS:
VLLM_ADDR_ARGS=()
if [[ -n "${REMOTE_DNS:-}" ]]; then
  VLLM_ADDR_ARGS=(--llm-server-address \
    "http://${REMOTE_DNS}:8100" \
    "http://${REMOTE_DNS}:8101" \
    "http://${REMOTE_DNS}:8102" \
    "http://${REMOTE_DNS}:8103")
fi

_DEFAULT_PYTHON="$(cd "${REPO_ROOT}/environments/{name}" && poetry env info --path 2>/dev/null || true)/bin/python"
PYTHON="${PRORL_ENVIRONMENT_PYTHON:-${_DEFAULT_PYTHON}}"

PYTHONPATH="${REPO_ROOT}/environments/{name}:${REPO_ROOT}/core:${PYTHONPATH:-}" \
"${PYTHON}" "${REPO_ROOT}/environments/{name}/scripts/start_server.py" \
  --host 0.0.0.0 --port 8006 \
  "${VLLM_ADDR_ARGS[@]}" \
  2>&1 | tee /tmp/s0-env.log
```

Then start it the same way as the ProRL reference:

```bash
nohup bash environments/{name}/scripts/start.sh > /tmp/s0-env.log 2>&1 &
echo $! > /tmp/env.pid
```

To switch to your new environment, set in `start_all.sh` or call `start_env_provider.sh`
with `ENVIRONMENT_ID={name}`.

---

## Part 5: Specific Notes for ROCK

ROCK (`alibaba/ROCK`) is described in ROLL's release notes as a reinforcement open
construction kit environment. Based on ROLL's architecture:

- ROCK exposes agentic multi-turn environments (tool use, games, construction tasks)
- It uses ROLL's GEM environment definition and aligns with ROLL's `DynamicSamplingScheduler`
- For integration with this fabric, ROCK would implement the `/process` HTTP endpoint
  as the bridge between the fabric's RolloutManager and ROCK's episode runner

**Hermes agent wiring:**

If ROCK's Hermes agent generates multi-turn trajectories, each turn must:
1. Send the current token context to vLLM via `POST /v{version}/generate`
   (where `{version}` is the pinned policy version from the episode snapshot)
2. Receive token IDs back (not text)
3. Append to the token array and pass back to the agent for the next turn

The `POST /process` handler coordinates this loop and returns the complete token
arrays at the end of the episode.

**Token pinning (BC-10):** All turns within one episode must use the same vLLM
endpoint version path. The policy version is read once at episode dispatch (BC-0)
and held for the entire episode.

---

## Part 6: Testing Your Integration

### 6.1 Smoke test

After starting all services including your new trainer or environment, run:

```bash
source /home/ubuntu/.prorl_creds.env
PYTHONPATH=./core bash tests/harness/smoke_test.sh
```

All 6 gates must pass:
- GATE-1: All service health probes pass
- GATE-2: LiveStore has ≥1 group (BC-16 warm-up — your environment generated one)
- GATE-3: Token IDs are `list[int]` (BC-1 — your environment sent valid tokens)
- GATE-4: `group_uid` consistent across siblings (BC-0 — your group dispatch is atomic)
- GATE-5: `behavior_policy_version` is `int` (BC-7 — your trainer or registry published a version)
- GATE-6: PolicyRegistry reachable (BC-9 — your trainer can publish)

### 6.2 Fast unit tests

```bash
PYTHONPATH=./core pytest tests/invariants/ tests/contracts/ -q
# Runs in ~1s, no real services needed
```

### 6.3 Contract boundary check

Import boundary violations show up as import errors. Verify:

```bash
# New trainer: must NOT import openhands or parquet loaders
PYTHONPATH=./core python -c "
import importlib
m = importlib.import_module('trainers.{name}.{name}_custom.fabric_adapter.live_store_batch')
print('Import boundary OK')
"

# New environment: must NOT import live_store or policy_registry
PYTHONPATH=./core python -c "
import importlib
m = importlib.import_module('environments.{name}.{name}.server')
print('Import boundary OK')
"
```

### 6.4 BC table reference

The full boundary condition table with failure signatures for every BC:

`plans-n-solutions/rollout_fabric.md` — Section "Boundary Conditions"

| BC | Boundary | Failure signature |
|---|---|---|
| BC-0 | One PolicyVersionSnapshot per group | Siblings see different versions → NaN loss |
| BC-1 | Token IDs as `int` on every wire | KL/entropy NaN within 2 training steps |
| BC-9 | `endpoints_failed > 0` = hard abort | Mixed-version pool → IS weights become lies |
| BC-11 | LiveStore wire is unpadded; trainer pads locally | `torch.stack` shape mismatch on step 1 |
| BC-13 | RolloutManager imports zero VERL/OpenHands code | Trainer can't be swapped |
| BC-14 | RolloutManager owns the parquet dataloader | Trainer becomes hidden orchestrator |
| BC-15 | Trainer connects only to LiveStore and PolicyRegistry | Trainer couples to environment |

---

## Part 7: Import Boundary Summary

| Service | May import | Must NOT import |
|---|---|---|
| RolloutManager | `httpx`, `grpcio`, `pyarrow` | `openhands`, `verl`, `torch`, `slime`, `roll` |
| TrainerAdapter | `torch`, VERL/slime/ROLL, `grpcio` | `openhands`, parquet loaders |
| EnvironmentProvider | `openhands`/your agent, `fastapi`, `httpx` | `rollout_fabric.live_store`, `rollout_fabric.policy_registry` |
| InferenceBackend | `vllm`/`sglang`, `fastapi` | training code, `openhands`, `torch` gradient ops |
