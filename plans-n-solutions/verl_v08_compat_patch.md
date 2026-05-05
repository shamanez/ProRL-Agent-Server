# VERL v0.8 Compatibility Patch Notes

## Context

The upstream VERL at `/tmp/verl` is a newer API (v0.8+) that introduced typed
dataclass configs via `omega_conf_to_dataclass`.  Our YAML (`ppo_trainer.yaml`)
was written for the old v0.4 API where config was accessed directly as raw
OmegaConf.  This file documents every breaking change and how it was fixed.

---

## Issues Fixed

### 1. `verl.workers.fsdp_workers` removed → `engine_workers`

**File:** `trainers/verl/verl_custom/workers/fsdp_workers.py`

**Problem:** `AsyncActorRolloutRefWorker` was in `verl.workers.fsdp_workers` in
v0.4; v0.8 renamed the module to `verl.workers.engine_workers` and the class to
`ActorRolloutRefWorker`.

**Fix:** Try import from old location, fall back to new:
```python
try:
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as _UpstreamAsyncWorker
except (ImportError, ModuleNotFoundError):
    from verl.workers.engine_workers import ActorRolloutRefWorker as _UpstreamAsyncWorker
```

---

### 2. `omega_conf_to_dataclass` now requires `_target_` in config

**File:** `trainers/verl/verl_custom/workers/fsdp_workers.py::init_model`

**Problem:** `engine_workers.init_model()` calls
`omega_conf_to_dataclass(self.config.actor)` (and ref/rollout) without a
`dataclass_type`.  The new API requires a `_target_:` field in the YAML
section or raises `AssertionError`.  Our YAML sections have no `_target_`.

**Fix:** Monkey-patch `omega_conf_to_dataclass` inside `init_model` to return
a `_ConfigProxy` (see §6) instead of asserting.

---

### 3. `actor_config.engine` missing (new field in `ActorConfig`)

**File:** `trainers/verl/verl_custom/workers/fsdp_workers.py::_inject_engine_fields`

**Problem:** New `ActorConfig.engine` field is set by `FSDPActorConfig.__post_init__`
via `self.engine = self.fsdp_config`.  Since we return raw OmegaConf (no
`__post_init__`), `.engine` is absent, causing `ConfigAttributeError`.

**Fix:** `_inject_engine_fields()` runs before `super().init_model()`:
- Reads `fsdp_config` from the YAML actor/ref section
- Creates `FSDPEngineConfig(**filtered_kwargs)` (known fields only)
- Converts to `OmegaConf.create(asdict(engine_obj))` (avoids `int=None` validation)
- Injects via `OmegaConf.update(section_cfg, 'engine', engine_cfg, merge=False)`

---

### 4. `HFModelConfig.hf_config` (a `Qwen3Config` object) rejected by OmegaConf

**File:** `trainers/verl/verl_custom/workers/fsdp_workers.py::_patched_omega_conf_to_dataclass`

**Problem:** For model configs (have `path`, no `strategy`), the patch
instantiates `HFModelConfig(**filtered)` which runs `__post_init__` loading
`hf_config = AutoConfig.from_pretrained(...)` (a `Qwen3Config`).  When
`actor_config.model_config = HFModelConfig_instance` is called on OmegaConf,
it tries to wrap the object — but `Qwen3Config` is not a supported primitive
type for `AnyNode`, causing `UnsupportedValueType`.

**Fix:** Return a `_ConfigProxy` for all non-model configs (actor/ref/rollout)
so assignment `actor_config.model_config = HFModelConfig_instance` bypasses
OmegaConf wrapping entirely (plain Python `object.__setattr__`).

---

### 5. `_ConfigProxy` — attribute-access wrapper replacing OmegaConf DictConfig

**File:** `trainers/verl/verl_custom/workers/fsdp_workers.py`

**Problem:** Throughout `engine_workers.init_model`, config sections are
accessed as attribute objects.  Returning raw OmegaConf causes mismatches when
Python objects (HFModelConfig, FSDPEngineConfig) are assigned as values.

**Fix:** `_ConfigProxy` class:
- Converts `OmegaConf.to_container(config)` recursively to Python attributes
- `__setattr__` uses `object.__setattr__` — no OmegaConf wrapping
- `__getattr__` returns `None` for missing attributes (new VERL fields absent
  from our YAML) rather than raising `AttributeError`
- `.get(key, default)` mimics OmegaConf's dict-like interface

---

### 6. Missing YAML fields accessed by `engine_workers` / `vllm_rollout`

**File:** `trainers/verl/verl_custom/trainer/config/ppo_trainer.yaml`

Fields added to the `actor_rollout_ref.rollout` section:

| Field | Default | Used at |
|---|---|---|
| `data_parallel_size` | `1` | `engine_workers.py:591` |
| `pipeline_model_parallel_size` | `1` | `engine_workers.py:591` |
| `expert_parallel_size` | `1` | `vllm_rollout.py:91` |
| `checkpoint_engine.update_weights_bucket_megabytes` | `2048` | `vllm_rollout.py:174` |

---

### 7. `zero_indexed_step` missing from optimizer config

**Addressed by:** `_ConfigProxy.__getattr__` returning `None` (falsy, treated
as step-0 default by the LR scheduler).  No YAML change needed.

---

## GPU Configuration

The machine has **8 × A100-SXM4-40GB** GPUs.  The trainer Docker container
starts with `--gpus all`.

FSDP uses `fsdp_size: -1` which means all visible GPUs → 8-way FSDP sharding
of Qwen3-4B (≈1.9 GB GPU memory per shard at BF16).

For the external vLLM pool (rollout inference), the pool runs on the remote
EC2 host, not on the training machine.  The trainer's `rollout` config section
controls only the **in-trainer log-prob recompute** path (which runs on these
8 GPUs), not the external rollout.

---

## Remaining Watch Points

- Any new `RolloutConfig` or `ActorConfig` field accessed as an integer/bool
  that isn't in our YAML will surface as `TypeError: unsupported operand ...`
  with `NoneType`.  Fix: add the field to `ppo_trainer.yaml` with its
  default value from the VERL dataclass.
- The `optimizer.zero_indexed_step = None` is silently accepted as falsy.
  If the LR scheduler behaves unexpectedly on checkpoint resume, add
  `zero_indexed_step: true` to the `actor.optim` section.
