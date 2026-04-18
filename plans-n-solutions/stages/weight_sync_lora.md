# Weight sync — LoRA first

**Status: NOT STARTED.** The baseline ([`baseline.md`](./baseline.md)) ships the decoupled topology with stale weights; this is the first doc that closes the staleness gap.

This branch owns **Phase 1 only**. Phases 2 and 3 are scoped here for orientation but are out of scope until Phase 1 lands.

| Phase | What | Scope of this branch |
|---|---|---|
| **1 — LoRA publish** (this branch) | Trainer trains a low-rank adapter; after every `save_freq` steps it publishes the adapter to each pool endpoint; pool calls `AsyncLLMEngine.add_lora(...)` and serves subsequent rollouts against `{base_weights + adapter}`. | **In scope.** |
| 2 — Full state-dict publish | Same publish protocol, payload is the full HF checkpoint shards. Pool swaps base weights (restart / `collective_rpc('update_weight', ...)` / warm-spawn). Needed once LoRA capacity is the bottleneck. | **Deferred.** |
| 3 — Replay buffer | Trajectory store + bounded sampler so inference runs continuously instead of lock-step with training. Orthogonal to how weights are published. | **Deferred.** |

---

## Why LoRA first

- **Publish payload is ~20–80 MiB** (rank-16 adapter for Qwen3-4B), vs ~8 GiB for the full state-dict. Every publish fits in a single HTTP request body; no presigned-URL hand-off, no shared disk.
- **Pool-side swap is in-place.** `AsyncLLMEngine.add_lora(LoRARequest(...))` installs the adapter alongside the base weights without unloading the engine or dropping in-flight requests. `remove_lora(lora_int_id)` retires the old version. No warm-spawn, no restart, no mixed-version window.
- **vLLM 0.18 API is present** (verified against `verlai/verl:vllm018.dev1`):
  - `LoRARequest(lora_name: str, lora_int_id: int, lora_path: str, base_model_name: str | None, tensorizer_config_dict: dict | None, load_inplace: bool)`
  - `AsyncLLMEngine.{add_lora, remove_lora, list_loras, pin_lora}`
- **verl already supports LoRA actor training.** `actor_rollout_ref.model.lora_rank=N` (+ `lora_alpha`, `lora_target_modules`) is a known config path; no trainer-side surgery is needed to turn it on.
- **Small blast radius.** If the protocol is wrong, the worst case is a pool serving the base model (known-good). Base weights are never mutated on the pool.

The unknowns LoRA defers (full state-dict streaming, base-weight hot-swap, merged-adapter checkpoints) are Phase 2 problems. This phase establishes the publish loop and the policy-version stamping that everything downstream reuses.

---

## Phase 1 — design sketch

### Trainer side

- Enable LoRA on the actor: `actor_rollout_ref.model.lora_rank=16`, `actor_rollout_ref.model.lora_alpha=32`, `actor_rollout_ref.model.lora_target_modules=[all-linear or explicit q/k/v/o/mlp list]`. Reference model stays full-rank (policy reference for KL).
- After `_save_checkpoint()` runs, call a new `_publish_lora_adapter(checkpoint_dir)`:
  1. Extract the adapter shards from the FSDP checkpoint (PEFT-style `adapter_model.safetensors` + `adapter_config.json`).
  2. Bump `policy_version` (trainer-owned counter).
  3. Fan out `POST /reload_lora` to every endpoint in `external_llm_endpoints` with either the raw adapter bytes or a presigned URL. Block the next training step until all endpoints return 200.
  4. On any endpoint failure, abort the run with a loud marker — mixed-version batches are a correctness bug, not a warning.
- Stamp `meta.policy_version` on every rollout job dispatched to the pool. Used for observability now; becomes the replay buffer's freshness key in Phase 3.

### Pool side

- New endpoint `POST /reload_lora` on `_vllm_child.py`:
  - Body: `{adapter_url: str, policy_version: int}` (or raw bytes for small ranks).
  - Download / write to local path → `LoRARequest(lora_name=f"pv{version}", lora_int_id=version, lora_path=<local>)` → `engine.add_lora(...)`.
  - After success, retire the previous adapter via `engine.remove_lora(old_int_id)`.
  - Return `200 {policy_version}` on success. Return `5xx` with a specific error string on failure.
- `_vllm_child.py` `POST /generate` must pass the currently-pinned `LoRARequest` into the sampler so served rollouts are against `{base + adapter}`. Without this wire-up the adapter is loaded but unused.
- `launch_remote_vllm_pool.sh publish <adapter_dir>` subcommand fans `/reload_lora` to all 4 children — lets the publish protocol be tested manually before the trainer-side hook lands.

### Policy-version ownership

Trainer-authoritative. The trainer mints the monotonic counter; each pool endpoint echoes back the version it installed. Pool-side never increments on its own. Keeps the single-writer discipline that makes mixed-version detection trivial (`min(meta.policy_version) == max(meta.policy_version)` per batch).

### Failure modes to surface, not hide

- **Partial reload:** one endpoint 200s, another 5xx → abort the run. Do not retry silently, do not let the batch contain mixed versions.
- **Adapter shape mismatch:** rank / target-module set changed between publish calls. The LoRA framework refuses the load; surface the engine error unmodified.
- **Pool restarted mid-run:** lost adapter state. Next publish reinstalls; in the meantime, rollouts served against base weights are stamped `policy_version=0` (the "adapter absent" sentinel) and the training step aborts on mixed-version detection.

---

## Phase 1 — files that change

| Path | Edit |
|---|---|
| `scripts/serving/_vllm_child.py` | Add `POST /reload_lora`; thread the active `LoRARequest` into `/generate`. |
| `scripts/serving/launch_remote_vllm_pool.sh` | Add `publish <adapter_dir>` subcommand that fans `/reload_lora` to all 4 children. |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | New `_publish_lora_adapter(checkpoint_dir)` called at the end of `_save_checkpoint`; fan-out over `self.config.actor_rollout_ref.rollout.external_llm_endpoints`. |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | Stamp `policy_version` on each job before dispatch to ProRL. |
| New: `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh` | Sibling of the frozen `run_proagent_qwn3_4B_instruct_remote_decoupled.sh`. Adds `model.lora_rank=16`, `publish_on_save=True`, `save_freq=5`, `trainer.experiment_name=weight-sync-decup-prorl`. |
| New: `scripts/_internal/s2_weightsync_docker.sh` | Sibling of the frozen `s1_remote_docker.sh`. |

Frozen-file rule: the baseline launchers (`s1_remote_docker.sh`, `run_proagent_qwn3_4B_instruct_remote_decoupled.sh`) are the reproduction artifacts for the baseline run and are not edited — new siblings only.

---

## Phase 1 — gates

| # | Gate | Pass |
|---|---|---|
| 1 | ≥ 4 successful `/reload_lora` across 20 steps at `save_freq=5` | All 4 pool children return 200 with monotonically increasing `policy_version` |
| 2 | Zero mixed-version batches | `min(meta.policy_version) == max(meta.policy_version)` per batch, every batch |
| 3 | `rollout_corr/kl` drops on the step following each publish | Visible dip in WandB vs the pre-publish baseline |
| 4 | `critic/rewards/mean` trends up (baseline `wdqqu52k` was flat until step 7) | Monotonic improvement across the 20 steps, modulo noise |
| 5 | Zero `POST /generate` errors during swap | Pool child logs show a clean `add_lora` → `remove_lora` hand-off; no 5xx spike |
| 6 | Baseline invariants still hold | `EXTERNAL BYPASS ACTIVE` present, zero Ray vLLM actors, all 8 local GPUs hold FSDP shards |

---

## WandB

`WANDB_API_KEY` is pinned in `/home/ubuntu/.prorl_creds.env`, sourced by both the host ProRL launcher and the docker trainer launcher.

| Field | Value |
|---|---|
| `trainer.project_name` | `ProAgent` (umbrella, unchanged) |
| `trainer.experiment_name` | `weight-sync-decup-prorl` |

New metrics to log (names inside the existing buckets):

| Key | Meaning |
|---|---|
| `weight_sync/policy_version` | Current trainer-minted version |
| `weight_sync/publish_latency_s` | Wall clock of the fan-out `/reload_lora` round |
| `weight_sync/endpoints_ok` | Count of endpoints returning 200 on the last publish (expected: `len(endpoints)`) |
| `rollout/staleness_steps` | `trainer.global_step - meta.policy_version` averaged across the batch |

---

## Entry points for the next session

1. Read this doc, then [`baseline.md`](./baseline.md) §Architecture for the HTTP-topology primer and the "Rollout time stats" table the new protocol must not regress.
2. Re-confirm the vLLM 0.18 LoRA surface against the installed image:
   ```bash
   docker run --rm verlai/verl:vllm018.dev1 python3 -c "
   from vllm.lora.request import LoRARequest
   from vllm import AsyncLLMEngine
   print(LoRARequest.__annotations__)
   print([m for m in dir(AsyncLLMEngine) if 'lora' in m.lower()])
   "
   ```
   Expected: `add_lora`, `remove_lora`, `list_loras`, `pin_lora`. Last-verified 2026-04-18.
3. Confirm verl's LoRA config path against `/tmp/verl` — grep for `lora_rank` in `trainer_integration/verl/verl_custom/**` and the upstream `verl/trainer/ppo/**`.
4. Enter plan mode and draft the publish protocol (trainer + pool side) before touching code.
