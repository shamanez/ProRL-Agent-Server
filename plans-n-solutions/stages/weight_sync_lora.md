# Weight sync — LoRA first

**Status: NOT STARTED.** The baseline ([`baseline.md`](./baseline.md)) ships the decoupled topology with stale weights. This doc closes the staleness gap — LoRA first.

---

## 1 — The core problem we're solving

In the baseline topology the trainer owns the actor/reference FSDP shards; the remote vLLM pool holds an **independent copy** of the base weights loaded at `start`. GRPO's policy updates mutate the trainer's copy every step, but the pool's copy never changes. By training step N the pool is N steps stale.

The symptom shows up as **importance-ratio drift** in the GRPO objective. The loss is

```
pg_loss = -E [ min( r(θ) · A,  clip(r(θ), 1−ε, 1+ε) · A ) ]
r(θ)    = π_θ(a|s) / π_behavior(a|s)
```

When `π_behavior` is the *pool's* stale policy and `π_θ` is the *trainer's* fresh policy, `r(θ)` diverges from 1 step-by-step. verl logs the diagnostic as `rollout_corr/ppl_ratio` — on the baseline run (`wdqqu52k`) it sat near 1.6 and grew; at `clip_ratio=0.2` the clip fraction rises and the gradient signal collapses.

The GRPO group-advantage normalization tolerates moderate staleness (that's why the baseline still produced a clean step-7 gradient) but the drift is monotonic and will eventually make training degenerate. **Weight sync closes the loop:** after every `save_freq` training steps, push the trainer's updated policy to every pool endpoint so `π_behavior ≈ π_θ` on the *next* batch.

### Why LoRA first

| | Full state-dict (Phase 2, deferred) | LoRA adapter (Phase 1, this branch) |
|---|---|---|
| Payload size for Qwen3-4B | ~8 GiB bf16 shards | ~20–80 MiB for rank-16 |
| Transport | presigned URL or chunked HTTP, needs shared disk or S3 | single HTTP body, no shared disk |
| Pool-side swap | restart / warm-spawn / `collective_rpc('update_weight', ...)` | `AsyncLLMEngine.add_lora(LoRARequest(...))` — in-place, no drops |
| Blast radius on failure | pool unusable until reload completes | worst case: pool serves base model (known-good) |
| verl support | partial (requires new publish protocol) | already present via `actor_rollout_ref.model.lora_rank=N` |
| Expressiveness | any update | rank-capped — adequate for moderate-step RL, not for full-pretrain-scale drift |

LoRA establishes the publish loop, policy-version stamping, mixed-batch detection, and WandB staleness metrics — every invariant Phase 2 will need. Once LoRA capacity is the bottleneck (measurably regressing reward), Phase 2 swaps the payload while reusing the protocol. Phase 3 (replay buffer) is orthogonal — it changes *what* the trainer consumes, not *how* weights are published.

**vLLM 0.18 LoRA API** (verified against `verlai/verl:vllm018.dev1` on 2026-04-18):
- `LoRARequest(lora_name: str, lora_int_id: int, lora_path: str, base_model_name: str | None, tensorizer_config_dict: dict | None, load_inplace: bool)`
- `AsyncLLMEngine.{add_lora, remove_lora, list_loras, pin_lora}`

---

## 2 — Phase scope on this branch

| Phase | What | Scope |
|---|---|---|
| **1 — LoRA publish** | Trainer trains a low-rank adapter; publishes after every `save_freq` steps; pool calls `AsyncLLMEngine.add_lora(...)` and serves subsequent rollouts against `{base + adapter}`. | **This branch.** |
| 2 — Full state-dict publish | Same publish protocol, payload is the full HF checkpoint shards. | Deferred. |
| 3 — Replay buffer | Trajectory store + bounded sampler so inference runs continuously instead of lock-step with training. | Deferred. |

---

## 3 — Fresh-box setup (run once per machine)

Everything the Phase 1 run depends on. Skip the steps where the artifact already exists.

### 3.1 — Credentials

`WANDB_API_KEY`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` are pinned in `/home/ubuntu/.prorl_creds.env`. Every launcher in this branch sources that file — **do not re-export inline**.

```bash
source /home/ubuntu/.prorl_creds.env
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/singularity_images
for v in HF_TOKEN WANDB_API_KEY OH_RUNTIME_SINGULARITY_IMAGE_REPO; do
  test -n "${!v}" && echo "$v OK" || echo "$v MISSING"
done
```

### 3.2 — Trainer-side (Docker + verl checkout + HF cache)

```bash
# Docker image (trainer stack: verl v0.8.0.dev, vLLM 0.18, PyTorch 2.6+).
docker image inspect verlai/verl:vllm018.dev1 >/dev/null 2>&1 \
  || docker pull verlai/verl:vllm018.dev1

# verl source tree (mounted into the container as /opt/verl).
[ -d /tmp/verl ] || git clone https://github.com/shamanez/verl /tmp/verl
git -C /tmp/verl fetch --tags && git -C /tmp/verl checkout 910ba344   # pinned

# Base-model HF cache (reused by the trainer inside the container via -v ~/.cache/huggingface:/root/.cache/huggingface).
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
```

### 3.3 — Data: SkyRL-v0-293 parquet + filtered copy

```bash
# Pulls ~/data/SkyRL-v0-293/{train,validation}.parquet (293 SWE-Bench rows).
bash scripts/_internal/s0_pull_data.sh

# Filter to the rows whose SIF has been built — otherwise the trainer aborts on missing images.
/opt/pytorch/bin/python3 scripts/_internal/filter_parquet_to_built_sifs.py \
    --source /home/ubuntu/data/SkyRL-v0-293/train.parquet \
    --sif-dir /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/singularity_images \
    --dest   /home/ubuntu/data/SkyRL-v0-293/train.filtered.parquet
# Repeat for validation.parquet → validation.filtered.parquet.
```

### 3.4 — Singularity SIFs (sandbox runtimes for ProRL agent jobs)

ProRL runs each agent turn inside a Singularity container built from a SWE-Bench Docker image. One `.sif` per problem, cached locally. `s0_build_sifs.sh` takes a `<start_idx> <end_idx>` range and calls `scripts/pull_swe_images.py` which shells out to `python -m openhands.runtime...` to convert Docker refs to `.sif` artifacts.

```bash
# Builds SIFs 1..50 into ./singularity_images/ (~30 min per 50 images). Run in batches.
bash scripts/_internal/s0_build_sifs.sh 1 50
ls singularity_images/*.sif | wc -l   # sanity: count should grow with each batch
```

For the 7-step baseline (`wdqqu52k`) 49 SIFs in the cache were enough because `train.filtered.parquet` restricted the dataset. Phase 1 uses the same filtered parquet.

### 3.5 — Remote vLLM pool (EC2 `vllm-instance`, one-time per host)

```bash
# Rsyncs _vllm_child.py + requirements-remote.txt to vllm-instance:~/vllm-pool/,
# creates a python3.12 venv, pip-installs vllm==0.18.*, and hf-downloads Qwen3-4B into ~/vllm-pool/hf-cache.
bash scripts/serving/launch_remote_vllm_pool.sh bootstrap
```

Before the first `start`: **open the EC2 security-group inbound for ports 8100–8103 from the trainer box's public IP**. Direct HTTP over public DNS — no tunnel, no TLS, SG lockdown is the only gate.

---

## 4 — Run

Three processes on two machines. Same as the baseline, plus a Phase 1 sibling launcher that turns LoRA on.

| # | Terminal | Command |
|---|---|---|
| 1 | trainer box, host | `bash scripts/_internal/s0_prorl.sh` |
| 2 | trainer box, host | `source /home/ubuntu/.prorl_creds.env && bash scripts/serving/launch_remote_vllm_pool.sh start` |
| 3 | trainer box, Docker | `bash scripts/_internal/s2_weightsync_docker.sh` ← **new sibling for Phase 1** (not yet written) |

The Phase 1 launcher pair to build (siblings of the frozen baseline pair — do not edit the originals):

- Inner Hydra script: `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh`
  - Copy `..._remote_decoupled.sh`, then add:
    - `actor_rollout_ref.model.lora_rank=16`
    - `actor_rollout_ref.model.lora_alpha=32`
    - `actor_rollout_ref.model.lora_target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` (or `all-linear`)
    - `+actor_rollout_ref.rollout.publish_on_save=True`
    - `trainer.save_freq=5`
    - `trainer.experiment_name=weight-sync-decup-prorl`
- Docker launcher: `scripts/_internal/s2_weightsync_docker.sh`
  - Copy `s1_remote_docker.sh`, swap the inner-script invocation and the container name (`s2-weightsync`), and forward Hydra overrides via `"$@"`.

### Before kicking off

```bash
# 1. All 4 remote children healthy.
for p in 8100 8101 8102 8103; do
  curl -sS -m 5 -o /dev/null -w "pool :$p = %{http_code}\n" \
    "http://ec2-54-145-77-207.compute-1.amazonaws.com:$p/health"
done
# Expect: four "pool :810x = 200" lines.

# 2. Local GPUs idle.
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 3. ProRL reachable.
curl -sf http://localhost:8006/health && echo "prorl OK"
```

---

## 5 — Check

### 5.1 — Live, during the run

```bash
tail -f /tmp/s2-weightsync.log                       # trainer (what the Docker launcher tees)
tail -f /tmp/prorl.log                               # ProRL server (job lifecycle + agent turns)
ssh vllm-instance 'tail -f /tmp/vllm-child-810*.log' # remote pool children (per-request log lines)
```

### 5.2 — WandB

Project / experiment: `ProAgent` / `weight-sync-decup-prorl` (set by the inner script).

**Existing baseline keys that must continue to behave:**

| Key | Signal |
|---|---|
| `actor/pg_loss`, `actor/grad_norm` | Gradient signal present; `grad_norm` > 0 on reward-variance steps. |
| `actor/kl_loss`, `actor/kl_coef` | KL regularization finite (`use_kl_loss=True`, `kl_loss_coef=0.001`). |
| `critic/rewards/{mean,max,min}` | Reward scoring; `wdqqu52k` first-reward step was 7. Phase 1 should reach a positive mean earlier. |
| `rollout_corr/ppl_ratio` | Importance-ratio proxy. **This is the staleness thermometer.** Should dip on the step *after* each `/reload_lora` publish. |
| `response_length/mean`, `prompt_length/mean` | Token distribution; watch for ceiling hits. |
| `timing_s/{step,gen,update_actor}` | Rollout wall clock vs actor update; publish overhead lands in a new bucket. |

**New Phase 1 keys.** The publish is the critical new path — each piece of its wall clock is logged separately so a regression is localizable.

| Key | Meaning | Expected (rank-16 Qwen3-4B, same-region EC2) |
|---|---|---|
| `weight_sync/policy_version` | Trainer-minted counter | monotonically increasing, bumps on every publish |
| `weight_sync/adapter_mib` | Size of the adapter bundle published this round | ~20–80 MiB at rank-16 (rises with rank, target_modules breadth) |
| `weight_sync/publish_latency_s` | End-to-end fan-out `/reload_lora` wall clock (trainer-side, max across endpoints) | < 5 s |
| `weight_sync/transfer_latency_s` | HTTP POST body transfer time to the slowest endpoint | < 2 s |
| `weight_sync/vllm_load_latency_s` | Pool-side `engine.add_lora(...)` wall clock (max across endpoints) | < 2 s — **this is the "vLLM idle time to load weights" the operator asked about** |
| `weight_sync/endpoints_ok` | Count returning 200 on last publish | `== len(external_llm_endpoints)` = 4 |
| `weight_sync/endpoints_failed` | Count returning non-200 | `== 0`; if > 0, the trainer aborts the run |
| `rollout/staleness_steps` | `trainer.global_step - meta.policy_version` averaged across the batch | ≤ `save_freq` after the first publish |

**Where each latency is measured.** Pool child returns `{policy_version, vllm_load_latency_ms, adapter_bytes}` in the `/reload_lora` 200 body; trainer subtracts `transfer_latency_s = publish_latency_s - vllm_load_latency_s` per endpoint and logs both. That separation is what tells us whether a slow publish is the network (widen the pipe / presigned S3) or the vLLM `add_lora` call (smaller rank / pinned adapter pool).

### 5.3 — Pool-side publish log (structured, one line per event)

Each `_vllm_child.py` emits a JSON line on `/reload_lora`:

```json
{"event": "reload_lora", "port": 8100, "policy_version": 3, "adapter_bytes": 41943040,
 "download_ms": 412, "add_lora_ms": 1186, "remove_lora_ms": 84, "ok": true}
```

Parseable after the run with:

```bash
ssh vllm-instance 'jq -c "select(.event==\"reload_lora\")" /tmp/vllm-child-*.log'
```

### 5.4 — Log greps (invariants the baseline established)

```bash
# Trainer still took the decoupled path.
grep -c 'EXTERNAL BYPASS ACTIVE' /tmp/s2-weightsync.log          # >= 1
grep -c 'async_llm_server_[0-9]' /tmp/s2-weightsync.log          # == 0 (no Ray vLLM actors)

# Pool applied each published adapter.
ssh vllm-instance 'grep -c "add_lora" /tmp/vllm-child-8100.log'  # matches number of publishes
ssh vllm-instance 'grep -c "remove_lora" /tmp/vllm-child-8100.log' # == publishes - 1

# No 5xx during the swap (Phase 1 gate 5).
ssh vllm-instance 'grep -E " 5[0-9][0-9] " /tmp/vllm-child-*.log'   # no lines
```

### 5.5 — Phase 1 gates (green = merge-ready)

| # | Gate | Pass |
|---|---|---|
| 1 | ≥ 4 successful `/reload_lora` across 20 steps at `save_freq=5` | All 4 endpoints return 200 with monotonically increasing `policy_version` |
| 2 | Zero mixed-version batches | `min(meta.policy_version) == max(meta.policy_version)` per batch, every batch |
| 3 | `rollout_corr/ppl_ratio` drops on the step following each publish | Visible dip in WandB vs the pre-publish baseline |
| 4 | `critic/rewards/mean` trends up (baseline `wdqqu52k` was flat until step 7) | Monotonic improvement across 20 steps, modulo noise |
| 5 | Zero `POST /generate` 5xx during swap | Clean `add_lora` → `remove_lora` hand-off in pool logs |
| 6 | Baseline invariants still hold | `EXTERNAL BYPASS ACTIVE` present, zero Ray vLLM actors, all 8 local GPUs hold FSDP shards |

---

## 6 — Phase 1 design sketch (detail)

### Trainer side

- Enable LoRA on the actor (config above). Reference model stays full-rank (policy reference for KL).
- After `_save_checkpoint()` runs, call a new `_publish_lora_adapter(checkpoint_dir)`:
  1. Extract the adapter shards from the FSDP checkpoint (PEFT-style `adapter_model.safetensors` + `adapter_config.json`).
  2. Bump `policy_version` (trainer-owned counter).
  3. Fan out `POST /reload_lora` to every endpoint in `external_llm_endpoints` with either the raw adapter bytes or a presigned URL. Block the next training step until all endpoints return 200.
  4. On any endpoint failure, abort the run with a loud marker — mixed-version batches are a correctness bug, not a warning.
- Stamp `meta.policy_version` on every rollout job dispatched to the pool.

### Pool side

- New endpoint `POST /reload_lora` on `_vllm_child.py`:
  - Body: `{adapter_url: str, policy_version: int}` (or raw bytes for small ranks).
  - Download / write to local path → `LoRARequest(lora_name=f"pv{version}", lora_int_id=version, lora_path=<local>)` → `engine.add_lora(...)`.
  - After success, retire the previous adapter via `engine.remove_lora(old_int_id)`.
  - Return `200 {policy_version}` on success. Return `5xx` with a specific error string on failure.
- `_vllm_child.py` `POST /generate` must pass the currently-pinned `LoRARequest` into the sampler so served rollouts are against `{base + adapter}`. Without this wire-up the adapter is loaded but unused.
- `launch_remote_vllm_pool.sh publish <adapter_dir>` subcommand fans `/reload_lora` to all 4 children — lets the publish protocol be tested manually before the trainer-side hook lands.

### Policy-version ownership

Trainer-authoritative. The trainer mints the monotonic counter; each pool endpoint echoes back the version it installed. Pool-side never increments on its own.

### Failure modes to surface, not hide

- **Partial reload:** one endpoint 200s, another 5xx → abort the run. Do not retry silently, do not let the batch contain mixed versions.
- **Adapter shape mismatch:** rank / target-module set changed between publish calls. The LoRA framework refuses the load; surface the engine error unmodified.
- **Pool restarted mid-run:** lost adapter state. Next publish reinstalls; in the meantime, rollouts served against base weights are stamped `policy_version=0` (the "adapter absent" sentinel) and the training step aborts on mixed-version detection.

---

## 7 — Files that change in Phase 1

| Path | Edit |
|---|---|
| `scripts/serving/_vllm_child.py` | Add `POST /reload_lora`; thread the active `LoRARequest` into `/generate`. |
| `scripts/serving/launch_remote_vllm_pool.sh` | Add `publish <adapter_dir>` subcommand that fans `/reload_lora` to all 4 children. |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | New `_publish_lora_adapter(checkpoint_dir)` called at the end of `_save_checkpoint`; fan-out over `self.config.actor_rollout_ref.rollout.external_llm_endpoints`. |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | Stamp `policy_version` on each job before dispatch to ProRL. |
| **New** `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh` | Sibling of the frozen `_remote_decoupled.sh`. LoRA rank/alpha/targets, `publish_on_save=True`, `save_freq=5`, experiment `weight-sync-decup-prorl`. |
| **New** `scripts/_internal/s2_weightsync_docker.sh` | Sibling of the frozen `s1_remote_docker.sh`. |

Frozen-file rule: the baseline launchers (`s1_remote_docker.sh`, `run_proagent_qwn3_4B_instruct_remote_decoupled.sh`) are the reproduction artifacts for the baseline run and are not edited — new siblings only.

---

## 8 — Entry points for the next session

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
   Expected: `add_lora`, `remove_lora`, `list_loras`, `pin_lora`.
3. Confirm verl's LoRA config path against `/tmp/verl` — grep for `lora_rank` in `trainer_integration/verl/verl_custom/**` and upstream `verl/trainer/ppo/**`.
4. Enter plan mode and draft the publish protocol (trainer + pool side) before touching code.
