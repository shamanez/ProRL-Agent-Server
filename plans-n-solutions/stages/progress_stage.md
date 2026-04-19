# Phase 1 LoRA weight-sync — as built

**Status:** DONE. 20-step run `w9nj4akn` completed 2026-04-19; 6/6 gates green.
**Branch:** `decoup-weight-sync`.
**Design doc:** [`weight_sync_lora.md`](./weight_sync_lora.md). **Timing:** [`timing_decoupled_4B.md`](./timing_decoupled_4B.md).

This doc is the operator's view: what shipped, how to reproduce, what happens during a swap, and how staleness is accounted.

---

## 1. How to reproduce

Three processes on two machines — exactly like the baseline, with Phase 1's sibling launcher on the trainer side.

| # | Host / context | Command |
|---|---|---|
| 1 | trainer box, host | `bash scripts/_internal/s0_prorl.sh` |
| 2 | trainer box, host | `source /home/ubuntu/.prorl_creds.env && bash scripts/serving/launch_remote_vllm_pool.sh start` |
| 3 | trainer box, Docker | `bash scripts/_internal/s2_weightsync_docker.sh` |

Pre-flight:

```bash
for p in 8100 8101 8102 8103; do
  curl -sS -m 5 "http://ec2-54-145-77-207.compute-1.amazonaws.com:$p/health"
done
# Expect four {"policy_version":0} responses — fresh pool, no adapter loaded.
curl -sf http://localhost:8006/health && echo prorl-up
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader  # GPUs idle
```

First-time setup for either host (Docker image, verl checkout, HF cache, SkyRL data, Singularity SIFs, EC2 pool bootstrap) is in [`weight_sync_lora.md`](./weight_sync_lora.md) §3.

Gates verified post-run:

```bash
# Gate 1: ≥ 4 successful /reload_lora across 20 steps at save_freq=5
ssh vllm-instance \
  'jq -c "select(.event==\"reload_lora\" and .ok)" /home/ec2-user/vllm-pool/child-8100.log' \
  | wc -l
# Expect ≥ 4.

# Gate 5: zero 5xx on /generate during swap
ssh vllm-instance 'grep -E " 5[0-9][0-9] " /home/ec2-user/vllm-pool/child-*.log' || echo clean

# Gate 6: bypass path taken, no Ray vLLM actors, FSDP on 8 local GPUs
grep -c 'EXTERNAL BYPASS ACTIVE' /tmp/s2-weightsync.log
```

Full gate list: [`weight_sync_lora.md`](./weight_sync_lora.md) §5.5.

---

## 2. What actually changed

6 modified, 7 new.

| Path | Role |
|---|---|
| `scripts/serving/_vllm_child.py` | New `POST /reload_lora` with drain + monotonicity + structured JSON log. `/generate` now threads a snapshot of `active_lora` into `engine.generate`. |
| `scripts/serving/_remote_vllm_runner.sh` | vLLM child launched with `--enable-lora --max-loras 2 --max-lora-rank 16 --max-cpu-loras 4`. `max-loras=2` lets old+new coexist during the swap window. |
| `scripts/serving/launch_remote_vllm_pool.sh` | New `publish <adapter_dir> [--policy-version N]` subcommand for manual fan-out. |
| `scripts/tests/test_external_vllm.py` | Obsolete 501 stub removed. |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | `__init__` adds `policy_version`, `_last_publish_step`, `_last_publish_metrics`. New `_publish_lora_adapter(local_global_step_folder)` fans HTTP multipart out to `external_llm_endpoints` after `_save_checkpoint`. Emits 8 `weight_sync/*` keys plus `rollout/staleness_steps` every step. |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | `AsyncLLMServerManager` carries `self.policy_version`; `DataProto2Messages` stamps it on every rollout message. |
| **new** `scripts/_internal/s2_weightsync_docker.sh` | Sibling of frozen `s1_remote_docker.sh`. |
| **new** `.../run_proagent_qwn3_4B_instruct_weightsync.sh` | Sibling launcher: `lora_rank=16`, `lora_alpha=32`, 7 target modules, `publish_on_save=True`, `save_freq=5`, `exclude_modules=null`. |
| **new** `scripts/tests/test_reload_lora.py` | Pool-side unit tests: happy path, invalid tarball, 409 monotonicity, concurrent swap serialization, drain-timeout 502, remove-failure 502. |
| **new** `tests/trainer/test_publish_lora_adapter.py` | Trainer-side unit tests: 4/4, 3/4 raises + no version bump, missing adapter raises, empty endpoints raises, monotonicity, 409-as-success, timeout counted as failure. |
| **new** `tests/rollout/test_policy_version_stamp.py` | Asserts every dict out of `DataProto2Messages` carries the manager's `policy_version`. |

Frozen files `s1_remote_docker.sh` and `..._remote_decoupled.sh` are untouched — baseline reproducibility preserved.

Fast-loop tests: `pytest -m "not integration and not slow and not real_data" tests/ scripts/tests/test_reload_lora.py -q` → green on host (importorskips hide the verl-only tests; those run inside the container).

---

## 3. What happens inside a vLLM child during a swap

This is the part worth being precise about. One pool child. One `/reload_lora` arrives while `/generate` requests are in flight.

```
_vllm_child.py:240 (POST /reload_lora)
├─ acquire _swap_lock                     # serializes concurrent reloads
├─ if version <= active_policy_version    # 409 — idempotent replay OK
│     return 409
├─ extract tarball → /tmp/lora_adapters/pv{N}/
├─ await engine.add_lora(new_request)     # ≈ 11–14 s GPU (Qwen3-4B rank-16).
│                                         # In-flight /generate unaffected:
│                                         # they hold a snapshot of the PRIOR
│                                         # LoRARequest and continue on it.
├─ active_lora = new_request              # commit — new /generate see this
├─ drain: await _inflight_cond.wait_for(  # block until every /generate that
│     lambda: _inflight[prior_id] == 0    # snapshotted the prior adapter has
│   )  with timeout=120 s                 # returned. NEW /generate already
│                                         # use the new adapter; they're not
│                                         # what we're waiting on.
├─ await engine.remove_lora(prior_id)     # free GPU slot for the old adapter
└─ return 200 {vllm_load_latency_ms=add_lora_ms, adapter_bytes, ...}
```

Two locks, three invariants:

| Lock / cond | Protects | Why |
|---|---|---|
| `_swap_lock: asyncio.Lock` | The `/reload_lora` critical section | Two parallel reloads would race on `active_lora` and `engine.add_lora`. Only one swap at a time per child. |
| `_inflight_cond: asyncio.Condition` | `active_lora` snapshot + `_inflight[lora_int_id]` counter | A generate that reads `active_lora` and a swap that calls `notify_all` must interleave atomically on the counter. See `_vllm_child.py:154-163` for the comment on why snapshot + counter must be under the same cond. |
| `max-loras=2` | vLLM adapter slot count | Old + new coexist between `add_lora` and `remove_lora`. Without this, `add_lora` would evict the old adapter while generates still reference it. |

Invariants:

1. A `/generate` request never observes a half-loaded adapter. Either it snapshotted the prior `LoRARequest` (which stays alive until drain) or it saw the new one (already fully loaded).
2. `remove_lora(prior_id)` runs only after `_inflight[prior_id] == 0`. On drain timeout we **skip** the remove and return 502 — the trainer aborts, the prior slot leaks until pool restart, but in-flight correctness is preserved.
3. Trainer is authoritative on `policy_version`: it only advances **after** all endpoints acknowledge (200 or 409). A partial failure raises `RuntimeError` and the run crashes loudly — mixed-version batches are a correctness bug.

### Does the trainer keep generating rollouts during the swap?

- **The pool does not pause `/generate` during `add_lora`.** In-flight requests continue against their snapshotted adapter; new requests queue through `engine.generate` normally and pick up the new adapter the moment `active_lora` commits.
- **The trainer's publish call is synchronous and blocks the next training step.** `_publish_lora_adapter` fans out over `ThreadPoolExecutor(max_workers=4)` with `timeout=60s` and waits on `as_completed` before returning. The next `generate_sequences` call — and therefore the next rollout batch — starts only after every endpoint has replied.
- Net effect on the 20-step run: publish wall clock is 13–16 s (step 5/10/15/20), of which ~2 s is network and ~11–14 s is `add_lora` GPU time. Rollout wall clock is 145–155 s/step. Publish overhead was ~60 s total across the 20 steps — ~2% of run wall clock.

Publish happens at the end of a training step, after `_save_checkpoint`, so the rollouts collected inside the step being closed were generated against the *prior* adapter — there is no mid-batch split by construction.

---

## 4. Async ratio / staleness accounting

### Definition

```
rollout/staleness_steps = max(0, global_step − _last_publish_step)
```

`_last_publish_step` is the trainer-side step counter at which the most recent successful publish committed (every 4-of-4 endpoints returned 200/409). It is **not** `policy_version` — that counts publishes (0, 1, 2, …) and would grow at a different rate than training steps, making the metric's numerator and denominator incompatible.

Shape of the signal in the 20-step run (`save_freq=5`):

| Step | Publishes landed | `policy_version` | `staleness_steps` |
|---|---|---|---|
| 1–4 | 0 | 0 | 1, 2, 3, 4 |
| 5 | **publish at end of step 5 → pv=1** | 1 | 0 |
| 6–9 | 1 | 1 | 1, 2, 3, 4 |
| 10 | publish → pv=2 | 2 | 0 |
| … | … | … | … |
| 20 | publish → pv=4 | 4 | 0 |

Mean staleness over the run ≈ 2 steps. Max ≈ 4. This is the knob for the async ratio: decrease `save_freq` to lower staleness (at the cost of more publish overhead per wall-clock unit), increase it if publish time ever becomes non-trivial.

### GRPO importance correction

verl already logs the correction ratio as `rollout_corr/ppl_ratio` = `π_θ(a|s) / π_behavior(a|s)`, where `π_behavior` is the pool's adapter at the time the rollout was sampled. It's the staleness thermometer: on-policy → 1; drifting stale → monotonic growth past 2; degenerate → clip fires on every token and the gradient signal collapses.

Baseline (`wdqqu52k`, frozen pool): `ppl_ratio ≈ 1.6` and growing past 2.0 by step 7.
Phase 1 (`w9nj4akn`, publish every 5 steps): `ppl_ratio` sits in [1.6, 1.7] and **resets with every publish** — the open-loop drift is capped by design.

### What about keeping earlier rollouts?

**Phase 1 does not.** Each training step consumes only the rollouts sampled from the most recent pool policy and then discards them. GRPO clip handles the within-step mismatch; cross-step reuse would need Phase 3 (below).

The pieces that already exist and would feed a replay buffer:

- Every rollout message carries `policy_version` (`async_server.py:1495`). Stored trajectories can be filtered / re-weighted by staleness without guessing.
- `rollout_corr/ppl_ratio` is logged per step — the off-policy correction term the replay sampler would apply.

Missing pieces for Phase 3:

- A bounded trajectory store (FIFO or priority by TD error).
- A sampler that draws by staleness budget and applies importance weights with clipping.
- An async-safe rollout worker that runs ahead of the trainer's clock instead of lock-step with it.

---

## 5. What comes next

Detail in [`../next_approach.md`](../next_approach.md).

- **Phase 2 — full state-dict publish.** Same protocol, bigger payload (~8 GiB bf16 for Qwen3-4B). Motivated when LoRA rank-16 capacity is measurably the bottleneck (reward curve plateaus before loss plateaus). Transport likely shifts to presigned S3 URL; pool-side swap stays identical in shape but uses a full-parameter update path instead of `add_lora`.
- **Phase 3 — replay buffer / truly async RL.** Decouple rollout wall clock from training wall clock. Off-policy correction (V-trace / IMPALA-style, or capped IS with a staleness budget). Reuses Phase 1's `policy_version` stamp and `rollout_corr/*` signals.
- **Polish on Phase 1 (optional, non-blocking).**
  - Benign warning `reload_lora prior adapter dir cleanup failed` fires once per swap after pv=1 because the PosixPath was already removed. Separate log line from the core `event=reload_lora ok:true`; doesn't affect gate status but is noise.
  - Scale `add_lora` GPU time vs rank and base model size to give Phase 2 planning a data point.
