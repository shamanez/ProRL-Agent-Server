# Handoff — current system

Single source of truth for a fresh session. Read top to bottom before touching anything. `CLAUDE.md` at repo root covers architectural invariants; this doc covers **how the live system works, how to run it, how to debug it, and what is currently broken**.

**Branch.** `full-async-optimization`. Baseline commit `01227c27` (branched off `full-async`). The primary next work is attacking the problems in [`stages/current_bottlenecks_and_problems.md`](stages/current_bottlenecks_and_problems.md) — that is the evidence sheet. Pick one lever per branch; don't stack.

---

## 0. What this repo runs

ProRLAgent Server + verl fork, configured as a **fully-async decoupled agentic RL** loop:

- Rollouts stream continuously into a bounded in-process replay store.
- Trainer samples from the store on its own cadence with clipped temporal importance-sampling correction and a hard staleness cutoff.
- Rank-16 LoRA adapters ship to the vLLM pool via `POST /reload_lora` every `SAVE_FREQ` steps.
- Both plain GRPO and DAPO (`filter_groups=True`) are wired.

Intellectual reference: `docs/README.md` (Arnal et al. 2026 distilled).

**Moment-of-truth run:** Run9 (n=16, DAPO `filter_groups=True`, K=4 staleness, 128-group FIFO buffer, LR=1e-6, rank-16 LoRA). Log `/tmp/s3-fullasync-n16-baseline.log`, monitor JSONL `/tmp/replay-monitor.jsonl`. Evidence: `stages/run9_n16_report.md`. Problem sheet: `stages/current_bottlenecks_and_problems.md`.

---

## 1. Topology — three processes, two machines

```
┌──────────────────────── trainer box (this machine) ────────────────────────┐
│  host, poetry venv:  ProRL FastAPI server           :8006  s0_prorl.sh     │
│  host, Docker:       GRPO trainer (8×A100 FSDP)            s3_fullasync    │
│                       verlai/verl:vllm018.dev1                             │
│                       verl @ /tmp/verl (shamanez/verl main, v0.8.0.dev)    │
└────────────────────────────────────────────────────────────────────────────┘
                                  │ HTTP
                                  ▼
┌──────────────────────── EC2 vllm-instance ─────────────────────────────────┐
│  4× vLLM children  :8100  :8101  :8102  :8103                              │
│  launch_remote_vllm_pool.sh start (orchestrated over SSH from trainer box) │
│  Qwen/Qwen3-4B-Instruct-2507, max_model_len=36864, --enable-lora           │
│  --max-loras 8 --max-lora-rank 32 --max-cpu-loras 16                       │
└────────────────────────────────────────────────────────────────────────────┘
```

The trainer container talks to ProRL on host `localhost:8006`; ProRL is called by OpenHands agents running inside vLLM-child turn loops, NOT directly by the trainer. `EXTERNAL BYPASS ACTIVE` in trainer logs confirms the decoupled topology is engaged.

**Frozen files — never edit, make siblings:**

- `scripts/_internal/s2_weightsync_docker.sh` — kept as a matched-`global_steps` baseline for A/B.
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh` — same.

---

## 2. Launch sequence (three terminals)

Run in order — the trainer pre-flight probes pool `/health`; do not parallelise start-up.

```bash
# Terminal 1 — trainer box (host, NOT Docker) — ProRL FastAPI
bash scripts/_internal/s0_prorl.sh
# Wait for: "Uvicorn running on http://0.0.0.0:8006"

# Terminal 2 — trainer box (host) — remote vLLM pool orchestration
source /home/ubuntu/.prorl_creds.env
bash scripts/serving/launch_remote_vllm_pool.sh start
# SSHs into vllm-instance, boots 4 children on 8100-8103.
# Wait for: 4× "ready" from /health.

# Terminal 3 — trainer box (Docker) — fully-async trainer
bash scripts/_internal/s3_fullasync_docker.sh
```

Stop order is reverse: kill the trainer container, then `launch_remote_vllm_pool.sh stop`, then kill ProRL.

Full runbook (env-knob matrix, smoke test, restart/resume, failure runbook): [`stages/how_to_run.md`](stages/how_to_run.md).

---

## 3. Credentials and hosts

| Secret/host | Where | Notes |
|---|---|---|
| `WANDB_API_KEY`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | `/home/ubuntu/.prorl_creds.env` | Sourced by `s0_prorl.sh`, `s3_fullasync_docker.sh`, `launch_remote_vllm_pool.sh`. **Never re-export inline. Never commit.** |
| Remote pool host | SSH alias `vllm-instance` (must exist in `~/.ssh/config`) | Public DNS currently `ec2-54-145-77-207.compute-1.amazonaws.com` — **hardcoded** in `run_proagent_qwn3_4B_instruct_fullasync.sh`. |
| Remote pool dir | `~/vllm-pool/` on `vllm-instance` | `pid-<port>.pid`, `child-<port>.log`, `venv/`, `hf-cache/`. Override with `REMOTE_POOL_DIR`. |
| Dataset | `/home/ubuntu/data/SkyRL-v0-293/{train,validation}.parquet` | 293 train / 23 val prompts. **Do not re-download.** |
| Singularity images | `./singularity_images` → `/opt/dlami/nvme/singularity_images` | DLAMI-specific symlink; gitignored. `OH_RUNTIME_SINGULARITY_IMAGE_REPO` points at it. |
| Trainer checkpoints | `outputs/` under repo root, written by root-owned Docker process | `sudo rm -rf outputs` to wipe. Gitignored. |

EC2 security group must allow inbound TCP 8100–8103 from the trainer box's public IP (and 22 for SSH).

---

## 4. Pointer table — where the work lives

| Role | Path | Notes |
|---|---|---|
| Trainer entrypoint (host) — ProRL | `scripts/_internal/s0_prorl.sh` | FastAPI on :8006, 64 init / 64 run workers, 1000s job timeout. |
| Trainer entrypoint (Docker) — fully-async | `scripts/_internal/s3_fullasync_docker.sh` | Default PRIMARY launcher. Env knobs: `TOTAL_TRAINING_STEPS`, `SAVE_FREQ`, `NUM_TRAJ`, `FILTER_GROUPS`, `TEST_FREQ`, `VAL_BEFORE_TRAIN`, `LOG_PATH`, `REMOTE_DNS`. |
| Hydra launcher | `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh` | Baked config: `lora_rank=32`, `lora_alpha=64`, `publish_on_save=True`, `replay.*`, `tis_imp_ratio_cap=5`, hardcoded EC2 DNS. |
| Replay store | `trainer_integration/verl/verl_custom/replay/trajectory_store.py` | FIFO deque max 128, K=4 staleness cap, pop-on-sample, single `threading.Lock`. |
| Continuous producer (daemon thread) | `trainer_integration/verl/verl_custom/replay/continuous_producer.py` | `start`/`stop(timeout)` cooperative exit (gotcha #19 fix). |
| GRPO trainer | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | `_publish_lora_adapter` after `_save_checkpoint`, policy_version sync at 1506-1515, metrics hook ~1685. |
| DAPO trainer | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py` | Same publish hook wired. `n_groups = max(1, train_batch_size // n)` at lines 86-87 (see problem #1). |
| Trainer class selector | `trainer_integration/verl/verl_custom/trainer/main_ppo.py:232-236` | `filter_groups.enable=True` → `RayPPOTrainerDAPO`, else plain. |
| Temporal IS correction | `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py:586-590` | Gated on `replay.use_temporal_is`. Ratio = `exp(old_log_prob − rollout_log_probs)`, clamped at `tis_imp_ratio_cap=5`. |
| Actor forward | `trainer_integration/verl/verl_custom/workers/actor/dp_actor.py` | `compute_log_prob` at T=1.0 (rollout at T=1.4 — see problem #3). |
| Rollout manager | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | `policy_version` stamping ~1495, EXTERNAL BYPASS ACTIVE path 408-425. |
| DAPO rollout dispatcher | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` | Producer-mode rebuild of `job_queue` (bug #16 fix). |
| Pool child (vLLM wrapper, FastAPI) | `scripts/serving/_vllm_child.py` | `active_lora` global, `POST /reload_lora`, `_swap_lock`, `_inflight_cond`, drain+swap. |
| Pool runner (per child) | `scripts/serving/_remote_vllm_runner.sh` | `--enable-lora --max-loras 8 --max-lora-rank 32 --max-cpu-loras 16`. |
| Pool orchestrator | `scripts/serving/launch_remote_vllm_pool.sh` | `start|stop|restart|publish` verbs. SSHs into `vllm-instance`. |
| Token-level client | `openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py` | INVARIANT — token-in/token-out. Never modify. |
| verl upstream (read-only ref) | `/tmp/verl/verl/workers/fsdp_workers.py:1210-1253`, `/tmp/verl/verl/utils/fsdp_utils.py:593` | PEFT save path (`layered_summon_lora_params`). |
| Experience replay reference | `docs/README.md` | Distilled Arnal et al. mapping: staleness/coupling/compute trade-off, `(W,T)` knobs, positive-bias sampling, `BufferStructure` diff. |
| Dataset | `/home/ubuntu/data/SkyRL-v0-293/{train,validation}.parquet` | 293 train / 23 val prompts. |

---

## 5. Observability

### WandB (project `ProAgent`)

| Key | Meaning |
|---|---|
| `weight_sync/policy_version` | Trainer counter, bumps on every successful publish. |
| `weight_sync/adapter_mib` | Tarball size on the wire. |
| `weight_sync/publish_latency_s` | Max endpoint wall-clock, trainer-side. |
| `weight_sync/transfer_latency_s` | `publish_latency_s − vllm_load_latency_s` ≈ network. |
| `weight_sync/vllm_load_latency_s` | Max pool-side `add_lora` ≈ GPU. |
| `weight_sync/endpoints_ok` / `endpoints_failed` | Partial-failure detector. Abort contract: `endpoints_failed > 0` raises. |
| `rollout/staleness_steps` | `global_steps − policy_version`. Pool-adapter-age. |
| `rollout_corr/ppl_ratio` / `log_ppl_diff` | Per-step importance ratio. |
| `replay/store_size`, `store_fill_ratio` | Groups held. 0 = producer-bound. |
| `replay/sample_age_steps_p50`, `_p95` | Buffer-age — how long groups sit in the store. |
| `replay/dropped_by_staleness_per_step` | K-cutoff evictions. |
| `is_weight/mean`, `_p99`, `_clip_fraction` | TIS correction magnitude. Target `p99 < 10`, `clip_fraction < 0.2`. |
| `timing_s/gen`, `_update_actor`, `_old_log_prob`, `_step` | Per-component wall-clock. |
| `critic/rewards/mean`, `critic/score/mean` | Learning signal. |
| `perf/mfu/actor`, `perf/max_memory_allocated_gb` | FSDP efficiency. |

`replay/sample_age_steps` (buffer-age) and `rollout/staleness_steps` (pool-adapter-age) measure different things — see gotcha #26 and problem #6.

### Log paths

- `/tmp/s3-fullasync.log` (trainer)
- `/tmp/s0-prorl.log` (ProRL)
- `ssh vllm-instance 'ls ~/vllm-pool/child-*.log'` (per-endpoint)
- `/tmp/replay-monitor.jsonl` (60-s rotating — producer iter + pool health aggregated by `/tmp/replay_monitor.py`)

### Pool health / LoRA state

```bash
for p in 8100 8101 8102 8103; do curl -sf http://$REMOTE_DNS:$p/health | jq .; done
# {"status":"ok","active_lora":"pv17","policy_version":17, ...}
```

Structured events on pool side (one JSON line per `/reload_lora`):

```bash
ssh vllm-instance 'jq -c "select(.event==\"reload_lora\")" ~/vllm-pool/child-8100.log' | tail -5
```

### Key signals to grep from trainer log

```bash
grep publish_lora_adapter /tmp/s3-fullasync.log
grep "dropped [0-9]* leftover jobs" /tmp/s3-fullasync.log | wc -l   # one per DAPO call
grep -c "did not exit within\|skipping _validate" /tmp/s3-fullasync.log   # §19 skips — should be ≤1 (at shutdown)
grep -c "Traceback" /tmp/s3-fullasync.log   # MUST be 0 during fit()
grep -E "resolved_ratio" /tmp/s3-fullasync.log | tail -20
grep -c 'EXTERNAL BYPASS ACTIVE' /tmp/s3-fullasync.log   # ≥ 1 = decoupled topology active
```

---

## 6. Current problems (pointer)

Evidence sheet: [`stages/current_bottlenecks_and_problems.md`](stages/current_bottlenecks_and_problems.md). Nine problems surface in the first 20 steps of the Run9 config.

| # | Problem | One-line symptom |
|---|---|---|
| 1 | `n_groups` floor | 1 group per trainer step — no cross-prompt gradient averaging. |
| 2 | Producer-bound regime | vLLM pool 100 % util, trainer A100s 0 % — 18.5× throughput mismatch. |
| 3 | `is_weight/clip_fraction ~60 %` | IS clamp saturated; dominant cause is T=1.4 vs T=1.0 mismatch, not real drift. |
| 4 | `response_length` saturating at 1536 | Systematic cap-hitting → corrupted reward signal. |
| 5 | Advantages computed twice | Scalar at push, re-computed on sample — same result, dead work. |
| 6 | Pool-age ≠ buffer-age | `rollout/staleness_steps` and `replay/sample_age_steps` diverge when producer iters are long. |
| 7 | Validation–producer race | First fit()-time §19 cooperative skip at step 10 — validation delayed. |
| 8 | Iter-3 wall regression | 80 min vs 53-min steady state — cause TBD. |
| 9 | Publish #2 latency 1.84× | `transfer_latency_s` 4× jump on second publish. |

**Ordering.** Problems group by root cause (numerical/config mismatch, producer-bound, architectural, network contention). Start by reading the evidence sheet, then decide ordering for your branch. Don't stack fixes across groups into one branch.

---

## 7. Gotchas (read every one before editing)

1. **`/codex:*` and the `codex-rescue` agent hang in plan mode / open-ended exploration.** They are diff-scoped tools. Always give them a concrete scope: working tree (default), `git diff base...HEAD`, or a specific commit range. Codified in `.claude/rules/codex-usage.md`. Free-form review: use `code-reviewer` or `python-reviewer`.
2. **`trainer.val_only=True` is silently ignored by the custom trainer.** Upstream verl (`/tmp/verl/verl/trainer/ppo/ray_trainer.py:~1325`) has `if self.config.trainer.get("val_only", False): return`; the fork does not. Baseline-val runs roll straight into training unless you `docker rm -f <name>` after the val prints.
3. **Remote EC2 DNS is hardcoded** in `run_proagent_qwn3_4B_instruct_fullasync.sh` and `..._weightsync.sh` as `ec2-54-145-77-207.compute-1.amazonaws.com`. Parameterize it (`REMOTE_DNS` env) before handing to a new environment. Same file: `openhands_base_url=http://localhost:8006` assumes ProRL on the same box.
4. **SSH alias `vllm-instance` must be in `~/.ssh/config`** on the trainer box before `launch_remote_vllm_pool.sh` will work. Script doesn't explain this in its error message.
5. **Autoflake strips module-level imports** during pre-commit. For imports referenced only in decorators, late-bound methods, or generated strings, use an inline import with `# noqa: PLC0415`. Precedent: `verl_custom/workers/fsdp_workers.py` (`_build_compute_log_prob`), `verl_custom/trainer/ppo/ray_trainer.py:1242` (inline `local_mkdir_safe`).
6. **Pool in-memory LoRA state survives trainer restarts** but not pool restarts. `policy_version=0` is "no adapter loaded" (base model). After a trainer resume (`trainer.resume_mode=auto`), `ray_trainer.py:1506-1515` syncs `self.policy_version` to the resumed `global_steps` so the first post-resume publish isn't rejected as non-monotonic (409). `ray_trainer_dapo.py` mirrors this — preserve both.
7. **DAPO vs plain trainer class selection** lives at `main_ppo.py:232-236`. Both classes must have the publish hook and any new hook wired. Easy to miss one.
8. **Token-in/token-out invariant** (`openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py`). Never decode and re-tokenize across turns — token boundaries shift, actor vs reference diverges, KL/entropy go NaN. The replay store **stores token IDs, not strings**; don't change that.
9. **`max-loras 8`** on the pool absorbs in-flight swap slots. Exceed it (e.g., by lowering drain timeout below survivable for a long request) and `remove_lora` fails — slot leaks. Pool tolerates up to 8 leaked slots before a child bounce.
10. **No `--no-verify`.** No `git push` without explicit approval. No force push. No modifying `dev_config/python/**`. No widening `pyproject.toml` pins without reading the pin comment (some are CVE-related).
11. **Untracked artifacts that look like in-progress work:** `outputs/` (root-owned, `sudo rm -rf`), `wandb/`, `/tmp/s*-*.log`, `singularity_images` (symlink — leave alone). All gitignored.
12. **`save_freq=1`** publishes every step. Fine debugging, wrong for real runs. For normal runs: `publish_latency_s × publishes_per_epoch < step_time × save_freq`.
13. **DAPO `filter_groups=True` runs take 2–3× the wall-clock of plain GRPO** — `generate_sequences_dapo` waits for `train_batch_size` *surviving* groups, and SWE-Gym drops ~50 % of groups to sign-shared rewards. Always smoke-test with `filter_groups=False` first; promote to `True` only after plain is green. Reverse order burns multi-hour debugging on bugs the fast path surfaces in minutes.
14. **`TrajectoryStore` concurrency is one `threading.Lock` serializing push / evict / sample+pop.** All mutations and reads acquire `self._lock`. Inside `sample_mini_batch` the sequence is `_evict_stale_locked → select → pop → detach records` as a single critical section, so a group cannot be observed-then-deleted out from under the caller. `_pack` runs outside the lock but only on the detached `records` list, so the returned `DataProto` cannot alias shared state.
15. **Sampling is consume-on-sample (queue semantics), not with-replacement.** `sample_mini_batch` pops chosen groups before returning. Rationale: when producer throughput falls below trainer throughput, with-replacement would train on the same 8 trajectories K+1 times — overfitting, not replay. Deviates from paper Fig-18 (which holds at a large effective buffer).
16. **`staleness_cutoff_k` is a producer-stall safety drop, not a reuse cap.** With pop-on-sample, a group sits in the store only between push and the next sample. K drops groups pushed but unconsumed for > K trainer steps (e.g., validation paused the sampler). Default `K=4` is conservative; not a "how many times can we reuse" knob.
17. **`OPENHANDS_NUM_WORKERS=32` is the sweet spot for a 4-child vLLM pool, not 64.** The pool saturates to ~100 % GPU util at ~32 concurrent clients (measured on 4× H100 with Qwen3-4B, LoRA rank 32, `gpu_memory_utilization=0.45`). Bumping to 64 made every client-turn slower. `replay.producer_batch_size` yaml key is **dead config** — actual prompts-per-call comes from `data.train_batch_size` (plain GRPO) or DAPO's internal dataloader (`filter_groups=True`). Re-measure if the pool grows to 8 children.
18. **`generate_sequences_dapo` leaves un-dispatched jobs in `self.job_queue` on every call.** The result-collection loop breaks when `num_completed_instances >= requested_batch_size`; mid-flight tasks are cancelled, queued-but-not-dispatched jobs stay. The classic path reuses them; producer mode can't (stale `asyncio` loop refs). **Fix:** producer-mode branch at top of `generate_sequences_dapo` drops leftovers by **rebuilding** `self.job_queue = asyncio.PriorityQueue()`. Do not replace with a `get_nowait` drain — stale loop refs make drain-only fragile.
19. **`ContinuousRolloutProducer.stop()` has no handle on a producer thread mid-`asyncio.run`.** `stop()` sets `self._stop_event` and joins with the given timeout. The event is only checked at the top of the worker `while not self._stop_event.is_set():` loop; once inside `self._generate_fn(...)` → `asyncio.run(generate_sequences_dapo)` the thread ignores the event until the call returns (up to `openhands_timeout × max_iterations` ≈ 45 min on SWE-Gym). **Fix shipped:** `stop()` now returns `bool`; on timeout it does NOT null `self._thread` and does NOT call `rollout_manager.sleep()`. Both `_stop_continuous_producer_if_needed` paths (ray_trainer.py, ray_trainer_dapo.py fit()) propagate the `False` and skip `_validate`, logging `_logger.warning('step=%d skipping _validate: producer stop timed out')`; next save boundary retries. Trade-off: validation metrics skipped at boundaries mid-producer-call. **Stop-timeout default raised to 300 s** via `+replay.stop_timeout_s=300` in `scripts/_internal/s3_fullasync_docker.sh` (Cut 4) — wide enough for most in-flight iterations to settle instead of the hard 10 s drop in `ray_trainer.py:1712-1713`. **Proper fix** (deferred): make `_generate_fn` cooperatively cancellable — add `producer.pause()` / `producer.resume()` (~40 LOC in `continuous_producer.py`). See problem #7.
20. **`rollout_manager.policy_version` is read across threads without a lock.** Continuous producer (daemon thread) reads it; trainer (main thread) writes it inside `_publish_lora_adapter` after a successful `/reload_lora` fanout. Relies on CPython GIL atomicity of single-int load/store. The benign race is temporal: a producer call in flight when publish lands stamps `behavior_policy_version = old_pv` on every trajectory of that call; the *next* producer call reads `new_pv`. That's exactly the TIS correction's input — not a bug. Do NOT rewrite as a lock or `threading.Event`.
21. **`TrajectoryRecord` prompt/response lengths vary across store entries.** Producer batches pad to call-local max lengths (`async_server.py:1343-1367`), so `group_A.prompt_ids.shape = (8, 3800)` and `group_B.prompt_ids.shape = (8, 4122)` can coexist in the same deque. Store uses raw variable-length tuples and re-pads at `sample_mini_batch` to a fresh per-sample max. `DataProto.concat → torch.cat(dim=0)` (verl `protocol.py:930`) would assert-fail on a dim-1 mismatch; re-padding is load-bearing.
22. **`/reload_lora` drain (`_vllm_child.py:163-166`) stamps `policy_version` atomically per `/generate` call.** In-flight agentic trajectories (tool-use loops) straddling a publish still see the *old* version because each turn's `/generate` completes against whichever version was active at that turn's arrival — correct (a mid-rollout switch would mix logprobs across two policies in one trajectory). Store stamps `behavior_policy_version = rollout_manager.policy_version_at_push_time`, which is the version at the *end* of the OpenHands session. For rank-16 adapters with publish_latency ≈ 30s and turn-time ≈ 20s the difference is within TIS clip. If `clip_fraction > 0.5` with K=4 staleness, investigate per-turn version stamping.
23. **`endpoints_failed > 0` abort contract (`ray_trainer.py:1348-1352`) is preserved in producer mode.** A warm buffer does not mask a broken pool: publish failure raises from the trainer's `_publish_lora_adapter` and the producer thread is stopped as part of trainer exit.
24. **Buffer is ephemeral — not checkpointed.** On trainer resume (`trainer.resume_mode=auto`), the store starts empty and re-warms from scratch. Pre-resume entries would be maximally stale anyway. Warm-up time ≈ `N / producer_throughput` steps.
25. **§19 cooperative skip can fire during `fit()` — not just at shutdown.** Run9 step 10 hit it for the first time at a scheduled validation boundary: `_validate()` wanted an exclusive pool, called `producer.stop(timeout=10s)`, the producer was mid-`generate_sequences_dapo` (53+ min call), timeout elapsed → validation skipped (no corruption, no traceback). With `save_freq=5` and `test_freq=10`, this slips the first in-training pass@k datapoint from step 10 to step 20. **Cut 4 mitigation (shipped):** default `stop_timeout_s=300` via `+replay.stop_timeout_s=300` Hydra override — wide enough to catch an iteration in its settle window; read at `ray_trainer.py:1712-1713` via `self.config.replay.get('stop_timeout_s', 10.0)`. **Principled fix (deferred):** add `producer.pause()` / `producer.resume()` that lets the worker finish its current call then pauses between calls (~40 LOC in `continuous_producer.py`). See problem #7.
26. **Pool-adapter-age (`rollout/staleness_steps`) and buffer-age (`replay/sample_age_steps`) diverge whenever producer-wall > `save_freq × burst_duration`.** Run9 step 9: `sample_age_p50=0` but `staleness_steps=4`. IS clipping correlates with pool-adapter-age, not buffer-age. Don't claim "staleness bounded" from `sample_age_p95 ≤ K` alone — add a companion gate `rollout/staleness_steps_p95 ≤ K + save_freq`. See problem #6.
27. **`is_weight/clip_fraction` is ~60 % in the n=16 regime, of which only ~20 % is real policy drift.** Decomposition of the ~0.55 mean log-ratio:
    - **~0.35** temperature mismatch: rollout at `T=1.4 top_p=0.95`, trainer's `old_log_prob` / `ref_log_prob` forward at `T=1.0` (dp_actor.py default).
    - **~0.20** vLLM ↔ FSDP numerical divergence: different kernels, fused ops, softmax paths. Systematic, not random.
    - **~0.05** LoRA load path at float16/bfloat16.
    - **~0.15** genuine policy drift from the adapter difference between push-time pv and trainer-update-time pv.

    **Phase-2.5 Cut 2 raises `tis_imp_ratio_cap` 2 → 5** so the clamp bounds genuine drift (~0.15 of the 0.55 mean log-ratio) rather than the T=1.4 / T=1.0 + kernel numerical floor (~0.55 − 0.15 = ~0.40) that is not a correctness bug. Principled followup (deferred): align the trainer's `old_log_prob` pass temperature to rollout temperature (single-line patch in `dp_actor.py`'s `compute_log_prob` — scale logits by `1/T` before log-softmax). Not a bug in the fork — the path assumes on-policy, where T-scaling cancels. With stored `rollout_log_probs`, the ratio is `exp(old − rollout)` and T-mismatch no longer cancels. See problem #3.
28. **Producer-wall outliers (80 min vs 53 min typical) are not yet instrumented to the prompt level.** Run9 iter 3 regressed completion 40 % → 27 %, wall 53 → 80 min; iter 4 recovered to 58 min. Plausible causes: dataset difficulty drift, post-publish policy regression, pool KV-cache fragmentation over long uptimes. Cannot distinguish without emitting per-prompt `(uid, resolved_ratio, wall_s)` in `DAPO_PRODUCER_CALL`. Also: publish #2 `transfer_latency_s` was 1.84× publish #1 (19.0 s vs 4.0 s); if publish #3 also > 30 s, systemic network contention. See problems #8, #9.

---

## 8. Testing

Fast loop — keep new unit tests out of `integration`, `slow`, `real_data` so they land here:

```bash
pytest -m "not integration and not slow and not real_data" tests/ -q
```

Replay-specific tests:

```bash
pytest tests/replay/ -q
pytest tests/trainer/test_trainer_buffer_integration.py tests/trainer/test_temporal_is_correction.py -q
```

Coverage on a subtree:

```bash
pytest --cov=trainer_integration.verl.verl_custom.replay --cov-report=term-missing tests/replay/
```

Lint / type / format:

```bash
make lint            # pre-commit on tracked files; never skip
make lint-scripts    # same, scoped to scripts/
```

Env vars for runtime tests:

```bash
export TEST_RUNTIME=singularity
export RUN_AS_OPENHANDS=False
export PYTHONPATH=.
```

See `.claude/rules/python-conventions.md` and `.claude/rules/testing-conventions.md` for the full rubric.

---

## 9. What NOT to do

- Don't touch `/tmp/verl` — pinned read-only upstream reference. Fork customizations live in `trainer_integration/verl/verl_custom/` as a patch package on top of the container's `verlai/verl:vllm018.dev1`. Edits to `/tmp/verl` are invisible to the trainer.
- Don't touch `openhands/llm/nvidia/qwen3.py` or `qwen2_5_vl.py`.
- Don't edit the frozen files in §1 (`s2_weightsync_docker.sh`, `..._weightsync.sh`). Make siblings.
- Don't modify `dev_config/python/**`.
- Don't widen `pyproject.toml` pins without reading the pin comment.
- Don't store decoded text across steps in the replay buffer.
- Don't use `--no-verify`, don't `git push --force`, don't push at all without explicit approval.
- Don't commit `outputs/`, `wandb/`, `/tmp/*.log`, `singularity_images`, or anything under `/home/ubuntu/.prorl_creds.env`.
- Don't re-download the SkyRL-v0-293 dataset.
- Don't run `/codex:*` without a concrete diff.
- Don't stack two unrelated fixes in one commit.

---

## 10. When in doubt

1. Re-read `CLAUDE.md` and this file.
2. Read the file in §4's pointer table closest to what you're changing.
3. If a design choice is 50/50, write both options with pros/cons and ask the user.
4. If a tool hangs, check §7 before retrying.
