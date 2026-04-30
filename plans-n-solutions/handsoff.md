# Handoff — current system

Single source of truth for a fresh session. Read top to bottom before touching anything. `CLAUDE.md` at repo root covers architectural invariants; this doc covers **how the live system works, how to run it, and how to debug it**.

---

## 0. What this repo runs

ProRLAgent Server + verl fork, configured as a **fully-async decoupled agentic RL** loop:

- Rollouts stream continuously into a bounded in-process replay store.
- Trainer samples from the store on its own cadence with clipped temporal importance-sampling correction and a hard staleness cutoff.
- Rank-32 LoRA adapters ship to the vLLM pool via `POST /reload_lora` every `SAVE_FREQ` steps.
- Both plain GRPO and DAPO (`filter_groups=True`) are wired; DAPO is the default.

Intellectual reference: `docs/README.md` (Arnal et al. 2026 distilled).

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
│  Qwen/Qwen3-4B-Instruct-2507, max_model_len=47616, --enable-lora           │
│  --max-loras 8 --max-lora-rank 32 --max-cpu-loras 16                       │
└────────────────────────────────────────────────────────────────────────────┘
```

The trainer container talks to ProRL on host `localhost:8006`; ProRL is called by OpenHands agents running inside vLLM-child turn loops, NOT directly by the trainer. `EXTERNAL BYPASS ACTIVE` in trainer logs confirms the decoupled topology is engaged.

**The mental model.** Producer's job is to **fill the buffer**. Trainer's job is to **drain the buffer** at its own cadence. They share nothing else.

- `data.gen_batch_size` controls the **producer**: how many prompts a single `generate_sequences_dapo` call attempts. With `rollout.n=8`, one call ships up to `gen_batch_size × n` trajectories (after zero-variance drops). The DAPO manager eagerly pushes each surviving group into the `TrajectoryStore` the moment it clears `filter_easy_hard_instance` — not at end of call.
- `data.train_batch_size` controls the **trainer**: how many groups it pulls from the buffer per step. Internally that's `train_batch_size × n` trajectories; after `compute_advantage` the loss path treats them as a flat tensor.

The two knobs are independent. The trainer never waits on the producer once the buffer is warm. The producer never knows or cares what the trainer is doing.

**Scaling the producer to fill faster.** Buffer-fill rate is bottlenecked by `vLLM throughput × concurrent agentic clients`. Scale either side:

- **More vLLM children** (currently 4 on the EC2 pool) → raw generation throughput.
- **More OpenHands workers** (currently `OPENHANDS_NUM_WORKERS=32`) → more concurrent agentic trajectories. The 4-child pool saturates at ~32 concurrent clients (§17); past that, scale the pool too.
- **Larger `gen_batch_size`** → longer hot phase per producer call, fewer call boundaries per hour. The `/start`–`/stop` boundary plus group cold-start (`n` trajectories from turn 0 before the first survivor) is the dominant dead time per call (§30).

Once buffer-fill rate ≥ trainer-drain rate, the trainer never waits and the loop converges on the near-on-policy regime.

**Where advantages are computed.** GRPO advantage = `(reward − group_mean) / group_std` over the `n` siblings of one prompt. This runs at the trainer on the sampled mini-batch (`ray_trainer_dapo.py:361`, `ray_trainer.py:2073`), so the buffer must hand back **whole groups intact** — `sample_mini_batch` never splits a group. The `TrajectoryRecord.advantage` field is `0.0` in the live path because eager-push happens before `compute_advantage`.

**Producer push semantics.** `filter_easy_hard_instance` (`async_server_dapo.py:763`) drops `resolved == 0` and `resolved == n` (zero-variance groups — see §6 gotcha #13) so the buffer only ever holds gradient-bearing groups. The producer does **not** compute advantages today — it only screens.

Defaults set in `scripts/_internal/s3_fullasync_docker.sh`, forwarded as `BATCH_SIZE` / `GEN_BATCH_SIZE` env vars and consumed by `run_proagent_qwn3_4B_instruct_fullasync.sh:31,37` → `+data.gen_batch_size=$GEN_BATCH_SIZE`. The DAPO server reads it at `async_server_dapo.py:251`.

**Frozen files — never edit, make siblings:**

- `scripts/_internal/s2_weightsync_docker.sh` — kept as a matched-`global_steps` lock-step baseline for A/B.
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
| Trainer entrypoint (Docker) — fully-async | `scripts/_internal/s3_fullasync_docker.sh` | Default PRIMARY launcher. Env knobs: `TOTAL_TRAINING_STEPS`, `SAVE_FREQ`, `NUM_TRAJ`, `FILTER_GROUPS`, `TEST_FREQ`, `VAL_BEFORE_TRAIN`, `LOG_PATH`, `REMOTE_DNS`, `GEN_BATCH_SIZE`. |
| Hydra launcher | `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh` | Baked config: `max_prompt_length=31232`, `max_response_length=16384`, `lora_rank=32`, `lora_alpha=64`, `publish_on_save=True`, `replay.*`, `tis_imp_ratio_cap=5`, hardcoded EC2 DNS. Pool `MAX_MODEL_LEN=47616`, FSDP `ppo_max_token_len_per_gpu=49152`. |
| Replay store | `trainer_integration/verl/verl_custom/replay/trajectory_store.py` | FIFO deque max 256, K=4 staleness cap, pop-on-sample, single `threading.Lock`. Exposes `total_pushes()` for the no-progress detector. Mechanics (record fields, push/sample step-by-step, temporal-IS plumbing): see [`stages/replay_dynamics.md`](stages/replay_dynamics.md). |
| Continuous producer (daemon thread) | `trainer_integration/verl/verl_custom/replay/continuous_producer.py` | `start`/`stop(timeout)` cooperative exit. Terminal `push_from_dataproto` skipped when `batch.meta_info['eager_pushed_all']`. Hosts `wait_until_with_progress` (no-progress detector). |
| GRPO trainer | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | `_publish_lora_adapter` after `_save_checkpoint`, policy_version sync at 1506-1515, metrics hook ~1685. `_start_continuous_producer_if_needed` wires eager-push closure into the DAPO manager. |
| DAPO trainer | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py` | Same publish hook wired. `n_groups = train_batch_size` (line ~86) — every sampled group carries gradient content because the ingest filter keeps zero-variance groups out of the buffer. |
| Trainer class selector | `trainer_integration/verl/verl_custom/trainer/main_ppo.py:232-236` | `filter_groups.enable=True` → `RayPPOTrainerDAPO`, else plain. |
| Temporal IS correction | `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py:586-590` | Gated on `replay.use_temporal_is`. Ratio = `exp(old_log_prob − rollout_log_probs)`, clamped at `tis_imp_ratio_cap=5`. |
| Actor forward | `trainer_integration/verl/verl_custom/workers/actor/dp_actor.py` | `compute_log_prob` rescales logits by `1/T` at line 210 / 290 (non-fused branch is the live path; `use_fused_kernels=False` by default). `T = self.config.rollout.temperature = 1.4` is plumbed via `verl_custom/workers/fsdp_workers.py:349` (old_log_prob) and upstream `verl/workers/fsdp_workers.py:1188` (ref_log_prob, including the LoRA `ref_in_actor` path). All three log-prob paths land in the same T-scaled distribution as vLLM's `processed_logprobs`. See §27 for the IS-ratio decomposition. |
| Advantage compute | `compute_advantage` in `ray_trainer.py:254` (called from `ray_trainer_dapo.py:361` and `ray_trainer.py:2073`) | GRPO group-mean/std over `n` siblings (`uid` keys the group). Runs on the sampled mini-batch — needs whole groups present. The `TrajectoryRecord.advantage` field is wired in `trajectory_store.py` but currently dead in the eager-push path (push happens pre-advantage). |
| Rollout manager | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | `policy_version` stamping ~1495, EXTERNAL BYPASS ACTIVE path 408-425. |
| DAPO rollout dispatcher | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` | Producer-mode rebuild of `job_queue` on every call (see §18). |
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

`replay/sample_age_steps` (buffer-age) and `rollout/staleness_steps` (pool-adapter-age) measure different things — see gotcha §26.

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
grep -c "did not exit within\|skipping _validate" /tmp/s3-fullasync.log   # §19 skips
grep -c "Traceback" /tmp/s3-fullasync.log   # MUST be 0 during fit()
grep -E "resolved_ratio" /tmp/s3-fullasync.log | tail -20
grep -c 'EXTERNAL BYPASS ACTIVE' /tmp/s3-fullasync.log   # ≥ 1 = decoupled topology active
```

---

## 6. Gotchas (read every one before editing)

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
12. **`save_freq=1`** publishes every step. Fine for debugging, wrong for real runs. For normal runs: `publish_latency_s × publishes_per_epoch < step_time × save_freq`.
13. **DAPO `filter_groups=True` is the default target path** — `scripts/_internal/s3_fullasync_docker.sh` defaults `FILTER_GROUPS=True`. The DAPO manager's eager-push seam (`self._push_fn` wired at `ray_trainer.py:_start_continuous_producer_if_needed`) fires each survivor into the replay store the moment it clears `filter_easy_hard_instance`, keeping the trainer drawing without waiting for whole iterations to finish. The plain-GRPO continuous-producer path has no equivalent seam; run `FILTER_GROUPS=False` only for throwaway smoke against the plain code paths.
14. **`TrajectoryStore` concurrency is one `threading.Lock` serializing push / evict / sample+pop.** All mutations and reads acquire `self._lock`. Inside `sample_mini_batch` the sequence is `_evict_stale_locked → select → pop → detach records` as a single critical section, so a group cannot be observed-then-deleted out from under the caller. `_pack` runs outside the lock but only on the detached `records` list, so the returned `DataProto` cannot alias shared state.
15. **Sampling is consume-on-sample (queue semantics), not with-replacement.** `sample_mini_batch` pops chosen groups before returning. Rationale: when producer throughput falls below trainer throughput, with-replacement would train on the same 8 trajectories K+1 times — overfitting, not replay. Deviates from paper Fig-18 (which holds at a large effective buffer).
16. **`staleness_cutoff_k` is a producer-stall safety drop, not a reuse cap.** With pop-on-sample, a group sits in the store only between push and the next sample. K drops groups pushed but unconsumed for > K trainer steps (e.g., validation paused the sampler). Default `K=4` is conservative; not a "how many times can we reuse" knob.
17. **`OPENHANDS_NUM_WORKERS=32` is the sweet spot for a 4-child vLLM pool, not 64.** The pool saturates to ~100 % GPU util at ~32 concurrent clients (measured on 4× H100 with Qwen3-4B, LoRA rank 32, `gpu_memory_utilization=0.45`). Bumping to 64 made every client-turn slower. `replay.producer_batch_size` yaml key is **dead config**. Prompts-per-call for plain GRPO comes from the trainer dataloader's `gen_batch_size`; DAPO survivor target per call comes from `data.gen_batch_size` (independent of `data.train_batch_size`). Re-measure if the pool grows to 8 children.
18. **`generate_sequences_dapo` leaves un-dispatched jobs in `self.job_queue` on every call.** The result-collection loop breaks when `num_completed_instances >= requested_batch_size`; mid-flight tasks are cancelled, queued-but-not-dispatched jobs stay. The classic path reuses them; producer mode can't (stale `asyncio` loop refs). **Fix:** producer-mode branch at top of `generate_sequences_dapo` drops leftovers by **rebuilding** `self.job_queue = asyncio.PriorityQueue()`. Do not replace with a `get_nowait` drain — stale loop refs make drain-only fragile. The eager-push seam fires at filter-clear time *inside the same call*, so all leftover-job reasoning stays intact (the rebuild still happens at the top of the next call, and any instance the eager path already pushed is also listed in `output_batch` as a regular survivor).
19. **`ContinuousRolloutProducer.stop()` has no handle on a producer thread mid-`asyncio.run`.** `stop()` sets `self._stop_event` and joins with the given timeout. The event is only checked at the top of the worker `while not self._stop_event.is_set():` loop; once inside `self._generate_fn(...)` → `asyncio.run(generate_sequences_dapo)` the thread ignores the event until the call returns (up to `openhands_timeout × max_iterations` ≈ 45 min on SWE-Gym). **Behavior:** `stop()` returns `bool`; on timeout it does NOT null `self._thread` and does NOT call `rollout_manager.sleep()`. Both `_stop_continuous_producer_if_needed` paths (ray_trainer.py, ray_trainer_dapo.py fit()) propagate the `False` and skip `_validate`, logging `_logger.warning('step=%d skipping _validate: producer stop timed out')`; next save boundary retries. Trade-off: validation metrics skipped at boundaries mid-producer-call. **Stop-timeout default is 300 s** via `+replay.stop_timeout_s=300` in `scripts/_internal/s3_fullasync_docker.sh` — wide enough for most in-flight iterations to settle.
20. **`rollout_manager.policy_version` is read across threads without a lock.** Continuous producer (daemon thread) reads it; trainer (main thread) writes it inside `_publish_lora_adapter` after a successful `/reload_lora` fanout. Relies on CPython GIL atomicity of single-int load/store. The benign race is temporal: a producer call in flight when publish lands stamps `behavior_policy_version = old_pv` on every trajectory of that call; the *next* producer call reads `new_pv`. That's exactly the TIS correction's input — not a bug. Do NOT rewrite as a lock or `threading.Event`.
21. **`TrajectoryRecord` prompt/response lengths vary across store entries.** Producer batches pad to call-local max lengths (`async_server.py:1343-1367`), so `group_A.prompt_ids.shape = (8, 3800)` and `group_B.prompt_ids.shape = (8, 4122)` can coexist in the same deque. Store uses raw variable-length tuples and re-pads at `sample_mini_batch` to a fresh per-sample max. `DataProto.concat → torch.cat(dim=0)` (verl `protocol.py:930`) would assert-fail on a dim-1 mismatch; re-padding is load-bearing.
22. **`/reload_lora` drain (`_vllm_child.py:163-166`) stamps `policy_version` atomically per `/generate` call.** In-flight agentic trajectories (tool-use loops) straddling a publish still see the *old* version because each turn's `/generate` completes against whichever version was active at that turn's arrival — correct (a mid-rollout switch would mix logprobs across two policies in one trajectory). Store stamps `behavior_policy_version = rollout_manager.policy_version_at_push_time`, which is the version at the *end* of the OpenHands session. For rank-32 adapters with publish_latency ≈ 30s and turn-time ≈ 20s the difference is within TIS clip. If `clip_fraction > 0.5` with K=4 staleness, investigate per-turn version stamping.
23. **`endpoints_failed > 0` abort contract (`ray_trainer.py:1348-1352`) is preserved in producer mode.** A warm buffer does not mask a broken pool: publish failure raises from the trainer's `_publish_lora_adapter` and the producer thread is stopped as part of trainer exit.
24. **Buffer is ephemeral — not checkpointed.** On trainer resume (`trainer.resume_mode=auto`), the store starts empty and re-warms from scratch. Pre-resume entries would be maximally stale anyway. Warm-up time ≈ `N / producer_throughput` steps.
25. **§19 cooperative skip can fire during `fit()` — not just at shutdown.** A scheduled validation boundary calls `_validate()`, which wants an exclusive pool, calls `producer.stop(timeout=stop_timeout_s)`. If the producer is mid-`generate_sequences_dapo` (multi-minute call) the timeout elapses → validation skipped (no corruption, no traceback). The `+replay.stop_timeout_s=300` Hydra override (`ray_trainer.py:1712-1713`, read via `self.config.replay.get('stop_timeout_s', 10.0)`) is wide enough to catch most iterations in their settle window.
26. **Pool-adapter-age (`rollout/staleness_steps`) and buffer-age (`replay/sample_age_steps`) diverge whenever producer-wall > `save_freq × burst_duration`.** A typical observation: `sample_age_p50=0` while `staleness_steps=4`. IS clipping correlates with pool-adapter-age, not buffer-age. Don't claim "staleness bounded" from `sample_age_p95 ≤ K` alone — add a companion gate `rollout/staleness_steps_p95 ≤ K + save_freq`.
27. **`is_weight/clip_fraction` decomposition.** Both rollout and the trainer's `old_log_prob` / `ref_log_prob` forward passes operate at `T = self.config.rollout.temperature = 1.4`: `verl_custom/workers/actor/dp_actor.py:210` (and 290 for the entropy path) rescales logits by `1/T` before log-softmax, with `T` plumbed in via `verl_custom/workers/fsdp_workers.py:349` (old_log_prob) and upstream `verl/workers/fsdp_workers.py:1188` (ref_log_prob, including the LoRA `ref_in_actor` path). `use_fused_kernels=False` by default, so the explicit `div_` branch is the live path; `is_lora` only swaps the adapter context, not the temperature plumbing. Temperature is therefore not a contributor to the IS log-ratio.

    Of the typical mean log-ratio ~0.55, the live sources, ordered by expected magnitude:
    - **vLLM ↔ FSDP numerical divergence** — different kernels, fused ops, softmax paths, FA-vs-paged-attention, mixed-precision casts. Systematic, not random. Largest contributor in practice.
    - **LoRA precision drift** — adapter loaded at bf16 in vLLM vs the FSDP-side forward pass (also bf16 but with different layernorm/attention numerics). Smaller than the kernel-divergence term but non-trivial under high `tis_imp_ratio_raw`.
    - **Genuine policy drift** — adapter difference between push-time `behavior_policy_version` and trainer-update-time policy. Bounded by `staleness_cutoff_k` × `save_freq`.

    **`tis_imp_ratio_cap=5`** is the live setting so the clamp bounds the kernel-numerics floor + drift comfortably; lower caps clip into the numerical floor and bias the gradient. The decomposition is qualitative — a quantitative split needs runtime instrumentation (per-source ratio histograms) and should be re-measured rather than trusted from memory.
28. **Eager push is the DAPO manager's job; the continuous producer skips its terminal push via `meta_info['eager_pushed_all']`.** Never call `store.push_from_dataproto(out_batch)` unconditionally from the continuous-producer worker loop under `filter_groups=True` — the DAPO manager pushes each survivor inside `request_from_openhands_dapo` the moment `filter_easy_hard_instance` clears it. If both fire, pop-on-sample (§15) breaks: the same `uid` appears twice, the first sample pops the first instance, the second sits in the buffer and is later popped under the same `uid`, corrupting `behavior_policy_version` / `created_at_step` tracking. The `eager_pushed_all` flag in `out_batch.meta_info` is the single source of truth; classic lockstep paths (`_push_and_sample_replay` in `ray_trainer.py`) never set it, so the flag defaults to False and the lockstep push fires normally. Manager-side seam is `self._push_fn` — the trainer wires it in `_start_continuous_producer_if_needed` and nulls it in `_stop_continuous_producer_if_needed`. A manager with `_push_fn is None` reverts to batch-level pushes at end of call.
29. **`n_groups = train_batch_size`, and groups are sampled intact.** The trainer draws `train_batch_size` independent groups per step from the buffer; `sample_mini_batch` never splits a group. Sampling whole groups is **load-bearing** today because GRPO `compute_advantage` normalizes by within-group mean/std, which needs the `n` siblings present. The ingest filter (`filter_easy_hard_instance`) drops zero-variance groups at push time, so every sampled group has gradient content. Both `ray_trainer_dapo.py` (DAPO path) and `ray_trainer.py` (plain-GRPO continuous-producer path) follow this.
30. **Producer call boundaries leave a dead gap (~26 min, no eager-push activity).** Between call N's last survivor (`stop_servers()` at `async_server_dapo.py:706`) and call N+1's first survivor, the trainer's buffer drains while: (a) HTTP `/stop` returns from OpenHands `localhost:8006`, (b) the producer thread loops back into `request_from_openhands_dapo`, (c) HTTP `/start` returns and agent runtimes warm up, (d) `gen_batch_size × n` jobs push to `self.job_queue` and dispatcher fans out, (e) the **first complete group** has to clear all `n` trajectories from turn 0 before any survivor can be filtered + pushed. Step (e) dominates — `/start`/`/stop` are seconds; group-cold-start is multi-turn agentic. Mitigation: raise `gen_batch_size` (`GEN_BATCH_SIZE=32` for `8 ×`) so the hot phase is longer per call and call boundaries are rarer per hour.
31. **Replay-store no-progress detector.** The helper `wait_until_with_progress(predicate, progress, no_progress_timeout)` lives in `verl_custom/replay/continuous_producer.py`; the store's monotonic `total_pushes()` accessor (`trajectory_store.py`) is the progress signal. The trainer resets the deadline whenever a new group lands and only aborts after `replay.no_progress_timeout_s` seconds with no push at all (default 1800 s, plumbed via `+replay.no_progress_timeout_s=1800` in `s3_fullasync_docker.sh`). Distinguishes "producer wedged" from "producer healthy but slow as the model learns longer trajectories." Test coverage: `tests/replay/test_continuous_producer.py::TestWaitUntilWithProgress`.

---

## 7. Testing

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

## 8. What NOT to do

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

## 9. When in doubt

1. Re-read `CLAUDE.md` and this file.
2. Read the file in §4's pointer table closest to what you're changing.
3. If a design choice is 50/50, write both options with pros/cons and ask the user.
4. If a tool hangs, check §6 before retrying.
