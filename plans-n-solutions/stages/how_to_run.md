# Runbook — how to execute a fully-async training run

Authoritative launch procedure for the current branch. Companion to `handsoff.md §2` — this doc adds the env-knob matrix, monitoring, stop/resume, and failure runbook.

Source of truth: `scripts/_internal/s3_fullasync_docker.sh` (outer) + `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh` (inner). **Do not invent new invocations.**

## Topology — three processes, two machines

| Role | Machine | Entry point |
|---|---|---|
| ProRL FastAPI server (:8006) | trainer box, host, poetry venv | `bash scripts/_internal/s0_prorl.sh` |
| Remote vLLM pool, 4 children (:8100-:8103), one per GPU | EC2 `vllm-instance` (SSH alias) | `bash scripts/serving/launch_remote_vllm_pool.sh start` |
| FSDP trainer, 8× A100 | trainer box, Docker `verlai/verl:vllm018.dev1` | `bash scripts/_internal/s3_fullasync_docker.sh` |

Network: trainer container → ProRL on host `localhost:8006`; trainer → vLLM pool on `ec2-54-145-77-207.compute-1.amazonaws.com:8100-8103`. ProRL is called by OpenHands agents running inside vLLM-child turn loops, NOT directly by the trainer.

## Credentials

Pinned in `/home/ubuntu/.prorl_creds.env` — do not re-export inline. Consumed by all three launchers via `source`.

Required keys:
- `WANDB_API_KEY`
- `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
- AWS keys for S3 adapter staging
- OpenHands/runtime tokens

## Pre-flight (first run in a new shell)

```bash
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server
git status           # expect clean or known-intentional edits only
git log -1 --oneline # confirm HEAD on full-async branch
ls /tmp/verl         # pinned verl checkout must be present (shamanez/verl, 910ba344)
source /home/ubuntu/.prorl_creds.env   # populates env for manual commands
```

## Launch sequence (three terminals)

### Terminal 1 — vLLM pool on EC2 (must be first, trainer blocks on /health)

```bash
# One-off (first time on a new EC2):
bash scripts/serving/launch_remote_vllm_pool.sh bootstrap

# Every run:
bash scripts/serving/launch_remote_vllm_pool.sh start
# Waits until all 4 /health endpoints return 200.
# Child logs: ssh vllm-instance 'tail -f /home/ec2-user/vllm-pool/child-8100.log'
# (or 8101/8102/8103)
```

Stop/restart:
```bash
bash scripts/serving/launch_remote_vllm_pool.sh stop
bash scripts/serving/launch_remote_vllm_pool.sh start
```

Security-group prerequisite: EC2 inbound 8100-8103 must be open from the trainer-box public IP.

### Terminal 2 — ProRL on trainer host

```bash
bash scripts/_internal/s0_prorl.sh
# Launches FastAPI on :8006. Log: /tmp/s0-prorl.log
# Workers: 64 init / 64 run / 1000s job timeout.
```

Verify:
```bash
curl -s localhost:8006/healthz
```

### Terminal 3 — FSDP trainer in Docker

```bash
# Default (PRIMARY run — plain GRPO, filter_groups=False):
bash scripts/_internal/s3_fullasync_docker.sh

# DAPO gate (filter_groups=True, ~2-3× wall-clock, only after plain run clean):
FILTER_GROUPS=True bash scripts/_internal/s3_fullasync_docker.sh

# Short smoke test (2 steps, save every step):
TOTAL_TRAINING_STEPS=2 SAVE_FREQ=1 bash scripts/_internal/s3_fullasync_docker.sh

# Custom scale:
TOTAL_EPOCHS=50 TOTAL_TRAINING_STEPS=500 SAVE_FREQ=5 \
  bash scripts/_internal/s3_fullasync_docker.sh
```

The outer docker script:
1. Sources `/home/ubuntu/.prorl_creds.env`.
2. Removes stale container `s3-fullasync` if present.
3. Starts container `verlai/verl:vllm018.dev1` with `--gpus all --network host --shm-size=16g`.
4. Installs `verl` + `verl_custom` editable.
5. Waits up to 300s for all 4 vLLM /health to return 200.
6. Launches inner `run_proagent_qwn3_4B_instruct_fullasync.sh` with Hydra overrides.
7. Tees output to `/tmp/s3-fullasync.log`.

## Env-knob matrix

| Var | Default | What it controls | When to override |
|---|---|---|---|
| `REPLAY_ENABLE` | `True` | Master switch for TrajectoryStore + temporal IS | Set `False` to fall back to lock-step (matches baseline `s2_weightsync_docker.sh`) |
| `BUFFER_SIZE` | `128` | Max surviving groups held in replay store (= 4 × train_batch_size × n) | Shrink if `sample_age_p95` near K; grow if replay reuse wanted |
| `STALENESS_CUTOFF_K` | `4` | Hard FIFO staleness eviction (steps) | Lower if IS clip fraction > 0.2 |
| `PRODUCER_BATCH_SIZE` | `4` | Groups per DAPO producer call (= train_batch_size) | Keep equal to train_batch_size |
| `USE_TEMPORAL_IS` | `True` | Gate for clipped IS correction in `core_algos.py` | Disable for pure on-policy A/B |
| `CONTINUOUS_PRODUCER` | `True` | Daemon producer thread (vs inline lock-step) | `False` reverts to lock-step rollout |
| `FILTER_GROUPS` | `False` | DAPO filter_groups.enable | Flip to `True` ONLY after filter=False run clean |
| `TOTAL_EPOCHS` | `10` | — | Scale up for full runs |
| `TOTAL_TRAINING_STEPS` | `500` | — | `2` for smoke, `5000+` for learning |
| `SAVE_FREQ` | `1` | Checkpoint + publish cadence | `5` for standard runs (handsoff gate: ≥ 4 reload_lora per 20 steps) |
| `LOG_PATH` | `/tmp/s3-fullasync.log` | — | Override per run for A/B logs |
| `REMOTE_DNS` | `ec2-54-145-77-207.compute-1.amazonaws.com` | vLLM pool public DNS | Change when pool moves |

Extra Hydra overrides can be appended after the launcher call:
```bash
bash scripts/_internal/s3_fullasync_docker.sh \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  +algorithm.filter_groups.enable=False
```

## Trainer-side config (baked into inner launcher)

Key Hydra values in `run_proagent_qwn3_4B_instruct_fullasync.sh`:

| Key | Value | Why |
|---|---|---|
| `algorithm.adv_estimator` | `grpo` | GRPO group baseline |
| `data.train_batch_size` | `4` | Prompts per step (× n=8 → 32 trajectories/step) |
| `data.gen_batch_size` | `1` | One prompt at a time through OpenHands |
| `data.max_prompt_length` | `31232` | 31k context |
| `data.max_response_length` | `1536` | Response budget per turn |
| `actor_rollout_ref.rollout.n` | `8` | Samples per prompt (DAPO-aligned) |
| `actor_rollout_ref.model.lora_rank` | `32` (note) | Trainer side; pool applies rank-16 adapter after quant |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | LoRA-safe LR |
| `actor_rollout_ref.actor.tis_imp_ratio_cap` | `2` | TIS clamp |
| `actor_rollout_ref.rollout.openhands_num_workers` | `32` | Sweet spot for 4-child pool (64 regresses per gotcha §17) |
| `actor_rollout_ref.rollout.max_iterations` | `30` | Max agent turns |
| `actor_rollout_ref.rollout.openhands_timeout` | `1000` | Per-job seconds |
| `actor_rollout_ref.rollout.temperature` | `1.4` | High exploration |
| `actor_rollout_ref.rollout.top_p` | `0.95` | — |
| `actor_rollout_ref.rollout.external_llm_endpoints` | 4× pool URLs | — |
| `actor_rollout_ref.rollout.publish_on_save` | `True` | LoRA auto-publish on save |
| `actor_rollout_ref.actor.use_kl_loss` | `False` | RLVR — no reward-model drift to anchor against |
| `actor_rollout_ref.actor.clip_ratio_low/high` | `0.2 / 0.28` | DAPO clip-higher |
| `trainer.n_gpus_per_node` | `8` | FSDP degree |
| `trainer.resume_mode` | `auto` | Picks latest `global_step_*` in STAGE2_OUT |
| `trainer.val_before_train` | `False` | In-run validation disabled |
| `trainer.test_freq` | `-1` | In-run eval disabled |

Output directory: `/workspace/outputs/ProAgent/fullasync` inside container (bind-mounted from host repo).

## Monitoring

### Live log tail

```bash
tail -f /tmp/s3-fullasync.log
```

### Key signals to grep

```bash
# Progress bar and step metrics
grep "Training Progress" /tmp/s3-fullasync.log | tail -3
grep "step:" /tmp/s3-fullasync.log | tail -5

# Weight-sync publish events (one per save_freq)
grep publish_lora_adapter /tmp/s3-fullasync.log

# Producer-mode bug #16 markers (one per DAPO call)
grep "dropped [0-9]* leftover jobs" /tmp/s3-fullasync.log | wc -l

# Gotcha §19 cooperative-stop skip — MUST BE ZERO
grep -c "did not exit within\|skipping _validate" /tmp/s3-fullasync.log

# Tracebacks — MUST BE ZERO
grep -c "Traceback" /tmp/s3-fullasync.log

# DAPO filter events (prompt, resolved_ratio)
grep -E "resolved_ratio" /tmp/s3-fullasync.log | tail -20
```

### vLLM pool

```bash
# Children up and serving
ssh vllm-instance 'for p in 8100 8101 8102 8103; do
  echo "=== $p ==="
  tail -5 /home/ec2-user/vllm-pool/child-$p.log
done'

# Latest reload_lora per child
ssh vllm-instance 'for p in 8100 8101 8102 8103; do
  grep reload_lora /home/ec2-user/vllm-pool/child-$p.log | tail -1
done'

# /health snapshot
for p in 8100 8101 8102 8103; do
  curl -sf "http://ec2-54-145-77-207.compute-1.amazonaws.com:$p/health"
done
```

### WandB

Project: `ProAgent`. Experiment: `fullasync-replay-prorl`. Key panels to track (handsoff §12):
- `replay/store_size`, `replay/store_fill_ratio`, `replay/sample_age_steps_p50/p95`, `replay/dropped_by_staleness_total`
- `is_weight/mean`, `is_weight/p99`, `is_weight/clip_fraction`
- `weight_sync/policy_version`, `weight_sync/endpoints_ok`, `weight_sync/endpoints_failed`
- `rollout/staleness_steps`
- `timing_s/gen`, `timing_s/update_actor`, `timing_s/step`
- `critic/rewards/mean`, `critic/score/mean`
- `perf/mfu/actor`, `perf/max_memory_allocated_gb`

## Stop / restart / resume

### Graceful stop
```bash
# Trainer: Ctrl-C on terminal 3 (SIGINT triggers cooperative shutdown).
# Container auto-removes (--rm) on exit.

# vLLM pool:
bash scripts/serving/launch_remote_vllm_pool.sh stop

# ProRL: Ctrl-C on terminal 2.
```

### Hard stop (container wedged)
```bash
docker rm -f s3-fullasync
ssh vllm-instance 'for p in 8100 8101 8102 8103; do
  pkill -f "port.*$p" || true
done'
```

### Resume

`trainer.resume_mode=auto` scans `/workspace/outputs/ProAgent/fullasync/global_step_*` and loads the latest. Replay buffer re-warms from empty (by design — pre-resume entries are maximally stale). Policy_version stamp is re-synced to `global_steps` at resume.

```bash
# Just relaunch — trainer picks up last checkpoint:
bash scripts/_internal/s3_fullasync_docker.sh
```

To resume from a specific step: wipe later step dirs first.
```bash
ls /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/outputs/ProAgent/fullasync/
# rm -rf global_step_{later-than-target}  # CAREFUL
```

**Current on-disk state (post-prep-100, 2026-04-26):** only `global_step_40` is preserved (all earlier prep-100 checkpoints were deleted to free disk during session shutdown). `resume_mode=auto` will pick it. If you need an earlier resume target you must re-train from scratch.

### Fresh start (discard checkpoints)

```bash
rm -rf outputs/ProAgent/fullasync
bash scripts/_internal/s3_fullasync_docker.sh
```

## Failure runbook (standing user instruction)

> "If there's an error just fix it and rerun it. 1) Stop trainer 2) Restart vLLM workers 3) Fix the error and commit, note it down 4) Retrain."

1. **Stop trainer**: Ctrl-C terminal 3, or `docker rm -f s3-fullasync`.
2. **Restart vLLM workers**:
   ```bash
   bash scripts/serving/launch_remote_vllm_pool.sh stop
   bash scripts/serving/launch_remote_vllm_pool.sh start
   ```
3. **Fix**: edit, `make lint`, `pytest -m "not integration and not slow and not real_data"`, commit (never `--no-verify`).
4. **Note**: add gotcha to `plans-n-solutions/handsoff.md §10` (or bump gotcha number if new).
5. **Rerun**: same command as before; `resume_mode=auto` picks up.

### Common failures

| Symptom | Root cause | Fix |
|---|---|---|
| `/health` timeout during pre-flight | Pool not up or SG blocks trainer IP | Run `launch_remote_vllm_pool.sh start`; confirm EC2 SG 8100-8103 inbound |
| `endpoints_failed > 0` in `weight_sync/*` | One vLLM child OOM'd or drained; partial publish → mixed policy versions | Abort (trainer does this automatically). Restart pool. Resume. |
| `did not exit within Ns; leaving thread running` | Producer stuck mid-`asyncio.run(generate_sequences)` during save | Handled automatically — caller skips validation, retries next boundary. Verify `skipping _validate` warning appears once, then normal progress. Fix `590f8281` applied in Cut 4.1. |
| Tracebacks with `NoneType.concat` in DAPO | Bug #16 (`all_input_batch` leak across producer calls) | Reset fires each call ("dropped N leftover jobs" marker). Count should equal producer call count. |
| tqdm frozen > 60 min with no producer-mode markers | Producer wedged | `docker rm -f s3-fullasync`, restart pool, resume |
| 5xx on `/generate` during publish | Pool drain race | Non-fatal under load (`drain_timed_out:true, ok:true`). Count should stay low. Concern if > 10% of calls. |

## Short smoke test (reproducible)

```bash
# 2 steps, save every step — minimal end-to-end exercise (~30-40 min)
TOTAL_TRAINING_STEPS=2 SAVE_FREQ=1 LOG_PATH=/tmp/smoke.log \
  bash scripts/_internal/s3_fullasync_docker.sh

# Expected:
grep "step:" /tmp/smoke.log        # 2 metric lines
grep publish_lora_adapter /tmp/smoke.log | wc -l  # 2 publishes
grep -c Traceback /tmp/smoke.log   # 0
```

## Frozen — do not edit

- `scripts/_internal/s2_weightsync_docker.sh` (matched-`global_steps` A/B baseline)
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh` (same)
- `dev_config/python/**` (lint/type/format configs — require explicit approval to change)
- `/tmp/verl/**` (pinned verl checkout at commit `910ba344`)

## Validation after a run

In-run validation is optional (`trainer.test_freq`, `trainer.val_before_train`). Offline A/B:

```bash
# Via eval-harness skill on validation.parquet (23 prompts, input_hash pass@k)
# Compares full-async checkpoint vs baseline at matched global_steps.
# Triggered from Claude Code: /eval-harness ... (see .claude/skills/)
```

Success signals (all should hold on a healthy run):
1. `weight_sync/endpoints_failed == 0` end-to-end
2. ≥ 4 `/reload_lora` events per 20 steps at `save_freq=5`
3. Zero 5xx on `/generate` during publishes
4. `replay/sample_age_steps_p95 ≤ K` (K=4) AND `rollout/staleness_steps_p95 ≤ K + save_freq`
5. `is_weight/p99 < 10`, `is_weight/clip_fraction < 0.2`  *(currently FAIL — problem #3)*
6. `critic/rewards/mean` trends up on long runs
7. Offline A/B: full-async ≥ baseline pass@k on validation.parquet
8. Both `filter_groups={False, True}` runs land clean
9. Zero tracebacks, zero §19 skipped-validates  *(currently FAIL at step-10 — problem #7)*
10. Token-in/token-out golden-file preserved

## Where this session leaves you (2026-04-26 sign-off)

### Last known-good state

- **HEAD commit:** `full-async-optimization` branch, post Cut 9 (no-progress detector).
- **Only preserved checkpoint:** `outputs/ProAgent/fullasync/global_step_40/` (full FSDP shards + LoRA adapter). All earlier prep-100 checkpoints were deleted; this is the only resume target.
- **WandB:** project `ProAgent`, experiment `fullasync-replay-prorl`. **Only run kept:** `z6yznr3z` (the prep-100 run, finished). 32 prior runs in the project were deleted.
- **vLLM pool:** healthy on all 4 children (`/health` 200, `MAX_MODEL_LEN=47616`).
- **ProRL FastAPI:** still running on host `:8006` (pid via `pgrep -f s0_prorl`).
- **Disk:** `/dev/root` 71% used (down from 92%); ~210 GB freed during cleanup.

### Three ways to pick up

**A. Resume training from step 40 with the no-progress fix in place.**
```bash
# vLLM pool already up; ProRL already up. Just relaunch the trainer:
TOTAL_TRAINING_STEPS=100 GEN_BATCH_SIZE=32 SAVE_FREQ=5 TEST_FREQ=-1 \
  bash scripts/_internal/s3_fullasync_docker.sh
```
`TEST_FREQ=-1` keeps the §19 cooperative-skip mechanism out of the picture entirely until the deferred pause/resume lands. `resume_mode=auto` picks `global_step_40` automatically.

**B. Evaluate the step-40 LoRA adapter (post-hoc pass@k against `validation.parquet`).**
The adapter is ready to load:
```
outputs/ProAgent/fullasync/global_step_40/actor/lora_adapter/
├── adapter_config.json   (peft 0.18.1, r=32, alpha=64, base=Qwen/Qwen3-4B-Instruct-2507)
└── adapter_model.safetensors  (253 MB)
```
The driver is not yet written. See `plans-n-solutions/stages/current_bottlenecks_and_problems.md` for the eval shape (T=0.6, top_p=0.95, n=2 against the live pool with this adapter loaded via `/load_lora`).

**C. Land Cut 7 (multi-producer fan-out) before any more training.**
Only do this if you want to address the producer call-boundary trough at the architectural level. Full file list and shape in `current_bottlenecks_and_problems.md` ("Next session — Cut 7"). Pair it with the deferred pause/resume implementation since both touch `continuous_producer.py`.

### Open TODOs (carry forward)

| TODO | Scope | Why deferred | Where it goes |
|---|---|---|---|
| **`continuous_producer.pause()` / `resume()`** | ~40 LOC in `verl_custom/replay/continuous_producer.py` | The §19 cooperative-skip mechanism prevented in-training pass@k for the entire prep-100 run (15 skips). The principled fix lets the worker finish its current call cleanly, pauses between calls, and lets `_validate()` run without races. | handsoff.md §19 / §25 |
| **Cut 7 — multi-producer fan-out** | `ray_trainer.py`, `async_server_dapo.py`, second OH server on `:8007` | Producer call-boundary dead gap (~26 min/call). Cut 8 (gen_batch_size 16→32) is the cheap mitigation; Cut 7 is the principled fix. | current_bottlenecks_and_problems.md "Next session" |
| **LoRA-only post-hoc eval driver** | New script under `scripts/eval/` | Need pass@k numbers from step 40 without spinning up the FSDP trainer. Loads `adapter_model.safetensors` into the pool via `/load_lora`, scores `validation.parquet`. | current_bottlenecks_and_problems.md (option B above) |
| **Temperature alignment in `dp_actor.compute_log_prob`** | `verl_custom/workers/actor/dp_actor.py` | `is_weight/clip_fraction` decomposition: ~0.35 of the 0.55 mean log-ratio is rollout-vs-train temperature mismatch (T=1.4 → T=1.0). Aligning the compute_log_prob temperature would cut ~60% of "drift" that isn't drift. Deferred because raising `tis_imp_ratio_cap` to 5 was sufficient for the prep run. | handsoff.md §27 |
| **Per-prompt instrumentation** | `nvidia/rollout/async_server_dapo.py` | Iter-3 wall regression (80 min vs 53 min) couldn't be diagnosed without `(uid, resolved_ratio, wall_s)` per prompt in `DAPO_PRODUCER_CALL`. | current_bottlenecks_and_problems.md problem #8 |
| **DAPO gate (`FILTER_GROUPS=True`)** | `s3_fullasync_docker.sh` env | Default already `True`; just confirm it stays clean across the next 100-step run. Plain-GRPO fallback is `FILTER_GROUPS=False`. | how_to_run.md |

### What "good" looks like on the next run

(Same target table as before; carry forward.)
- `weight_sync/endpoints_failed == 0`
- `replay/sample_age_steps_p95 ≤ 4`
- `is_weight/clip_fraction < 0.25` (post-Cut-2 cap raise)
- `response_length/clip_ratio` trends down or stays bounded
- `critic/rewards/mean` trends up across 50+ steps
- Zero tracebacks, zero `Replay store made no forward progress` (Cut 9 guardrail)

## Related docs

- `plans-n-solutions/handsoff.md` — topology, pointer table, gotchas, credentials
- `plans-n-solutions/stages/current_bottlenecks_and_problems.md` — open problems
- `plans-n-solutions/stages/run9_n16_report.md` — moment-of-truth run evidence
- `plans-n-solutions/stages/replay_dynamics.md` — producer/store/trainer interaction
- `plans-n-solutions/stages/latencies.md` — per-component latency / TPS breakdown
- `openhands/nvidia/README.md` — FastAPI job lifecycle
- `openhands/llm/nvidia/README.md` — token-in/token-out invariant
- `CLAUDE.md` — architectural invariants
