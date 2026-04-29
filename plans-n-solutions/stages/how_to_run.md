# Runbook — how to execute a fully-async training run

Authoritative launch procedure. Companion to `handsoff.md §2` — this doc adds the env-knob matrix, monitoring, stop/resume, and failure runbook. For the producer-store-trainer mechanics behind these knobs (record fields, push/sample step-by-step, temporal-IS, `replay/*` metric reads), see [`replay_dynamics.md`](replay_dynamics.md).

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
# Default (DAPO with filter_groups=True):
bash scripts/_internal/s3_fullasync_docker.sh

# Plain-GRPO fallback (only for throwaway smoke against the plain code paths):
FILTER_GROUPS=False bash scripts/_internal/s3_fullasync_docker.sh

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

The two important ones are `GEN_BATCH_SIZE` (producer-side — how fast we fill the buffer) and `BATCH_SIZE` / `train_batch_size` (trainer-side — how fast we drain). They are **independent**: the trainer never waits on the producer once the buffer is warm, the producer never waits on the trainer.

| Var | Default | What it controls | When to override |
|---|---|---|---|
| `GEN_BATCH_SIZE` | `BATCH_SIZE × 4` (= 16) | **Producer**: prompts attempted per `generate_sequences_dapo` call. Each call ships up to `GEN_BATCH_SIZE × n` trajectories (after zero-variance drops). | Raise to fill the buffer faster: `32` (= 8 × `BATCH_SIZE`) for longer hot phase per call; cuts call-boundary dead time per hour. Real ceiling is set by vLLM pool throughput + `OPENHANDS_NUM_WORKERS` (handsoff §17, §30). |
| `BATCH_SIZE` (`train_batch_size`) | `4` | **Trainer**: groups drained from the buffer per step. Internally `BATCH_SIZE × n = 32` trajectories per step. | Don't tune unless you know why — affects FSDP compute shape and group-mean variance. |
| `BUFFER_SIZE` | `256` | Max surviving groups held in replay store (FIFO). | Shrink if `sample_age_p95` near `K`; grow if you intend replay reuse (currently the regime is near-on-policy, so the buffer rarely fills). |
| `STALENESS_CUTOFF_K` | `4` | Hard FIFO staleness eviction (steps). | Lower if `is_weight/clip_fraction > 0.2`. |
| `FILTER_GROUPS` | `True` | DAPO `filter_groups.enable`. Producer drops `resolved == 0` and `resolved == n` (zero-variance groups carry no GRPO gradient). | `False` only for throwaway plain-GRPO debugging. |
| `USE_TEMPORAL_IS` | `True` | Gate for clipped IS correction in `core_algos.py`. | Disable for a pure on-policy A/B. |
| `REPLAY_ENABLE` | `True` | Master switch for TrajectoryStore + temporal IS. | `False` reverts to lock-step (matches baseline `s2_weightsync_docker.sh`). |
| `CONTINUOUS_PRODUCER` | `True` | Daemon producer thread (vs inline lock-step). | `False` reverts to lock-step rollout. |
| `TOTAL_EPOCHS` | `10` | — | Scale up for full runs. |
| `TOTAL_TRAINING_STEPS` | `500` | — | `2` for smoke, `5000+` for learning. |
| `SAVE_FREQ` | `1` | Checkpoint + publish cadence. | `5` for standard runs (gate: ≥ 4 reload_lora per 20 steps). |
| `LOG_PATH` | `/tmp/s3-fullasync.log` | — | Override per run for A/B logs. |
| `REMOTE_DNS` | `ec2-54-145-77-207.compute-1.amazonaws.com` | vLLM pool public DNS. | Change when pool moves. |

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
| `data.train_batch_size` | `4` | **Trainer-side.** Groups pulled from the buffer per step (× n=8 → 32 trajectories/step). |
| `data.gen_batch_size` | `BATCH_SIZE × 4` (= 16) | **Producer-side.** Prompts attempted per producer call; raise to fill the buffer faster (`GEN_BATCH_SIZE=32` for 8 ×). |
| `data.max_prompt_length` | `31232` | 31k context |
| `data.max_response_length` | `16384` | Response budget per turn |
| `actor_rollout_ref.rollout.n` | `8` | Samples per prompt (DAPO-aligned) |
| `actor_rollout_ref.model.lora_rank` | `32` | Trainer side; pool applies rank-16 adapter after quant |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | LoRA-safe LR |
| `actor_rollout_ref.actor.tis_imp_ratio_cap` | `5` | TIS clamp (wide enough for T-mismatch + kernel floor; see handsoff §27) |
| `actor_rollout_ref.rollout.openhands_num_workers` | `32` | Sweet spot for 4-child pool (64 regresses per handsoff §17) |
| `actor_rollout_ref.rollout.max_iterations` | `30` | Max agent turns |
| `actor_rollout_ref.rollout.openhands_timeout` | `1000` | Per-job seconds |
| `actor_rollout_ref.rollout.temperature` | `1.4` | High exploration |
| `actor_rollout_ref.rollout.top_p` | `0.95` | — |
| `actor_rollout_ref.rollout.external_llm_endpoints` | 4× pool URLs | — |
| `actor_rollout_ref.rollout.publish_on_save` | `True` | LoRA auto-publish on save |
| `actor_rollout_ref.actor.use_kl_loss` | `False` | RLVR — no reward-model drift to anchor against |
| `actor_rollout_ref.actor.clip_ratio_low/high` | `0.2 / 0.28` | DAPO clip-higher |
| `+replay.stop_timeout_s` | `300` | Cooperative-stop window for the producer thread (handsoff §19, §25) |
| `+replay.no_progress_timeout_s` | `1800` | Replay-store no-progress detector (handsoff §31) |
| `trainer.n_gpus_per_node` | `8` | FSDP degree |
| `trainer.resume_mode` | `auto` | Picks latest `global_step_*` in STAGE2_OUT |
| `trainer.val_before_train` | `False` | In-run validation disabled by default |
| `trainer.test_freq` | `-1` | In-run eval disabled by default |

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

# Producer-mode leftover-job rebuild marker (one per DAPO call)
grep "dropped [0-9]* leftover jobs" /tmp/s3-fullasync.log | wc -l

# §19 cooperative-stop skip — should be rare
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

Project: `ProAgent`. Experiment: `fullasync-replay-prorl`. Key panels (handsoff §5):
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
4. **Note**: add gotcha to `plans-n-solutions/handsoff.md §6` (or bump gotcha number if new).
5. **Rerun**: same command as before; `resume_mode=auto` picks up.

### Common failures

| Symptom | Root cause | Fix |
|---|---|---|
| `/health` timeout during pre-flight | Pool not up or SG blocks trainer IP | Run `launch_remote_vllm_pool.sh start`; confirm EC2 SG 8100-8103 inbound |
| `endpoints_failed > 0` in `weight_sync/*` | One vLLM child OOM'd or drained; partial publish → mixed policy versions | Abort (trainer does this automatically). Restart pool. Resume. |
| `did not exit within Ns; leaving thread running` | Producer stuck mid-`asyncio.run(generate_sequences)` during save | Handled automatically — caller skips validation, retries next boundary. Verify `skipping _validate` warning appears, then normal progress. |
| Tracebacks with `NoneType.concat` in DAPO | `all_input_batch` leak across producer calls (handsoff §18) | Reset fires each call ("dropped N leftover jobs" marker). Count should equal producer call count. |
| tqdm frozen with no producer-mode markers | Producer wedged | `docker rm -f s3-fullasync`, restart pool, resume |
| 5xx on `/generate` during publish | Pool drain race | Non-fatal under load (`drain_timed_out:true, ok:true`). Count should stay low. Concern if > 10% of calls. |
| `Replay store made no forward progress` | No new groups pushed for `no_progress_timeout_s` (default 1800 s) | Producer is genuinely wedged. Stop trainer, restart pool, resume. |

## Short smoke test (reproducible)

```bash
# 2 steps, save every step — minimal end-to-end exercise
TOTAL_TRAINING_STEPS=2 SAVE_FREQ=1 LOG_PATH=/tmp/smoke.log \
  bash scripts/_internal/s3_fullasync_docker.sh

# Expected:
grep "step:" /tmp/smoke.log        # 2 metric lines
grep publish_lora_adapter /tmp/smoke.log | wc -l  # 2 publishes
grep -c Traceback /tmp/smoke.log   # 0
```

## Frozen — do not edit

- `scripts/_internal/s2_weightsync_docker.sh` (matched-`global_steps` lock-step A/B baseline)
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh` (same)
- `dev_config/python/**` (lint/type/format configs — require explicit approval to change)
- `/tmp/verl/**` (pinned verl checkout at commit `910ba344`)

## Validation after a run

In-run validation is optional (`trainer.test_freq`, `trainer.val_before_train`). Offline A/B uses the eval-harness skill against `validation.parquet` (23 prompts, input_hash pass@k) at matched `global_steps`.

Success signals (all should hold on a healthy run):
1. `weight_sync/endpoints_failed == 0` end-to-end
2. ≥ 4 `/reload_lora` events per 20 steps at `save_freq=5`
3. Zero 5xx on `/generate` during publishes
4. `replay/sample_age_steps_p95 ≤ K` (K=4) AND `rollout/staleness_steps_p95 ≤ K + save_freq`
5. `is_weight/p99 < 10`, `is_weight/clip_fraction < 0.25` (with `tis_imp_ratio_cap=5`)
6. `critic/rewards/mean` trends up on long runs
7. Offline A/B: full-async ≥ baseline pass@k on validation.parquet
8. Both `filter_groups={False, True}` runs land clean
9. Zero tracebacks
10. Token-in/token-out golden-file preserved

## Related docs

- `plans-n-solutions/handsoff.md` — topology, pointer table, gotchas, credentials
- `openhands/nvidia/README.md` — FastAPI job lifecycle
- `openhands/llm/nvidia/README.md` — token-in/token-out invariant
- `CLAUDE.md` — architectural invariants
