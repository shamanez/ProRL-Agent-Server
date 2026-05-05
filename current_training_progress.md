# Training Progress Report

**Report timestamp:** 2026-05-05 06:41 UTC

---

## Current Step Count

- **Completed steps:** 1 (from the original trainer run, wandb run `4jlj31ed`)
- **Current trainer state:** STOPPED — needs manual restart
- **New wandb run created:** `zv33vvxt` (the second container that died at step 0)

The original trainer container completed step 1 (~22 min per step, step 1 finished at ~05:46 UTC).
A second container was launched at 06:33 UTC by another agent session running `start_all.sh`,
which died before completing step 1 due to a vLLM health probe failure (the start_all.sh
killed the existing vLLM processes then could not reach them at `vllm-instance:8100`).

---

## Latest Loss Values

Not available — the trainer was killed before completing step 2. Step 1 metrics are in
wandb run `4jlj31ed` at https://wandb.ai/shamanework-pl/ProAgent/runs/4jlj31ed

---

## Weight Sync Status

**NO — policy_registry.db has 0 entries.**

The trainer is configured with `save_freq=5` and `lora_rank=32`. The first LoRA checkpoint
and `publish_policy_version` call will happen at step 5. Step 1 has been completed but
no checkpoint was saved (save_freq=5 means saves at steps 5, 10, 15...).

---

## Errors Found and Fixed

### Critical Bug: Stale pytest manifests poisoning PolicyVersionCache

**Root cause:** The policy subscription file poller reads `/tmp/prorl_policy_manifest.json`.
Pytest tests (specifically tests for `policy_registry`) write to this same path as part
of test fixtures. Multiple pytest runs (pytest-5 through pytest-11) wrote versions 1, 3,
5, 7 to the manifest during the training run.

**Symptom:** RolloutManager's `PolicyVersionCache` reached version=7 (from pytest-5 at
06:12 UTC). With `policy_version=7`, ProRL routes inference to `/v7/generate` on vLLM.
Since no LoRA adapter version 7 exists, vLLM returned HTTP 410 Gone. All episodes after
06:12 returned empty `response_token_ids` — groups dropped — LiveStore stuck at 2 groups
— trainer blocked waiting for 4 groups (BATCH_SIZE=4). Confirmed in `/tmp/s0-prorl.log`:
`httpx.HTTPStatusError: Client error '410 Gone' for url '.../v7/generate'`

**Fix applied:**
1. Wrote `version=0` to `/tmp/prorl_policy_manifest.json`.
2. Restarted RolloutManager (PID 1030397 killed, new PID 1316270) with
   `--policy-manifest-path /tmp/prorl_policy_manifest_prod.json` — a dedicated path
   that pytest does NOT write to. This prevents future pytest runs from poisoning
   the production PolicyVersionCache.
3. Confirmed new RM started at `version=0` and is dispatching successfully.

### Cascade failure: start_all.sh killed all services at 06:33 UTC

A second Claude agent session called `bash scripts/services/start_all.sh` at ~06:33 UTC.
The script killed the existing vLLM EC2 processes and launched new ones, then ran a health
probe against `http://vllm-instance:8100/health`. DNS resolution failed within 120s, so
the script executed its shutdown path, cleanly killing LiveStore, PolicyRegistry, and the
new trainer container.

**Fix applied:** Restarted both LiveStore (PID 1321160) and PolicyRegistry (PID 1321400)
manually at 06:39 UTC. Both sockets are healthy. The new vLLM pool is at
`policy_version=0` (fresh state, no LoRA adapters).

---

## Service Health Status

| Service | Status | Details |
|---|---|---|
| EnvironmentProvider (ProRL :8006) | HEALTHY | `running`, 1 pending job |
| InferenceBackend (vLLM :8100-8103) | HEALTHY | All 4 at `policy_version=0`, fresh |
| LiveStore (`/tmp/prorl_live_store.sock`) | HEALTHY | Fresh restart, 0 groups, PID 1321160 |
| PolicyRegistry (`/tmp/prorl_policy_registry.sock`) | HEALTHY | Fresh DB, PID 1321400 |
| RolloutManager (PID 1316270) | HEALTHY | `version=0`, 3 episodes, 0 groups pushed |
| TrainerAdapter (`s3-fullasync`) | STOPPED | Needs manual restart |

---

## RolloutManager State (as of 06:40 UTC)

- PID: 1316270
- Manifest: `/tmp/prorl_policy_manifest_prod.json` (isolated from pytest)
- Policy version: 0 (correct — base model, no LoRA)
- Episodes total: 3 (fresh session, started 06:36 UTC)
- Groups pushed: 0 (group_size=4, need 4 episodes to complete one group)
- First group ETA: ~06:46 UTC

---

## Action Required (HUMAN INTERVENTION NEEDED)

**The TrainerAdapter must be restarted after the RolloutManager pushes the first group.**
Per BC-16, trainer should not start until LiveStore has at least 1 group.
First group ETA: ~06:46 UTC (estimate).

```bash
# Wait until LiveStore has >= 1 group, then:
bash scripts/_internal/s3_fullasync_docker.sh \
  data.train_files=[/data/SkyRL-v0-293/train.ready.parquet] \
  data.val_files=[/data/SkyRL-v0-293/train.ready.parquet] \
  >> /tmp/s3-fullasync.log 2>&1 &
echo $! > /tmp/trainer.pid
```

The trainer will start from step 0 (no checkpoint — save_freq=5 was never reached in
the disrupted runs). New WandB run: `zv33vvxt`.

---

## Root Cause Prevention Recommendation

The test suite writes to `/tmp/prorl_policy_manifest.json` (same path as the production
manifest). Fix: in pytest fixtures for `policy_registry` tests, use a `tmp_path`-scoped
manifest path instead of `DEFAULT_MANIFEST_PATH`. This is the same pattern already used
for test sockets and test DB paths in the same test suite.

Relevant files:
- `/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/rollout_manager/policy_subscription.py` (line 20: `DEFAULT_MANIFEST_PATH`)
- `/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/policy_registry/file_registry.py` (line 21: `DEFAULT_MANIFEST_PATH`)
