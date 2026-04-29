# errors_solutions.md

Concise log of errors encountered during runs and the resolution applied. Append-only; one entry per distinct issue. Keep entries short — error → root cause → fix.

Format:

```
## YYYY-MM-DD HH:MM <branch / context>
**Error:** one-line symptom (paste the key log line, no full tracebacks)
**Root cause:** one sentence — why it happened
**Fix:** one or two lines — what changed or what command was run
**File(s):** path:line where applicable
```

---

## 2026-04-28 14:21 full-async-optimization-final-cut — switching to new vllm-instance (ec2-3-87-168-160…)

**Error:** Inner Hydra launcher contained `external_llm_endpoints=[http://ec2-54-145-77-207…:8100..8103]` — hardcoded to the previous EC2 host (handsoff gotcha §3).
**Root cause:** Old DNS baked into `run_proagent_qwn3_4B_instruct_fullasync.sh:132`; `s3_fullasync_docker.sh` already plumbed `REMOTE_DNS` into the container but the inner script didn't read it.
**Fix:** Parameterized to `${REMOTE_DNS:-ec2-54-145-77-207.compute-1.amazonaws.com}` so the existing env-knob in `s3_fullasync_docker.sh` propagates through.
**File(s):** `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh:132`

---

## 2026-04-28 14:22 full-async-optimization-final-cut — pool launcher /health probe failed against new instance

**Error:** `[start] ERROR: http://ec2-54-145-77-207.compute-1.amazonaws.com:8100/health did not become healthy within 300 s` even though the children booted on the new EC2 host (`Uvicorn running on http://0.0.0.0:8100` was visible in the SSH-streamed remote log).
**Root cause:** `launch_remote_vllm_pool.sh` defaults `REMOTE_DNS` to the old hostname. SSH alias `vllm-instance` correctly pointed at the new host (so children started fine), but the local `/health` probe used the default.
**Fix:** Re-invoke with `REMOTE_DNS=ec2-3-87-168-160.compute-1.amazonaws.com` set inline. Verified all 4 children healthy via both `ssh vllm-instance 'curl … /health'` and direct probe through the public DNS.
**File(s):** `scripts/serving/launch_remote_vllm_pool.sh:38` (default), invocation pattern.

---

## 2026-04-28 14:21 full-async-optimization-final-cut — stale orphan `s0_prorl.sh` from prior session

**Error:** Two `s0_prorl.sh` PIDs after a single launch. Investigation showed one was started Apr23 (PPID=1, python child long dead, only `bash + tee` orphans).
**Root cause:** A previous session's ProRL was killed without reaping its bash wrapper. Pipeline (`python … | tee`) kept the bash blocked even after python died because tee was still alive on init's adoption.
**Fix:** `kill <bash-pid> <tee-pid>`. Confirmed only one ProRL bound to :8006 (pid 986757).
**File(s):** N/A — session-state artifact.
