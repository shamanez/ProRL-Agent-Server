# Stage 1 — External vLLM standalone

**Status:** Not started. Depends on Stage 0.1 gating metrics passing.

**Stack assumption (from Stage 0.1, non-negotiable):**
- Docker image for the trainer: `verlai/verl:vllm018.dev1`
- vLLM: 0.18 (both host-side standalone and the trainer-side client)
- verl: v0.8.0.dev (`shamanez/verl` main) at `/tmp/verl`
- Runtime OOM mitigations already in place: `PYTORCH_ALLOC_CONF=expandable_segments:True`, `gpu_memory_utilization=0.45`
- Token-level invariant: never modify `openhands/llm/nvidia/qwen3.py`

---

## Goal

Prove that ProRL can route SWE-Bench rollouts to a **standalone vLLM process** (started outside Ray, outside the trainer container) and return a valid result. No training code runs in this stage — it isolates the first coupling break: inference off the trainer actor.

A pass here means Stage 2 (decoupled training loop) can assume vLLM is reachable via HTTP at a stable endpoint, independent of the trainer lifecycle.

---

## Where it runs

2 GPUs out of the 8. No training is happening in this stage, so the other 6 are idle.

| GPU | Role | Port |
|---|---|---|
| 0 | vLLM supervisor 0 | 8100 |
| 1 | vLLM supervisor 1 | 8101 |

ProRL stays on the host at `:8006` (same as Stage 0.1).

---

## What gets built

Three new files, all under paths that don't exist yet. No edits to existing trainer code.

### 1. `scripts/serving/vllm_launcher.py`

Supervisor script. Spawns **one vLLM 0.18 OpenAI-compatible server** as a child process, pins it to a GPU, and exposes three HTTP routes of its own on the supervisor port:

| Route | Method | Behavior |
|---|---|---|
| `/health` | GET | 200 when child vLLM responds to `/v1/models`, 503 otherwise |
| `/generate` | POST | Proxies to child vLLM's `/v1/completions` with token-in / token-out headers matching the token-level invariant in `openhands/llm/nvidia/qwen3.py` |
| `/reload_weights` | POST | Returns **501 Not Implemented** for this stage (placeholder for Stage 4) |

Child launched with vLLM 0.18 CLI:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<snapshot> \
    --port <child_port> \
    --gpu-memory-utilization 0.45 \
    --max-model-len 17920 \
    --enforce-eager \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192
```

Supervisor owns the child via `subprocess.Popen`. On `SIGTERM` / parent death: send `SIGTERM` to the child, wait 30 s, then `SIGKILL`. No zombie on ungraceful shutdown.

### 2. `scripts/serving/launch_external_vllm_pool.sh`

Takes `--gpus 0,1 --ports 8100,8101` and launches one supervisor per (gpu, port). Each supervisor gets `CUDA_VISIBLE_DEVICES=<n>` and its own child port (supervisor port + 1000, e.g. 9100 / 9101). Writes PID files to `/tmp/vllm-sup-<port>.pid` for clean teardown.

### 3. `scripts/tests/test_external_vllm.py`

Smoke test covering the pass criteria in the Test plan below. Uses the existing ProRL client path, not a new one.

---

## Test plan

All six must pass to close Stage 1.

| # | Test | Pass criterion | How to check |
|---|---|---|---|
| 1 | Supervisors boot | Both `/health` return 200 within 60 s of `launch_external_vllm_pool.sh` | `for p in 8100 8101; do curl -sf http://localhost:$p/health; done` |
| 2 | ProRL registers endpoints | `/add_llm_server` for each supervisor, then `/start`, then `/status` shows both endpoints under `llm_servers` | `curl -s http://localhost:8006/status \| python3 -m json.tool` |
| 3 | End-to-end SWE-Bench rollout | 2 instances submitted via `scripts/run_swe.py` return a `report` dict with a `resolved` field (boolean) | `scripts/tests/test_external_vllm.py` asserts this |
| 4 | vLLM logs show traffic | `POST /v1/completions` count > 0 on each supervisor's child log | `grep -c "POST /v1/completions" /tmp/vllm-child-810[01].log` |
| 5 | `/reload_weights` stub | Returns HTTP 501 with JSON body `{"detail": "Not implemented in Stage 1"}` | `curl -X POST http://localhost:8100/reload_weights` |
| 6 | No trainer code touched | `git diff --stat HEAD trainer_integration/` is empty | single command check |

---

## Execution (self-contained — runs from a fresh session)

From project root `/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server`.

### Step 1 — Clean state + start ProRL

```bash
# Kill anything listening on 8006, 8100, 8101
for port in 8006 8100 8101 9100 9101; do
  fuser -k -9 "$port/tcp" 2>/dev/null || true
done
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'vllm_launcher'    2>/dev/null || true
sleep 2

source /home/ubuntu/.prorl_creds.env
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/singularity_images

nohup poetry run python scripts/start_server.py \
  --host 0.0.0.0 --port 8006 \
  --max-init-workers 64 --max-run-workers 64 \
  --timeout 1000 \
  > /tmp/s1-prorl.log 2>&1 &
```

### Step 2 — Launch the vLLM pool (built in this stage)

```bash
bash scripts/serving/launch_external_vllm_pool.sh \
  --gpus 0,1 --ports 8100,8101
```

Wait for both supervisors to report healthy:

```bash
for p in 8100 8101; do
  until curl -sf "http://localhost:$p/health" >/dev/null; do sleep 2; done
  echo "supervisor :$p healthy"
done
```

### Step 3 — Register with ProRL

```bash
for p in 8100 8101; do
  curl -sX POST "http://localhost:8006/add_llm_server" \
    -H 'Content-Type: application/json' \
    -d "{\"url\": \"http://localhost:$p\"}"
done
curl -sX POST "http://localhost:8006/start"
curl -s http://localhost:8006/status | python3 -m json.tool
```

### Step 4 — Run the smoke test

```bash
poetry run python scripts/tests/test_external_vllm.py
```

Exit code 0 = all six criteria pass.

### Step 5 — Teardown

```bash
curl -sX POST http://localhost:8006/stop || true
for f in /tmp/vllm-sup-*.pid; do
  [ -e "$f" ] && kill -TERM "$(cat "$f")" 2>/dev/null || true
done
kill $(cat /tmp/s1-prorl.pid 2>/dev/null) 2>/dev/null || true
```

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Port conflicts between supervisor port and child vLLM port | Fixed gap: supervisor on N, child on N+1000 |
| Long cold start while HF weights download | Pre-warm: model is already at `/home/ubuntu/.cache/huggingface/`. If missing: `huggingface-cli download Qwen/Qwen3-4B-Instruct-2507` before Step 2 |
| Supervisor child leaked on crash | `vllm_launcher.py` installs `atexit` + signal handlers that SIGKILL the child |
| Token-level routing drift | Supervisor `/generate` passes `prompt_ids` through verbatim; does not re-tokenize. Matches `openhands/llm/nvidia/qwen3.py` expectations |
| vLLM 0.18 flag regression vs 0.8.5 | Launcher uses 0.18-only args; no `--disable-mm-preprocessor-cache`; no deprecated flags. Validated against `vllm --help` inside the Docker image |

---

## Out of scope for this stage

- Dynamic weight updates → Stage 4 (`/reload_weights` is a 501 stub here)
- Trainer sends rollouts through this supervisor → Stage 2
- Multi-node → covered separately once single-node passes
- Eviction / sleep mode → Stage 3

---

## Solution

*To be filled in by the agent that completes this stage. Document:*
- *Final vLLM 0.18 CLI flag set used*
- *Any deviations from the plan and why*
- *Smoke test run output + timing*
- *WandB or log URL if applicable*
- *Any Stage 1 bugs that surfaced, same table format as Stage 0.1 Problem log*
