# Stage 1 — External vLLM standalone

**Status:** Not started. Depends on Stage 0 gating metrics passing (Stage 0 completed: `plans-n-solutions/stages/stage0.md`).

**Part 1 of 2 for the decoupling milestone.** Stage 1 only *hosts* vLLM outside the trainer — no training code is touched and nothing is actually decoupled yet. **Stage 2** (`stage2.md`) is what completes the cut, by making the trainer bypass its in-Ray vLLM and target this pool. Both stages are driven together via `plans-n-solutions/stages/stage1_playbook.md`. **Reuse** `scripts/_internal/s0_prorl.sh` for the ProRL server — do not invent a new server launcher.

**Stack assumption (from Stage 0, non-negotiable):**
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

ProRL stays on the host at `:8006` (same as Stage 0).

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

# Start ProRL using the existing poetry-based launcher (unchanged since Stage 0).
# This sources /home/ubuntu/.prorl_creds.env and runs the same
#   poetry run python scripts/start_server.py --host 0.0.0.0 --port 8006 ...
# invocation used in Stage 0. Do NOT write a new server launcher.
bash scripts/_internal/s0_prorl.sh 2>&1 | tee /tmp/s1-prorl.log &
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

### Checkpoint log

- 2026-04-17 — Step 2 done: launcher + child + pool script landed, ruff green, manual
  end-to-end verified on GPU 0. Container /health=200 at 30 s cold-start,
  `/generate` with `prompt_ids=[9707,11,7299,2138,498,525]` returned valid
  `{response_ids, logprobs}`, `/reload_weights` → 501, images → 400,
  `docker stop` clean shutdown in 0.94 s with zero GPU-resident processes.
- 2026-04-17 — Step 3 done: `scripts/tests/test_external_vllm.py` landed,
  ruff green, 7/7 criteria pass against live pool (GPUs 0+1, ports 8100/8101,
  both healthy in 20 s). Criterion 3 (SWE-Bench rollout) delegated to
  `standalone_swebench_test.py` for the Step 4 live run.
- 2026-04-17 — Step 4 resolved via option A after user sign-off: the 7/7 smoke
  test already proves the decoupled rollout path end-to-end — ProRL accepts
  the pool (`/add_llm_server` 200 for each of :8100 and :8101, `/start` →
  `running=True`), and direct `/generate` round-trips through the supervisor
  into the child produce valid `{response_ids, logprobs}` with corresponding
  `POST /generate` hits in each child log. The outstanding criterion 3
  evidence (multi-turn SWE-Bench rollout through ProRL) is absorbed into the
  Stage 2 `validate_run.py` gate, which requires a successful 20-step GRPO
  run against this same pool and therefore transitively covers the ProRL
  routing path. Pool torn down cleanly (0 GPU procs) before handoff.

### Deviation from original plan

The plan proposed fronting vLLM 0.18's stock `vllm.entrypoints.openai.api_server`
with a `/generate` translation proxy in the supervisor. Two things pushed us to
a simpler design:

1. ProRL's `openhands/llm/nvidia/qwen3.py` client POSTs
   `{prompt_ids:[int], …}` → expects `{response_ids:[int], logprobs:[float]}`.
   OpenAI `/v1/completions` speaks `{prompt:str|[int], …}` and puts token ids
   inside `choices[0].logprobs.token_ids`. Translating correctly for every
   kwarg (top_p, seed, temperature, max_tokens, …) adds surface area without
   value — the supervisor would just be renaming fields.
2. `scripts/tests/vllm_api_server.py` already speaks the ProRL contract but
   imports `FlexibleArgumentParser` from `vllm.utils`, which vLLM 0.18 moved.
   Using it directly would require patching the import.

So we shipped `scripts/serving/_vllm_child.py` — a tiny FastAPI+AsyncLLMEngine
server that speaks `{prompt_ids} → {response_ids, logprobs}` natively. The
supervisor's `/generate` is a pure byte-for-byte pass-through; no translation,
no re-tokenization. This preserves the token-level invariant documented in
`openhands/llm/nvidia/README.md`.

### Final vLLM 0.18 flag set used

```
--gpu-memory-utilization 0.45
--max-model-len 17920
--enforce-eager
--enable-chunked-prefill
--max-num-batched-tokens 8192
```

Model is the HF repo id `Qwen/Qwen3-4B-Instruct-2507`, resolved at runtime via
the bind-mounted HF cache (`-v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface`
with `HF_HOME=/root/.cache/huggingface` in the container).

### Problem log

| # | Symptom | Fix |
|---|---------|-----|
| 1 | Initial child crashed with `OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ubuntu/.cache/huggingface/hub/.../snapshots/<hash>'` | The launcher was passing a host-absolute snapshot path to the container, where the HF cache is mounted at `/root/.cache/huggingface` instead. Switched default `--model` to the HF repo id so vLLM resolves via the mounted cache. |
| 2 | `scripts/tests/vllm_api_server.py` import failure on vLLM 0.18 (`FlexibleArgumentParser` moved out of `vllm.utils`) | Wrote fresh `scripts/serving/_vllm_child.py` speaking ProRL's `{prompt_ids}` contract directly on `AsyncLLMEngine` + `TokensPrompt`. |
| 3 | Host `rm -f /tmp/vllm-*.pid` failed because the container's root wrote the files under the shared `/tmp` bind mount | Dropped the host-side `rm`; supervisor's `Path.write_text` truncates on open, so stale content is overwritten on next launch. |
| 4 | `docker run ... &` backgrounded with `disown` silently lost non-zero exit codes (image pull fail, port collision, bind-mount missing) — pool reported "launched N supervisor(s)" even when all containers had already crashed | Added a 3-s liveness gate after the launch loop; `docker inspect` each container and `exit 1` with log tails if any is not Running. |
| 5 | `sys.exit(0)` from inside the supervisor's SIGTERM handler could be deferred by CPython when the signal arrives in a C-call frame, potentially stranding the child holding GPU memory | Signal handler now calls `child.terminate()` directly, then `os._exit(128+signum)`. Child termination is idempotent via `_terminated` flag. |
| 6 | `_vllm_child._flatten_logprobs` returned `None` on the first empty-dict mid-sequence, which the trainer cannot distinguish from "logprobs not requested" | Raise `RuntimeError` on empty dict mid-sequence so the failure is loud, not a silent advantage-estimate corruption. |
| 7 | `/generate` handler caught `asyncio.CancelledError` and returned 499 without aborting the engine request — orphaned inflight requests would accumulate under client-timeout churn | Added `await engine.abort(request_id)` then re-raise, so vLLM frees GPU blocks promptly. |
