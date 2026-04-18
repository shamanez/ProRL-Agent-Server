# `scripts/serving/`

External vLLM pool for decoupled rollouts. Two deployment modes:

- **Local pool** (Stage 1, Cuts A + B) — supervisors run in Docker on the same box as the trainer, GPUs 4–7. Launcher: `launch_external_vllm_pool.sh`. See [`plans-n-solutions/stages/stage1.md`](../../plans-n-solutions/stages/stage1.md).
- **Remote pool** (Stage 1, Cut C) — children run on a separate EC2 host, trainer talks over public HTTP. Launcher: `launch_remote_vllm_pool.sh`. See [`plans-n-solutions/stages/stage1_remote_pool.md`](../../plans-n-solutions/stages/stage1_remote_pool.md).

Both modes share the same token-level `{prompt_ids} → {response_ids, logprobs}` contract served by `_vllm_child.py`. They differ only in supervision (local has a `vllm_launcher.py` FastAPI supervisor; remote runs the child directly under `nohup`).

## Files

| File | Role |
|---|---|
| `vllm_launcher.py` | Supervisor FastAPI app (runs inside the verl Docker image). One `subprocess.Popen` owns a child `_vllm_child.py` on port `N+1000`. Routes: `GET /health`, `POST /generate` (thin pass-through), `POST /reload_weights` (501 stub). Child is started with `prctl(PR_SET_PDEATHSIG, SIGTERM)` so the kernel reaps it when the supervisor dies — including via `SIGKILL`. Clean shutdown: `SIGTERM/SIGINT → child SIGTERM → 30 s wait → SIGKILL`. |
| `_vllm_child.py` | vLLM 0.18 `AsyncLLMEngine`-backed `/generate` server. Accepts `{prompt_ids, …}` and returns `{response_ids, logprobs}` — same contract as `openhands/llm/nvidia/qwen3.py` expects. No text detokenization anywhere on the hot path. |
| `launch_external_vllm_pool.sh` | Host shell launcher. `docker run` one `verlai/verl:vllm018.dev1` container per (GPU, port) pair. |

## Port discipline

- Supervisor: port `N` (e.g. `8100`). Registered with ProRL via `POST /add_llm_server`.
- Child vLLM: port `N+1000` (e.g. `9100`). Loopback only.
- Child log: `/tmp/vllm-child-<N>.log` on the host (via `-v /tmp:/tmp`).
- Supervisor log: `/tmp/vllm-sup-<N>.log` on the host.
- PID files: `/tmp/vllm-sup-<N>.pid` (host docker-client), `/tmp/vllm-child-<N>.pid` (child vLLM inside container).

## Design note — why not the OpenAI server

ProRL's `openhands/llm/nvidia/qwen3.py` client POSTs to `/generate` with
`{prompt_ids: [int], …}` and expects `{response_ids: [int], logprobs: [float]}`.
That is *not* OpenAI `/v1/completions`, and vLLM 0.18's stock
`vllm.entrypoints.api_server` speaks `{prompt: str} -> {text: [str]}`
(different contract). We therefore ship `_vllm_child.py` — a tiny wrapper
around `AsyncLLMEngine` that serves the exact contract the client speaks.
The supervisor proxies `/generate` byte-for-byte to the child; no
translation, no re-tokenization, which preserves the token-level invariant
documented in `openhands/llm/nvidia/README.md`.

## Part A (smoke) use

```bash
# Terminal 1 — ProRL on host (unchanged Stage 0 launcher)
bash scripts/_internal/s0_prorl.sh

# Terminal 2 — 2 supervisors on GPUs 0,1
bash scripts/serving/launch_external_vllm_pool.sh --gpus 0,1 --ports 8100,8101

# Wait for health, then register with ProRL
for p in 8100 8101; do
  until curl -sf "http://localhost:$p/health" >/dev/null; do sleep 2; done
  curl -sX POST http://localhost:8006/add_llm_server \
    -H 'Content-Type: application/json' -d "{\"url\":\"http://localhost:$p\"}"
done
curl -sX POST http://localhost:8006/start

# Smoke test
poetry run python scripts/tests/test_external_vllm.py
```

## Part B (trainer bypass) use

Same launcher, different GPUs/ports:

```bash
bash scripts/serving/launch_external_vllm_pool.sh --gpus 4,5,6,7 --ports 8100,8101,8102,8103
```

The trainer runs on GPUs `0-3` via `scripts/_internal/s2_decoupled_docker.sh`.

## Teardown

```bash
for p in 8100 8101 8102 8103; do
  docker stop "vllm-sup-$p" 2>/dev/null || true
done
rm -f /tmp/vllm-sup-*.pid /tmp/vllm-child-*.pid
```

## Remote pool (Stage 1 — Cut C)

When the vLLM pool has to live on a different box than the trainer (EC2
`vllm-instance`, 4 × 23 GiB), use the remote orchestrator instead of the
Docker launcher. Direct HTTP over the public EC2 DNS — no SSH tunnel, no
supervisor on the remote. Full plan + problem log:
[`plans-n-solutions/stages/stage1_remote_pool.md`](../../plans-n-solutions/stages/stage1_remote_pool.md).

| File | Role |
|---|---|
| `launch_remote_vllm_pool.sh` | Orchestrator with `bootstrap` / `start` / `stop` subcommands. `bootstrap` rsyncs `_vllm_child.py` + the remote runner + `requirements-remote.txt` to `vllm-instance:~/vllm-pool/`, creates a python3.12 venv, pip-installs, and `hf download`s the model into `~/vllm-pool/hf-cache`. `start` SSHes in and `nohup`s 4 children on GPUs 0–3, then polls `/health` over the public DNS. `stop` kills the PIDs (plus a `VLLM::EngineCore` orphan sweep) and rsyncs remote child logs back to `/tmp/vllm-child-<port>.log` so `scripts/validate_run.py` keeps working. |
| `_remote_vllm_runner.sh` | Remote-side helper invoked under `nohup` by the orchestrator. Sources the venv, exports `CUDA_VISIBLE_DEVICES`, `PYTORCH_ALLOC_CONF`, `HF_HOME`, writes its PID file, and `exec`s `_vllm_child.py`. |
| `requirements-remote.txt` | Pins `vllm==0.18.*` + the FastAPI stack the remote venv needs. |
| `teardown_remote_vllm_pool.sh` | One-liner wrapper → `launch_remote_vllm_pool.sh stop`. |

Use:

```bash
# One-time (idempotent)
bash scripts/serving/launch_remote_vllm_pool.sh bootstrap

# User: open EC2 security-group inbound 8100-8103 from trainer-box public IP.

# Start, verify health, run training (via scripts/_internal/s1_remote_docker.sh),
# then teardown.
bash scripts/serving/launch_remote_vllm_pool.sh start
bash scripts/serving/teardown_remote_vllm_pool.sh
```

Overrides via env: `REMOTE_HOST`, `REMOTE_DNS`, `REMOTE_POOL_DIR`,
`REMOTE_PYTHON`, `MODEL`, `GPU_MEM_UTIL` (default 0.85), `MAX_MODEL_LEN`.

### Child launch flags (remote)

Each remote `_vllm_child.py` is invoked by `_remote_vllm_runner.sh` with:

```
--gpu-memory-utilization $GPU_MEM_UTIL   # default 0.85
--max-model-len $MAX_MODEL_LEN           # default 17920
--dtype bfloat16
--enable-chunked-prefill
--enable-prefix-caching                  # big win for GRPO n=4 shared prefixes
--max-num-batched-tokens 16384
--max-num-seqs 128
```

No `--enforce-eager` — CUDA graphs are on, adds ~30–60 s one-time capture to `start` but gives ~30–50% throughput lift on Qwen3-4B. The 300 s health budget in `s1_remote_docker.sh` covers the capture.
