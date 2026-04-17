# `scripts/serving/`

Stage 1 external vLLM pool. Runs outside the trainer so ProRL can route rollouts
to a standalone inference endpoint. See `plans-n-solutions/stages/stage1.md` for
the full stage plan and `plans-n-solutions/stages/stage1_playbook.md` for the
combined Stages 1 + 2 runbook.

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

## Stage 1 use

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

## Stage 2 use

Same launcher, different GPUs/ports:

```bash
bash scripts/serving/launch_external_vllm_pool.sh --gpus 4,5,6,7 --ports 8100,8101,8102,8103
```

The trainer (Stage 2) runs on GPUs `0-3` via `scripts/_internal/s2_decoupled_docker.sh`.

## Teardown

```bash
for p in 8100 8101 8102 8103; do
  docker stop "vllm-sup-$p" 2>/dev/null || true
done
rm -f /tmp/vllm-sup-*.pid /tmp/vllm-child-*.pid
```
