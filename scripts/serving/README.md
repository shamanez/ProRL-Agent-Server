# `scripts/serving/`

Remote vLLM pool for decoupled rollouts. Children run on a separate EC2 host (`vllm-instance`, 4 × 23 GiB GPUs); the trainer talks to them over public HTTP. Topology, credentials, and launch sequence: [`plans-n-solutions/handsoff.md`](../../plans-n-solutions/handsoff.md).

Token-level contract (same as `openhands/llm/nvidia/qwen3.py` expects): `POST /generate` with `{prompt_ids: [int], …}` returns `{response_ids: [int], logprobs: [float]}`. No text detokenization anywhere on the hot path — this preserves the token-level invariant documented in `openhands/llm/nvidia/README.md`. Stock `vllm.entrypoints.api_server` speaks a different contract (`{prompt: str} -> {text: [str]}`), so we ship `_vllm_child.py` as a thin wrapper around `AsyncLLMEngine` that serves the exact contract the client speaks.

## Files

| File | Role |
|---|---|
| `_vllm_child.py` | vLLM 0.18 `AsyncLLMEngine`-backed `/generate` server. Accepts `{prompt_ids, …}` and returns `{response_ids, logprobs}`. Exposes `POST /reload_lora` for trainer-driven adapter swaps. |
| `launch_remote_vllm_pool.sh` | Orchestrator with `bootstrap` / `start` / `stop` subcommands. `bootstrap` rsyncs `_vllm_child.py` + the remote runner + `requirements-remote.txt` to `vllm-instance:~/vllm-pool/`, creates a python3.12 venv, pip-installs, and `hf download`s the model into `~/vllm-pool/hf-cache`. `start` SSHes in and `nohup`s 4 children on GPUs 0–3, then polls `/health` over the public DNS. `stop` kills the PIDs (plus a `VLLM::EngineCore` orphan sweep) and rsyncs remote child logs back to `/tmp/vllm-child-<port>.log`. |
| `_remote_vllm_runner.sh` | Remote-side helper invoked under `nohup` by the orchestrator. Sources the venv, exports `CUDA_VISIBLE_DEVICES`, `PYTORCH_ALLOC_CONF`, `HF_HOME`, writes its PID file, and `exec`s `_vllm_child.py`. |
| `requirements-remote.txt` | Pins `vllm==0.18.*` + the FastAPI stack the remote venv needs. |
| `teardown_remote_vllm_pool.sh` | One-liner wrapper → `launch_remote_vllm_pool.sh stop`. |

## Use

```bash
source /home/ubuntu/.prorl_creds.env

# One-time (idempotent): rsync code + create venv + cache the base model on the remote.
bash scripts/serving/launch_remote_vllm_pool.sh bootstrap

# Before first `start`: open EC2 security-group inbound 8100-8103 from the trainer public IP.

# Bring the 4 children up (~90 s including CUDA graph capture); verify with /health.
bash scripts/serving/launch_remote_vllm_pool.sh start
for p in 8100 8101 8102 8103; do
  curl -sS -m 5 -o /dev/null -w "pool :$p = %{http_code}\n" \
    "http://ec2-54-145-77-207.compute-1.amazonaws.com:$p/health"
done

# Teardown (rsyncs child logs back to /tmp/vllm-child-<port>.log on the trainer host).
bash scripts/serving/teardown_remote_vllm_pool.sh
```

Env overrides: `REMOTE_HOST`, `REMOTE_DNS`, `REMOTE_POOL_DIR`, `REMOTE_PYTHON`, `MODEL`, `GPU_MEM_UTIL` (default 0.85), `MAX_MODEL_LEN` (default 32768).

## Child launch flags

Each remote `_vllm_child.py` is invoked by `_remote_vllm_runner.sh` with:

```
--gpu-memory-utilization $GPU_MEM_UTIL   # default 0.85
--max-model-len $MAX_MODEL_LEN           # default 32768
--dtype bfloat16
--enable-chunked-prefill
--enable-prefix-caching                  # big win for GRPO n=4 shared prefixes
--max-num-batched-tokens 16384
--max-num-seqs 128
```

No `--enforce-eager` — CUDA graphs are on, adds ~30–60 s one-time capture to `start` but gives ~30–50% throughput lift on Qwen3-4B. The 300 s health budget in `scripts/_internal/s2_weightsync_docker.sh` covers the capture.
