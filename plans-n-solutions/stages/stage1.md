# Stage 1 - External vLLM standalone

**Status: NOT STARTED**

## Plan

Launch vLLM as a standalone process outside Ray. Prove ProRL can route to it and return a valid SWE-Bench result. No training - isolates the first coupling break (serving off-actor).

### What gets built

1. **`scripts/serving/vllm_launcher.py`** - supervisor that spawns vLLM as a child, exposes `/health`, `/generate` (proxied), `/reload_weights` (stub, returns 501).
2. **`scripts/serving/launch_external_vllm_pool.sh`** - takes `--gpus 0,1 --ports 8100,8101`, launches N supervisors with GPU pinning.
3. **`scripts/tests/test_external_vllm.py`** - smoke test: 2 SWE-Bench instances through ProRL.

### GPU plan

Inference only, 2 GPUs:

| GPU | Role | Port |
|---|---|---|
| 0 | vLLM supervisor 0 | 8100 |
| 1 | vLLM supervisor 1 | 8101 |

### Test plan

| # | Test | Pass criteria |
|---|---|---|
| 1 | Supervisors boot | Both `/health` return 200 |
| 2 | ProRL routes to external vLLM | `/add_llm_server` + `/start` + `/status` shows endpoints |
| 3 | End-to-end SWE-Bench rollout | 2 instances return a `report` with `resolved` field |
| 4 | vLLM logs show traffic | `POST /generate` count > 0 per supervisor |
| 5 | `/reload_weights` stub | Returns 501 |
| 6 | No trainer code touched | `git diff trainer_integration/` is empty |

### Risks

- Port conflicts between inner/outer ports
- Long cold-start for HF weight download (pre-warm with `huggingface-cli download`)
- Supervisor child process leak on ungraceful shutdown

## Solution

*Not started yet.*
