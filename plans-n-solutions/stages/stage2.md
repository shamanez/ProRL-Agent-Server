# Stage 2 - Trainer bypass (stale weights)

**Status: NOT STARTED**

## Plan

Make the trainer skip in-Ray vLLM startup and target Stage 1's external pool. Run 20 GRPO steps with intentionally stale weights to verify the decoupled plumbing independently of weight sync.

### Files modified

- `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` - new `external_llm_endpoints` config; skip `start_llm_servers()`, no-op `wake_up()`/`sleep()`
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_decoupled.sh` - new fork with 4-GPU trainer config
- `scripts/validate_run.py` (new) - automated 20-step gate checker

### GPU plan

| Pool | GPUs | Role |
|---|---|---|
| Trainer (FSDP) | 0-3 | `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| External vLLM | 4-7 | 4 supervisors, TP=1, ports 8100-8103 |

### Success criteria

- 20 steps completed, validator green with `--expect-weight-publishes 0`
- External pool served all rollout traffic
- No Ray "actor not found" traces
- `make lint` + fast test loop green

### Risks

- DP-size mismatch vs endpoint count (mitigated by ProRL's heap LB)
- Growing KL(actor, rollout) from stale weights (expected, not a bug)
- Ray picks wrong GPUs (enforce via `CUDA_VISIBLE_DEVICES`)

## Solution

*Not started yet.*
