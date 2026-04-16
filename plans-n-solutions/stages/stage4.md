# Stage 4 - Policy Registry + blue-green cutover

**Status: NOT STARTED**

## Plan

Replace Stage 3's stop-the-world reload with blue-green: warm pool B while pool A serves, cut over atomically, drain A. Tag every rollout with `policy_version`.

### New files

- `openhands/nvidia/policy_registry.py` (~150 LOC) - version lifecycle: created, warming, active, draining, retired
- `scripts/serving/orchestrator.py` (~200 LOC) - manages pool A/B, drives warm/swap/drain
- `tests/nvidia/test_policy_registry.py` - state machine tests

### Files modified

- `openhands/nvidia/async_server.py` - stamp `policy_version` on jobs, atomic heap swap
- `scripts/start_server.py` - new `POST /swap_llm_servers` route
- `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` - POST to orchestrator instead of direct fan-out

### GPU plan

| Pool | GPUs | Ports | Initial state |
|---|---|---|---|
| A | 4-5 | 8100, 8101 | active |
| B | 6-7 | 8102, 8103 | idle |

### Success criteria

- 20 steps, >= 4 cutovers
- Every rollout has a `policy_version` tag
- Zero "No LLM server addresses" errors (atomic swap works)
- Registry shows full state-machine progression per version

### Risks

- Dual-pool VRAM (4 engines on 4 GPUs - fits with Qwen3-4B)
- Drain timeout kills long-running SWE jobs
- Orchestrator is a SPOF (accepted for this stage)

## Solution

*Not started yet.*
