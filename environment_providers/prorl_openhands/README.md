# ProRL / OpenHands EnvironmentProvider

**Source directory:** `../../openhands/` (the ProRL SWE-Bench implementation)

This is the current default EnvironmentProvider. It implements the `POST /process`
contract (see `../README.md`) for SWE-Bench tasks via Singularity containers.

## Python environment

Requires the **full** poetry env (`rollout-fabric-*`). Set `PRORL_OPENHANDS_PYTHON`
to point at it. Never use the fabric-core minimal venv for this service.

## Launch

```bash
bash scripts/adapters/start_env_prorl.sh
# Health: GET http://localhost:8006/status → {"status": "running"}
```

## Frozen files (DO NOT EDIT)

These files maintain the token-ID integrity invariant (BC-1). Re-tokenizing decoded
text across multi-turn episodes shifts token boundaries and causes KL/entropy NaN:

- `openhands/llm/nvidia/qwen3.py`
- `openhands/llm/nvidia/qwen2_5_vl.py`
- `scripts/inference/_vllm_child.py`
- `openhands/nvidia/async_server.py`

To add support for a different model, create a sibling file — do not edit these.

## Key internal modules

- `openhands/nvidia/async_server.py` — FastAPI request handler (frozen)
- `openhands/nvidia/registry.py` — agent handler registry
- `openhands/llm/nvidia/qwen3.py` — token-in/token-out LLM backend (frozen)
- `openhands/runtime/singularity/` — Singularity container lifecycle
