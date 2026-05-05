# openhands/nvidia/ — EnvironmentProvider Implementation

This directory is the **ProRL SWE-Bench EnvironmentProvider** — one concrete
implementation of `schemas.protocols.environment_provider.EnvironmentProvider`.

For the navigation index and the `POST /process` contract, see
`../../environment_providers/prorl_openhands/README.md`.

## Role in the fabric

```
RolloutManager ──POST /process──► ProRL FastAPI :8006
                                       │
                                  (token-in / token-out loop)
                                       │
                                  vLLM :8100-8103  (via qwen3.py / qwen2_5_vl.py)
                                       │
                                  Singularity sandbox
```

The RolloutManager calls `POST /process` via plain `httpx` (no openhands imports in
the RolloutManager — BC-13). The EnvironmentProvider owns the full tool-call loop,
tokenization, detokenization, and sandbox lifecycle.

## Frozen files — DO NOT EDIT

These files implement the token-ID invariant (BC-1). Re-tokenizing decoded text
across multi-turn episodes shifts token boundaries; actor vs reference diverges;
KL/entropy go NaN; PPO/GRPO/DAPO collapses.

| File | Why frozen |
|------|-----------|
| `qwen3.py` | Token-in/token-out LLM backend for Qwen3 |
| `qwen2_5_vl.py` | Token-in/token-out LLM backend for Qwen2.5-VL |
| `async_server.py` | FastAPI handler — depends on token ID alignment |
| `../../scripts/inference/_vllm_child.py` | vLLM child process server |

To add a different model: create a sibling file (e.g. `llama4.py`). Never edit
the frozen files.

## What is NOT ProRL-specific

The following concepts are required by ANY EnvironmentProvider but are NOT
openhands-specific:
- Serving `POST /process` on a TCP port
- Returning integer token IDs (BC-1)
- Not calling LiveStore or PolicyRegistry

To plug in ROCK or OpenReward, see `../../environment_providers/`.
