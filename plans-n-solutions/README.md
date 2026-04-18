# Decoupled Rollouts + Weight Sync

Branch `decoup-weight-sync` carries two things and nothing else: the decoupled multi-machine baseline (trainer ↔ remote vLLM pool over HTTP, stale weights) and the plan for closing the staleness gap, LoRA first.

| Doc | Status | Role |
|---|---|---|
| [`stages/baseline.md`](./stages/baseline.md) | DONE | Fork-point. WandB run `wdqqu52k`, 7 steps, 8/8 gates green. HTTP topology, rollout throughput, architecture, problem log. |
| [`stages/weight_sync_lora.md`](./stages/weight_sync_lora.md) | NEXT | Phase 1 plan: trainer trains a LoRA adapter, publishes it after every `save_freq` steps via `AsyncLLMEngine.add_lora`, pool serves subsequent rollouts against `{base + adapter}`. Phases 2 (full state-dict) and 3 (replay buffer) are deferred. |

## Hard constraints

- Single-box trainer: `8 × A100-SXM4-40GB`, no Slurm.
- Trainer runs in Docker: `verlai/verl:vllm018.dev1` (vLLM 0.18, PyTorch 2.6+).
- verl source: `/tmp/verl`, `shamanez/verl` main at commit `910ba344` (v0.8.0.dev).
- Remote pool: EC2 box `vllm-instance`, 4 × 23 GiB GPUs, ports 8100–8103.
- Token-level invariant: never modify `openhands/llm/nvidia/qwen3.py` or `qwen2_5_vl.py`.
- Config / pins: never modify `dev_config/python/**`; don't widen `pyproject.toml` pins without reading the pin comment.

## Frozen files (baseline reproduction depends on them — new siblings only)

- `scripts/_internal/s1_remote_docker.sh`
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_remote_decoupled.sh`

Phase 1 adds siblings `scripts/_internal/s2_weightsync_docker.sh` and `.../run_proagent_qwn3_4B_instruct_weightsync.sh`; the originals stay untouched.

## How the next session starts

Follow [`next_approach.md`](./next_approach.md) — concise kickoff with the planner-agent invocation, the skills/agents worth using, the environment gate, and commit discipline.
