# Decoupled Rollouts + Weight Sync

Branch `decoup-weight-sync` carries the decoupled multi-machine baseline (trainer FSDP ↔ remote vLLM pool over HTTP, previously stale) and the first closed-loop weight-sync phase built on top of it.

| Doc | Status | Role |
|---|---|---|
| [`stages/baseline.md`](./stages/baseline.md) | DONE | Fork-point. WandB run `wdqqu52k`, 7 steps, 8/8 gates green. HTTP topology, rollout throughput, architecture, problem log. |
| [`stages/weight_sync_lora.md`](./stages/weight_sync_lora.md) | DONE | Phase 1 design + run book. Trainer publishes rank-16 LoRA adapters after every `save_freq` steps via `AsyncLLMEngine.add_lora`. |
| [`stages/progress_stage.md`](./stages/progress_stage.md) | DONE | Phase 1 as-built: what shipped, how to reproduce, what happens inside vLLM during a swap, and the async-ratio / staleness accounting. |
| [`stages/timing_decoupled_4B.md`](./stages/timing_decoupled_4B.md) | DONE | First empirical timing — Qwen3-4B decoupled, 20 steps, 4 publishes. WandB run `w9nj4akn`. |
| [`stages/phase1_summary.html`](./stages/phase1_summary.html) | DONE | Visual summary — system diagram, vLLM in-flight reload mechanics, FSDP→PEFT→EC2 pipeline, timing table. Open in a browser. |
| [`next_approach.md`](./next_approach.md) | NEXT | Phase 2 (full state-dict) + Phase 3 (replay buffer / truly async) scope. |

## Hard constraints

- Single-box trainer: 8 × A100-SXM4-40GB, no Slurm.
- Trainer runs in Docker: `verlai/verl:vllm018.dev1` (vLLM 0.18, PyTorch 2.6+).
- verl source: `/tmp/verl`, `shamanez/verl` main at commit `910ba344` (v0.8.0.dev).
- Remote pool: EC2 `vllm-instance`, 4 × 23 GiB GPUs, ports 8100–8103.
- Token-level invariant: never modify `openhands/llm/nvidia/qwen3.py` or `qwen2_5_vl.py`.
- Config / pins: never modify `dev_config/python/**`; don't widen `pyproject.toml` pins without reading the pin comment.

## Frozen files (baseline reproduction depends on them — new siblings only)

- `scripts/_internal/s1_remote_docker.sh`
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_remote_decoupled.sh`

Phase 1 adds siblings `scripts/_internal/s2_weightsync_docker.sh` and `.../run_proagent_qwn3_4B_instruct_weightsync.sh`; the originals are untouched.
