# Issue — Phase 1: LoRA weight sync for the decoupled trainer ↔ remote vLLM pool

> **How to use this doc.** Paste the following prompt into a fresh planning session:
>
> > *You are the best software architect in the world. Read `plans-n-solutions/next_approach.md` and work out a perfect step-by-step plan to solve the issue described there. Cite file paths and line numbers for every edit site. Call out open questions. Do not write code — produce a plan only.*
>
> The doc below is the full issue brief. Everything the planner needs is either stated here or reachable from the pointers in §Context.

---

## Problem

On branch `decoup-weight-sync` the trainer runs FSDP on 8 × A100-40GB locally while the vLLM pool runs on a separate EC2 host over HTTP (see [`stages/baseline.md`](./stages/baseline.md) — WandB run `wdqqu52k`, 7 steps, 8/8 gates green). The pool loads Qwen3-4B once at `start` and serves from those frozen weights for the entire run. By training step N the pool is N steps stale; GRPO's importance ratio `r(θ) = π_θ(a|s) / π_behavior(a|s)` drifts from 1 (verl logs it as `rollout_corr/ppl_ratio` — baseline sat near 1.6 and grew), the PPO clip fraction rises, the gradient signal eventually degenerates.

**Close the loop with a publish protocol.** After every `save_freq` training steps the trainer ships an updated policy to every pool endpoint so `π_behavior ≈ π_θ` on the next batch. **Phase 1 does this with LoRA** (rank-16 adapter, ~20–80 MiB payload, in-place `AsyncLLMEngine.add_lora(...)`). Phase 2 (full state-dict) and Phase 3 (replay buffer) are deferred — do not design them now.

---

## Deliverables the plan must produce

1. **Pool side — `POST /reload_lora` on `scripts/serving/_vllm_child.py`** (currently a 501 stub). Body: `{adapter_url, policy_version}` (or raw bytes for small ranks). Download → `LoRARequest(lora_name=f"pv{v}", lora_int_id=v, lora_path=<local>)` → `engine.add_lora(...)` → retire prior adapter via `engine.remove_lora(old_int_id)`. Return `200 {policy_version, vllm_load_latency_ms, adapter_bytes}` on success, `5xx` with a specific error string on failure. Thread the currently-pinned `LoRARequest` into `POST /generate` so served rollouts are against `{base + adapter}`.
2. **Pool orchestrator — `launch_remote_vllm_pool.sh publish <adapter_dir>` subcommand** that fans `/reload_lora` to all 4 children. Enables manual testing before the trainer-side hook lands.
3. **Trainer side — `_publish_lora_adapter(checkpoint_dir)` in `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`**, called at the end of `_save_checkpoint`. Extracts the PEFT-style `adapter_model.safetensors` + `adapter_config.json` from the FSDP checkpoint, bumps a trainer-owned `policy_version`, fan-outs `/reload_lora` over `self.config.actor_rollout_ref.rollout.external_llm_endpoints`, **blocks the next training step until all endpoints return 200**, and **aborts the run on any partial failure** (mixed-version batches are a correctness bug, not a warning). Stamp `meta.policy_version` on every rollout job in `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` before dispatch.
4. **Observability** (detail in `stages/weight_sync_lora.md` §5.2 / §5.3): the eight WandB keys (`weight_sync/policy_version`, `adapter_mib`, `publish_latency_s`, `transfer_latency_s`, `vllm_load_latency_s`, `endpoints_ok`, `endpoints_failed`, `rollout/staleness_steps`) and the structured JSON log line the pool child emits on every `/reload_lora`. Separating `transfer_latency_s` from `vllm_load_latency_s` is non-negotiable — it is the signal that tells us whether a slow publish is the network or `add_lora`.
5. **Sibling launchers (new files — frozen originals must NOT be edited):**
   - `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh` — sibling of `..._remote_decoupled.sh`. Adds `actor_rollout_ref.model.{lora_rank=16, lora_alpha=32, lora_target_modules=[...]}`, `+actor_rollout_ref.rollout.publish_on_save=True`, `trainer.save_freq=5`, `trainer.experiment_name=weight-sync-decup-prorl`.
   - `scripts/_internal/s2_weightsync_docker.sh` — sibling of `s1_remote_docker.sh`.

---

## Success criteria (Phase 1 gates)

Enumerated in [`stages/weight_sync_lora.md`](./stages/weight_sync_lora.md) §5.5. In one sentence: 20 training steps complete with `save_freq=5`; ≥ 4 successful `/reload_lora` events; zero mixed-version batches; `rollout_corr/ppl_ratio` dips on the step after each publish; `critic/rewards/mean` trends up (baseline was flat until step 7); zero `POST /generate` 5xx during swap; baseline invariants (`EXTERNAL BYPASS ACTIVE`, zero Ray vLLM actors, 8 local GPUs hold FSDP shards) still hold.

---

## Constraints and guardrails

- **Topology is fixed.** Trainer inside Docker (`verlai/verl:vllm018.dev1`, vLLM 0.18), 8 A100-40GB FSDP locally; pool on EC2 `vllm-instance`, 4 children on ports 8100–8103; ProRL on host :8006. Do not propose changing it.
- **Frozen files** (baseline reproduction — new siblings only): `scripts/_internal/s1_remote_docker.sh`, `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_remote_decoupled.sh`.
- **Must-preserve invariants**: the token-level `{prompt_ids} → {response_ids, logprobs}` contract (`openhands/llm/nvidia/qwen3.py`), the `EXTERNAL BYPASS ACTIVE` path (`trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py:408-425`), and the WandB keys the baseline already logs (`actor/*`, `critic/*`, `rollout_corr/ppl_ratio`).
- **Cut order (hard).** Pool-side `/reload_lora` first → orchestrator `publish` subcommand → trainer-side `_publish_lora_adapter` → sibling launchers → 20-step run. Each cut must run green before the next.
- **No `--no-verify`. No `git push` without explicit user approval.** Stage-boundary commits only.
- **Policy-version ownership is trainer-authoritative.** Pool echoes what it installed; never increments on its own.

---

## Context (read before planning)

| Path | Why |
|---|---|
| `plans-n-solutions/stages/weight_sync_lora.md` | The full Phase 1 design sketch, fresh-box setup, run book, and observability spec. Everything in this issue brief is expanded there. |
| `plans-n-solutions/stages/baseline.md` | §Architecture (HTTP topology + bypass path), §"Rollout time stats" (throughput baseline the new protocol must not regress). |
| `scripts/serving/_vllm_child.py` | The 501 `/reload_weights` stub and the current `/generate` wiring. |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py:408-425` | `EXTERNAL BYPASS ACTIVE` path — do not break. |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | `_save_checkpoint` is the hook site for `_publish_lora_adapter`. |
| `scripts/serving/launch_remote_vllm_pool.sh` | Orchestrator — `publish` subcommand lands here. |
| `../CLAUDE.md` | Repo invariants, frozen files, credential pin, autoflake precedents. |
| `docs/decoupling-walkthrough.md` | Architectural background: the weight handoff via tensor aliasing in the colocated path, i.e. what decoupling replaces. |

vLLM 0.18 LoRA surface (verified 2026-04-18 against the image): `LoRARequest(lora_name, lora_int_id, lora_path, base_model_name, tensorizer_config_dict, load_inplace)` + `AsyncLLMEngine.{add_lora, remove_lora, list_loras, pin_lora}`.

---

## Known unknowns the plan must resolve

1. **Adapter extraction from an FSDP checkpoint** — does verl's `_save_checkpoint` already dump PEFT-format shards when `lora_rank>0`, or do we need a merge step? Confirm against `/tmp/verl` before proposing the extraction code.
2. **Transport** — raw HTTP body of the safetensors shard (simple, one hop) vs presigned S3 URL (adds a dependency but survives larger future ranks). Pick one with rationale; the protocol's 200 body is identical either way.
3. **`rollout.n=4` + adapter swap mid-batch** — GRPO groups 4 samples/prompt into one advantage group. If a publish lands mid-batch, the group contains mixed versions. Plan must either (a) block dispatch during publish or (b) defer publish to batch boundaries. State the choice and justify.
4. **Adapter shape change between publishes** — if `lora_target_modules` or rank were to shift, `add_lora` refuses. Decide whether Phase 1 locks them at run start or surfaces the engine error; document the choice.
5. **Pool restart mid-run** — lost adapter state on the pool. Plan the recovery path (is `policy_version=0` the "adapter absent" sentinel? does the trainer detect and abort, or re-publish on next step?).

---

## Mandatory review gates (do not skip for this phase)

Phase 1 is correctness-sensitive: a partial `/reload_lora` failure or a mid-batch version split is a data-correctness bug, not a warning. Treat these as gates, not options.

1. **Codex review of the plan** — before any code is written. `Agent(subagent_type="codex:codex-rescue")` with the completed plan + this brief + the five known unknowns. Ask it specifically to attack: (a) the mid-batch publish strategy (block-vs-defer), (b) the abort-on-partial-failure contract, (c) policy-version ownership, (d) the PEFT-from-FSDP extraction path. Capture its objections in the plan before cutting.
2. **Codex review at each cut boundary** — after (i) pool-side `/reload_lora` lands, (ii) the orchestrator `publish` subcommand lands, (iii) trainer-side `_publish_lora_adapter` lands. Same agent, diff-scoped. Each cut must be green on Codex review before the next cut starts.
3. **`silent-failure-hunter`** on the partial-reload abort path and the mixed-version detection in `async_server.py` — before the 20-step run.
4. **`security-reviewer`** on `/reload_lora` if the transport choice is URL-based (SSRF surface on the pool child).

## Other tooling the planner should consider (optional, at most 1–2 per turn)

| Need | Tool |
|---|---|
| Orient in unfamiliar verl/vLLM modules | `Skill("repo-architecture")` |
| Surgical changes, avoid rabbit holes | `Skill("karpathy-guidelines")` |
| Fast codebase search (LoRA inside verl/vLLM, PEFT checkpoint format) | `Agent(subagent_type="Explore")` |
| Test-first for `_publish_lora_adapter` and `/reload_lora` | `Skill("tdd-workflow")` / `Agent(subagent_type="tdd-guide")` |
| Context compaction between cuts | `Skill("strategic-compact")` |
| Verify loop (lint + fast pytest) before commit | `Skill("verification-loop")` |
| CUDA OOM / tensor shape errors during the run | `Agent(subagent_type="pytorch-build-resolver")` |
