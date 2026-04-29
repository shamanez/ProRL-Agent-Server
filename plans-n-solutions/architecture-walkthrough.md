# Architecture walkthrough — doc map for coding tasks

You are about to change something in this repo. Read this file first; it tells you which of the ~5 reference docs to open and in what order. Every other doc is single-purpose; this one is the index.

## The doc collection

| Doc | What it is | When to open |
|---|---|---|
| [`README.md`](../README.md) | One-screen project pitch + system shape. | First time you land here. |
| [`CLAUDE.md`](../CLAUDE.md) | Architectural invariants (token-in/token-out, registry, runtime, RL trainer integration) + repo conventions. | Before your first edit in any unfamiliar module. |
| [`handsoff.md`](handsoff.md) | Single source of truth for the running system: topology, launch sequence, credentials, observability, **pointer table** (every concept → file:line), **gotchas** (32 numbered traps). | Anytime you're touching the trainer / producer / store / pool / weight-sync stack. |
| [`stages/how_to_run.md`](stages/how_to_run.md) | Runbook: env-knob matrix, smoke test, restart/resume, failure runbook. | When you need to run, restart, or debug a run. |
| [`stages/replay_dynamics.md`](stages/replay_dynamics.md) | Code-grounded mechanics of producer / store / trainer: every `TrajectoryRecord` field, push/sample step-by-step, temporal-IS plumbing, `replay/*` metrics. | When you're modifying anything in `verl_custom/replay/`, `verl_custom/nvidia/rollout/`, or the trainer's sample seam. |
| [`stages/latencies.md`](stages/latencies.md) | Per-component TPS and wall-clock budgets; tells you where to spend tuning effort. | When the question is "is this slow / where is the bottleneck". |
| [`stages/current_bottlenecks_and_problems.md`](stages/current_bottlenecks_and_problems.md) | Open work queue. One problem per branch. | Picking up a new task. |

## Reading order by task type

| Task | Read in this order |
|---|---|
| Pick up an open problem from the queue | `handsoff.md` (skim §§1–5, gotchas) → `stages/current_bottlenecks_and_problems.md` (the problem) → the supporting doc the problem links to |
| Add a new agent task type (`AgentHandler`) | `CLAUDE.md` (Registry + AgentHandler section) → an existing handler under `openhands/nvidia/swe_agent/` or `math_coder/` → `openhands/nvidia/registry.py` |
| Change anything in the LLM client path | `CLAUDE.md` (Token-in/token-out invariant) → `openhands/llm/nvidia/qwen3.py` — this path is **frozen by invariant**, ask before editing |
| Tune the replay loop / change buffer semantics | `stages/replay_dynamics.md` end-to-end → `handsoff.md` gotchas §14, §19, §20, §21, §29, §31 → `verl_custom/replay/trajectory_store.py` |
| Modify the producer (rollout / push path) | `stages/replay_dynamics.md` §§1–4 → `verl_custom/nvidia/rollout/async_server_dapo.py` (filter + eager-push) → `handsoff.md` gotchas §19, §20, §22, §25–28 |
| Modify the trainer sample / advantage path | `stages/replay_dynamics.md` §§5–8 → `verl_custom/trainer/ppo/ray_trainer_dapo.py:_acquire_training_batch_dapo` → `verl_custom/trainer/ppo/ray_trainer.py:compute_advantage` |
| Touch weight-sync / LoRA publish | `handsoff.md` (publish hook in pointer table, gotcha §22) → `verl_custom/trainer/ppo/ray_trainer.py:_publish_lora_adapter` |
| Run a training job / debug a stalled run | `stages/how_to_run.md` end-to-end → `handsoff.md` §2 (launch sequence) |
| Performance / where-is-time-going question | `stages/latencies.md` → `stages/replay_dynamics.md` §10 (fill-faster levers) |

## Pre-commit invariant checklist

Five invariants every change must preserve. Each has a short rationale and a code anchor for spot-checks; the gotchas in `handsoff.md` add finer-grained traps.

1. **Token-in / token-out** — re-tokenizing decoded text across turns shifts boundaries → actor vs reference diverges → KL/entropy NaN → PPO collapses. Enforced in `openhands/llm/nvidia/qwen3.py` and `qwen2_5_vl.py` (paired clients, frozen).
2. **Pop-on-sample, no reuse** — a sampled group is removed from the buffer; it cannot be sampled again at the same trainer step. Prevents off-policy double-counting. Enforced at `trajectory_store.py:sample_mini_batch` (records popped from the deque).
3. **Zero-variance groups never enter the buffer** — `resolved ∈ {0, n}` ⇒ `group_std=0` ⇒ zero advantage ⇒ wasted FSDP step. Enforced at `async_server_dapo.py:filter_easy_hard_instance` (runs before eager-push).
4. **Staleness gate at sample time** — groups with `current_step − created_at_step > staleness_cutoff_k` are dropped before sampling, so the temporal-IS correction never has to fix up an unbounded version lag. Enforced at `trajectory_store.py:num_fresh_groups` and `_evict_stale_locked` (K=4 in production).
5. **Monotonic `policy_version` on the pool** — the pool rejects `/reload_lora` with HTTP 409 if `new_version ≤ active_policy_version`. Trainer is the source of truth (`ray_trainer.py:_publish_lora_adapter`); abort on any `endpoints_failed > 0`.

If your change might cross any of these, make it explicit in the PR description and prove it via a test. If you can't tell, ask.
