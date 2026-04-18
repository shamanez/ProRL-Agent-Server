# Next approach — how to kick off Phase 1 (LoRA weight sync)

Fresh session. **Plan mode first.** No edits before `ExitPlanMode` is approved.

## Step 1 — Load context (read-only, in this order)

1. [`stages/weight_sync_lora.md`](./stages/weight_sync_lora.md) — the problem, setup, run, check, design.
2. [`stages/baseline.md`](./stages/baseline.md) §Architecture + §"Rollout time stats" — what the new protocol must not regress.
3. `../CLAUDE.md` — repo invariants, frozen files, credential pin.

Skim `scripts/serving/_vllm_child.py` (holds the 501 `/reload_weights` stub — Phase 1 replaces it with `/reload_lora`) and `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py:408-425` (the `EXTERNAL BYPASS ACTIVE` path — do not break it).

## Step 2 — Environment gate (run, then read the exit of each)

```bash
source /home/ubuntu/.prorl_creds.env
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/singularity_images

# Pool health (start the pool if any port is non-200)
for p in 8100 8101 8102 8103; do
  curl -sS -m 5 -o /dev/null -w "pool :$p = %{http_code}\n" \
    "http://ec2-54-145-77-207.compute-1.amazonaws.com:$p/health"
done

# vLLM 0.18 LoRA API against the image (must print add_lora/remove_lora/list_loras/pin_lora)
docker run --rm verlai/verl:vllm018.dev1 python3 -c "
from vllm.lora.request import LoRARequest
from vllm import AsyncLLMEngine
print(LoRARequest.__annotations__)
print([m for m in dir(AsyncLLMEngine) if 'lora' in m.lower()])
"

# Trainer GPUs idle
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

## Step 3 — Invoke the planner (subagent)

Plan the publish protocol before editing any file. Prefer the `Plan` subagent (it has read-only tools and cannot accidentally edit):

```
Agent(subagent_type="Plan",
  description="Phase 1 LoRA publish protocol",
  prompt="""
We are on branch decoup-weight-sync. Read plans-n-solutions/stages/weight_sync_lora.md
(the problem, setup, run, check are all there) and plans-n-solutions/stages/baseline.md §Architecture.

Design the Phase 1 LoRA publish protocol end to end. Deliverables:
  1. Trainer-side: _publish_lora_adapter(checkpoint_dir) hook wired into _save_checkpoint;
     policy_version counter; fan-out over external_llm_endpoints; abort-on-partial-failure.
  2. Pool-side: POST /reload_lora on _vllm_child.py (download → add_lora → retire prior);
     pass the pinned LoRARequest into /generate so served rollouts use {base + adapter}.
  3. Observability wiring for the eight WandB keys listed in §5.2 of weight_sync_lora.md,
     plus the structured pool-side JSON log line in §5.3.
  4. Sibling launchers: run_proagent_qwn3_4B_instruct_weightsync.sh (inner Hydra) and
     s2_weightsync_docker.sh (outer Docker) — frozen originals must NOT be edited.

Cite file paths + line numbers for every edit site. Call out open questions at the end.
Do not write code; produce a plan only.
""")
```

Alternatives by situation:

| Need | Tool |
|---|---|
| Plan a feature end-to-end before coding | `Agent(subagent_type="Plan" / "planner" / "architect")` |
| Orient in an unfamiliar verl/vLLM module before editing | `Skill("repo-architecture")` |
| Keep surgical, avoid rabbit holes during the plan | `Skill("karpathy-guidelines")` |
| Second opinion / rescue when stuck, or handing off a tricky fix | `Agent(subagent_type="codex:codex-rescue")` |
| Codex CLI readiness + review gate toggle | `Skill("codex:setup")` |
| Compact context at the end of plan mode / between Cut A and Cut B | `Skill("strategic-compact")` |
| Test-first for `_publish_lora_adapter` + `/reload_lora` | `Skill("tdd-workflow")` / `Agent(subagent_type="tdd-guide")` |
| Fast codebase search for LoRA references inside verl/vLLM | `Agent(subagent_type="Explore")` |
| Python review after edits | `Agent(subagent_type="python-reviewer")` |
| Audit the partial-reload abort + mixed-version detection paths | `Agent(subagent_type="silent-failure-hunter")` |
| SSRF / download-from-URL review on `/reload_lora` | `Agent(subagent_type="security-reviewer")` |
| Training crashes / CUDA OOM / tensor shape errors | `Agent(subagent_type="pytorch-build-resolver")` |
| Tight verify loop (lint + fast pytest) before commit | `Skill("verification-loop")` |

Use at most one or two in a turn — avoid ceremony.

## Step 4 — Gate on ExitPlanMode

Only after the user approves the plan, start editing. First edit: `_vllm_child.py` — the 501 stub is the obvious first wire-up site because it isolates the protocol from the trainer side.

## Step 5 — Cut order (hard rule)

- Land the pool-side `/reload_lora` first (`launch_remote_vllm_pool.sh publish <adapter_dir>` as the manual test harness).
- Then the trainer-side `_publish_lora_adapter` hook.
- Then the sibling launcher pair.
- Then a 20-step run against Phase 1 gates in `weight_sync_lora.md` §5.5.

## Step 6 — Commit discipline

- Stage-boundary commits only; no mid-plan commits.
- Never `--no-verify`. Never `git push` without explicit user approval.
- Frozen launchers (`s1_remote_docker.sh`, `run_proagent_qwn3_4B_instruct_remote_decoupled.sh`) stay untouched — new siblings only.
