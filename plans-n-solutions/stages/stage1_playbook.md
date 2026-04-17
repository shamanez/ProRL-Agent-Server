# Decoupling Milestone — Stages 1 + 2 Execution Runbook

**Status:** Not started. Stage 0 complete (`stage0.md`).

From this point forward, **all subsequent stages run against the upgraded stack** (verl v0.8.0.dev + vLLM 0.18 + Docker `verlai/verl:vllm018.dev1`).

---

## What this milestone does

Decouples vLLM inference from the GRPO trainer. Two stages, landed together:

| Stage | What it does | Touches trainer? |
|---|---|---|
| **1 — External vLLM standalone** | Hosts vLLM as an out-of-Ray process + supervisor HTTP wrapper on GPUs 0–1. Proves ProRL can route rollouts to it. | No. `git diff --stat HEAD trainer_integration/` stays empty. |
| **2 — Trainer bypass (stale weights)** | Trainer on GPUs 0–3 skips in-Ray vLLM startup, routes rollouts to 4 external supervisors on GPUs 4–7. Runs 20 GRPO steps with intentionally stale weights. **This is what actually decouples.** | Yes — adds `external_llm_endpoints` config; `start_llm_servers()` skipped; `wake_up()` / `sleep()` become no-ops when external. |

Shipping Stage 1 alone does not decouple anything. Shipping Stage 2 alone has no endpoint to talk to. They close as a single milestone.

---

## Hard rules (read once, obey throughout)

| Rule | Why |
|---|---|
| **Plan-first.** Step 1 runs inside plan mode; the plan-mode approval green-lights Steps 2–9. After that, stop only at `[COMMIT GATE]` (Steps 5, 9) and `[GO/NO-GO]` (Step 6). | Stage 0 cost 26 issues because changes landed before the design was pressure-tested. |
| **Reuse the existing launchers.** ProRL server = `bash scripts/_internal/s0_prorl.sh` (poetry, unchanged). Stage-2 trainer = **new sibling** `scripts/_internal/s2_decoupled_docker.sh` mirroring `s0_baseline_docker.sh` line-for-line except for the inner Hydra script. | No ad-hoc `docker run`. No new server wrappers. |
| **Frozen baseline files — do not edit:** `scripts/_internal/s0_prorl.sh`, `scripts/_internal/s0_baseline_docker.sh`, `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh`. Stage 2 creates *siblings*, never modifications. | Stage 0 reproduction must stay intact so any regression is obvious. |
| **Token-level invariant is sacred.** `openhands/llm/nvidia/qwen3.py` and `qwen2_5_vl.py` are read-only. | Re-tokenizing decoded text across turns collapses GRPO (see `openhands/llm/nvidia/README.md`). |
| **Stage 1 OOM = record and continue.** Stage 2 OOM = STOP, the milestone failed. | Stage 1 proves the architecture; memory tuning is Stage 3/4. Stage 2 is the gate. |
| **Codex is the adversarial reviewer** at plan gates and commit gates. `/codex:adversarial-review --background` for design, `/codex:review --background` for code quality. | Catches design mistakes before they become code. |
| **Compact at clean boundaries.** Invoke the `strategic-compact` skill after Steps 5 and 8 (before the long training runs that follow). | Prevents harness auto-compaction mid-step, which loses in-flight state. |
| **Write status snapshots.** At each checkpoint (end of Steps 2, 4, 6, 8), append a timestamped one-liner to the active stage's Solution section: `YYYY-MM-DDTHH:MMZ — <what just passed / what just failed / what's next>`. | If context compresses, the next session re-hydrates from the file, not from guessing. |
| **Never `git push`** without explicit user approval. Never commit with `--no-verify`. | Carry-over from repo rules. |

---

## Prerequisites

From project root `/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server`. Verify each with the right-column gate before proceeding.

| Prereq | Bootstrap | Gate |
|---|---|---|
| Stage 0 complete | See `stage0.md` | `git log --oneline -20 \| grep -q '13f95697' && echo OK` |
| Branch + clean tree | `git checkout de-coupled` | `git branch --show-current` → `de-coupled`; `git status --short` → empty |
| Credentials loaded | Create `/home/ubuntu/.prorl_creds.env` (see `stage0.md` Prerequisites) | `source /home/ubuntu/.prorl_creds.env && test -n "$WANDB_API_KEY" && echo OK` |
| Image repo env var | `export OH_RUNTIME_SINGULARITY_IMAGE_REPO=$PWD/singularity_images` | `test -n "$OH_RUNTIME_SINGULARITY_IMAGE_REPO"` |
| Docker image | `docker pull verlai/verl:vllm018.dev1` | `docker image inspect verlai/verl:vllm018.dev1 >/dev/null && echo OK` |
| verl checkout | `git clone https://github.com/shamanez/verl.git /tmp/verl && cd /tmp/verl && git checkout 910ba344` | `cd /tmp/verl && git log --oneline -1` shows `910ba344` |
| 49 SIF images | `scripts/pull_swe_images.py …` (see `stage0.md`) | `ls singularity_images/*.sif \| wc -l` ≥ 49 |
| Codex CLI ready | `/codex:setup` inside Claude Code | `node /home/ubuntu/.claude/plugins/cache/openai-codex/codex/1.0.3/scripts/codex-companion.mjs setup --json \| jq '.auth.loggedIn'` → `true` |

---

## Execution

Each step is a phase of work. Hard stops: `[COMMIT GATE]` (Steps 5, 9) and `[GO/NO-GO]` (Step 6). Everything else flows under the plan-mode approval from Step 1.

### Step 1 — Plan both stages (you are already in plan mode)

Context to read, in order, before drafting the plan:

1. `CLAUDE.md` — invariants
2. `plans-n-solutions/README.md` — stage map
3. `plans-n-solutions/stages/stage0.md` — upgraded stack + problem log pattern
4. `plans-n-solutions/stages/stage1.md` — Decoupling-milestone plan + combined test criteria (Part A: 6 smoke gates; Part B: 8 trainer-run gates)
5. `docs/decoupling-walkthrough.md` — why Stage 0 is colocated (FSDP + vLLM in one Ray actor)
7. `docs/decoupled-rollout-architecture.html` — target architecture

Then, still inside plan mode:

1. Delegate a pressure-test: `Agent(subagent_type="architect", …)` with prompt *"Pressure-test both stages. Stage 1: supervisor↔child boundary, lifecycle, port discipline (N+1000 gap), signal propagation, `SIGTERM→30s→SIGKILL`, token-level invariant preservation in `/generate` proxy. Stage 2: every control-plane path where the trainer assumes colocated vLLM; `wake_up()` / `sleep()` no-op correctness; DP-size vs endpoint-count mismatch; external-pool death mid-step."*
2. Run `/codex:adversarial-review --background` with the plan as focus: *"Challenge the decoupling design across Stages 1 + 2. What breaks the bypass assumption? Where does the supervisor leak child processes? Where does the no-op `wake_up/sleep` silently corrupt training state?"*
3. Fold feedback in.
4. Exit plan mode via `ExitPlanMode` — the plan-mode approval is the green light for Steps 2–9.

### Step 2 — Stage 1 launcher implementation

Files created:

- `scripts/serving/vllm_launcher.py` — supervisor FastAPI app. One `subprocess.Popen` (process-group-owned) for vLLM 0.18 OpenAI server at port N+1000. Routes: `GET /health` (200 iff child `/v1/models` OK, else 503), `POST /generate` (proxies to child `/v1/completions`, passes `prompt_ids` verbatim — does NOT re-tokenize), `POST /reload_weights` (returns 501 JSON `{"detail": "Not implemented in Stage 1"}`). `atexit` + `SIGTERM`/`SIGINT` handlers `SIGTERM` child, wait 30 s, `SIGKILL`.
- `scripts/serving/launch_external_vllm_pool.sh` — takes `--gpus 0,1 --ports 8100,8101`; fans out one supervisor per (GPU, port) with `CUDA_VISIBLE_DEVICES=<n>` and child port = supervisor port + 1000. PID files at `/tmp/vllm-sup-<port>.pid`.

Child vLLM launch (per `stage1.md`):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<snap> \
  --port <child_port> \
  --gpu-memory-utilization 0.45 \
  --max-model-len 17920 \
  --enforce-eager \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192
```

Validation inside this step:

```bash
python scripts/serving/vllm_launcher.py --help   # parses
poetry run ruff check scripts/serving/ scripts/tests/
```

Then:

1. `/code-review` on the diff.
2. `Agent(subagent_type="silent-failure-hunter", …)` specifically on the signal/cleanup paths.
3. `/codex:review --background` for code-quality pass.
4. **Checkpoint:** append `<timestamp> — Step 2 done: launcher + pool script landed, lint green.` to `stage1.md` Solution.

### Step 3 — Stage 1 smoke test (TDD)

Files created:

- `scripts/tests/test_external_vllm.py` — asserts all six criteria from `stage1.md` Test plan.

Order (write test FIRST):

1. Invoke the `tdd` skill with scope: *"Write `scripts/tests/test_external_vllm.py` asserting the 6 criteria in `stage1.md`. Integration test, no mocking."*
2. Run the test; it fails (no supervisor running). Red.
3. Start the pool: `bash scripts/serving/launch_external_vllm_pool.sh --gpus 0,1 --ports 8100,8101`.
4. Wait for health: `for p in 8100 8101; do until curl -sf http://localhost:$p/health >/dev/null; do sleep 2; done; done`.
5. Re-run the test. Green.
6. Run `/verify` (lint + fast test loop).

### Step 4 — Stage 1 live integration + gating

Exactly the commands in `stage1.md` Steps 1–4, using the existing ProRL launcher.

```bash
# Terminal 1 — ProRL host (unchanged launcher)
bash scripts/_internal/s0_prorl.sh

# Terminal 2 — external vLLM pool
bash scripts/serving/launch_external_vllm_pool.sh --gpus 0,1 --ports 8100,8101

# Terminal 3 — register + start + verify
for p in 8100 8101; do
  curl -sX POST http://localhost:8006/add_llm_server \
    -H 'Content-Type: application/json' -d "{\"url\":\"http://localhost:$p\"}"
done
curl -sX POST http://localhost:8006/start
curl -s http://localhost:8006/status | python3 -m json.tool

# Run smoke test (2 SWE-Bench instances end-to-end)
poetry run python scripts/tests/test_external_vllm.py
```

Pass all six criteria:

| # | Criterion | Pass = |
|---|---|---|
| 1 | Supervisors boot healthy | Both `/health` return 200 in < 60 s |
| 2 | ProRL registers endpoints | `/status` shows both under `llm_servers` |
| 3 | SWE-Bench rollout succeeds | 2 instances return `{"resolved": bool, …}` |
| 4 | vLLM logs show traffic | `grep -c 'POST /v1/completions' /tmp/vllm-child-810[01].log` > 0 per child |
| 5 | `/reload_weights` stub | `curl -X POST http://localhost:8100/reload_weights` → HTTP 501 + `{"detail": "Not implemented in Stage 1"}` |
| 6 | Trainer code untouched | `git diff --stat HEAD trainer_integration/` empty |

If any fails: `/codex:rescue --background investigate why criterion N fails`. Do not hand-debug past 20 min.

**Checkpoint:** append `<timestamp> — Step 4 done: criteria [1-6 or the failing set] green.` to `stage1.md` Solution.

### Step 5 — Stage 1 commit `[COMMIT GATE]`

1. Fill in `stage1.md` Solution section (real vLLM 0.18 CLI, timings, any surfaced bugs — same table format as `stage0.md` Problem log).
2. `make lint`.
3. Show the staged diff (`git diff --stat HEAD` + `git status --short`) and the proposed commit message. **Ask the user for "commit" before running `git commit`.** Do not push.

```bash
git add scripts/serving/ scripts/tests/test_external_vllm.py \
        plans-n-solutions/stages/stage1.md plans-n-solutions/README.md
git commit -m "Stage 1: external vLLM standalone

- scripts/serving/vllm_launcher.py — FastAPI supervisor; /health, /generate, /reload_weights (501)
- scripts/serving/launch_external_vllm_pool.sh — GPU-pinned pool launcher
- scripts/tests/test_external_vllm.py — 6-criterion integration test
- Reuses scripts/_internal/s0_prorl.sh for ProRL host launch (unchanged)
- No trainer code touched; git diff --stat HEAD trainer_integration/ empty"
```

After commit, invoke the `strategic-compact` skill — clean boundary before Stage 2's 2-hour run.

### Step 6 — Stage 1 → Stage 2 handoff `[GO/NO-GO]`

Stage 2 starts a ~2-hour 20-step training run on all 8 GPUs. Before proceeding:

1. Refresh the Stage 2 scope against what Stage 1 actually revealed. Re-read these files for any Step-1 plan deltas:
   - `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` — confirm `start_llm_servers()`, `wake_up()`, `sleep()` touch-points match the plan from Step 1
   - `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` — the Hydra knobs to mirror into the sibling script
   - `scripts/_internal/s0_baseline_docker.sh` — the Docker-run block to mirror
2. If Stage 1 surfaced anything that invalidates the Stage 2 plan from Step 1, re-enter plan mode (`EnterPlanMode`) and redraft just the Stage 2 portion.
3. Report a short readiness summary to the user:
   - Stage 1 artifacts landed (commit hash, smoke test green)
   - Stage 2 plan diffs since Step 1 (if any)
   - All 8 GPUs free, `/tmp/s2-decoupled.log` path ready
4. **Ask the user for "go" before Step 7.** If GPUs are busy or timing is wrong, wait.

**Checkpoint:** append `<timestamp> — Step 6 done: Part A committed <hash>, Part B plan confirmed, GPUs free.` to `stage1.md` Solution.

### Step 7 — Stage 2 implementation

Files created or edited:

| File | Action | Change |
|---|---|---|
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | **edit** | Add `external_llm_endpoints: list[str] = []` config field; early-return from `start_llm_servers()` when set (log a loud marker); `wake_up()` / `sleep()` become no-ops when external |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_decoupled.sh` | **new sibling** | `cp run_proagent_qwn3_4B_instruct.sh` + diff-edit: `trainer.n_gpus_per_node=4`; add `+actor_rollout_ref.rollout.external_llm_endpoints=[http://127.0.0.1:8100,http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103]`. Everything else identical. |
| `scripts/_internal/s2_decoupled_docker.sh` | **new sibling** | `cp s0_baseline_docker.sh` + diff-edit: `CNAME=s2-decoupled`; add `-e CUDA_VISIBLE_DEVICES=0,1,2,3` to `docker run`; swap inner `run_proagent_qwn3_4B_instruct.sh` for `run_proagent_qwn3_4B_instruct_decoupled.sh`. Same image, same mounts, same env, same `--network=host`, same `PYTORCH_ALLOC_CONF=expandable_segments:True`. |
| `scripts/validate_run.py` | **new** | CLI: `scripts/validate_run.py [--expect-weight-publishes N] <logfile>`. Asserts all 5 hard gates from a log file; checks for the loud "external bypass active" marker; counts weight-publish events. Exit 0 = all green. |

Review chain:

1. `/code-review` on the diff.
2. `Agent(subagent_type="python-reviewer", …)` on the `async_server.py` edits.
3. `Agent(subagent_type="silent-failure-hunter", …)` on the no-op `wake_up()` / `sleep()` paths.
4. `Agent(subagent_type="code-reviewer", …)` on `s2_decoupled_docker.sh` (escaping, idempotence, env propagation).
5. `/codex:review --background` for code-quality pass.
6. `make lint` (fix anything pre-commit flags — never `--no-verify`).

### Step 8 — Stage 2 20-step validation run

Three terminals:

```bash
# Terminal 1 — ProRL host (existing launcher, unchanged)
bash scripts/_internal/s0_prorl.sh

# Terminal 2 — 4 external vLLM supervisors on GPUs 4-7
bash scripts/serving/launch_external_vllm_pool.sh --gpus 4,5,6,7 --ports 8100,8101,8102,8103
# Wait for health
for p in 8100 8101 8102 8103; do
  until curl -sf "http://localhost:$p/health" >/dev/null; do sleep 2; done
done

# Register with ProRL + start
for p in 8100 8101 8102 8103; do
  curl -sX POST http://localhost:8006/add_llm_server \
    -H 'Content-Type: application/json' -d "{\"url\":\"http://localhost:$p\"}"
done
curl -sX POST http://localhost:8006/start

# Terminal 3 — decoupled trainer (SIBLING launcher, same pattern as s0)
bash scripts/_internal/s2_decoupled_docker.sh 2>&1 | tee /tmp/s2-decoupled.log
```

Monitor wandb for the 5 hard gates. Expected cadence: ~350–400 s per step. 20 steps ≈ 2 hours.

| # | Metric | Pass |
|---|---|---|
| 1 | `training/global_step` | ≥ 20 |
| 2 | `actor/grad_norm` | finite, > 0, < 1e6 every step |
| 3 | `critic/rewards/mean` | not identically zero across 20 steps |
| 4 | Advantage variance (`critic/advantages/max - min`) | > 0 |
| 5 | `actor/kl` | finite every step |
| 6 | External pool served rollouts | `POST /v1/completions` count > 0 on each of the 4 supervisors; zero Ray vLLM actor spawn logs |
| 7 | `scripts/validate_run.py --expect-weight-publishes 0 /tmp/s2-decoupled.log` | exit 0 |

If any fails: stop, `/codex:rescue --background investigate why gate N failed`, document in `stage1.md` Solution.

**Checkpoint:** append `<timestamp> — Step 8 done: 20/20 steps, gates [list] green, WandB <url>.` to `stage1.md` Solution. Then invoke `strategic-compact` before the commit gate.

### Step 9 — Stage 2 commit + docs `[COMMIT GATE]`

1. Fill in `stage1.md` Solution section with real numbers + WandB URL (same table format as `stage0.md` Problem log).
2. Run `update-docs` skill — refreshes `README.md` + `plans-n-solutions/README.md` status row.
3. `make lint`.
4. Show the staged diff (`git diff --stat HEAD` + `git status --short`) and the proposed commit message. **Ask the user for "commit" before running `git commit`.** After committing, **ask again for "push" before `git push`.**

```bash
git add trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py \
        trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_decoupled.sh \
        scripts/_internal/s2_decoupled_docker.sh scripts/validate_run.py \
        plans-n-solutions/stages/stage1.md plans-n-solutions/README.md README.md
git commit -m "Stage 2: trainer bypass with stale weights

- verl_custom/nvidia/rollout/async_server.py — external_llm_endpoints field
  short-circuits start_llm_servers(); wake_up()/sleep() no-op when external
- scripts/_internal/s2_decoupled_docker.sh — sibling of s0_baseline_docker.sh
  (same image/mounts/env, CUDA_VISIBLE_DEVICES=0,1,2,3, inner script swapped)
- run_proagent_qwn3_4B_instruct_decoupled.sh — sibling of baseline Hydra
  script (n_gpus_per_node=4, external_llm_endpoints=[...])
- scripts/validate_run.py — 20-step gate checker

WandB: <url>
Decoupling milestone complete: trainer on GPUs 0-3, vLLM pool on 4-7."
```

---

## Out-of-scope / do not touch

| Path | Reason |
|---|---|
| `scripts/_internal/s0_baseline_docker.sh` | Stage 0 baseline reproduction — **frozen**. Stage 2 creates a sibling. |
| `scripts/_internal/s0_prorl.sh` | Server launcher — **reuse as-is**. No new server wrapper. |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` | Baseline Hydra — **frozen**. Stage 2 creates a `_decoupled.sh` sibling. |
| `openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py` | Token-level invariant. |
| `dev_config/python/**` | Linter / formatter / type-checker configs — ask user first. |
| `pyproject.toml` pins | Security / bug pins — read the pin comment before widening. |
| `CLAUDE.md`, `plans-n-solutions/stages/stage0.md`, `.claude/commands/continue-decoupling.md` | Document previous stage; amend only if this milestone reveals an error. |
| `trainer_integration/verl/verl_custom/**/*.py` during Stage 1 | Criterion #6 requires `git diff --stat HEAD trainer_integration/` empty after Stage 1. |

---

## When you get stuck

1. Timebox 20 min per blocker.
2. Re-read the relevant section of `docs/decoupling-walkthrough.md` to confirm the assumption.
3. `/codex:rescue --background investigate <one-sentence blocker>`.
4. While Codex works, append the blocker to the active stage file's Solution section with a timestamp.
5. On resume: `/codex:result <task_id>`; fold into the plan.

If the blocker is structural (not a typo / missing import), **stop and bring it to the user** rather than stacking code on top of a broken premise.

---

## Success criteria for the full milestone

All of the below must hold before calling the decoupling milestone done:

- Stage 1: all 6 criteria in Step 4 🟢; Solution section filled; single commit
- Stage 2: all 7 criteria in Step 8 🟢; Solution section filled; single commit
- `make lint` + fast test loop green
- User has reviewed and said "push" before any `git push`

---

## Reference

- Previous stage: `plans-n-solutions/stages/stage0.md`
- Decoupling-milestone plan: `plans-n-solutions/stages/stage1.md`
- Architecture: `docs/decoupling-walkthrough.md`, `docs/decoupled-rollout-architecture.html`
- Resume-from-cold command: `.claude/commands/continue-decoupling.md`
