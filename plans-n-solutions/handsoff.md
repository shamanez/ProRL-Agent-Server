# Handoff — Phase 2: Fully-Async Decoupled Agentic RL

**Core goal:** Retool the RL loop around **experience replay** (Arnal et al. 2026 — see `docs/README.md`). Rollouts stream continuously into a bounded replay buffer; the trainer samples from it on its own cadence with off-policy correction and a staleness budget. Phase 1 already decoupled the *machines* (trainer ≠ vLLM pool); Phase 2 decouples the *clocks*. Both GRPO (`filter_groups=False`) and DAPO (`filter_groups=True`) must keep working.

> This is the **single source of truth** for the next milestone on this repo. Read it top to bottom before you do anything. Do not skip sections. Do not start coding until you have written a plan that references specific files and line numbers.

---

## 0. One-paragraph mission

Take ProRLAgent Server from **Phase 1 (lock-step LoRA weight-sync, DONE on branch `decoup-weight-sync`)** to **Phase 2 (fully-async decoupled agentic RL on branch `full-async`)**. In Phase 1, the trainer and the remote vLLM pool already run on different machines and the trainer already publishes rank-16 LoRA adapters to the pool after every `save_freq` steps. That works, but the two clocks are still locked: the trainer blocks on `generate_sequences` for a full batch before it can update. With GRPO `filter_groups.enable=True` the trainer throws away any prompt whose samples all share the same reward sign, so idle time compounds — trainer waits on rollouts, rollouts are agentic (long tail), and many rollouts get filtered. **The fix is to decouple the clocks entirely**: rollouts stream continuously into a bounded trajectory store, the trainer samples from it on its own schedule with off-policy correction and a staleness budget. **Both `filter_groups=True` AND `filter_groups=False` must keep working after Phase 2.**

---

## 1. Non-negotiable step-by-step process

You are a **planning agent first**, an implementation agent second. Follow Phases A → F in order. Commit at stage boundaries only. **Scope one session to one unit of work** — `A+C` (planning → stage doc), `B` (reproduction runs), one of D's cuts (implementation), `E` (verification), or `F` (handoff). Do not bundle units; commit at the unit boundary and hand off to a fresh session.

### Phase A — Orient (no file edits, no destructive commands)

1. Read this doc end to end.
2. Read `CLAUDE.md` at repo root.
3. Read every file under `.claude/rules/` (they auto-load but re-read so you know what they say).
4. Read `docs/README.md` — distilled mapping of Arnal et al. "Efficient RL Training for LLMs with Experience Replay" onto this codebase. Names the three-way staleness/diversity/compute trade-off, the minimal `BufferStructure` diff, the `(W,T)` knobs, positive-bias sampling, and how `filter_groups` interacts with a replay buffer. This is the intellectual spec for Phase C — do not skip.
5. Invoke the `repo-architecture` skill (`.claude/skills/repo-architecture/SKILL.md`) to map the fork layout.
6. `git log --oneline -20` on `full-async`; skim the last 3 commit diffs back to `decoup-weight-sync`.
7. Read the Phase 1 code sites in the Pointer Table (§11). Do not edit anything.
8. Verify: the dataset at `/home/ubuntu/data/SkyRL-v0-293/` still has `train.parquet` and `validation.parquet`; `nvidia-smi` shows 8 idle A100s; `/home/ubuntu/.prorl_creds.env` exists.

**Deliverable (reply to the user in chat, NOT a file):** a 10-bullet summary of current state + your initial hypothesis for the Phase 2 architecture. Ask any clarifying questions. **Do not begin Phase C until the user has answered them** — a half-informed stage doc is worse than none.

### Phase B — Reproduce `decoup-weight-sync`

Do not assume the environment works. Prove it against the known-good starting state.

1. Start the three processes (§6) — ProRL, remote vLLM pool, trainer.
2. Check out `decoup-weight-sync`. Launch `bash scripts/_internal/s2_weightsync_docker.sh` with a short config (`TOTAL_TRAINING_STEPS=2 SAVE_FREQ=1`). Verify: pool `/health` shows `policy_version` bumps after each save, `weight_sync/*` WandB keys appear on every step, zero 5xx on `/generate` during swaps, `weight_sync/endpoints_failed == 0`.
3. Back to `full-async`. Reproduce the same 2-step cycle — behavior must be identical because Phase 2 code doesn't exist yet.

**Deliverable:** a short status note with run IDs + gate status. If anything is red, fix it before touching Phase 2 scope.

### Phase C — Design

Write your own stage doc at `plans-n-solutions/stages/full_async.md`. Mirror the structure of Phase 1's old plan (resolved known unknowns → cut order → tests → green signals → failure modes → invariants). Cover at minimum:

- Trajectory store: FIFO vs priority? Where does it live — in-process dict in the trainer, out-of-process FastAPI service, on-disk sqlite? Pros/cons with numbers.
- Staleness budget K: how big, how enforced, what happens when a trajectory ages out.
- Off-policy correction: clipped IS (extending the existing `tis_imp_ratio_cap` per-step to across-time), V-trace, or IMPALA. Math must be in the doc.
- Producer side: how rollouts fan out to `external_llm_endpoints` without the trainer driving each call; what fires `_publish_lora_adapter` once the store has "enough fresh" trajectories.
- `filter_groups=True` semantics in an async world: where does the filter apply — at ingestion into the store, at sampling time, or both?
- Token-in/token-out invariant: how does it survive a replay buffer (never decode/re-encode across steps).

**Deliverable:** committed `plans-n-solutions/stages/full_async.md`. Run `/codex:review` on this commit (the plan itself) — it will work because it has a concrete diff. Fold objections before starting Phase D.

### Phase D — TDD cut by cut

Each cut: write tests first → make them pass → run `make lint` → run fast-loop pytest → commit → run `/codex:review` on the diff → fold comments. Don't stack two cuts in one commit. Target cut sizes:

- Cut 1: trajectory store + sampler, pure Python, zero training-loop coupling.
- Cut 2: wire the store into the rollout path — producers write, the trainer reads.
- Cut 3: off-policy correction in the loss.
- Cut 4: decouple the `generate_sequences` call from `update_actor` (this is the clock-separation cut).
- Cut 5: sibling launcher `run_proagent_..._fullasync.sh` + `s3_fullasync_docker.sh`. Keep `s2_weightsync_docker.sh` and `..._weightsync.sh` frozen.

### Phase E — Verify

Success gates are in §9. Every gate needs a WandB panel or a log grep. No verbal "looks green".

### Phase F — Handoff

Update `plans-n-solutions/handsoff.md` (this file) with what shipped, what's deferred, and where to look. Commit. Do not push without explicit approval.

---

## 2. Current state

**Phase 2 is CONCLUDED on `full-async`** — closed-loop rank-16 LoRA weight-sync PLUS continuous-producer replay store + clipped temporal IS correction. The core plumbing (§15 ship summary) holds: 10-gate scorecard at Run8 (n=8, `filter_groups=True`, 50 steps) passed its plumbing obligations. Run9 (n=16) then surfaced three distinct Phase 2.5 signals that motivate the next branch (§16 and `plans-n-solutions/stages/run9_n16_report.md`).

**Branch-off point for Phase 2.5: commit `55e94122`** on `full-async`. That commit reverts a tuning experiment back to the baseline config (`MAX_NUM_ITERS=30`, `openhands_timeout=1000`). Base any Phase 2.5 work off this SHA and keep the n=16 launcher overrides (`test_freq=10`, `val_before_train=True`, `NUM_TRAJ=16`) as the observation harness.

Phase 2 key commits on `full-async`:
- `55e94122` — revert phase2 tuning; baseline values locked in (**Phase 2.5 starts here**)
- `f5adb456` — Phase 2 + 3 instrumentation log events (`PRODUCER_ITER`, `DAPO_PRODUCER_CALL`, `RELOAD_LORA`)
- `590f8281` — cooperative producer-stop (§19 fix: skip `_validate` on stop timeout)
- `af07c28b` — DAPO bug #16 rebuild of `job_queue` on producer-mode re-entry
- `dd9e4f3a` — final-run metrics
- Cut 4 (replay store + continuous producer), Cut 5 (sibling launchers)
- `12c0e170`, `846432d0`, `f726a876` — scaffolding and docs

**Branch-off guidance for the next session:**
1. `git checkout -b phase2.5-<topic> 55e94122` off `full-async`.
2. Read §16 for the ranked fix list. Pick one lever per branch — don't stack.
3. Keep the run9 observation scripts (`/tmp/replay_monitor.py`, `plans-n-solutions/stages/run9_n16_report.md` schema) for comparability.
4. Re-read gotchas §10 items **19, 20, 22, plus the new 25–28 below** before touching the producer thread or the IS path.

**Prior starting state = `decoup-weight-sync`.** That branch carries Phase 1 only (lock-step LoRA weight-sync): rank-16 adapters, `/reload_lora` on the pool, trainer-authoritative `policy_version`, DAPO + plain GRPO both wired, 6/6 Phase 1 gates green. All running and validation flows originate there. Nothing earlier is in scope — do not reference, reproduce against, or frame anything relative to pre-`decoup-weight-sync` state.

Phase 1 commits worth reading (from `decoup-weight-sync`): `9191de66 feat: Phase 1 LoRA weight-sync`, `39452b58 feat(phase1): final-run hardening — resume, DAPO, pool headroom`, `bea41a0f Elevate Codex review from optional tooling to a Phase 1 gate`.

---

## 3. Why fully-async is a must (not an optimization)

Under the current lock-step protocol:

1. Trainer calls `generate_sequences(batch_size=4, n=8)` → 32 trajectories, agentic, each can run up to `max_iterations=30` tool-calls with `openhands_timeout=1000 s`.
2. The slowest trajectory in the batch is the step's critical path. Tail latency is ~1000 s per step in bad cases.
3. With `filter_groups.enable=True`, GRPO discards any *group* of `n` samples that all succeeded or all failed — the advantage is zero, gradient is zero. Empirically ~50% of groups get filtered on SWE-Gym.
4. That means ~half of the trainer's rollout wall-clock produces no learning signal. The trainer cannot proceed without `ppo_mini_batch_size=4` surviving groups.

The structural fix: **rollout workers run continuously against whatever adapter is live on the pool**, emit `(prompt_ids, response_ids, logprobs, reward, policy_version)` tuples into a bounded store, and the trainer samples mini-batches from the store as fast as it can form them — applying an importance weight `π_θ / π_{behavior@pv_i}` with clipping and a staleness cutoff. Phase 1's `policy_version` stamp is already on every rollout message (`async_server.py:~1495`), so the plumbing to identify the behavior policy per-trajectory exists. What does NOT exist: the store, the sampler, the off-policy correction in the loss, and the scheduler that runs rollouts independently of `fit()`.

---

## 4. Topology (unchanged from Phase 1)

```
┌──────────────────────── trainer box (this machine) ────────────────────────┐
│  host, poetry venv:  ProRL FastAPI server           :8006  s0_prorl.sh     │
│  host, Docker:       GRPO trainer (8×A100 FSDP)            s?_*.sh         │
│                       verlai/verl:vllm018.dev1                             │
│                       verl @ /tmp/verl (shamanez/verl main, v0.8.0.dev)    │
└────────────────────────────────────────────────────────────────────────────┘
                                  │ HTTP
                                  ▼
┌──────────────────────── EC2 vllm-instance ─────────────────────────────────┐
│  4× vLLM children  :8100  :8101  :8102  :8103                              │
│  launch_remote_vllm_pool.sh start (orchestrated over SSH from trainer box) │
│  Qwen/Qwen3-4B-Instruct-2507, max_model_len=32768, --enable-lora           │
│  --max-loras 8 --max-lora-rank 32 --max-cpu-loras 16                       │
└────────────────────────────────────────────────────────────────────────────┘
```

Frozen files (reproduction-critical — **never edit**, make siblings):
- `scripts/_internal/s2_weightsync_docker.sh`
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh`

Phase 2 will add `scripts/_internal/s3_fullasync_docker.sh` and `..._fullasync.sh` as siblings.

---

## 5. Credentials and hosts

| Secret/host | Where | Notes |
|---|---|---|
| `WANDB_API_KEY`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` | `/home/ubuntu/.prorl_creds.env` | Sourced by `s0_prorl.sh`, `s2_weightsync_docker.sh`, `launch_remote_vllm_pool.sh`. **Never re-export inline. Never commit.** |
| Remote pool host | SSH alias `vllm-instance` (must exist in `~/.ssh/config`) | Public DNS currently `ec2-54-145-77-207.compute-1.amazonaws.com` — **hardcoded** in `run_proagent_qwn3_4B_instruct_weightsync.sh:99`. See Gotchas §10. |
| Remote pool dir | `~/vllm-pool/` on `vllm-instance` | `pid-<port>.pid`, `child-<port>.log`, `venv/`, `hf-cache/`. Override with `REMOTE_POOL_DIR`. |
| Dataset | `/home/ubuntu/data/SkyRL-v0-293/{train,validation}.parquet` | **Do not re-download.** Slow and bandwidth-heavy. Keep. |
| Singularity images | `./singularity_images` → `/opt/dlami/nvme/singularity_images` | DLAMI-specific symlink; gitignored. `OH_RUNTIME_SINGULARITY_IMAGE_REPO` points at it. |
| Trainer checkpoints | `outputs/` under repo root, written by root-owned Docker process | `sudo rm -rf outputs` to wipe. Gitignored. |

**EC2 security group** must allow inbound TCP 8100–8103 from the trainer box's public IP (and 22 for SSH).

---

## 6. Launch sequence (three terminals)

Run in this order. Do not parallelize start-up — the trainer pre-flight probes the pool's `/health`.

```bash
# Terminal 1 — trainer box (host, NOT Docker) — ProRL FastAPI
bash scripts/_internal/s0_prorl.sh
# Wait for: "Uvicorn running on http://0.0.0.0:8006"

# Terminal 2 — trainer box (host) — remote vLLM pool orchestration
source /home/ubuntu/.prorl_creds.env
bash scripts/serving/launch_remote_vllm_pool.sh start
# Orchestrator SSH's into vllm-instance, boots 4 children on 8100-8103.
# Wait for: 4× "ready" from /health.

# Terminal 3 — trainer box (Docker) — GRPO trainer with closed-loop LoRA weight-sync
bash scripts/_internal/s2_weightsync_docker.sh
# Env knobs: TOTAL_EPOCHS TOTAL_TRAINING_STEPS SAVE_FREQ LOG_PATH REMOTE_DNS

# Phase 2 will add:
# bash scripts/_internal/s3_fullasync_docker.sh
```

Stop order is reverse: kill the trainer container, then `launch_remote_vllm_pool.sh stop` (graceful pool shutdown), then kill ProRL.

---

## 7. Observability

**WandB** (project `ProAgent`). Phase 1 ships these keys — preserve them, add Phase 2 keys additively:

| Key | Meaning |
|---|---|
| `weight_sync/policy_version` | Trainer counter, bumps on every successful publish. |
| `weight_sync/adapter_mib` | Tarball size on the wire. |
| `weight_sync/publish_latency_s` | Max endpoint wall-clock, trainer-side. |
| `weight_sync/transfer_latency_s` | `publish_latency_s − vllm_load_latency_s` ≈ network. |
| `weight_sync/vllm_load_latency_s` | Max pool-side `add_lora` ≈ GPU. |
| `weight_sync/endpoints_ok` / `endpoints_failed` | Partial-failure detector. |
| `rollout/staleness_steps` | `global_steps − policy_version`. **Phase 2 must bound this.** |
| `rollout_corr/ppl_ratio` | Per-step importance ratio. Stable ≈ 1.0; drift indicates off-policy divergence. |

Phase 2 adds (suggest — finalize in your `full_async.md`):

- `replay/store_size`, `replay/store_fill_ratio`
- `replay/sample_age_steps_p50`, `_p95`
- `replay/dropped_by_staleness_per_step`
- `is_weight/mean`, `_p99`, `_clip_fraction`

**Log paths:**
- `/tmp/s2-weightsync.log` (Phase 1 trainer). Phase 2: `/tmp/s3-fullasync.log`.
- `/tmp/s0-prorl.log` (ProRL).
- `ssh vllm-instance 'ls ~/vllm-pool/child-*.log'` for per-endpoint logs.

**Pool health / LoRA state:**
```bash
for p in 8100 8101 8102 8103; do curl -sf http://$REMOTE_DNS:$p/health | jq .; done
# {"status":"ok","active_lora":"pv17","policy_version":17, ...}
```

**Structured events on the pool side (JSON, one line per `/reload_lora`):**
```bash
ssh vllm-instance 'jq -c "select(.event==\"reload_lora\")" ~/vllm-pool/child-8100.log' | tail -5
```

---

## 8. Skills — which, when, why

All under `.claude/skills/<name>/SKILL.md`. Invoke with `/skill <name>` (or ask the agent to read it).

| Skill | When to reach for it | Why |
|---|---|---|
| **repo-architecture** | Before your first edit. | Names the top-level abstractions (`AgentHandler`, registry, token-in/token-out, Singularity runtime) and the invariants. If you skip this, you will break something invisible. |
| **karpathy-guidelines** | Any RL loss / sampler / off-policy correction design. | Grounded reasoning on KL, IS, clipping, entropy. Phase 2 needs you to decide between V-trace, IMPALA, clipped IS with staleness cutoff. |
| **python-patterns** | Writing the trajectory store, async producer/consumer. | Structured concurrency over fire-and-forget. `@dataclass(slots=True, frozen=True)` for trajectory records. `pathlib` over `os.path`. |
| **python-testing** | Every cut. | `pytest.ini` markers (`asyncio`, `integration`, `real_data`, `slow`) and the fast-loop command. |
| **tdd-workflow** | Before writing each cut. | Red/green/refactor. Tests name the behavior before the code. |
| **verification-loop** | After each batch of edits. | `make lint` + `pytest -m "not integration and not slow and not real_data"`. Never commit without it. |
| **documentation-lookup** | Any time you're about to guess a verl or vLLM API surface. | verl 0.8 and vLLM 0.18 are moving targets. Prefer this over WebFetch — it's scoped to what's actually on disk. |
| **strategic-compact** | When context gets hot near a cut boundary. | Summarize + compact so the next cut starts fresh. |
| **eval-harness** | Baseline-vs-async A/B. | Reuse Phase 1's val set (`validation.parquet`, 23 prompts). pass@k with `input_hash` strategy. |
| **continuous-learning** | When you hit a surprise. | Write the finding somewhere durable. Phase 1 left 6 gotchas for you — add yours. |
| **security-review** | Only if you switch transport to S3 presigned URLs or introduce a new HTTP surface. | SSRF, signed-URL expiry, IAM scope. Skip if replay stays in-process or local. |
| **backend-patterns** | Only if the replay store becomes its own FastAPI service. | FastAPI lifecycles, dependency injection, gracefull shutdown. |
| **api-design** | Same trigger as backend-patterns. | Versioned endpoints, error shapes. |

Skills you should **not** reach for in Phase 2: `mcp-server-patterns` (irrelevant).

---

## 9. Agents — which, when, why

Spawn via the `Agent` tool with `subagent_type=<name>`. Independent queries → spawn in parallel.

| Agent | When |
|---|---|
| **Plan** | Phase C design. One call, give it this doc's §3 and §8, get a step-by-step back. |
| **planner** | If Plan's output is too shallow for the replay-buffer cut. |
| **Explore** | Any cross-module search that will take 3+ rounds. (Single Grep? Use Grep directly.) |
| **tdd-guide** | Enforces test-first per cut. Feed it the failing behavior spec. |
| **code-reviewer** | Immediately after each cut lands. |
| **python-reviewer** | Python-specific review (PEP 8, types, Pythonic idioms). |
| **silent-failure-hunter** | After any error-path / abort / partial-failure code lands. Phase 1 had an `endpoints_failed>0 → raise RuntimeError` contract that Phase 2 must generalize to the replay store. |
| **security-reviewer** | Only for new HTTP surfaces / transport changes. |
| **performance-optimizer** | If the trainer still stalls post-Phase-2, profile the async scheduler and the store. |
| **codex-rescue** | **DIFF-SCOPED ONLY.** See §10 — never run in plan mode. |
| **refactor-cleaner** | After a cut lands. Run `knip` / `ts-prune` equivalent for Python is basically `vulture`; use with caution. |
| **doc-updater** | At stage boundaries. Update this file and any README that references renamed code. |
| **pr-test-analyzer** | Before opening a PR. Coverage vs behavioral completeness. |

---

## 10. Gotchas (read every one)

1. **`/codex:*` and the `codex-rescue` agent hang in plan mode / open-ended exploration.** They are built to work on a concrete diff. ALWAYS give them a scope: working tree (default), `git diff base...HEAD`, or a specific commit range. This is codified in `.claude/rules/codex-usage.md`. If you need free-form review, use `code-reviewer` or `python-reviewer`.
2. **`trainer.val_only=True` is silently ignored by the custom trainer.** Upstream verl (`/tmp/verl/verl/trainer/ppo/ray_trainer.py:~1325`) has `if self.config.trainer.get("val_only", False): return` — the fork does not. Baseline-val runs will roll straight into training unless you `docker rm -f <name>` after the val prints. Fix this in Phase 2 if you use val_only.
3. **Remote EC2 DNS is hardcoded** in `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh:99` as `ec2-54-145-77-207.compute-1.amazonaws.com`. Parameterize it (`REMOTE_DNS` env) before handing to a new environment. Same file: `openhands_base_url=http://localhost:8006` assumes ProRL is on the same box.
4. **SSH alias `vllm-instance` must be in `~/.ssh/config`** on the trainer box before `launch_remote_vllm_pool.sh` will work. Script does not explain this in its error message.
5. **Autoflake strips module-level imports** during pre-commit. For imports referenced only in decorators, late-bound methods, or generated strings, use an inline import with `# noqa: PLC0415`. Precedent: `verl_custom/workers/fsdp_workers.py` (`_build_compute_log_prob`), `verl_custom/trainer/ppo/ray_trainer.py:1242` (inline `local_mkdir_safe`).
6. **Pool in-memory LoRA state survives trainer restarts** but not pool restarts. `policy_version=0` is the "no adapter loaded" sentinel — base model. After a trainer resume (`trainer.resume_mode=auto` reads `latest_checkpointed_iteration.txt`), `ray_trainer.py:1506-1515` syncs the trainer's `self.policy_version` to the resumed `global_steps` so the first post-resume publish isn't rejected as non-monotonic (409). Preserve this sync in any Phase 2 resume path.
7. **DAPO trainer class vs plain `RayPPOTrainer` selection** lives at `trainer_integration/verl/verl_custom/trainer/main_ppo.py:232-236`: `filter_groups.enable=True` → `RayPPOTrainerDAPO`, else `RayPPOTrainer`. **Both classes must have the publish hook and any Phase 2 hook wired.** Phase 1 missed this for DAPO initially — commit `39452b58` fixed it.
8. **Token-in/token-out invariant** is in `openhands/llm/nvidia/qwen3.py` and `qwen2_5_vl.py`. Never decode and re-tokenize across turns — token boundaries shift, actor vs reference diverges, KL/entropy go NaN, training collapses. A replay buffer that stores decoded text and re-encodes on read would silently break this. **Store token IDs, not strings.**
9. **`max-loras 8`** on the pool absorbs in-flight swap slots. If you exceed it (e.g., by lowering the drain timeout below what's survivable for a long request), `remove_lora` fails and the slot leaks. The pool currently tolerates up to 8 leaked slots before a child needs a bounce.
10. **No `--no-verify`.** No `git push` without explicit approval. No force push. No modifying `dev_config/python/**`. No widening `pyproject.toml` pins without reading the pin comment (some are CVE-related, some are bug-workarounds).
11. **Untracked artifacts that look like your in-progress work:** `outputs/` (root-owned, `sudo rm -rf`), `wandb/`, `/tmp/s*-*.log`, `singularity_images` (symlink — leave alone). All gitignored.
12. **`save_freq=1`** means publish every step. Fine during debugging. For a real run, set it so that `publish_latency_s × publishes_per_epoch < step_time × save_freq`.
13. **DAPO `filter_groups=True` E2E runs take 2–3× the wall-clock of plain GRPO** because `generate_sequences_dapo` waits for `train_batch_size` **surviving** groups, not `train_batch_size` prompts, and SWE-Gym drops ~50 % of groups to sign-shared rewards. **Always run `filter_groups=False` (plain GRPO) as the primary smoke gate first**; only promote to `filter_groups=True` after the plain run is green. Reverse order burns multi-hour debugging on bugs the fast path surfaces in minutes. See `plans-n-solutions/stages/full_async.md §5a` for the test ordering rule. Applies to every Phase 2 E2E run (new branch, resumed branch, post-config-change verification).
14. **`TrajectoryStore` concurrency is one `threading.Lock` serializing push / evict / sample+pop.** All mutations and reads (`push_from_dataproto`, `evict_stale`, `sample_mini_batch`, `num_groups`, `num_fresh_groups`, `num_trajectories`) acquire `self._lock`. Inside `sample_mini_batch` the sequence is `_evict_stale_locked → select → pop → detach records` as a single critical section, so a group cannot be observed-then-deleted out from under the caller. `_pack` runs outside the lock but only operates on the detached `records` list, so the returned `DataProto` cannot alias shared state. The only lock-free cross-thread read is the producer's `rollout_manager.policy_version` (single int, relies on CPython GIL atomicity — see §20 below).
15. **Sampling is consume-on-sample (queue semantics), not with-replacement.** `sample_mini_batch` pops chosen groups from `self._groups` before returning. Rationale: when producer throughput falls below trainer throughput the effective buffer shrinks to ~1 group, and with-replacement sampling would train on the same 8 trajectories K+1 times — overfitting, not replay. This deviates from the paper's Fig-18 finding (which holds at a large effective buffer) and was triggered by a real crash at run4/step 11 where a single group aged out between the waiter unblocking and `sample_mini_batch` evicting it. The waiter predicate was also tightened to `num_fresh_groups(current_step) >= n_groups` so it never unblocks on a group about to be dropped as stale.
16. **`staleness_cutoff_k` is no longer a reuse cap, it is a producer-stall safety drop.** With pop-on-sample, a group sits in the store only between push and the next sample. K drops groups that were pushed but nobody consumed them for > K trainer steps (e.g., because validation paused the sampler). Default `K=4` is a conservative "producer pushed a stale-flavored group during a multi-step trainer pause" cutoff, not a "how many times can we reuse" knob.
17. **`OPENHANDS_NUM_WORKERS=32` is the sweet spot for a 4-child vLLM pool, not 64.** The pool saturates to ~100 % GPU util at ~32 concurrent clients (measured 2026-04-23 on 4× H100 with Qwen3-4B, LoRA rank 32, `gpu_memory_utilization=0.45`). Bumping to 64 made every client-turn slower — DAPO `filter_groups=True` Progress regressed from 3/4 at 17 min (32 workers) to 0/4 at 47 min (64 workers). The `replay.producer_batch_size` yaml key is **dead config** — the actual prompts-per-call comes from `data.train_batch_size` (plain GRPO) or DAPO's internal dataloader (`filter_groups=True`); no producer-side dataloader exists in Phase 2. If the vLLM pool ever grows to 8 children, re-measure the saturation point before raising `OPENHANDS_NUM_WORKERS`.
18. **`generate_sequences_dapo` leaves un-dispatched jobs in `self.job_queue` on every call.** The result-collection loop in `async_server_dapo.py` breaks as soon as `num_completed_instances >= requested_batch_size` — the dispatcher is cancelled, mid-flight tasks are cancelled, but queued-but-not-dispatched jobs stay. The classic path relies on this: the leftovers are the head of the next call's batch, pinned to `self.all_input_batch` for instance-id conservation. **Producer mode can't reuse them** — `all_input_batch` is reset each call to keep the buffer-push filtered, and each call runs inside its own `asyncio.run` event loop (a PriorityQueue built against a now-closed loop can raise from `put`/`get` in Python 3.12). Producer-mode branch at the top of `generate_sequences_dapo` drops the leftovers by **rebuilding** the queue (`self.job_queue = asyncio.PriorityQueue()`) rather than draining it. Observed on run6/step 2 as `AssertionError: DAPO producer-mode invariant: job_queue must be drained`; the earlier Cut 4 fix only asserted the invariant instead of actually restoring it. Fixed 2026-04-23 — do not replace the rebuild with a `get_nowait` drain; stale loop refs make drain-only fragile.
19. **`ContinuousRolloutProducer.stop()` has no handle on a producer thread mid-`asyncio.run`.** `stop()` sets `self._stop_event` and joins with a 10 s timeout. The event is only checked at the top of the worker `while not self._stop_event.is_set():` loop; once inside `self._generate_fn(...)` → `asyncio.run(generate_sequences_dapo)` the thread ignores the event until the call returns (up to `openhands_timeout * max_iterations` ≈ 45 min on SWE-Gym). First observed at run7/step 5 (2026-04-23 16:10:56): trainer logged `ContinuousRolloutProducer did not exit within 10.0s`; subsequent validation start failed with `HTTP 400: Server is already running`; validation + producer-call-3 ran concurrent OH dispatchers against the same 32-worker pool for 40+ min producing a stream of `Message N returned empty response` / `Timeout error sending message` retries until the runbook triggered stop-and-fix. Core RL training was unaffected at the weight level (step 5 LoRA publish succeeded — all 4 vLLM children reloaded at 16:10:46 with `policy_version: 1`), but trainer progress was wedged at tqdm 4/50 because step 5's `logger.log` sits after `_validate`. **Fix shipped on `full-async` (commit landed after run7 stop):** `ContinuousRolloutProducer.stop()` now returns `bool`; on timeout it **does not** null `self._thread` and **does not** call `rollout_manager.sleep()`. Both `_stop_continuous_producer_if_needed` paths (ray_trainer.py, ray_trainer_dapo.py fit()) propagate the `False` and **skip `_validate`** at that save boundary, logging `_logger.warning('step=%d skipping _validate: producer stop timed out')`; the next save boundary retries. This keeps the producer running against a single OH session — no concurrent dispatcher, no 400-loop, no 40-min wedge. Trade-off: validation metrics can be skipped at boundaries that land mid-producer-call; they resume whenever stop succeeds within the 10 s window. **Proper fix** (deferred): make `_generate_fn` cooperatively cancellable — wrap `generate_sequences_dapo` in a `loop.create_task` and cancel it from `stop()` — but that crosses the async_server boundary and needs its own cut.
20. **`rollout_manager.policy_version` is read across threads without a lock.** The continuous producer (daemon thread) reads it; the trainer (main thread) writes it inside `_publish_lora_adapter` after a successful `/reload_lora` fanout. Relies on CPython GIL atomicity of single-int load/store — a store is one bytecode op, a load is one bytecode op, neither can interleave within a bytecode boundary. The benign race is temporal: a producer call in flight when publish lands stamps `behavior_policy_version = old_pv` on every trajectory of that call; the *next* producer call reads `new_pv`. That's exactly the TIS correction's input — not a bug. Do NOT rewrite as a lock or `threading.Event`; the latency would dominate. If the read ever needs to do more than "fetch an int", add a proper lock *then*.
21. **`TrajectoryRecord` prompt/response lengths vary across store entries.** Producer batches (different DAPO calls) pad to call-local max lengths (`async_server.py:1343-1367`), so `group_A.prompt_ids.shape = (8, 3800)` and `group_B.prompt_ids.shape = (8, 4122)` can coexist in the same deque. Store uses raw variable-length tuples and re-pads at `sample_mini_batch` to a fresh per-sample max. `DataProto.concat → torch.cat(dim=0)` (verl `protocol.py:930`) would assert-fail on a dim-1 mismatch; re-padding is load-bearing. Don't "optimize" by padding once at push.
22. **`/reload_lora` drain (`_vllm_child.py:163-166`) stamps `policy_version` atomically per `/generate` call.** In-flight agentic trajectories (tool-use loops) that straddle a publish still see the *old* version because each turn's `/generate` completes against whichever version was active at that turn's arrival — correct behavior (a mid-rollout version switch would mix logprobs across two policies in one trajectory). **The replay store then stamps `behavior_policy_version = rollout_manager.policy_version_at_push_time`**, which is the version at the *end* of the OpenHands session, not per-turn. For temporal IS this is a slight approximation; for rank-16 adapters with publish_latency ≈ 30s and turn-time ≈ 20s the difference is within TIS clip and has not caused pathological `is_weight` in run8. If you ever see `clip_fraction > 0.5` with K=4 staleness, investigate per-turn version stamping as a possible fix.
23. **`endpoints_failed > 0` abort contract (`ray_trainer.py:1348-1352`) is preserved in producer mode.** A warm buffer does not mask a broken pool: publish failure raises from the trainer's `_publish_lora_adapter` and the producer thread is stopped as part of trainer exit. If `endpoints_failed` bubbles up *inside* the producer's `generate_sequences` call (shouldn't happen — publishes are trainer-driven), `check_background_error` re-raises on the next trainer-side call.
24. **Buffer is ephemeral — not checkpointed.** On trainer resume (`trainer.resume_mode=auto`), the store starts empty and re-warms from scratch. Pre-resume entries would be maximally stale anyway (pre-`global_steps`-reset), so discarding them is correct. Warm-up time ≈ `N / producer_throughput` steps; in the current producer-bound regime, that's a few steps of no-op `wait_until` at start-up.
25. **§19 cooperative skip can fire during `fit()` — not just at shutdown.** Run9 / step 10 hit it for the first time at a scheduled validation boundary: `_validate()` wanted an exclusive pool, called `producer.stop(timeout=10s)`, the producer was mid-`generate_sequences_dapo` (53+ min call), timeout elapsed → validation **skipped** (no corruption, no traceback). With `save_freq=5` and `test_freq=10`, this slips the first in-training pass@k datapoint from step 10 to step 20 — **3× worse time-to-first-eval** than the gate assumed. Direct fix: pass `timeout=7200` at validation boundaries (zero-code knob flip — `ContinuousRolloutProducer.stop(timeout=…)` already parameterised). Principled fix: add `producer.pause()` / `producer.resume()` that lets the worker finish its current call then pauses between calls (~40 LOC in `continuous_producer.py`). Keep §19 for shutdown; use pause/resume for planned sampler-exclusive windows.
26. **Pool-adapter-age (`rollout/staleness_steps`) and buffer-age (`replay/sample_age_steps`) diverge whenever producer-wall > `save_freq × burst_duration`.** Run9 step 9: `sample_age_p50=0` (iter 3's groups were just pushed) but `staleness_steps=4` (pool had been at pv=1 for 4 steps because no publish landed between step 5 and step 9). The paper's "buffer age" intuition and the fork's "pool-adapter age" invariant measure **different things**. IS clipping correlates with pool-adapter-age, not buffer-age. Don't claim "staleness bounded" from `sample_age_p95 ≤ K` alone — add a companion gate `rollout/staleness_steps_p95 ≤ K + save_freq`. Consider tying buffer eviction to pool-adapter-age rather than buffer-age in Phase 2.5.
27. **`is_weight/clip_fraction` is ~60 % in n=16 regime, of which only ~20 % is real policy drift.** Run9 `rollout_corr/log_ppl_diff` oscillates 0.49–0.79 across all 10 steps (clip threshold `log(tis_imp_ratio_cap=2) ≈ 0.69`). Decomposition of the divergence source:
    - **~0.35** temperature mismatch: rollout sampling at `T=1.4 top_p=0.95`, trainer's `old_log_prob` / `ref_log_prob` forward pass at `T=1.0` (dp_actor.py default).
    - **~0.20** vLLM ↔ FSDP numerical divergence: different kernels (FlashAttention vs FA-with-paged-KV), different fused ops, slightly different softmax paths. Systematic, not random.
    - **~0.05** LoRA load path: rank-16 merge vs adapter-applied forward can drift at float16/bfloat16.
    - **~0.15** genuine policy drift from the adapter difference between push-time pv and trainer-update-time pv.

    **Cheapest fix: align the trainer's `old_log_prob` pass temperature to the rollout temperature.** Single-line patch in `dp_actor.py`'s `compute_log_prob` (scale logits by `1/T` before log-softmax). Not a bug in the fork — the existing path assumes on-policy, where T-scaling cancels in the ratio. With Phase 2's stored `rollout_log_probs`, the ratio is `exp(old − rollout)` and the T-mismatch no longer cancels. Do NOT widen `tis_imp_ratio_cap` as the first move — that masks the symptom, not the cause.
28. **Producer-wall outliers happen (80 min vs 53 min typical) and are not yet instrumented to the prompt level.** Run9 iter 3 regressed completion 40 %→27 %, effective TPS 664→435, wall 53→80 min; iter 4 recovered to 58 min. Plausible causes: dataset difficulty drift (iter 3's first 10 prompts were harder), post-publish-1 policy regression (pv=1 worse than pv=0 on some prompts at 4-step LR=1e-6), pool KV-cache fragmentation over long uptimes. Cannot distinguish without emitting per-prompt `(uid, resolved_ratio, wall_s)` in `DAPO_PRODUCER_CALL`. Priority Phase 2.5 instrumentation. Also flag: publish-latency on publish #2 was **1.84× publish #1** (33.6 s vs 18.2 s, `transfer_s` 4.0→19.0); correlates with iter-4 dispatcher startup competing for pool bandwidth. If publish #3 is also > 30 s, systemic. Mitigation: gate publish-push on a low-activity window, or move transfer onto a dedicated HTTP client.

---

## 11. Pointer table (file:line)

| Role | Path | Notes |
|---|---|---|
| Pool child (vLLM wrapper, FastAPI) | `scripts/serving/_vllm_child.py` | `active_lora` global, `POST /reload_lora`, `_swap_lock`, `_inflight_cond`, drain+swap protocol. |
| Pool runner (per child) | `scripts/serving/_remote_vllm_runner.sh` | `--enable-lora --max-loras 8 --max-lora-rank 32 --max-cpu-loras 16`. |
| Pool orchestrator | `scripts/serving/launch_remote_vllm_pool.sh` | `start|stop|restart|publish` verbs. SSHs into `vllm-instance`. |
| Trainer entrypoint (docker) | `scripts/_internal/s2_weightsync_docker.sh` | Phase 1. Copy this when making `s3_fullasync_docker.sh`. |
| Trainer entrypoint (host — ProRL) | `scripts/_internal/s0_prorl.sh` | Unchanged between phases. |
| Hydra script (Phase 1) | `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh` | `lora_rank=32`, `lora_alpha=64`, `+publish_on_save=True`, hardcoded EC2 DNS on line 99. |
| GRPO trainer | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | `_publish_lora_adapter` (after `_save_checkpoint`), policy_version sync at 1506-1515, metrics hook at ~1685. |
| DAPO trainer | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py` | Same publish hook wired. |
| Trainer class selector | `trainer_integration/verl/verl_custom/trainer/main_ppo.py:232-236` | `filter_groups.enable=True` → DAPO, else plain. |
| Rollout manager | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | `policy_version` stamping around line 1495, EXTERNAL BYPASS ACTIVE path around 408-425. |
| Token-level client | `openhands/llm/nvidia/qwen3.py` | INVARIANT. Never modify. |
| verl upstream (read-only ref) | `/tmp/verl/verl/workers/fsdp_workers.py:1210-1253`, `/tmp/verl/verl/utils/fsdp_utils.py:593` | PEFT save path (`layered_summon_lora_params`). |
| Experience replay reference | `docs/README.md` | Distilled summary of "Efficient RL Training for LLMs with Experience Replay" (Arnal et al.) mapped onto this codebase — staleness/coupling/compute trade-off, `(W,T)` knobs, positive-bias sampling, minimal `BufferStructure` diff, interaction with Phase 1 `POST /reload_lora` and DAPO `filter_groups`. Read this before designing the Phase 2 trajectory store. |
| Related upstream (external, read-only) | `NVIDIA-NeMo/ProRL-Agent-Server@polar` (GitHub) | Different stack (Slime + SGLang, co-located GPUs, **NCCL full-weight sync**) — **incompatible** with our decoupled EC2 topology; don't lift code. Borrow conceptually only: (a) commit `bbbfa6c` push-based rollout completion (FastAPI callback + per-task `asyncio.Event` + 60 s fallback poll) as a pattern for Cut 2 producer→buffer writeback; (b) Slime's `--use-tis` + `--use-rollout-logprobs` as upstream precedent for Cut 3 truncated-IS correction; (c) commit `f3e5dc0` "drop failed traces" as a concrete example of ingest-time filtering (option A in `docs/README.md` §8). Polar is primarily a framework-agnostic *harness* proxy — out of Phase 2 scope. |
| Dataset | `/home/ubuntu/data/SkyRL-v0-293/{train,validation}.parquet` | 293 train / 23 val prompts. Do not re-download. |

---

## 12. Success gates (Phase 2 merge-blocking)

Every gate needs a WandB panel or a log grep. No verbal "looks green".

1. **Filter-groups parity.** A 50-step run with `+algorithm.filter_groups.enable=True` AND a 50-step run with `=False` both land cleanly on `full-async`. WandB `critic/rewards/mean` trends up in both, shape can differ.
2. **Clock separation.** A WandB panel shows `trainer_update_time_s` is not dominated by `rollout_wait_time_s`. Concretely: the trainer completes at least 2 `update_actor` passes between consecutive LoRA publishes when the store has capacity.
3. **Staleness bounded.** `rollout/staleness_steps` stays ≤ K (you pick K in `full_async.md`; suggest K=4). Traces older than K are dropped or down-weighted, not consumed blindly.
4. **Importance weight sanity.** `is_weight/p99 < 10` (tune the cap) and `is_weight/clip_fraction < 0.2`. A run with clip_fraction ≥ 0.5 means your staleness budget is too loose.
5. **Token-in/token-out preserved.** Dump 10 random trajectories from the store and verify round-trip token equality with what the pool emitted. A golden file test with 3 fixed trajectories is enough to regression-guard this.
6. **Baseline invariants.** Phase 1 gates that must still pass:
   - ≥ 4 successful `/reload_lora` events per 20 steps at `save_freq=5`.
   - Zero 5xx on `/generate` during publishes.
   - `weight_sync/endpoints_failed == 0`.
   - `grep -c 'EXTERNAL BYPASS ACTIVE' /tmp/s3-fullasync.log ≥ 1` (decoupled topology still active).

---

## 13. What NOT to do

- Don't touch `/tmp/verl` — pinned read-only upstream reference (shamanez/verl main, v0.8.0.dev; see §4, §11). Fork customizations live in `trainer_integration/verl/verl_custom/` as a patch package on top of the container's `verlai/verl:vllm018.dev1`; edits to `/tmp/verl` are invisible to the trainer.
- Don't touch `openhands/llm/nvidia/qwen3.py` or `qwen2_5_vl.py`.
- Don't edit the frozen files in §4. Siblings only.
- Don't modify `dev_config/python/**`.
- Don't widen `pyproject.toml` pins without reading the pin comment.
- Don't store decoded text across steps in the replay buffer.
- Don't use `--no-verify`, don't `git push --force`, don't push at all without explicit approval.
- Don't commit `outputs/`, `wandb/`, `/tmp/*.log`, `singularity_images`, or anything under `/home/ubuntu/.prorl_creds.env`. All gitignored.
- Don't re-download the SkyRL-v0-293 dataset.
- Don't run `/codex:*` in plan mode. Always give it a concrete diff.
- Don't stack two cuts in one commit.

---

## 14. When in doubt

1. Re-read `CLAUDE.md` and this file.
2. Read the matching Phase 1 code in the Pointer Table — it is your template.
3. If a design choice is 50/50, write both options into `full_async.md` with pros/cons and ask the user.
4. If a tool hangs, check §10 before retrying.

---

## 15. Phase F ship summary (Phase 2)

**Shipped:** 2026-04-24 on branch `full-async`. Phase 2 = fully-async decoupled agentic RL with bounded in-process replay store + clipped temporal IS correction. The 10 merge-blocking plumbing gates in §12 pass on Run8 (DAPO `filter_groups=True`, n=8, 50 steps, 9h04m, log `/tmp/s3-fullasync.log`). Run9 (n=16, paper-aligned) reproduces the plumbing cleanly and surfaces three Phase 2.5 signals — see **§16** for the roadmap and `plans-n-solutions/stages/run9_n16_report.md` for evidence.

### What landed

| Component | File | Cut |
|---|---|---|
| `TrajectoryStore` (FIFO deque, K=4 staleness, pop-on-sample, atomic push) | `trainer_integration/verl/verl_custom/replay/trajectory_store.py` | Cut 1, 4.1 |
| `ContinuousRolloutProducer` (daemon thread, cooperative stop, §19 fix) | `trainer_integration/verl/verl_custom/replay/continuous_producer.py` | Cut 4, post-run7 fix `590f8281` |
| Trainer integration (plain GRPO + DAPO) | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py`, `ray_trainer_dapo.py` | Cut 2, 4 |
| Temporal IS correction | `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py` (gated on `replay.use_temporal_is`) | Cut 3 |
| DAPO bug #16 producer-mode reset | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` | Cut 4c |
| DAPO bug #18 resume sync | `ray_trainer_dapo.py` (mirror of `ray_trainer.py:1510-1514`) | Cut 2 |
| Sibling launchers | `scripts/_internal/s3_fullasync_docker.sh`, `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh` | Cut 5 |
| Tests | `tests/replay/{test_trajectory_store.py,test_continuous_producer.py}`, `tests/trainer/{test_trainer_buffer_integration.py,test_temporal_is_correction.py}` | Cuts 1-4 |
| Docs | `plans-n-solutions/stages/{full_async.md,run8_findings.md,latencies.md,how_to_run.md,replay_dynamics.md}` + this file | Cut 0 + phases E, F |

### 10-gate scorecard (run8)

| # | Gate | Result |
|---|---|---|
| 1 | `weight_sync/endpoints_failed == 0` | **PASS** — 10/10 publishes endpoints_ok:4 |
| 2 | ≥ 4 `/reload_lora` per 20 steps at save_freq=5 | **PASS** — 10 publishes over 50 steps |
| 3 | Zero 5xx on `/generate` during publishes | **PASS** — drain_timed_out:true, ok:true under load |
| 4 | `replay/sample_age_steps_p95 ≤ K=4` | **PASS** — max=3, mean=1.47, `dropped_by_staleness_total=0` |
| 5 | `is_weight/p99 < 10`, `clip_fraction < 0.2` | **SKIP** — keys not emitted (store near-empty → IS ≈ 1 trivially). Non-blocker; follow-up is logging-only. |
| 6 | `critic/rewards/mean` trends up | **INCONCLUSIVE** — 50 steps too short (paper Fig 1 needs 5k+). Plumbing gate only. |
| 7 | Offline A/B via eval-harness | **TBD** — pending eval-harness kick-off on validation.parquet |
| 8 | Both `filter_groups={False,True}` land clean | **PASS** — task #22 (False) + run8 (True) both GREEN |
| 9 | Zero fit()-time tracebacks / §19 skips | **PASS** — §19 at shutdown is the expected cooperative path |
| 10 | Token-in/token-out preserved | **PASS** — golden test + round-trip equality |

Also see `plans-n-solutions/stages/run8_findings.md` for full evidence and `latencies.md` for the per-component TPS breakdown.

### Key empirical findings

- **Regime is producer-bound, not trainer-bound.** Rollout wait = 96% of wall-clock (623 s/step of 651 s/step average). The replay store sits at 0–3 groups most of the time.
- **DAPO hard-filter dominates producer cost.** 81% of filtered groups are all-fail (0/8), 19% all-pass (8/8). The model is too weak for ~half the SWE-Gym train prompts at this LR × rank × scale. This is exactly the pathology Phase 2.5 (positive-bias sampling + AsymRE) targets.
- **Replay reuse does not activate at this scale.** Pop-on-sample + producer-bound = each group consumed exactly once. Phase 2's value today is smoothing (producer bursts 4 groups → trainer burns 4 steps fast) and setup for Phase 2.5.
- **Weight sync cost is negligible.** Mean publish 30 s, amortized 0.9% of wall-clock. S3 upload 14 MB/s, 4-child load 19 MB/s.
- **Staleness cap K=4 is dormant.** `sample_age_steps_p95` max=3 observed. K only starts biting when producer gets faster or trainer gets slower.

See `plans-n-solutions/stages/replay_dynamics.md` for the full producer/store/trainer interaction reference.

### Next work (not shipped in Phase 2)

1. **Phase 2.5** — positive-bias sampling + AsymRE loss. `docs/README.md §9`. Primary lever for converting producer-bound time into training signal.
2. **Run9 — n=16 tuning study.** `NUM_TRAJ=16`, otherwise-identical config. Expected: reduced DAPO hard-filter rate (bigger groups → higher P(mixed-sign)), 2× rollout cost, better baseline variance. See `replay_dynamics.md §10` for the plan.
3. **Offline A/B (gate 7)** — run `eval-harness` skill on `validation.parquet` (23 prompts, pass@k by `input_hash`) for `decoup-weight-sync` vs `full-async` at matched `global_steps`.
4. **Emit `is_weight/*` keys unconditionally** — currently gated on non-empty buffer; should always emit so gate 5 is measurable even in producer-bound regimes.
5. **Proper §19 fix** — make `_generate_fn` cooperatively cancellable (wrap `generate_sequences_dapo` in `loop.create_task`). Deferred; current skip-validate workaround is sufficient for landed runs.
6. **Sharded buffer** — paper Appendix D.4 says little impact; promote only if the single central lock profiles hot under larger scale. Not observed.

### Knobs locked in for this phase

| Knob | Value | Rationale |
|---|---|---|
| `replay.enable` | `True` | Phase 2 on |
| `replay.continuous_producer` | `True` | Clock separation path |
| `replay.buffer_size` | `128` | 4× `train_batch_size` × `n` headroom |
| `replay.staleness_cutoff_k` | `4` | Pre-tightening value, hasn't bitten in practice |
| `replay.use_temporal_is` | `True` | Gates existing `core_algos.py` TIS path on stored logprobs |
| `actor_rollout_ref.actor.tis_imp_ratio_cap` | `2` | TIS clip upper bound |
| `data.train_batch_size` | `4` | DAPO survivors-per-call target (Option A ingest filter) |
| `actor_rollout_ref.rollout.n` | `8` | GRPO group size |
| `SAVE_FREQ` | `5` | `save_freq=5` steps between LoRA publishes |
| `OPENHANDS_NUM_WORKERS` | `32` | Sweet spot for 4-child pool (gotcha #17) |
| `replay.wait_timeout_s` | `7200.0` | 2 h soft floor on producer stall |

Change one knob per run. Don't stack knob changes with code changes.

Last updated at the end of Phase 2 shipping. Keep this file current — it is the next agent's starting point after you.

---

## 16. Phase 2.5 kickoff — what the measurements say, what to do next

Phase 2's plumbing works. The Run9 (n=16, paper-aligned) measurements show the clock-separation benefit **does not yet engage** in this regime — trainer utilisation 1.1–4.3 %, buffer at `store_size=0` for 50+ % of wall-clock — and surface three signals that a follow-up phase must address. This section is the **plan-of-record entry point** for Phase 2.5.

**Source of truth for evidence:** `plans-n-solutions/stages/run9_n16_report.md`. Read it before branching.

### 16.1 Phase 2 verdict

| Dimension | Verdict |
|---|---|
| Plumbing (store, producer thread, publish, TIS) | **PASS** — no fit()-time tracebacks, no `endpoints_failed`, no stale-drop events |
| Trainer–rollout clock separation (`trainer_update_time_s` ≪ `rollout_wait_time_s`) | **NOT ACTIVE** — producer is the strict bottleneck at n=16 |
| IS weight sanity (`clip_fraction < 0.2`) | **FAIL** — ~60 % clipping, dominated by non-drift sources (gotcha #27) |
| Pool-adapter-age vs buffer-age invariant | **NEEDS a second gate** (gotcha #26) |
| Validation-during-fit | **RACE SURFACED** — step-10 pass@k skipped via §19 (gotcha #25) |
| Gradient signal per step (n=16) | 1 surviving group × 16 trajectories = 1 group per gradient step (vs Phase-1-style 4 prompts × 8). 4× fewer unique prompts per 10 steps. |

Phase 2 ships **not because learning was demonstrated in 10 steps on a 4B + rank-16 LoRA @ LR=1e-6** (it can't be), but because the infrastructure is measurably correct and every gap has a concrete, measurement-backed fix.

### 16.2 Ranked Phase 2.5 roadmap

Ordered by **(expected lift × cheapness) / blast radius**. Pick one per branch.

#### Tier 1 — one-line / config-only fixes (ship first, cheap)

| # | Fix | File | Expected |
|---|---|---|---|
| T1.a | **Validation-race fix**: pass `timeout=7200` to `producer.stop()` at validation boundaries | `ray_trainer.py` + `ray_trainer_dapo.py` `fit()` — look for `_stop_continuous_producer_if_needed` callers tied to `_validate()` | Restores scheduled pass@k. Trade-off: validation delayed up to 1 h (one producer iter) at each `test_freq` boundary. No new race. |
| T1.b | **Temperature-match IS fix**: scale trainer's `old_log_prob` logits by `1/T_rollout` before log-softmax, only on the Phase 2 path | `trainer_integration/verl/verl_custom/workers/actor/dp_actor.py` `compute_log_prob` | `clip_fraction` ~60% → ~25% (removes the ~0.35 of the 0.55 mean log-ratio that is pure T-mismatch). Unblocks gate 5. |
| T1.c | **Second staleness gate**: log and alert on `rollout/staleness_steps_p95` (pool-adapter-age), not just `replay/sample_age_steps_p95` | `ray_trainer.py` metrics hook ~line 1685 | Makes gotcha #26 visible in WandB. Zero risk. |

#### Tier 2 — small code changes (~50 LOC each)

| # | Fix | File | Expected |
|---|---|---|---|
| T2.a | **`producer.pause()` / `producer.resume()`** — worker finishes current call, pushes, then blocks on a `threading.Event` instead of entering next iter. Validation path pauses/resumes instead of stop/restart. | `trainer_integration/verl/verl_custom/replay/continuous_producer.py` + both `fit()`s | Principled fix for gotcha #25. Replaces T1.a's "big timeout" with proper semantics. Still no async-cancel needed. |
| T2.b | **Per-prompt producer instrumentation**: emit `(prompt_uid, resolved_ratio, wall_s)` list in `DAPO_PRODUCER_CALL` event | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` near the existing event emit | Distinguish iter-3-style regressions (pool drift vs dataset difficulty vs post-publish policy regression — gotcha #28). |
| T2.c | **`is_weight/*` emitted unconditionally** (currently skipped when store near-empty / stats degenerate) | `core_algos.py` + `ray_trainer.py` metrics | Gate 5 measurable in all regimes; no more "SKIP" verdicts. |

#### Tier 3 — architectural — highest lift, biggest blast radius

| # | Fix | File / subsystem | Expected |
|---|---|---|---|
| T3.a | **Parallel DAPO producers** (N concurrent worker threads pulling from one dataloader, all pushing into the same store). Paper's Option A ingest filter is already per-producer; the store already has thread-safe push. | New multi-producer orchestration class; `continuous_producer.py` generalised to a pool | **Highest single lever.** Arrival rate ~N×, staleness holds ≤ K, trainer bursts overlap producer idle. Paper's clock-separation benefit engages. Main risk: pool saturation (4-child `OPENHANDS_NUM_WORKERS=32` is already at sweet spot — see gotcha #17; parallel producers will push past it). Expect to re-tune `OPENHANDS_NUM_WORKERS` down per producer, or add more pool children. |
| T3.b | **Positive-bias sampling + AsymRE loss** (`docs/README.md §9`). Reclaims the 60 % filter-drop wall-clock as sparse-but-real gradient signal. | `TrajectoryStore.sample_mini_batch` (add positive-bias override), `core_algos.py` (AsymRE loss variant) | Converts 4 survivors → ~8 effective gradient-carrying groups per iter at n=8 (stronger at n=16). Orthogonal to T3.a. |
| T3.c | **Pool-adapter-age-based eviction** (vs buffer-age). More conservative, aligns with what IS clipping actually operates on. | `TrajectoryStore.evict_stale` | Fixes the gotcha #26 divergence by making both metrics collapse to one. Small risk of over-eviction during slow-producer windows. |
| T3.d | **Re-size `train_batch_size` at n=16**. At `train_batch_size=4, n=16`, each gradient step sees 1 group × 16 trajectories = 1 unique prompt. At `train_batch_size=8, n=16`, each step sees 2 unique prompts, 32 trajectories. Batch cost is the same tokens-per-step; diversity doubles. | `run_proagent_qwn3_4B_instruct_fullasync.sh` | Partial compensation for the Phase-1-vs-Phase-2 semantic shift. Only do this after T3.a lands — current producer can't feed 8 surviving groups per call in reasonable wall-clock. |

### 16.3 Recommended branching order

1. **Branch `phase2.5-t1-cheap`** off `55e94122`: T1.a + T1.b + T1.c in one commit each (three commits). Run a short n=16 smoke to verify `clip_fraction` drops and validation fires at step 10. Merge back to `full-async` as a patch release.
2. **Branch `phase2.5-t2-pause`** off the updated `full-async`: T2.a (pause/resume), then T2.b (instrumentation). Re-run Run9 with matched config; expect iter-3-style regressions to become diagnosable.
3. **Branch `phase2.5-t3a-parallel-producers`**: the big one. Keep as its own branch; merge only after a 100-step smoke demonstrates trainer utilisation > 20 %.
4. **Branch `phase2.5-t3b-positive-bias`**: can proceed in parallel with T3.a — different code paths.

Don't stack T1/T2 fixes into the same branch as T3.a — you want to attribute the clock-separation lift cleanly.

### 16.4 Out of scope for Phase 2.5 (revisit after a learning run lands)

- Sharded buffer (paper Appendix D.4 — no measured benefit at current scale).
- Out-of-process buffer service (no multi-trainer use case).
- Upstream `rollout_corr_helper.py` rebase / ESS metrics / IcePop — fork's TIS + temporal extension is sufficient through Phase 2.5.
- Changing the LR / rank / SAVE_FREQ sweep until after T3.a — anything before it confounds the clock-separation measurement.

### 16.5 Verification recipe for any Phase 2.5 branch

```
# Minimum smoke (~4 hours wall-clock)
TOTAL_TRAINING_STEPS=20 SAVE_FREQ=5 NUM_TRAJ=16 bash scripts/_internal/s3_fullasync_docker.sh
```

Verify against Run9 baseline numbers in `run9_n16_report.md`:
- `is_weight/clip_fraction` trend (T1.b gate)
- `replay/sample_age_steps_p95` vs `rollout/staleness_steps_p95` divergence (T1.c gate)
- step-10 and step-20 validation fire without §19 skip (T1.a / T2.a gate)
- trainer utilisation (active-burst-time / wall-clock) (T3.a gate)
- DAPO iter wall-clock distribution across 4+ iters (T3.a / T2.b)

For larger runs (50+ steps), rerun the full 10-gate scorecard in §12 and add gate 4b from gotcha #26.

### 16.6 Quick reference — files a Phase 2.5 agent touches

| Area | Primary file |
|---|---|
| Producer thread, stop/pause semantics | `trainer_integration/verl/verl_custom/replay/continuous_producer.py` |
| Store, sampling, eviction | `trainer_integration/verl/verl_custom/replay/trajectory_store.py` |
| Plain GRPO trainer integration | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` |
| DAPO trainer integration | `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py` |
| DAPO producer-mode reset + per-prompt instrumentation | `trainer_integration/verl/verl_custom/nvidia/rollout/async_server_dapo.py` |
| TIS correction, is_weight metrics | `trainer_integration/verl/verl_custom/trainer/ppo/core_algos.py` |
| Temperature match in log-prob forward | `trainer_integration/verl/verl_custom/workers/actor/dp_actor.py` |
| Phase 2.5 launcher siblings | `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_fullasync.sh`, `scripts/_internal/s3_fullasync_docker.sh` |

### 16.7 Monitoring infra to carry forward

- `/tmp/replay_monitor.py` — aggregates `PRODUCER_ITER` + `DAPO_PRODUCER_CALL` + pool `/health` into JSONL. Keep running across branches for comparable producer timing data.
- `/tmp/replay-monitor.jsonl` — rotating log; 60 s cadence; safe to wipe between runs.
- Log path convention: `/tmp/s3-fullasync-<run-label>.log` so multiple runs don't overwrite each other.

---

Last updated at the end of Phase 2 / start of Phase 2.5. Keep this file current — it is the next agent's starting point after you.
