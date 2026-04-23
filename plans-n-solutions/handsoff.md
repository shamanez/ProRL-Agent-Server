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

**Starting state = `decoup-weight-sync`.** That branch carries the closed-loop Phase 1 LoRA weight-sync: rank-16 adapters, `/reload_lora` on the pool, trainer-authoritative `policy_version`, DAPO + plain GRPO both wired, 6/6 Phase 1 gates green. All running and validation flows originate there. Nothing earlier is in scope — do not reference, reproduce against, or frame anything relative to pre-`decoup-weight-sync` state.

`full-async` (this branch) was cut from `decoup-weight-sync` HEAD. No Phase 2 code yet — you write it.

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

Last updated at the start of Phase 2. Keep this file current — it is the next agent's starting point after you.
