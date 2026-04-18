# Stage 2 — Weight sync + replay buffer

**Status: NOT STARTED.** Stage 1 (all three cuts — local smoke, local decoupled trainer, remote HTTP pool) ships the decoupled topology with stale weights. Stage 2 is the next step: close the staleness gap, and let inference run continuously instead of in lock-step with training.

Two deliverables, landed together because the replay buffer is only interesting once the trainer can publish fresh weights:

| Cut | What it does | Trainer edits? | Pool edits? |
|---|---|---|---|
| **A — Weight publish / reload** | Trainer streams actor state_dict to each pool endpoint every K steps; pool atomically swaps weights without dropping in-flight requests. | Yes (new `_publish_policy_version()` in `ray_trainer.py`). | Yes (new `POST /reload_weights` in `_vllm_child.py` + pool orchestrator routing). |
| **B — Replay buffer** | Rollouts stream into a disk-backed trajectory store; trainer samples from a bounded window with a staleness cap. Inference no longer blocks on training. | Yes (sample/consume split in the GRPO loop). | No. |

---

## Why this milestone now

Stage 1 Cut C proved two things that make Stage 2 tractable:

1. **GRPO tolerates stale weights better than expected.** The step-7 gradient signal arrived cleanly with 6 steps of staleness already accumulated on the pool. The group-advantage normalization absorbs the importance-ratio shift; `rollout_corr/ppl_ratio` stayed near 1.6 throughout.
2. **Cross-machine HTTP is not the bottleneck.** Per-step throughput is within ±10% of a colocated pool. Adding a weight-sync round-trip every 5–10 steps costs a few seconds of downtime, not a throughput collapse.

So the plan is: publish weights every K steps (Cut A) and run the rollout pool continuously (Cut B) to amortize the pool's idle time across training steps.

---

## Cut A — weight publish

### Design sketch (to be pressure-tested in plan mode)

- **Publish protocol:** after `trainer._save_checkpoint()` runs (already triggered by `save_freq`), fan out a `POST /reload_weights` to every endpoint in `external_llm_endpoints`. Payload: either a presigned URL pointing at an HF-format snapshot in `trainer.default_local_dir`, or the raw state_dict shards streamed over HTTP. URL is simpler; streamed shards avoid the shared-disk assumption.
- **Pool-side swap:** `_vllm_child.py` currently holds an `AsyncLLMEngine` over one set of weights. Options, in order of preference:
  1. `engine.collective_rpc('update_weight', ...)` — vLLM 0.18 API, no restart, drops in-flight requests if the weight-shape signature changes. Verified to exist in `vllm>=0.7`; confirm against the image.
  2. Warm-spawn a new `AsyncLLMEngine` on the same GPU with the new weights, atomically rebind the FastAPI router, tear down the old engine after drain. Requires a second `gpu_mem_util` slot ≈ 0.4 each.
  3. Kill + restart the child with `subprocess.Popen`, gated on `/health`. Simplest; costs a full cold-load (~30–60 s on Qwen3-4B).
- **Publish metadata:** every reload increments `policy_version`; trainer stamps rollouts with the version they were generated against (`meta.policy_version` in the ProRL job). Needed for Cut B's staleness filter.
- **Failure mode:** if any endpoint fails to reload, the batch is scored against a mix of policy versions. Fix: block the training step on publish-to-all-endpoints success; on failure, abort the run with a loud marker and require operator intervention.

### Files likely to change (verify before editing)

| Path | Expected edit |
|---|---|
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | New `_publish_policy_version(checkpoint_dir)` called at the end of `_save_checkpoint`; fan-out over `self.config.actor_rollout_ref.rollout.external_llm_endpoints`. |
| `scripts/serving/_vllm_child.py` | Implement `POST /reload_weights` (currently 501 stub, see `stage1.md` §Final flag set). Argument: `{weights_url: str, policy_version: int}` or `{state_dict: bytes}`. Returns 200 + `{policy_version}` on success. |
| `scripts/serving/launch_remote_vllm_pool.sh` | Orchestrator subcommand `publish <checkpoint_dir>` that fans `/reload_weights` to the 4 remote children (useful for manual testing before the trainer-side path lands). |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | Stamp `policy_version` on each job before dispatch to ProRL. |
| New: `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_weightsync.sh` | Sibling of the Stage 1 Cut C inner script with `+publish_on_save=True` and `save_freq=5`. Do not edit the frozen Stage 1 Cut C script. |
| New: `scripts/_internal/s2_weightsync_docker.sh` | Sibling of `s1_remote_docker.sh`. |

### Gates

| # | Gate | Pass |
|---|---|---|
| A1 | ≥ 4 successful `/reload_weights` across 20 steps at `save_freq=5` | Pool children return 200 with monotonically increasing `policy_version` |
| A2 | `rollout_corr/kl` drops on the step following each publish | Visible dip in WandB vs the pre-publish baseline |
| A3 | Zero mixed-version batches | `min(meta.policy_version) == max(meta.policy_version)` per batch |
| A4 | `critic/rewards/mean` trends up (not flat as in Stage 1 Cut C) | Monotonic improvement across the 20 steps, modulo noise |
| A5 | Zero `POST /generate` errors during swap | Pool child logs show a clean handoff; no 5xx spike |

---

## Cut B — replay buffer

### Design sketch

- **Trajectory store:** one JSONL file per completed job, written inside ProRL's eval stage (`openhands/nvidia/async_server.py` already flushes a final-result dict; extend that write). File rotation by count or total bytes. gzip on close. Keyed by `(policy_version, job_id, timestamp)`.
- **Replay sampler:** bounded deque of recent trajectories; the trainer's `fit()` loop in `verl_custom/trainer/ppo/ray_trainer.py` calls `sampler.next_batch()` instead of driving rollouts directly. Freshness rule: reject trajectories older than `max_staleness` policy versions.
- **Inference pump:** a separate coroutine keeps the pool at target QPS by draining the ProRL server's outbound trajectory stream. Decouples "how fast can rollouts be produced" from "how fast can the trainer consume them."
- **Serving-to-training ratio (`serving/w_over_t`):** how many trajectories the pool produces per training step. Target ≥ 1 so the buffer never starves.

### Files likely to change

| Path | Expected edit |
|---|---|
| New: `openhands/nvidia/trajectory_store.py` | Per-episode JSONL writer with rotation + gzip. Module-level `logger = logging.getLogger(__name__)`. |
| `openhands/nvidia/async_server.py` | Call `trajectory_store.write(job_details)` at the end of the eval stage (`JobDetails.event.set()` gates the write). |
| New: `trainer_integration/verl/verl_custom/nvidia/rollout/replay_buffer.py` | Bounded sampler with freshness window; exposes `next_batch(size)`. |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | New `rollout.mode=replay` branch in `fit()` that reads from the buffer instead of calling `openhands_generate_sequences()`. |

### New WandB metrics

| Key | Meaning |
|---|---|
| `replay/buffer_size` | Current trajectory count in the deque |
| `replay/staleness_mean` | Mean `policy_version` lag across the sampled batch |
| `replay/fresh_ratio` | Fraction of sampled trajectories within `max_staleness` |
| `serving/w_over_t` | Rollouts produced per training step (target ≥ 1) |

### Gates

| # | Gate | Pass |
|---|---|---|
| B1 | 20 training steps complete with `rollout.mode=replay` | `training/global_step` reaches 20 |
| B2 | `replay/buffer_size` stays ≤ `max_buffer_size` | No unbounded growth; old trajectories evict |
| B3 | `replay/staleness_mean` bounded by `max_staleness` | Freshness rule is enforced, not just logged |
| B4 | `serving/w_over_t ≥ 1` over last 10 steps | Pool outpaces or keeps up with trainer |
| B5 | Cut A gates A1–A5 still hold under replay | Composition works |

---

## Risks and known unknowns

- **vLLM `update_weight` API drift.** Confirm against the image, not the upstream README — 0.18 may differ from the main-branch docs.
- **Partial reload failure leaves endpoints on mixed versions.** Must be detected and surfaced; silent mixed-version batches are a correctness bug.
- **`rollout.n=4` + replay.** GRPO groups the 4 samples-per-prompt into one advantage group. Replayed trajectories must preserve the group grouping; sampling individual trajectories out of a group breaks the advantage normalization.
- **Drain race on swap.** Pool side must finish in-flight `/generate` requests before swapping weights, or responses get stamped with the wrong `policy_version`.
- **`policy_version` assignment on the pool side.** Who owns the version counter — trainer (authoritative) or pool (serving) — must be decided before Cut A lands.
- **Disk / IO for trajectory store.** Under the Stage 1 Cut C workload (16 trajectories × ~1 200 tokens response + tool traces), ~5–20 MiB/step. Rotate aggressively.

---

## Reuse from Stage 1 Cut C

- Trainer topology (8-GPU FSDP + 4 remote pool endpoints) stays as-is; Stage 2's work layers on top.
- `external_llm_endpoints` config is the same; Stage 2 adds `+publish_on_save=True` and `rollout.mode=replay`.
- `launch_remote_vllm_pool.sh` orchestration (bootstrap / start / stop) is reused; `publish` becomes a new subcommand.
- The 5-proof set from Stage 1 (`EXTERNAL BYPASS ACTIVE`, zero Ray vLLM actors, etc.) must keep passing — Stage 2 doesn't touch the bypass path.

---

## Entry points for the next session

1. Read this doc, then `stage1_remote_pool.md` §Architecture for the HTTP-topology primer.
2. Read `scripts/serving/_vllm_child.py` (the 501 stub is there) and `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py:408-425` (the bypass path).
3. Confirm vLLM 0.18 `update_weight` / `collective_rpc` surface against the installed image: `docker run --rm verlai/verl:vllm018.dev1 python3 -c "from vllm import LLM; help(LLM)"`.
4. Enter plan mode and draft Cut A first. Do not start Cut B until Cut A gates A1–A5 are green on a 20-step run.

Operator kickoff: `.claude/commands/continue-weight-sync.md`.
