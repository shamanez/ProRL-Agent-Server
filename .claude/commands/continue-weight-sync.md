---
description: Resume the decoupled-rollouts staged implementation at Stage 2 (weight sync + replay buffer).
---

# Continue Decoupled Rollouts — Stage 2 (weight sync + replay buffer)

Fresh session. Stages 0 and 1 are DONE. Stage 1 shipped in three cuts (A local smoke, B local decoupled trainer, C remote HTTP pool). The trainer talks to a remote vLLM pool over HTTP and GRPO trains against intentionally stale weights. Your job this milestone is to close that staleness gap.

## First-action gate — Claude Code plan mode

**Enter Claude Code plan mode as the first action of this session. Do not write or edit code before plan mode is entered and approved.**

A new session that skips plan mode is out of spec. The mechanical sequence is:

1. Read the five docs below (context load only, no edits).
2. Run the environment check block and print the `STATUS` preamble.
3. Ask the user to confirm the plan-mode entry prompt.
4. Enter plan mode. Draft Cut A. `ExitPlanMode` is the gate for the first code edit.
5. Only after `ExitPlanMode` is approved do you touch a file.

If you are running this command and your harness cannot enter Claude Code plan mode natively, stop and tell the user — do not fall through to ad-hoc coding.

## Stage context (read these first, in order)

1. `CLAUDE.md` — repo invariants (token-level, frozen files, linter/pin rules).
2. `plans-n-solutions/README.md` — 0/1/2 status table, stage map, frozen-file list, "How to advance to the next stage".
3. `plans-n-solutions/stages/stage1_remote_pool.md` — **the current working state** (Stage 1 Cut C, formerly "Stage 1.5"). Read §Architecture for the HTTP-topology primer; read §"Rollout time stats" for the throughput baseline Stage 2 must not regress.
4. `plans-n-solutions/stages/stage2_weight_sync_and_replay.md` — **the plan for this session**. Cut A (weight publish) and Cut B (replay buffer) are described with file-level entry points.
5. `plans-n-solutions/stages/stage1.md` §Decoupling proofs — the 5 invariants (plus the 8 Cut-C gates) that must keep passing.

## Environment check

```bash
# Credentials + env vars
source /home/ubuntu/.prorl_creds.env
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/singularity_images
for v in HF_TOKEN WANDB_API_KEY OH_RUNTIME_SINGULARITY_IMAGE_REPO; do
  test -n "${!v}" && echo "$v OK" || echo "$v MISSING"
done

# Docker image + verl checkout
docker image inspect verlai/verl:vllm018.dev1 >/dev/null 2>&1 \
  && echo "docker image OK" || echo "docker image MISSING — pull verlai/verl:vllm018.dev1"
test -d /tmp/verl && echo "/tmp/verl OK" || echo "/tmp/verl MISSING"

# Remote pool health (Stage 1 Cut C infra)
for p in 8100 8101 8102 8103; do
  curl -sS -m 5 -o /dev/null -w "pool :$p = %{http_code}\n" \
    "http://ec2-54-145-77-207.compute-1.amazonaws.com:$p/health"
done
# If any returns a non-200, bring the pool up: bash scripts/serving/launch_remote_vllm_pool.sh start

# Local GPUs idle?
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

## Report status, then ask

Before entering plan mode or drafting any code:

```
DECOUPLED ROLLOUTS — STATUS
============================
Last completed stage: 1 (Cut C — remote HTTP pool, WandB wdqqu52k)
Next milestone:       Stage 2 — weight sync + replay buffer
Driver doc:           plans-n-solutions/stages/stage2_weight_sync_and_replay.md
Cut A (weight publish): not started
Cut B (replay buffer):  not started
Environment:          [OK / issues found]
Remote pool:          [4/4 healthy / needs start / down]
============================
```

Then ask the user **verbatim** (do not paraphrase):

> "Entering Claude Code plan mode. I'll draft Cut A (weight publish) first — confirm the vLLM 0.18 `update_weight` surface against the installed image, then lay out the `_reload_weights` / publisher protocol. Reply 'plan' to proceed, or tell me to adjust scope."

Wait for the user's reply before entering plan mode.

## Execution rules (binding)

1. **Plan mode first.** Do not write code before the plan is approved. `ExitPlanMode` is the gate.
2. **Confirm the vLLM 0.18 weight-update API against the image, not the upstream README.** The `_vllm_child.py` 501 stub is the place to wire it in.
3. **Cut A before Cut B.** Cut B's freshness rule is meaningless without a working publish loop. Cut A's gates (A1–A5) must be green on a 20-step run before starting Cut B.
4. **Don't break Stage 1 (any cut).** The `EXTERNAL BYPASS ACTIVE` path stays intact; the 5 decoupling proofs from Cut B and the 8 gates from Cut C must keep passing.
5. **Sibling launchers, not edits to frozen files.**
   - Inner Hydra: `run_proagent_qwn3_4B_instruct_weightsync.sh` (new sibling of `_remote_decoupled.sh`).
   - Docker launcher: `s2_weightsync_docker.sh` (new sibling of `s1_remote_docker.sh`).
6. **Codex adversarial review at plan gates.** `/codex:adversarial-review --background` for Cut A design; `/codex:review --background` before commit.
7. **Never `--no-verify`. Never `git push` without explicit user approval.** A branch exists only once — do not force-push.
8. **Commit discipline.** Stage-boundary commits only, not mid-plan. Message format matches Stage 1 history (see `git log --oneline stable..HEAD`).

## Known unknowns to resolve during the plan

- vLLM 0.18 `AsyncLLMEngine.collective_rpc('update_weight', ...)` exists? If not, which swap strategy (warm-spawn vs. kill+restart)?
- Who owns the `policy_version` counter — trainer side or pool side?
- `rollout.n=4` group-integrity constraint under replay — can we sample individual trajectories, or must the 4-of-a-kind group be atomic?
- State-dict streaming vs. presigned-URL hand-off — what's already mounted on both trainer and pool box?
- Backpressure: when the pool publishes faster than the trainer can consume, does the replay buffer evict oldest or pause inference?

## Key files for reference

| Path | Role |
|---|---|
| `scripts/serving/_vllm_child.py` | Pool child server; holds the 501 `/reload_weights` stub. |
| `scripts/serving/launch_remote_vllm_pool.sh` | Remote-pool orchestrator; add a `publish` subcommand for manual testing. |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py:408-425` | `EXTERNAL BYPASS ACTIVE` path — do not break. |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` | `_save_checkpoint` is the publish trigger site. |
| `openhands/nvidia/async_server.py` | Eval-stage completion — trajectory-store write goes here. |
| `scripts/_internal/s1_remote_docker.sh` | Stage 1 Cut C launcher — copy-edit this into `s2_weightsync_docker.sh`. |

## Artifacts already in place — do not rebuild

- Docker image `verlai/verl:vllm018.dev1` (pulled).
- verl checkout at `/tmp/verl` (commit `910ba344`).
- Qwen3-4B weights on the trainer host at `~/.cache/huggingface/`.
- Qwen3-4B weights on the remote pool host at `~/vllm-pool/hf-cache/`.
- 49 Singularity `.sif` images at `singularity_images/`.
- SkyRL parquet at `/home/ubuntu/data/SkyRL-v0-293/{train,validation}.filtered.parquet`.
- Remote pool bootstrap already done — `launch_remote_vllm_pool.sh start` brings it up in ≈90 s.
