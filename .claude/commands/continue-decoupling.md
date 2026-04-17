---
description: Resume the decoupled-rollouts staged implementation from where it left off.
---

# Continue Decoupled Rollouts Implementation

You are continuing a multi-stage project to decouple vLLM inference from the GRPO trainer. The session is fresh — assume no prior context.

## The decoupling milestone (Stage 1)

The "decoupled rollouts" milestone lives in a single stage (`stage1.md`) with **two parts that ship together**:

| Part | What it does | Touches trainer? |
|---|---|---|
| **A — External vLLM standalone** | Hosts vLLM as an out-of-Ray process + supervisor HTTP wrapper. Proves ProRL can route to it. | No. |
| **B — Trainer bypass (stale weights)** | Trainer skips in-Ray vLLM startup, targets Part A's pool, runs 20 GRPO steps. **This is what actually decouples.** | Yes — add `external_llm_endpoints` config + skip `start_llm_servers()`. |

Shipping Part A alone decouples nothing. Shipping Part B alone has no endpoint to talk to. Treat both as one deliverable.

**Launcher reuse (non-negotiable):**
- **ProRL server** = `bash scripts/_internal/s0_prorl.sh` (poetry, unchanged).
- **Part-B trainer** = `bash scripts/_internal/s2_decoupled_docker.sh` — a **new sibling** of `s0_baseline_docker.sh` with the same Docker image, same mounts, same `--network=host`; only the inner Hydra script changes.

## Step 1: Read the plan and current status

Read in order:

1. `plans-n-solutions/README.md` — status table, stage map, gating standard
2. `docs/SETUP.md` — infrastructure, 40 GB config, upgraded stack (verl v0.8 / vLLM 0.18)
3. `plans-n-solutions/stages/stage0.md` — last completed stage; problem log + bootstrap
4. `plans-n-solutions/stages/stage1_playbook.md` — **the playbook covering the full decoupling milestone**. This is the driver document.
5. `plans-n-solutions/stages/stage1.md` — merged decoupling-milestone plan + Part-A smoke criteria + Part-B 20-step gates
6. `docs/decoupling-walkthrough.md` — why Stage 0 is colocated

## Step 2: Check the environment

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
df -h /home
source /home/ubuntu/.prorl_creds.env
export OH_RUNTIME_SINGULARITY_IMAGE_REPO=/home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server/singularity_images
for v in HF_TOKEN WANDB_API_KEY OH_RUNTIME_SINGULARITY_IMAGE_REPO; do
  test -n "${!v}" && echo "$v OK" || echo "$v MISSING"
done

# Upgraded stack sanity
docker image inspect verlai/verl:vllm018.dev1 >/dev/null 2>&1 \
  && echo "docker image OK" || echo "docker image MISSING — pull verlai/verl:vllm018.dev1"
test -d /tmp/verl && echo "/tmp/verl OK" || echo "/tmp/verl MISSING — clone shamanez/verl main there"

# Codex readiness
node /home/ubuntu/.claude/plugins/cache/openai-codex/codex/1.0.3/scripts/codex-companion.mjs setup --json \
  | jq '{codex_ready:.codex.available, logged_in:.auth.loggedIn}'
# both should be true; if logged_in=false run `! codex login`
```

## Step 3: Report status and confirm

Before doing any work, report:

```
DECOUPLED ROLLOUTS - STATUS
============================
Last completed stage: [N] - [name]  ([wandb URL])
Next milestone:       Decoupling (Stage 1, Parts A + B)
Driver doc:           plans-n-solutions/stages/stage1_playbook.md
Per-stage plan:       plans-n-solutions/stages/stage1.md
Environment:          [OK / issues found]
Codex:                [ready / needs login]
============================
```

Then ask: "Ready to start the decoupling milestone (Stage 1, Parts A → B)? Say 'go' to proceed or tell me what to adjust."

## Step 4: Execute the milestone

**Follow the phased workflow in `stage1_playbook.md` §4.** It is the single source of truth for how to drive the decoupling milestone. Key rules:

- **Plan mode first.** Invoke `/plan` and get explicit user approval before any code.
- **Phase-by-phase commits.** Do NOT combine Part A and Part B into one commit; do NOT combine phases within a part.
- **Launcher reuse.** Server launch is `s0_prorl.sh`. Part-B trainer launch is the new sibling `s2_decoupled_docker.sh`. No ad-hoc `docker run` invocations.
- **Codex at every phase gate.** `/codex:adversarial-review --background` before commit. `/codex:review --background` for code-quality pass.
- **Verify assumptions** before writing code (read referenced files in the current v0.8 / vLLM 0.18 stack — v0.4 paths are gone).
- **Document what breaks** in the Solution section of each stage file, same table format `stage0.md` uses.

## Key constraints (upgraded stack — Stage 0 baseline)

- **Trainer runs in Docker:** `verlai/verl:vllm018.dev1` (vLLM 0.18, PyTorch 2.6+).
- **verl source:** `/tmp/verl` holds `shamanez/verl` main (v0.8.0.dev, commit `910ba344`). Installed editable inside the container via `pip install --no-deps -e /opt/verl` (script mounts `/tmp/verl` → `/opt/verl`).
- **verl_custom:** installed via `pip install --no-deps -e /workspace/trainer_integration/verl` inside the container. Custom worker subclass + config monkey-patches live here; upstream `/tmp/verl` stays read-only.
- **ProRL runs on the host**, not in the Docker container. Trainer container joins host networking to reach `http://localhost:8006`.
- **40 GB tuning (A100-40GB × 8):**
  - `gpu_memory_utilization=0.45` (lowered from 0.6 after backward-pass OOM on step 5)
  - `ulysses_sequence_parallel_size=2`
  - `max_prompt_length=16384`, `max_response_length=1536`
  - `save_freq=10`, `total_training_steps=20` for validation runs
  - `+actor_rollout_ref.actor.calculate_entropy=false` (required — custom `compute_log_prob` override relies on this)
- **Env vars inside the container:**
  - `PYTORCH_ALLOC_CONF=expandable_segments:True` — required to avoid activation-memory fragmentation OOM on 40 GB cards
  - `PYTHONPATH=/workspace`
  - `WANDB_API_KEY`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` forwarded from the host
- **GPU layout for Part B:** trainer FSDP on GPUs 0–3, external vLLM on 4–7 (4 supervisors, ports 8100–8103).
- **Never modify:**
  - `scripts/_internal/s0_baseline_docker.sh` (Stage 0 baseline reproduction — frozen; Part B creates a sibling)
  - `scripts/_internal/s0_prorl.sh` (server launcher — reuse as-is)
  - `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh` (baseline Hydra — Part B creates a `_decoupled.sh` sibling)
  - `dev_config/python/**` (linter/formatter/type-checker configs — ask user first)
  - `pyproject.toml` pins (read the comment line before widening anything)
  - `openhands/llm/nvidia/qwen3.py` and `qwen2_5_vl.py` (token-level invariant)
- **Pre-commit autoflake will strip module-level imports it thinks are unused.** If you add an import that's only referenced inside a decorator, late-bound method, or newly-generated string, wrap it in a closure or use an inline import with `# noqa: PLC0415`. Precedent: `verl_custom/workers/fsdp_workers.py` (`_build_compute_log_prob`), `verl_custom/trainer/ppo/ray_trainer.py:1242` (inline `local_mkdir_safe`).
- **Never commit with `--no-verify`.** If a hook fails, fix it.
- **Do not `git push`** without explicit user approval.

## Stage 0 artifacts you can reuse

Already in place — do NOT rebuild:

- 49 Singularity `.sif` images at `singularity_images/`
- Training parquet at `/home/ubuntu/data/SkyRL-v0-293/{train,validation}.filtered.parquet`
- Docker image `verlai/verl:vllm018.dev1` (pulled)
- verl checkout at `/tmp/verl`
- Qwen3-4B weights at `/home/ubuntu/.cache/huggingface/`
- Baseline launchers `scripts/_internal/s0_prorl.sh` and `scripts/_internal/s0_baseline_docker.sh` (reuse server; sibling the trainer)

## If you need to rerun the Stage 0 baseline

Follow `plans-n-solutions/stages/stage0.md` Steps 1–6. The whole chain (clean state → ProRL host launch → Docker trainer launch → monitor → gating metrics → commit) is self-contained there.
