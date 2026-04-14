# Decoupling Walkthrough — Investigation Prompt

This file contains (1) **how to run** a read-only Claude Code investigation of the trainer ↔ vLLM decoupling, and (2) **the prompt itself** to paste into the session.

At the end, Claude will produce two files inside this repo:

- `docs/decoupling-walkthrough.md`  — human-readable, code-cited walkthrough
- `docs/decoupling-walkthrough.html` — self-contained HTML with syntax-highlighted code blocks

---

## How to execute this prompt

### Step 1 — Go inside the repo

```bash
cd ~/Documents/Decupled-RL/ProRL-Agent-Server
```

### Step 2 — Start Claude Code

```bash
claude
```

(If `claude` isn't on your PATH, run it however you normally do — `npx claude-code`, the desktop app, etc.)

### Step 3 — Confirm you're on Opus with max thinking

The session tooling we just installed assumes Opus + max reasoning effort. Inside the Claude Code prompt, type:

```
/model
```

and pick **Opus 4.6 (1M context) — max effort**. This matches the `model: opus` setting on every agent in `.claude/agents/`.

### Step 4 — Enter plan mode (optional but recommended for read-only safety)

Press **Shift+Tab** until the banner shows `plan mode on`. In plan mode Claude cannot write files *during* the exploration — only the final deliverable step will write `docs/decoupling-walkthrough.{md,html}`. You'll be asked to approve before any writes happen.

Alternative if you want Claude to write the two deliverables without asking: skip plan mode and rely on the read-only constraints inside the prompt.

### Step 5 — Paste the prompt

Copy **everything between the two `===PROMPT===` markers below** and paste it into Claude Code as a single message.

### Step 6 — Claude will work through the investigation

Expect it to:

1. Apply the `repo-architecture` skill (installed in `.claude/skills/repo-architecture/`) to orient.
2. Run `/plan` internally or call the `planner` / `architect` agents to structure the investigation.
3. Launch parallel `Explore` subagents to trace code in `openhands/`, `trainer_integration/verl/`, and the weight-sync glue.
4. Ask approval to write the two output files.

### Step 7 — Open the deliverables

```bash
open docs/decoupling-walkthrough.md
open docs/decoupling-walkthrough.html
```

---

## What to expect in the output

The walkthrough should be organized as an **8-step trace** of a single training iteration on an **8×A100/H100 node**, with the split:

- **FSDP trainer:** all 8 GPUs (sharded params + optimizer state)
- **vLLM rollout:** 4 instances × TP=2 = 8 GPU-slots *colocated* with FSDP on the same Ray actors (not separate GPUs)

Every step must be grounded in a `path/to/file.py:<line>` citation with the exact code snippet quoted.

The final section must explicitly answer the question **"After each GRPO update, how do the vLLM actors end up serving the *new* weights?"** — with concrete code, not hand-waving.

---

## ===PROMPT===

You are a senior systems architect performing a **read-only** code investigation of the `ProRL-Agent-Server` repository you are currently inside. Do not edit, commit, push, or run any command that mutates state outside of writing the two final deliverable files (`docs/decoupling-walkthrough.md` and `docs/decoupling-walkthrough.html`). No `git` writes, no `poetry` installs, no `pytest` runs, no starting servers. Read, grep, and think only.

### Your task

Produce a step-by-step walkthrough of **how the RL trainer decouples from the vLLM inference servers**, with special emphasis on **how updated model weights reach the vLLM actors after every training step**. The audience is a new senior engineer joining the team; they have read `CLAUDE.md` but nothing deeper.

Concretely they want to understand, for a single 8-GPU node:

- **Boot:** what comes up first, in what order, on which GPUs
- **Rollout:** how a prompt becomes token IDs and logprobs via HTTP
- **Training:** what a single GRPO/PPO iteration does
- **Weight handoff:** the non-obvious part — after the optimizer updates the FSDP-sharded weights, how do the vLLM actors start serving the new weights on the next rollout? Is there a `state_dict()` transfer? An NCCL broadcast? A file round-trip? Something else?
- **What would change** if the vLLM servers moved off this node

### How to work

1. **Orient first.** Apply the `repo-architecture` skill at `.claude/skills/repo-architecture/SKILL.md`. Read `CLAUDE.md` at the repo root before anything else.

2. **Plan the investigation.** Either call the `planner` agent (from `.claude/agents/planner.md`) or produce a short internal plan listing the file paths you intend to read and the questions each read must answer. You may use the `architect` agent for the "if vLLM moved off-node" hypothetical.

3. **Explore in parallel.** Launch up to **3 `Explore` subagents in parallel** (single message, multiple tool calls). Suggested splits:
   - Agent A — ProRL Agent Server side: `scripts/start_server.py`, `openhands/nvidia/async_server.py`, `openhands/nvidia/registry.py`, `openhands/server/routes/`, `openhands/llm/nvidia/qwen3.py` and `qwen2_5_vl.py`. Trace the `POST /process` → agent loop → `POST /generate` path. Get exact file:line citations.
   - Agent B — Trainer side: `trainer_integration/verl/verl_custom/trainer/main_ppo.py`, `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` (especially the `fit()` method), `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py`, and the shell script `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh`. Map what one iteration does.
   - Agent C — Weight sync mechanism: grep the whole tree for `enable_sleep_mode`, `wake_up`, `sleep_`, `memory_saver`, `load_model`, `state_dict`, `param_offload`, `ExternalRayDistributedExecutor`, `collective_rpc`, `AsyncActorRolloutRefWorker`. For each hit, return file:line and 5–20 verbatim lines. Specifically test the hypothesis: *"vLLM and FSDP run inside the same Ray worker processes, share the same CUDA context, and so `wake_up()` after a training step reloads weights from the GPU tensors that FSDP just wrote — no serialize/deserialize."* Report whether the code supports this or contradicts it.

4. **Synthesize.** Each claim in the final document MUST be backed by a `path/to/file.py:<line_range>` citation and a short quoted code block. If a function is defined upstream (in the `vllm` or `verl` pip packages), say so clearly and cite the closest in-repo call-site.

5. **Write the deliverables.** Exit plan mode if active, then write exactly two files:
   - `docs/decoupling-walkthrough.md`
   - `docs/decoupling-walkthrough.html`

   Do not write any other files. Do not modify `CLAUDE.md`, `.claude/`, or any code. If `docs/` doesn't exist, create it; otherwise write inside it.

### Structure of `docs/decoupling-walkthrough.md`

Exactly these sections, in order:

1. **TL;DR** — 5–7 bullets. One sentence each. Covers: what's colocated, what's not, how weights get reused, which files are load-bearing.

2. **Hardware layout for this investigation** — diagram-by-text showing the 8-GPU node with FSDP on all 8 GPUs, vLLM instances at TP=2 colocated on the same Ray actors, reference FSDP params also present, and all comms via Ray + local CUDA IPC. Cite the config flags from `run_proagent_qwn3_4B_instruct.sh` (`TP_SIZE`, `NNODES`, `trainer.n_gpus_per_node`, `actor_rollout_ref.rollout.tensor_model_parallel_size`, `actor_rollout_ref.actor.fsdp_config.param_offload`, `+actor_rollout_ref.rollout.enable_memory_saver=True`).

3. **Boot sequence** — numbered steps, each with a file:line citation:
   - Ray cluster init (from `main_ppo.py`)
   - FSDP worker group creation & `init_model()`
   - `AsyncLLMServerManager` instantiation
   - vLLM async engines spawned via `ExternalRayDistributedExecutor`
   - vLLM calls `load_model` via `collective_rpc` **on the same Ray actors that hold the FSDP model**
   - ProRL Agent Server started separately as a FastAPI process (port 8006)
   - `POST /add_llm_server` registers each vLLM endpoint with the ProRL server

4. **One rollout step (end-to-end trace)** — 8–10 sub-steps. For a single prompt flowing through:
   - Trainer calls `async_rollout_manager.wake_up()` before generation (`ray_trainer.py` around line 1080)
   - Trainer issues `generate_sequences(gen_batch)` — for `async_manager=openhands`, this POSTs to `http://localhost:8006/process`
   - ProRL Agent Server enqueues into init → run → eval pipeline (cite `openhands/nvidia/async_server.py`)
   - Agent loop inside `run` stage calls the Qwen3 LLM client (`openhands/llm/nvidia/qwen3.py`)
   - That client issues `httpx.post(f'{base_url}/generate', json={'prompt_ids': input_ids, ...})` — cite exact lines
   - The vLLM FastAPI inside `trainer_integration/.../vllm_async_server.py` returns `response_ids` + `logprobs`
   - Tokens flow back, agent continues loop; state is preserved as `output_ids` to maintain the token-in/token-out invariant (cite the code that avoids re-tokenization)
   - Rollout results return to the trainer as a `DataProto`
   - Trainer calls `async_rollout_manager.sleep()` to free vLLM VRAM before the update

5. **One training step (after rollout)** — cite `ray_trainer.py`:
   - Reward computation (`SWEBenchRewardManager.__call__`) — cite file:line
   - Old log-prob / ref log-prob compute
   - Advantage (GRPO)
   - Actor update via FSDP — `self.actor_rollout_wg.update_actor(batch)` — cite exact line

6. **The weight handoff — the crucial section** — explain clearly, with citations:
   - vLLM was configured with `enable_sleep_mode=True` in `AsyncEngineArgs` (cite `vllm_async_server.py` line)
   - vLLM's `ExternalRayDistributedExecutor` called `load_model` via `collective_rpc` on the **same Ray actors** where FSDP lives (cite `vllm_async_server.py:43–85`)
   - When the trainer calls `sleep()`, vLLM aborts requests, resets prefix cache, then suspends the engine (`vllm_async_server.py:239–247`) — releasing KV cache and allowing FSDP to dominate GPU memory
   - FSDP runs the optimizer step in-place in GPU tensors
   - When the trainer calls `wake_up()` at the top of the next iteration, vLLM re-activates and finds the model tensors **already updated in-place** at the same CUDA addresses — no state_dict transfer, no NCCL broadcast, no file round-trip
   - Explicitly quote the comment at `ray_trainer.py:939–942` that confirms this: "*Sleep and wake up the async rollout manager. This syncs weights to vllm server. Also release GPU memory.*"
   - Note that `param_offload=True` in the FSDP config (`run_proagent_qwn3_4B_instruct.sh:51`) keeps this sharing cheap

7. **What would break if vLLM moved to a different node** — short section, 4–6 bullets. No speculation beyond what the code implies. The answer should center on: `ExternalRayDistributedExecutor` looks up Ray named actors by prefix (`{wg_prefix}WorkerDict_...`) in the local cluster; `wake_up()` has no tensor payload because it assumes shared CUDA. A remote worker would require an explicit weight-materialization path that does not exist today.

8. **Experimental check you can run locally** — a small read-only investigation recipe (grep/read, no state mutation) that a reader can execute to convince themselves of the walkthrough's claims. Examples: "to see that vLLM and FSDP share actors, run `grep -n 'WorkerDict' trainer_integration/verl/verl_custom/nvidia/rollout/vllm_async_server.py`" and show expected output.

9. **File & line index** — flat table of every file:line cited in the document, so the reader can jump around.

### Structure of `docs/decoupling-walkthrough.html`

Same content as the `.md`, rendered as a self-contained HTML file (no CDN assets; inline CSS; monospace font for code; subtle syntax highlighting via plain `<span>` classes is fine; do not pull remote JS). Put a table of contents at the top. Keep it under ~200 KB.

### Constraints (repeat)

- **READ-ONLY**. Do not modify anything except `docs/decoupling-walkthrough.md` and `docs/decoupling-walkthrough.html`.
- **Every claim needs a citation.** If you cannot find code to back a claim, say so explicitly in the document rather than asserting it.
- **Do not invent line numbers.** Re-read the file if you're unsure. Cite a range (`L1040–L1082`) when the snippet spans lines.
- **Use the installed tooling.** `.claude/skills/`, `.claude/agents/`, `.claude/commands/`, and `.claude/rules/` already exist. Prefer the `repo-architecture` skill for orientation and the `planner`/`architect` agents for structuring the investigation.
- **Keep the walkthrough tight.** Target: `.md` between 800 and 1600 lines, `.html` under 200 KB. Prefer code blocks over prose when code already explains itself.

Begin.

## ===PROMPT===

---

## Troubleshooting

- **Claude skipped the `repo-architecture` skill** → Start a fresh session and re-paste; or prepend `Apply the .claude/skills/repo-architecture skill before doing anything else.` to the prompt.
- **Claude started editing code** → You're not in plan mode. Exit, press Shift+Tab to enter plan mode, and paste the prompt again. The READ-ONLY constraints in the prompt will still be respected but plan mode is a safety net.
- **HTML file is bloated with remote JS/CSS** → The prompt forbids remote assets; if Claude slipped, ask it to regenerate the `.html` with inline CSS only.
- **Claude refuses to write outputs** → It probably asked for approval; check the Claude Code UI for a pending plan/edit approval.
