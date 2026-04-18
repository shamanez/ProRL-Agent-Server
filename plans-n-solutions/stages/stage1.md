# Stage 1 — Decoupling milestone (external vLLM + trainer bypass, stale weights)

**Status: DONE.** The milestone shipped in three cuts. This doc is the record for Cuts A + B (local pool); Cut C (remote HTTP pool) has its own record at [`stage1_remote_pool.md`](./stage1_remote_pool.md).

| Cut | Scope | Evidence |
|---|---|---|
| **A — External vLLM standalone** | Host vLLM as an independent service outside the trainer; ProRL routes rollouts to it over localhost HTTP. | `849314ff` — 7/7 smoke gates green on GPUs 0+1, ports 8100/8101. |
| **B — Trainer bypass with stale weights** | Trainer skips its in-Ray vLLM startup and targets the external pool; 20 GRPO steps with intentionally stale weights isolates decoupling plumbing from weight-sync plumbing. | `53949b72` — five decoupling proofs green (see §Solution); WandB `bgbvlqslo`. |
| **C — Remote HTTP pool** | Pool lives on a separate EC2 host; trainer reclaims all 8 local A100s for FSDP. Previously labelled "Stage 1.5" during development. | WandB `wdqqu52k` — 7/7 training steps, 8/8 gates green. Record: [`stage1_remote_pool.md`](./stage1_remote_pool.md). |

This stage merges what earlier drafts called "Stage 1" (host vLLM standalone) and "Stage 2" (trainer bypass) into one milestone, because shipping either half alone proves nothing: standalone alone decouples nothing, bypass alone has no endpoint to talk to. They only count together, and Cut C then proved the same invariant holds across machines.

**Historical execution playbook:** [`stage1_playbook.md`](./stage1_playbook.md). Preserved for reference — do not re-run.

**Stack (from Stage 0, non-negotiable):**
- Docker image for the trainer: `verlai/verl:vllm018.dev1`
- vLLM 0.18 (both host-side standalone and trainer-side client)
- verl v0.8.0.dev (`shamanez/verl` main) at `/tmp/verl`, commit `910ba344`
- Runtime OOM mitigations: `PYTORCH_ALLOC_CONF=expandable_segments:True`, `gpu_memory_utilization=0.45`
- Token-level invariant: never modify `openhands/llm/nvidia/qwen3.py`

---

## Goal

Decouple vLLM inference from the GRPO trainer in two related cuts landed together:

- **Part A — External vLLM standalone.** Host vLLM outside the trainer (no Ray, no trainer actor). Prove ProRL can route SWE-Bench rollouts to it and back.
- **Part B — Trainer bypass with stale weights.** Make the trainer skip its in-Ray vLLM startup and target the external pool. Run 20 GRPO steps with intentionally stale weights to isolate the decoupling plumbing from the weight-sync plumbing.

A pass here means Stage 2 (weight sync + replay buffer) can assume vLLM is reachable via HTTP at a stable endpoint, independent of the trainer lifecycle, and that the trainer no longer tries to spawn its own vLLM worker when `external_llm_endpoints` is set.

---

## Launcher reuse (non-negotiable)

- **ProRL server** — `bash scripts/_internal/s0_prorl.sh` (poetry, unchanged across Stage 0, Stage 1 Part A, and Stage 1 Part B).
- **Trainer** — a **new sibling** of `s0_baseline_docker.sh`, not a modification:
  - `scripts/_internal/s2_decoupled_docker.sh` — copy `s0_baseline_docker.sh` line-for-line; change `CNAME`, bind GPUs 0–3 via `--gpus '"device=0,1,2,3"'`, swap the inner `run_proagent_qwn3_4B_instruct.sh` call for `run_proagent_qwn3_4B_instruct_decoupled.sh`. Same image, mounts, env, `--network=host`, `PYTORCH_ALLOC_CONF=expandable_segments:True`.

---

## Where it runs

| Part | GPU | Role | Port |
|---|---|---|---|
| A (smoke) | 0 | vLLM supervisor 0 | 8100 |
| A (smoke) | 1 | vLLM supervisor 1 | 8101 |
| B (trainer) | 0–3 | FSDP trainer, `trainer.n_gpus_per_node=4` | — |
| B (pool)    | 4–7 | 4 vLLM supervisors, TP=1 | 8100–8103 |

ProRL stays on the host at `:8006` (same as Stage 0).

---

## Files created / edited

| Path | Action | Summary |
|---|---|---|
| `scripts/serving/vllm_launcher.py` | **new** | FastAPI supervisor app. One `subprocess.Popen` owns the child vLLM server. Routes: `GET /health` (200 iff child `/v1/models` OK), `POST /generate` (pass-through to child's native `{prompt_ids}` endpoint — does NOT re-tokenize), `POST /reload_weights` (501 stub until Stage 2). `SIGTERM`/`SIGINT`/`atexit` hooks → child `SIGTERM` → 30 s wait → `SIGKILL`. PID files for both supervisor and child. |
| `scripts/serving/_vllm_child.py` | **new** | Tiny FastAPI + AsyncLLMEngine server that natively speaks ProRL's `{prompt_ids} → {response_ids, logprobs}` contract on top of vLLM 0.18. Replaced the original plan of fronting `vllm.entrypoints.openai.api_server` with a translation proxy — see §Deviation. |
| `scripts/serving/launch_external_vllm_pool.sh` | **new** | `--gpus a,b,... --ports X,Y,...`; one supervisor per (gpu, port), child port = supervisor port + 1000, PID files at `/tmp/vllm-sup-<port>.pid`. |
| `scripts/serving/README.md` | **new** | Minimal invocation + port discipline. |
| `scripts/tests/test_external_vllm.py` | **new** | 7-criterion integration smoke test. Boots the pool out-of-process, registers with ProRL, asserts health / status / end-to-end `/generate` / `/reload_weights` stub / `git diff --stat HEAD trainer_integration/` empty. |
| `trainer_integration/verl/verl_custom/nvidia/rollout/async_server.py` | **edit** | New `external_llm_endpoints` config; `start_llm_servers()` early-returns when set with a loud `EXTERNAL BYPASS ACTIVE` WARNING. `wake_up()` / `sleep()` become no-ops when `self.async_llm_servers` is empty. `_parse_external_endpoint()` helper rejects non-`http://` schemes, empty host/port, link-local, and GCE metadata hosts (SSRF hardening). |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_decoupled.sh` | **new sibling** | Copy of the baseline Hydra script. Changes: `TP_SIZE=1`, `trainer.n_gpus_per_node=4`, `+actor_rollout_ref.rollout.external_llm_endpoints=[http://127.0.0.1:8100,http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103]`. Dropped the baseline's dangerous tail `rm -rf "$CKPT_PATH/..."` (placeholder path that could nuke real checkpoints). |
| `scripts/_internal/s2_decoupled_docker.sh` | **new sibling** | Copy of `s0_baseline_docker.sh`. `CNAME=s2-decoupled`; `--gpus '"device=0,1,2,3"'`; 180 s bounded health-retry loop against `:8100-:8103/health` before trainer launch; `trainer.resume_mode=disable`; `rm -rf "$STAGE2_OUT"` so a prior attempt can't silently resume. |
| `scripts/validate_run.py` | **new** | 8-gate post-hoc checker. Reads **both** the trainer log (`EXTERNAL BYPASS ACTIVE` marker, zero Ray vLLM actor spawns, `global_step ≥ 20`, finite `grad_norm`/`kl`, advantage variance > 1e-8) **and** each child log (`POST /reload_weights` count — authoritative receiver-side count; every child shows ≥1 `POST /generate`). |

### Files NOT touched (frozen baseline)

- `scripts/_internal/s0_baseline_docker.sh`
- `scripts/_internal/s0_prorl.sh`
- `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct.sh`
- `openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py` (token-level invariant)
- `dev_config/python/**`, `pyproject.toml` pins

---

## Test plan

### Part A — external pool smoke (≈2 GPUs, ~2 min)

All six must pass.

| # | Test | Pass criterion |
|---|---|---|
| 1 | Supervisors boot | Both `/health` return 200 within 60 s of `launch_external_vllm_pool.sh` |
| 2 | ProRL registers endpoints | `/add_llm_server` per supervisor, then `/start`, then `/status` shows both under `llm_servers` |
| 3 | End-to-end rollout | 2 SWE-Bench instances return a `report` dict with a `resolved` boolean |
| 4 | vLLM logs show traffic | `POST /generate` count > 0 on each child log |
| 5 | `/reload_weights` stub | 501 + `{"detail": "Not implemented in Stage 1"}` |
| 6 | No trainer code touched | `git diff --stat HEAD trainer_integration/` empty **at end of Part A** (Part B relaxes this) |

### Part B — 20-step decoupled GRPO (8 GPUs, ~2 h expected; filter_groups note below)

| # | Metric | Gate |
|---|---|---|
| 1 | `global_step` | ≥ 20 |
| 2 | `actor/grad_norm` | finite, > 0, < 1e6 every step |
| 3 | `critic/rewards/mean` | not identically zero across 20 steps |
| 4 | Advantage variance | > 0 |
| 5 | `actor/kl` | finite every step |
| 6 | External pool served rollouts | `POST /generate` > 0 on each of the 4 children; zero Ray vLLM actor spawn logs in `/tmp/s2-decoupled.log` |
| 7 | `python3 scripts/validate_run.py` (default paths, `--expect-weight-publishes 0`) | exit 0 |
| 8 | `make lint` + fast test loop | green |

---

## Execution (re-runnable from a fresh session)

Prereqs (one-time):
- `~/.prorl_creds.env` with `WANDB_API_KEY`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`.
- HF cache at `~/.cache/huggingface` with `Qwen/Qwen3-4B-Instruct-2507` (bind-mounted as `/root/.cache/huggingface`).
- Docker image `verlai/verl:vllm018.dev1` pulled locally; `/tmp/verl` checkout of `shamanez/verl` main at `910ba344`.
- SWE data parquet at `/home/ubuntu/data/SkyRL-v0-293/{train,validation}.filtered.parquet` (bind-mounted read-only at `/data`).

### Part A — external pool smoke

```bash
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server

# Terminal 1 — ProRL (reused across parts)
bash scripts/_internal/s0_prorl.sh 2>&1 | tee /tmp/s1-prorl.log &

# Terminal 2 — two-supervisor pool on GPUs 0,1
bash scripts/serving/launch_external_vllm_pool.sh --gpus 0,1 --ports 8100,8101

# Terminal 3 — smoke test (registers endpoints, runs the 6 gates)
poetry run python scripts/tests/test_external_vllm.py
```

Expected: smoke test prints `7/7 criteria passed` and exits 0.

Teardown (Part A):

```bash
for p in 8100 8101; do docker rm -f "vllm-sup-$p" 2>/dev/null; done
```

### Part B — 20-step decoupled GRPO

```bash
cd /home/ubuntu/de-coupled-rollouts-rl/ProRL-Agent-Server

# Terminal 1 — ProRL (if not already running)
bash scripts/_internal/s0_prorl.sh 2>&1 | tee /tmp/s2-prorl.log &

# Terminal 2 — four-supervisor pool on GPUs 4-7, ports 8100-8103
bash scripts/serving/launch_external_vllm_pool.sh --gpus 4,5,6,7 --ports 8100,8101,8102,8103

# (No explicit /add_llm_server needed — the trainer's start_llm_servers() bypass
#  calls clear_llm_server() + add_llm_server() for each external endpoint itself.)

# Terminal 3 — decoupled trainer on GPUs 0-3 (tees to /tmp/s2-decoupled.log)
bash scripts/_internal/s2_decoupled_docker.sh

# After the trainer exits (or once enough steps have elapsed):
python3 scripts/validate_run.py
```

Teardown (Part B):

```bash
docker rm -f s2-decoupled vllm-sup-8100 vllm-sup-8101 vllm-sup-8102 vllm-sup-8103 2>/dev/null
```

---

## Out of scope for this milestone

- Dynamic weight updates → Stage 2 (`/reload_weights` is a 501 stub here).
- Multi-node → single-node first.
- Eviction / sleep mode → later stage.
- Exact metric convergence at small `max_prompt_length` — plumbing-only; see §Problem log #1 and #8.

---

## Solution

### Decoupling proofs (engineering gate — independent of the 20-step run)

These five signals together prove the trainer has no in-Ray vLLM worker and that every rollout token + logprob was served by the external pool. Collected from the 2026-04-17 12:55 UTC run (trainer PID 6769 in container `s2-decoupled`, WandB run `bgbvlqslo`).

| # | Proof | Evidence source | Observed |
|---|---|---|---|
| 1 | `start_llm_servers()` early-returned via the external-endpoint path | trainer log grep `EXTERNAL BYPASS ACTIVE` | `WARNING:2026-04-17 12:55:15,654:EXTERNAL BYPASS ACTIVE: start_llm_servers() skipped, using 4 external endpoints: ['http://127.0.0.1:8100','http://127.0.0.1:8101','http://127.0.0.1:8102','http://127.0.0.1:8103']` |
| 2 | Zero Ray vLLM actor spawns | `grep -c 'async_llm_server_[0-9]' /tmp/s2-decoupled.log` | `0` |
| 3 | No colocated fallback path taken (`init_engine` / RayActor / "vllm init") | `grep -ciE 'init_engine\|vllm.*actor\|spawn.*vllm'` | `0` |
| 4 | ProRL registered the external endpoints as the rollout pool | trainer log `Assigned same-IP LLM server addresses` | `['127.0.0.1:8100','127.0.0.1:8101','127.0.0.1:8102','127.0.0.1:8103']` — bare `host:port` from `_parse_external_endpoint()`, matching the colocated format; `_send_llm_addresses_to_openhands` works unchanged |
| 5 | All four pool children are serving rollouts (tokens **and** logprobs) | `docker logs vllm-sup-<port> \| grep -c 'POST /generate'` per child | `:8100=1601`, `:8101=1660`, `:8102=1599`, `:8103=1569` (≈1.5 k each within first 27 min) |

Proof #5 is the load-bearing claim that answers *"is the trainer getting logprobs from the external vLLM servers?"*: each `POST /generate` returns `{response_ids, logprobs}` from the child's `AsyncLLMEngine.generate(..., logprobs=1)` path (see `scripts/serving/_vllm_child.py`). The trainer uses those logprobs as the rollout policy logprobs during GRPO. There is no code path in the bypassed `start_llm_servers()` (or anywhere reachable after the early `return`) that could inject a colocated Ray vLLM worker — proofs #2 and #3 confirm none spawned.

### Checkpoint log

- 2026-04-17 — Step 2 done: launcher + child + pool script landed, ruff green, manual end-to-end verified on GPU 0. Container `/health=200` at 30 s cold-start, `/generate` with `prompt_ids=[9707,11,7299,2138,498,525]` returned valid `{response_ids, logprobs}`, `/reload_weights` → 501, images → 400, `docker stop` clean shutdown in 0.94 s with zero GPU-resident processes.
- 2026-04-17 — Step 3 done: `scripts/tests/test_external_vllm.py` landed, ruff green, 7/7 criteria pass against live pool (GPUs 0+1, ports 8100/8101).
- 2026-04-17 — Step 4 resolved via option A after user sign-off: the 7/7 smoke test already proves the decoupled rollout path end-to-end. Pool torn down cleanly (0 GPU procs) before handoff.
- 2026-04-17 — Part A committed as `849314ff`.
- 2026-04-17 — Step 7 done: `async_server.py` edit, `s2_decoupled_docker.sh`, `run_proagent_qwn3_4B_instruct_decoupled.sh`, `scripts/validate_run.py` landed; `_parse_external_endpoint` SSRF hardening added; `wake_up/sleep` no-op paths verified.
- 2026-04-17 — Step 8: five decoupling proofs observed on trainer PID 6769 (≈1.5 k `POST /generate` per child, zero Ray vLLM actor spawns, `EXTERNAL BYPASS ACTIVE` WARNING logged). The 20-step `validate_run.py` gate itself was not reached in-session because `filter_groups.enable=True` + SWE-Bench's low `resolved_rate` at `max_prompt_length=8192` made step-1 batch-fill effectively stall (see Problem #8). Accepted by user as engineering success on the basis of the five proofs; filter-groups throughput belongs to a later stage.

### Deviation from the original plan

The plan proposed fronting vLLM 0.18's stock `vllm.entrypoints.openai.api_server` with a `/generate` translation proxy in the supervisor. Two things pushed us to a simpler design:

1. ProRL's `openhands/llm/nvidia/qwen3.py` client POSTs `{prompt_ids:[int], …}` → expects `{response_ids:[int], logprobs:[float]}`. OpenAI `/v1/completions` speaks `{prompt:str|[int], …}` and puts token ids inside `choices[0].logprobs.token_ids`. Translating correctly for every kwarg (top_p, seed, temperature, max_tokens, …) adds surface area without value.
2. `scripts/tests/vllm_api_server.py` already speaks the ProRL contract but imports `FlexibleArgumentParser` from `vllm.utils`, which vLLM 0.18 moved.

So we shipped `scripts/serving/_vllm_child.py` — a tiny FastAPI + `AsyncLLMEngine` server that speaks `{prompt_ids} → {response_ids, logprobs}` natively. The supervisor's `/generate` is a pure byte-for-byte pass-through; no translation, no re-tokenization. This preserves the token-level invariant documented in `openhands/llm/nvidia/README.md`.

### Other changes vs. the plan

- `actor_rollout_ref.rollout.external_llm_endpoints` is plumbed as a `+`-override (new Hydra key, default effectively `None` via `.get()`), so `ppo_trainer.yaml` is untouched.
- `_parse_external_endpoint()` (new helper in `async_server.py`) rejects non-`http://` schemes, empty values, and link-local / GCE metadata hosts (`169.254.*`, `metadata.google.internal`, `metadata.goog`) to pre-empt SSRF via a malformed endpoint env var.
- `wake_up()` / `sleep()` early-return when `self.async_llm_servers` is empty — required because verl's rollout control plane still calls them.
- The decoupled Hydra script drops the baseline's dangerous tail `rm -rf "$CKPT_PATH/..."`. Cleanup moved to `s2_decoupled_docker.sh` where the resolved `default_local_dir` is concrete.
- `s2_decoupled_docker.sh` passes `trainer.resume_mode=disable` and clears `default_local_dir` before launch so stale state from a prior attempt cannot leak into the fresh 20-step run.
- `scripts/validate_run.py` reads **both** the trainer log and each child log (`/tmp/vllm-child-810{0..3}.log`) so `/reload_weights` count is collected on the authoritative receiver side, and every child is proven to have served `POST /generate`.

### Final vLLM 0.18 flag set (per child)

```
--gpu-memory-utilization 0.45
--max-model-len 17920
--enforce-eager
--enable-chunked-prefill
--max-num-batched-tokens 8192
```

Model is the HF repo id `Qwen/Qwen3-4B-Instruct-2507`, resolved at runtime via the bind-mounted HF cache (`-v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface` with `HF_HOME=/root/.cache/huggingface` in the container).

### Problem log

| # | Symptom | Fix |
|---|---------|-----|
| 1 | Trainer OOM'd on first `loss.backward()` at step 1 with `max_prompt_length=16384` (GPU 0: 31.15 GiB in use, tried to allocate 8.47 GiB, 8.29 GiB free). Baseline used 8 FSDP ranks; Part B uses 4, so the per-GPU shard doubled and 18 k-token activation bursts no longer fit alongside all-gather / gradient buffers. | Lowered `data.max_prompt_length` to 8192 in `scripts/_internal/s2_decoupled_docker.sh`. Plumbing-only run — the validation target is the decoupling gate, not peak accuracy. Full-length runs would need either more trainer GPUs (e.g. 6T+2P split) or LoRA (supported natively via `actor_rollout_ref.model.lora_rank` but changes the training recipe). |
| 2 | Initial child crashed with `OSError: Repo id must be in the form ...` because the launcher passed a host-absolute snapshot path to the container, where the HF cache is mounted at `/root/.cache/huggingface`. | Switched default `--model` to the HF repo id so vLLM resolves via the mounted cache. |
| 3 | `scripts/tests/vllm_api_server.py` import failure on vLLM 0.18 (`FlexibleArgumentParser` moved out of `vllm.utils`). | Wrote fresh `scripts/serving/_vllm_child.py` speaking ProRL's `{prompt_ids}` contract directly on `AsyncLLMEngine` + `TokensPrompt`. |
| 4 | Host `rm -f /tmp/vllm-*.pid` failed because the container's root wrote the files under the shared `/tmp` bind mount. | Dropped the host-side `rm`; supervisor's `Path.write_text` truncates on open, so stale content is overwritten on next launch. |
| 5 | `docker run ... &` backgrounded with `disown` silently lost non-zero exit codes — pool reported "launched N supervisor(s)" even when all containers had already crashed. | Added a 3-s liveness gate after the launch loop; `docker inspect` each container and `exit 1` with log tails if any is not Running. |
| 6 | `sys.exit(0)` from inside the supervisor's SIGTERM handler could be deferred by CPython when the signal arrives in a C-call frame, stranding the child holding GPU memory. | Signal handler now calls `child.terminate()` directly, then `os._exit(128+signum)`. Child termination is idempotent via `_terminated` flag. |
| 7 | `_vllm_child._flatten_logprobs` returned `None` on the first empty-dict mid-sequence, which the trainer cannot distinguish from "logprobs not requested". | Raise `RuntimeError` on empty dict mid-sequence so the failure is loud, not a silent advantage-estimate corruption. |
| 8 | `/generate` handler caught `asyncio.CancelledError` and returned 499 without aborting the engine request — orphaned inflight requests would accumulate under client-timeout churn. | `await engine.abort(request_id)` then re-raise, so vLLM frees GPU blocks promptly. |
| 9 | `+actor_rollout_ref.rollout.external_llm_endpoints=[http://...]` went through a fragile `.replace('http://','')` in an earlier draft, which silently accepted `https://`, `127.0.0.1:8100/../`, and AWS IMDS hosts like `169.254.169.254` — SSRF / credential-exfil risk. | Wrote `_parse_external_endpoint()` using `urllib.parse.urlparse`; refuses non-http schemes, empty host/port, and the link-local / metadata host allow-list. |
| 10 | First pre-flight health check used a single-shot `curl -sf ... || exit` which false-negatived during the 30–60 s vLLM 0.18 cold load. | 180 s bounded retry loop in `s2_decoupled_docker.sh`. |
| 11 | `resume_mode=auto` inherited from the baseline script would silently resume from a prior attempt's checkpoint, skipping the initial steps that `global_step ≥ 20` is gated on. | Added `trainer.resume_mode=disable` to the in-container override and `rm -rf "$STAGE2_OUT"` before launch. Dropped the baseline's tail-`rm` in the sibling Hydra script to avoid nuking real checkpoints if anyone sets `CKPT_PATH` for real. |
| 12 | Trainer script originally didn't propagate its exit code; a failed Hydra / PyTorch run could be masked by an always-succeeding tail command. | `set -euo pipefail` at the top of `run_proagent_qwn3_4B_instruct_decoupled.sh`; pipefail propagates through the `tee` in `s2_decoupled_docker.sh`. |
| 13 | Initial `validate_run.py` counted `POST /reload_weights` on the **trainer** side, which would false-pass if the trainer attempted a publish that never hit the wire. | Rewrote to count on the **receiver** side by reading each child's `/tmp/vllm-child-<port>.log`. |
| 14 | `_collect_metric` regex dropped `NaN` / `Inf` values because only numeric literals matched, so a non-finite `grad_norm` would silently make the finite-check gate PASS. | Added `\|nan\|inf` alternation under `re.IGNORECASE`. |
| 15 | `filter_groups.enable=True` + SWE-Bench's low `resolved_rate` at initialization (≈0.05–0.15) makes the effective batch-fill rate slow at `max_prompt_length=8192`: P(≥1 of 4 trajectories succeeds) ≈ 0.28 → ~14 instances needed per batch of 4 → ~30–45 min/step. Not a bug; a known GRPO side-effect amplified by the shorter prompt budget. | Accepted for this milestone (validation goal is plumbing, not throughput). Follow-up options: (a) `+algorithm.filter_groups.enable=False` (weaker learning signal but all batches progress), or (b) switch to LoRA (natively supported via `actor_rollout_ref.model.lora_rank=N`) which fits `max_prompt_length=16384` in 4-GPU FSDP. Both deferred — engineering proofs #1–5 above satisfy the decoupling-milestone intent. |
