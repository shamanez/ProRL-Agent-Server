# Stage 1 — Cut C: Remote vLLM pool (direct HTTP)

**Status: DONE.** Part A (infra smoke) + Part B (GRPO training against a remote pool) both shipped. Stale-weight tolerance confirmed: gradient signal arrived on the first rewarded step, no trainer/pool coordination issues.

> This was previously labelled "Stage 1.5" during development. It is folded into Stage 1 as **Cut C** of the decoupling milestone. Cuts A and B (local external pool + trainer bypass) are recorded in [`stage1.md`](./stage1.md). All three cuts ship the same decoupling invariant; only the physical topology and the machine that owns the pool GPUs change.

Reuses the Stage 1 Cut B `EXTERNAL BYPASS ACTIVE` path, `_parse_external_endpoint()`, and the `_vllm_child.py` token-level contract verbatim. The only change is physical topology: the pool now lives on a separate EC2 host (`vllm-instance`, public DNS `ec2-54-145-77-207.compute-1.amazonaws.com`, 4 × 23 GiB) and the trainer reclaims all 8 local A100s for FSDP.

---

## What shipped

| Cut | GPUs | What it proved |
|---|---|---|
| Part A | 4 × 23 GiB on EC2 | Token-level `/generate` contract round-trips over public DNS; clean teardown |
| Part B | 8 × A100-40GB local (FSDP, TP=2, SP=2) + 4 remote pool endpoints | GRPO signal propagates end-to-end over HTTP with stale weights |

---

## Where it runs

| Role | Machine | GPUs | Ports |
|---|---|---|---|
| ProRL server | local trainer box | — | `:8006` |
| Trainer | local trainer box | 0–7 (FSDP) | — |
| vLLM pool | `vllm-instance` (EC2) | 0–3 (one `_vllm_child.py` each, TP=1) | `8100–8103` |

Topology: trainer → public EC2 DNS → remote `:810x`. EC2 security-group inbound for `8100–8103` is restricted to the trainer box's public IP.

---

## Files

| Path | Role |
|---|---|
| `scripts/serving/launch_remote_vllm_pool.sh` | Orchestrator. `bootstrap` / `start` / `stop` subcommands. |
| `scripts/serving/_remote_vllm_runner.sh` | Remote-side helper invoked under `nohup`; sources venv, exports env, `exec`s `_vllm_child.py`. |
| `scripts/serving/requirements-remote.txt` | Pins `vllm==0.18.*` + FastAPI stack for the remote venv. |
| `scripts/serving/teardown_remote_vllm_pool.sh` | One-liner wrapper → `launch_remote_vllm_pool.sh stop`. |
| `scripts/_internal/s1_remote_docker.sh` | Part B trainer launcher. Sibling of `s2_decoupled_docker.sh`: `--gpus all`, remote DNS health probe, 300 s health budget. |
| `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_remote_decoupled.sh` | Part B inner Hydra script. `TP=2`, `SP=2`, `n_gpus_per_node=8`, `external_llm_endpoints` = 4 remote URLs. |

No trainer Python code touched beyond what Stage 1 already landed. The `EXTERNAL BYPASS ACTIVE` path in `async_server.py` is topology-agnostic.

---

## Part A gates (infra smoke) — green

| # | Gate | Result |
|---|---|---|
| S1 | `/health` 200 on all 4 ports via public DNS | :8100–:8103 all 200, ~2–5 ms |
| S2 | Token-level `/generate` round-trips | prompt_ids `[9707,11,7299,2138,498,525]` → `response_ids:[264,6584,2975,18189]`, `logprobs:[-1.49,-2.26,-2.72,-0.07]` |
| S3 | Remote GPUs carry load | 4 procs holding 13.8 GiB each at `gpu_mem_util=0.60`; 19 GiB each at 0.85 |
| S4 | Clean teardown | PIDs killed, `VLLM::EngineCore` orphans swept, `/tmp/vllm-child-<port>.log` rsynced back |

---

## Part B gates — green

WandB run `wdqqu52k`, 7 training steps completed, run exited cleanly on epoch boundary (`train.filtered.parquet` has 30 rows × `batch=4` → 7 steps/epoch × `total_epochs=1` → 7 steps; the `total_training_steps=20` cap never tripped). Launcher now sets `total_epochs=3` so the 20-step cap is reachable if someone re-runs.

| # | Gate | Result |
|---|---|---|
| 1 | `EXTERNAL BYPASS ACTIVE` marker present | ✅ 1 hit, 4 external endpoints registered |
| 2 | Zero Ray vLLM actor spawns | ✅ `grep -c 'async_llm_server_[0-9]'` = 0 |
| 3 | All 4 remote GPUs served rollouts | ✅ 19 GiB resident each during the run |
| 4 | All 8 local GPUs held FSDP shards | ✅ `max_memory_allocated_gb=24.7` per rank |
| 5 | Gradient signal propagates | ✅ `actor/grad_norm` finite every step; spike to 9.91 at step 7 (first non-zero reward) |
| 6 | `actor/kl_loss` finite | ✅ 0.0008–0.0027 range |
| 7 | First reward arrives | ✅ step 7: `critic/rewards/mean=0.0625`, `max=1.0` (1/16 rollouts resolved) |
| 8 | No cross-machine data-path errors | ✅ zero connection/timeout/5xx events in `/tmp/s1-remote.log` |

### Rollout time stats (WandB run `wdqqu52k`)

All values from the per-step log line in `/tmp/s1-remote.log`.

| Step | `timing_s/step` | `timing_s/gen` | `timing_s/update_actor` | Tokens | Throughput |
|---|---|---|---|---|---|
| 1 | 112.8 | 68.4 | 25.3 | 97 578 | 108 tok/s |
| 2 | 95.3 | 64.6 | 22.0 | 84 726 | 111 tok/s |
| 3 | 110.0 | 79.2 | 22.2 | 88 486 | 100 tok/s |
| 4 | 187.8 | 156.3 | 22.7 | 108 030 | 72 tok/s |
| 5 | 177.9 | 147.0 | 22.3 | 91 572 | 64 tok/s |
| 6 | 108.9 | 77.0 | 23.2 | 88 108 | 101 tok/s |
| 7 | 201.5 | 169.7 | 22.9 | 145 915 | 90 tok/s (longest prompts; first reward) |

**Observations**

- `timing_s/gen` (rollout wall clock) dominates at 70–90% of step time.
- `timing_s/update_actor` is rock-steady at ~22–25 s — FSDP backward is not the bottleneck.
- Cross-machine HTTP tax is near-invisible: `/health` RTT is 2–5 ms and the per-token throughput at the trainer (64–111 tok/s) tracks what a colocated pool returned in Stage 1 Cut B within ±10%.
- `perf/cpu_memory_used_gb` flat at ~158 GiB; `max_memory_allocated_gb` per FSDP rank ~24.7 GiB. `param_offload=True` + `optimizer_offload=True` keep local GPUs quiet during `gen` — local `nvidia-smi` shows ~1 GiB idle on each of 8 GPUs mid-rollout, confirming rollouts truly live on the remote pool.
- `actor/grad_norm` trajectory: `0.046, 0.066, 0.040, 0.035, 0.023, 0.045, 9.913`. Steps 1–6 have identically-zero rewards (all-negative GRPO groups → zero advantages → near-zero gradient from KL term only). Step 7 is the first group with reward variance, producing a real policy-gradient signal. `filter_groups.enable=False` (inner script:122) keeps those zero-variance steps in the batch, matching prior run `e6496fdt`.

---

## Architecture — hosting vLLM servers over HTTP

This stage is the architectural proof that **trainer and inference can live on independent machines with nothing between them but HTTP**. The cost is minimal and the wins are structural.

**What the trainer sees.** The `external_llm_endpoints` config is the only thing that changes; FSDP, GRPO advantage estimation, reward pipeline, and ProRL job dispatch all run bit-identical to the colocated case. The bypass in `async_server.py:408-425` swaps one list (Ray-spawned local vLLM actors) for another (pre-existing HTTP endpoints) and the downstream code — `start_llm_servers()`, `wake_up()`, `sleep()`, rollout dispatch, DP-rank → endpoint assignment — either no-ops or works unchanged.

**What HTTP buys.**
- **Heterogeneous hardware.** The trainer runs on 8 × A100-40GB; the pool runs on 4 × 23 GiB GPUs of whatever kind the cloud provider has cheapest. Neither side cares about the other's layout.
- **Independent scaling.** Pool concurrency is a CLI flag (`--max-num-seqs`, `--max-num-batched-tokens`). Adding more endpoints adds DP fan-out on the trainer side — the `len(external_llm_endpoints) == rollout_dp_size` contract is the only invariant (inner script §TP_SIZE comment).
- **Independent lifecycle.** Pool can be rebooted, upgraded, or relocated without touching the trainer. The 300 s health probe in `s1_remote_docker.sh` is the only coupling; the trainer blocks on it before step 1 and never again.
- **Cost model.** Pool hardware spins up only while training is active; the trainer box doesn't reserve 4 GPUs for rollouts it's not always using.
- **Security surface is small and explicit.** `_parse_external_endpoint()` rejects non-`http://` schemes and link-local / GCE-metadata hosts. EC2 security-group inbound is pinned to the trainer's public IP. No SSH tunnel, no TLS — adequate for a test topology; Stage 2+ can harden.

**What HTTP doesn't buy (yet).**
- **Weight freshness.** The pool loads Qwen3-4B once at `start` and serves from those weights for the whole run. By step N the pool is N steps behind. This stage accepts staleness as the deliberate tradeoff; Stage 2 closes it.
- **Fault tolerance.** A pool crash kills the run. No retry, no failover endpoint. Adequate for a 20-step smoke; production needs health-aware routing.
- **Throughput at saturation.** Cut C ran with `gpu_mem_util=0.85`, `--enable-prefix-caching`, `--enable-chunked-prefill`, `--max-num-seqs=128`. Prefix caching is the big lever for GRPO's `rollout.n=4` repeated prompts; further gains (continuous batching across batches, speculative decoding) are future work.

**Conclusion.** The decoupled topology is ready to carry weight-sync and replay-buffer work (Stage 2). Nothing about the current plumbing forces the pool to be colocated or even same-AZ — the trainer talks to an opaque HTTP endpoint, and that endpoint can be anywhere that returns `{response_ids, logprobs}` within the ProRL handler timeout.

---

## Problem log

### Part A

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| A1 | `hf download` failed with `PermissionError` on `~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507` | AMI's default cache dir is `root`-owned from a prior sudo install | Pinned `HF_HOME=~/vllm-pool/hf-cache` in bootstrap + runner |
| A2 | After `stop`, remote `nvidia-smi` still showed 4 procs holding ~13.8 GiB | vLLM 0.18 spawns worker subprocesses with argv `VLLM::EngineCore`; they re-parent to init on parent exit and outlive `pkill -f _vllm_child.py` | Added `pkill -f 'VLLM::EngineCore'` sweep to `do_stop` |
| A3 | `huggingface-cli` deprecation warning | CLI name changed in recent `huggingface-hub` | Switched bootstrap to `hf download` |

### Part B

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| B1 | `assert len(seqlen_list) >= k_partitions` in `_balance_batch` at step 1 | Inner script set `+data.gen_batch_size=1` but `world_size=8`, `rollout.n=4` → post-gen batch = 4 items < 8 partitions | Changed `+data.gen_batch_size=$BATCH_SIZE` in `run_proagent_qwn3_4B_instruct_remote_decoupled.sh` so post-gen batch = `BATCH_SIZE × rollout.n = 16 >= 8` |
| B2 | Actor update OOM at `entropy_from_logits` with `ppo_max_token_len_per_gpu=32768` | Softmax allocates `tokens × vocab × 4B = ~5.5 GiB` per forward; 32 K budget exceeded 40 GB shard headroom | Reverted actor-side override to 16 384 (log-prob keys can stay at 32 K — forward-only, no backward) |
| B3 | OOM persisted at 16 K when `calculate_entropy=True` | `entropy_checkpointing=True` only saves activations during backward — the forward-pass softmax still allocates the full `[tokens × vocab]` tensor | Set `+actor_rollout_ref.actor.calculate_entropy=false`; `actor/entropy_loss` now logs 0. Trade-off accepted. Restoring entropy logs requires either smaller per-GPU tokens or a chunked-entropy kernel |
| B4 | Container exited after step 7 despite `total_training_steps=20` | `train.filtered.parquet` has 30 rows; `batch=4` → 7 steps per epoch; `total_epochs=1` → epoch boundary hit before the 20-step cap | Bumped launcher `trainer.total_epochs=3` so 20-step cap is reachable. The 7-step run is still the canonical success record |

---

## Reproduce

Part A (one-time bootstrap is idempotent):

```bash
source /home/ubuntu/.prorl_creds.env
bash scripts/serving/launch_remote_vllm_pool.sh bootstrap   # one-time
bash scripts/serving/launch_remote_vllm_pool.sh start       # 4 children healthy in ~90 s
# verify: curl -sf http://ec2-54-145-77-207.compute-1.amazonaws.com:8100/health
bash scripts/serving/teardown_remote_vllm_pool.sh
```

Part B (three terminals on the trainer box):

```bash
# Terminal 1 — ProRL FastAPI server
bash scripts/_internal/s0_prorl.sh

# Terminal 2 — remote pool
source /home/ubuntu/.prorl_creds.env
bash scripts/serving/launch_remote_vllm_pool.sh start

# Terminal 3 — decoupled trainer (8 local A100s, 4 remote endpoints)
bash scripts/_internal/s1_remote_docker.sh
# log: /tmp/s1-remote.log
```

Teardown:

```bash
docker rm -f s1-remote-decoupled 2>/dev/null
bash scripts/serving/teardown_remote_vllm_pool.sh
```

---

## Next

Stage 2 — weight sync + replay buffer. Plan: [`stage2_weight_sync_and_replay.md`](./stage2_weight_sync_and_replay.md). Fresh-session kickoff: `.claude/commands/continue-weight-sync.md`.
