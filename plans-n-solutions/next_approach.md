# What comes next after Phase 1

Phase 1 (LoRA weight-sync) landed on `decoup-weight-sync`. Run `w9nj4akn`, 20 steps, 6/6 gates green. The decoupled topology now closes the loop: trainer publishes rank-16 LoRA adapters to the remote vLLM pool after every `save_freq` steps, `rollout_corr/ppl_ratio` stays bounded instead of drifting.

This doc frames the next two phases and lists the small residual polish from Phase 1.

---

## Phase 2 — full state-dict publish

**Why.** LoRA rank-16 caps expressiveness. When reward plateaus before loss plateaus, or when KL regularization fights the adapter's low-rank subspace, Phase 1 is the bottleneck. Phase 2 swaps the payload — same publish protocol, same trainer-authoritative `policy_version` ownership, same partial-failure abort contract — but ships the full HF checkpoint (~8 GiB bf16 for Qwen3-4B).

**Shape of the change.**

| Layer | Phase 1 | Phase 2 |
|---|---|---|
| Payload | `adapter_model.safetensors` + `adapter_config.json`, ~122 MiB gzipped | full HF shards, ~8 GiB bf16 (Qwen3-4B) |
| Transport | HTTP multipart body in one POST | presigned S3 URL in POST body, pool downloads → loads |
| Pool-side swap | `engine.add_lora(LoRARequest)` → drain → `remove_lora(prior)` | `collective_rpc('update_weight', ...)` or AsyncLLMEngine equivalent against CPU-loaded shards; drain mechanism reuses Phase 1's `_inflight_cond` |
| `max-loras` slot | Needed (old+new coexist) | Not needed — full-model update replaces in place |

**Open questions worth attacking before writing code.**

1. Does `AsyncLLMEngine` in vLLM 0.18 expose a public `update_weight` path that doesn't require spinning up a separate `WorkerWrapperBase`? If not, we need `collective_rpc` on the Ray-less child path (the pool runs bare FastAPI, not Ray).
2. How long is the pool unavailable during the swap? Phase 1's `add_lora` is ~11–14 s and *does not* block `/generate`. Full-weight update will block — need to decide whether the trainer tolerates that wait synchronously or if a second hot-spare pool member takes rollouts during the swap.
3. S3 presigned URLs introduce SSRF surface on the pool child. Security review required before merging (`security-reviewer` agent).
4. Transfer-latency budget scales ~60× over Phase 1 (8 GiB vs 122 MiB at same ~60 MB/s). At `save_freq=1` that's minutes; may dictate a minimum `save_freq` or a chunked / resumable transfer.

**Reuses from Phase 1 (no re-design).** `policy_version` ownership, the 8 `weight_sync/*` WandB keys + `rollout/staleness_steps`, partial-failure abort, structured JSON log line, `_swap_lock` + `_inflight_cond` discipline.

---

## Phase 3 — replay buffer / truly async RL

**Why.** Phase 1 publishes in lock-step with training: step N ends, publish, step N+1 starts. Publish overhead scales with publish frequency. For future multi-node / multi-pool topologies (decentralized rollouts), the trainer and rollout clocks need to decouple entirely — rollouts run continuously, trainer draws from a bounded trajectory store.

**Pieces that already exist** (from Phase 1, nothing to redo):

- `policy_version` stamp on every rollout message (`async_server.py:1495`).
- `rollout_corr/ppl_ratio` per-step importance ratio, already logged.
- `rollout/staleness_steps` metric — the denominator for staleness-budget filtering.

**Pieces that don't exist and need design work.**

1. **Bounded trajectory store.** FIFO, or priority-by-TD-error. Sizing driven by staleness budget (keep trajectories ≤ K versions stale, drop older).
2. **Staleness-budgeted sampler.** Draw by priority, apply importance weight `w_i = π_θ / π_behavior^{pv_i}` with clipping (`tis_imp_ratio_cap` already exists per-step in Phase 1 — Phase 3 generalizes it across time).
3. **Off-policy correction math.** V-trace / IMPALA, or clipped IS with a hard staleness cutoff. The choice affects whether we need the value head to evaluate off-policy trajectories, which cascades to how the reference model is used.
4. **Async rollout worker lifecycle.** Today `generate_sequences` is called once per step by the trainer's Ray controller. Phase 3 needs rollouts running independently and emitting into the store; the trainer drives `update_actor` on its own clock.

**Prerequisite signal Phase 3 must preserve.** The token-level `{prompt_ids, response_ids, logprobs}` contract (`openhands/llm/nvidia/qwen3.py`) stays invariant. A replay buffer that decodes/re-encodes text across steps would re-shift token boundaries and reintroduce the same off-policy bug staleness correction is supposed to fix.

---

## Phase 1 residual polish (non-blocking)

- `_vllm_child.py` emits one benign `reload_lora prior adapter dir cleanup failed` WARNING per swap after `pv=1` — the PosixPath was already removed by the previous swap's `shutil.rmtree`. Separate log line from the ok event; doesn't move gates. Fix: only attempt cleanup on the *last-but-one* adapter dir, or `ignore_errors=True`.
- Scan `add_lora` GPU time vs (rank, hidden_dim, num_layers) so Phase 2 planning has an empirical curve rather than an extrapolation.
- Wire test: assert `policy_version` on rollout messages matches the pool's `/health` reply at dispatch time (cross-validation rather than sender-side assertion only).

---

## Constraints that carry forward

- Topology stays fixed: trainer FSDP local (8×A100-40GB Docker), vLLM pool on EC2 (4 children, ports 8100–8103), ProRL on host `:8006`. Phase 2+3 ride on top.
- Frozen files: `scripts/_internal/s1_remote_docker.sh`, `trainer_integration/verl/verl_custom/nvidia/scripts/run_proagent_qwn3_4B_instruct_remote_decoupled.sh`. New siblings only.
- Trainer-authoritative `policy_version`. Pool only echoes.
- Partial publish failure = `RuntimeError` = abort. Mixed-version batches are a correctness bug, not a warning.
- Token-in/token-out contract (`openhands/llm/nvidia/qwen3.py`, `qwen2_5_vl.py`) is invariant.
- No `--no-verify`, no `git push` without explicit user approval, stage-boundary commits only.

---

## Pointers for whoever picks this up next

| Path | Why |
|---|---|
| [`stages/weight_sync_lora.md`](./stages/weight_sync_lora.md) | Phase 1 design — protocol, failure modes, success gates. Phase 2 reuses all of this. |
| [`stages/progress_stage.md`](./stages/progress_stage.md) | Phase 1 as-built — what each file does, what happens inside a swap, how staleness is accounted. |
| [`stages/timing_decoupled_4B.md`](./stages/timing_decoupled_4B.md) | First empirical numbers. Phase 2 budget negotiations start here. |
| [`stages/baseline.md`](./stages/baseline.md) | Pre-weight-sync baseline; the "before" picture for importance-ratio drift. |
| `../CLAUDE.md` | Repo invariants, frozen files, credential pin. |
| `scripts/serving/_vllm_child.py` | Pool-side reload implementation — the drain/swap protocol Phase 2 extends. |
| `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py:1266+` | `_publish_lora_adapter` — the publish fan-out Phase 2 generalizes. |
