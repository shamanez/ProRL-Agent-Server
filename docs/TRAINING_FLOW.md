# Training Flow

Step-by-step data path from the SkyRL parquet dataset through all six services to a completed gradient step. Read `CLAUDE.md` for the boundary conditions and startup sequence. Read `docs/TRAINING_OPERATIONS.md` for operational runbook.

```
  train.ready.parquet
       │
       │  ParquetDataLoader (RolloutManager owns this — BC-14)
       ▼
  RolloutManager ──POST /process──► EnvironmentProvider (ProRL :8006 + Singularity)
  (num_parallel_groups × group_size     │  • Singularity sandbox executes the SWE task
   concurrent HTTP calls)               │  • vLLM call per assistant turn (token IDs only)
                                        ▼
                                 InferenceBackend (vLLM :8100-8103 EC2)
                                        │  /vN/generate  (N = policy_version slot)
                                        │  response_token_ids + per-token logprobs
                                        ▼
  RolloutManager ◄──────────── ProRL returns episode result
       │           messages [{role, token_ids, logprobs}], reward, resolved
       │
       │  build_training_sample() — no decode/re-tokenize (BC-1)
       │  is_zero_variance_group() filter
       │
       │  gRPC push_group
       ▼
  LiveStore (UDS /tmp/prorl_live_store.sock)
  bounded FIFO, pop-on-sample, staleness eviction
       │
       │  gRPC get_batch  (server-side blocking)
       ▼
  TrainerAdapter (VERL FSDP, Docker 8×A100)
       │  pack_unpadded_groups() → tensors
       │  compute_advantage (GRPO/DAPO)
       │  actor update (FSDP)
       │  _save_checkpoint()
       │
       │  gRPC publish_policy_version
       ▼
  PolicyRegistry (UDS /tmp/prorl_policy_registry.sock)
       │  fanout: POST /reload_lora to all 4 vLLM children
       │  write manifest (/tmp/prorl_policy_manifest.json)
       ▼
  RolloutManager ← FilePollingPolicySubscription polls manifest at 1 Hz
                   PolicyVersionCache updated → next group uses new version
```

---

## 1. Dataset — `ParquetDataLoader`

**File:** `core/rollout_fabric/rollout_manager/dataloader.py`

The input is `train.ready.parquet` — a subset of SkyRL-v0-293 filtered to tasks with built Singularity images. `ops/data/filter_parquet_to_built_sifs.py` produces this file; it must be regenerated whenever new SIFs are added.

`ParquetDataLoader` is an infinitely cycling iterator:
- Reads all rows via PyArrow; **shuffles within each file** on every pass with a seeded RNG
- Advances files in round-robin when a file is exhausted
- `state_dict / load_state_dict` support for checkpoint resume

Each row: `{prompt, data_source, ability, instance: {instance_id, FAIL_TO_PASS, ...SWE-bench fields}}`. Only the inner `instance` dict is sent to ProRL.

The trainer has no parquet path, no `StatefulDataLoader`, no knowledge of task IDs (BC-14).

---

## 2. RolloutManager — Producer

**Files:** `core/rollout_fabric/rollout_manager/main.py`, `loop.py`, `episode_builder.py`

Runs as `python -m rollout_fabric.rollout_manager.main`. Zero VERL/OpenHands imports (BC-13).

**Wired at startup:**
- `LiveStoreClient` — gRPC, pushes groups
- `PolicyVersionCache` + `FilePollingPolicySubscription` — manifest poll at 1 Hz
- `ParquetDataLoader` — owns the dataset
- `ProRLClient` — plain `httpx` to `:8006/process`
- `RolloutManagerLoop` — the production loop

**`_run_one_group()` — called by each of `num_parallel_groups` worker threads:**

**Step 1 — Read one task row (under `_dl_lock`)**

The lock ensures the same parquet row is not given to two workers simultaneously.

**Step 2 — Take ONE policy snapshot (BC-0)**

```python
snap = self._cache.snapshot()
created_at_step = snap.version
```

Read once before any episode starts. All `group_size` siblings are dispatched with this exact `snap.version` — no trajectory ever spans two policy versions.

**Step 3 — Run `group_size` episodes in parallel**

Each worker spawns `group_size` episode threads, all calling:
```python
ep = self._prorl.run_episode(instance, snap.version)
```

With default `num_parallel_groups=8` and `group_size=4`, this is **32 concurrent HTTP calls** to ProRL, saturating its async worker pool.

**Step 4 — Build `TrainingSample` objects**

`episode_builder.build_training_sample()` extracts from each episode result:
- `prompt_token_ids` — from the first message (role=user/system)
- `response_token_ids` — concatenated across all assistant turns
- `response_loss_mask` — `1` on assistant tokens, `0` on tool/observation tokens
- `behavior_log_probs` — per-token, aligned to `response_token_ids` (zeros on tool positions)
- `reward`, `truncated`, `behavior_policy_version = snap.version`, `created_at_step = snap.version`

Samples with empty `response_token_ids` are dropped (all-zero loss mask crashes the actor update).

**Step 5 — Bind siblings with a shared `group_uid`**

```python
group_samples = build_group(valid_samples, group_uid)
```

All N samples share one `group_uid`. LiveStore and trainer treat the group atomically — the store never splits siblings across two `get_batch` calls (§3.2).

**Step 6 — Zero-variance filter**

```python
if is_zero_variance_group(group_samples):   # all rewards identical → std=0
    return                                   # advantage undefined; no learning signal
```

Archive tee receives the group before this filter (BC-12), so the archive sees everything.

**Step 7 — Push to LiveStore**

```python
self._store.push_group(group_samples)
```

One atomic gRPC call. Calling `push_group` twice for the same group would corrupt `created_at_step` tracking under pop-on-sample — never do it.

---

## 3. EnvironmentProvider — ProRL (`:8006`)

**Frozen files (never edit):** `environments/prorl_openhands/openhands/llm/nvidia/qwen3.py`, `async_server.py`

FastAPI server running the OpenHands SWE-bench agent in a Singularity container. For each `POST /process`:

**vLLM routing by `policy_version`:** The instance carries `policy_version=N`. ProRL's LLM client (`qwen3.py`) routes to `http://vllm_host:810(N%4)/vN/generate` — each episode is **pinned to one vLLM slot** for its entire lifetime. A publish mid-episode does not affect in-flight generation.

**Token-in / token-out (BC-1, FROZEN):** `qwen3.py` sends `prompt_token_ids` (integer list) to vLLM; gets back `response_token_ids` + per-token `logprobs`. These integers flow directly into the response — they are **never decoded to text and re-tokenized**. Re-tokenization shifts token boundaries; actor vs. reference diverges; KL/entropy go NaN within 2 training steps.

**The agent tool loop:** The agent calls tools (file ops, bash) inside a Singularity sandbox. Each tool result becomes a tool-turn message. The conversation alternates: `system/user → assistant → tool → assistant → tool → …`. The full conversation is returned as `messages`.

**Reward:** ProRL evaluates the final state against FAIL_TO_PASS tests and returns `reward` (float), `resolved` (bool), `finish` (bool).

**Response contract:**
```json
{
  "messages": [{"role": "user|assistant|tool", "content": "...", "token_ids": [int, ...], "logprobs": [float, ...]}],
  "reward": 1.0,  "resolved": true,  "finish": true,  "error": null
}
```

---

## 4. InferenceBackend — vLLM Pool (`:8100-8103`, remote EC2)

**Frozen file:** `inference/vllm/scripts/_vllm_child.py`

Four vLLM children on a remote EC2 node:
- Serves `/vN/generate` — N is the policy version slot (enables the per-episode pinning protocol)
- Holds a LoRA LRU cache; `/reload_lora` installs a new adapter tarball atomically
- Returns `409` for a non-monotonic version (install v3 when v5 is active) — treated as hard failure (BC-9)

The vLLM pool has no knowledge of policy version semantics — it just routes by URL. The `policy_version % 4` routing is handled inside ProRL's `qwen3.py`.

---

## 5. LiveStore — Decoupling Buffer

**Files:** `core/rollout_fabric/live_store/store_core.py`, `server.py`, `client.py`

gRPC server on Unix domain socket `/tmp/prorl_live_store.sock`. The only connection between producer and trainer.

**Key parameters:**
- `max_size=256` — maximum groups in the buffer (`deque(maxlen=256)`)
- `staleness_cutoff_k` — a group is stale when `current_step - group.created_at_step > k`
- `no_progress_timeout_s=1800` — if no push occurs for 30 minutes while trainer is waiting → `NoProgressError`

**`push_group` (producer side):**
All samples must share `group_uid` (enforced). Thread-safe via `Condition`; notifies any blocked `get_batch`. When the deque hits `max_size`, the oldest group is silently evicted — natural backpressure from a full buffer.

**`get_batch` (trainer side) — the most critical method:**
```
while len(fresh_groups) < n_groups:
    evict_stale(current_step)      # drop groups where age > staleness_cutoff_k
    if enough_groups: break
    if no_push_for > 1800s: raise NoProgressError
    wait on condition variable
```
- **Server-side blocking:** The gRPC call blocks the thread inside the server until groups arrive. Trainer's timeout is 90 minutes — the no-progress detector (1800s) is the abort path.
- **Pop-on-sample:** Groups are atomically removed from the deque. Never returned twice.
- **Staleness eviction:** Every `get_batch` call evicts groups where `current_step - created_at_step > k` before checking availability.

**Staleness trade-offs:**

| `staleness_cutoff_k` | Effect |
|---|---|
| `k=0` | Fully on-policy — only groups from the current step survive. Impractical without perfect sync. |
| `k=4` | Near on-policy — only groups from the last 4 policy versions survive. Risk of starvation if rollout is slow. |
| `k=1000` | Off-policy-friendly — buffer stays full, no starvation. Current default. |

---

## 6. TrainerAdapter — Consumer

**Files:** `trainers/verl/verl_custom/trainer/ppo/ray_trainer_dapo.py`, `fabric_adapter/live_store_batch.py`, `fabric_adapter/pad.py`

Pure consumer. Connects only to LiveStore and PolicyRegistry (BC-15). Has no parquet path, no ProRL URL, no vLLM URL.

**Each training step:**

**A. Sample from LiveStore**
```python
sampled = sample_mini_batch(
    self.trajectory_store,           # LiveStoreClient
    n_groups=train_batch_size,       # e.g. 4 groups per step
    current_step=self.global_steps,
)
```
`get_batch` blocks server-side until `n_groups` fresh groups are available, then pops them atomically.

**B. Pad to tensors — `pack_unpadded_groups()`**

The wire carries unpadded `tuple[int, ...]`. Padded to `(B, prompt_cap + response_cap)`:

| Tensor | Shape | Notes |
|---|---|---|
| `input_ids` | `(B, P+R)` | prompt left-padded + response |
| `attention_mask` | `(B, P+R)` | 0 on pad positions |
| `position_ids` | `(B, P+R)` | cumsum of attention_mask |
| `loss_mask` | `(B, R)` | 1 on assistant tokens, 0 on tool/pad |
| `rollout_log_probs` | `(B, R)` | behavior policy log probs from vLLM |
| `reward` | `(B,)` | scalar per sample |

Over-sized records raise `RuntimeError` — silent truncation is not allowed.

**C. Reward computation**
`SWEBenchRewardManager` passes through the reward already in the sample. KL penalty applied here if `use_kl_in_reward`.

**D. Advantage computation (GRPO/DAPO)**
```python
compute_advantage(batch, adv_estimator='grpo', num_repeat=group_size)
```
Within each group: `advantage_i = (reward_i - mean(rewards)) / std(rewards)`. Zero-variance groups are filtered upstream for this reason — `std=0` makes all advantages undefined.

**E. Old log-prob recompute**
```python
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
```
Actor (under current weights) recomputes log-probs for collected sequences. The difference `rollout_log_probs` vs `old_log_probs` gives the importance ratio for PPO/GRPO clip.

**F. Actor update**
```python
actor_output = self.actor_rollout_wg.update_actor(batch)
```
FSDP gradient step across 8 A100s. Loss computed only on `loss_mask=1` positions (assistant tokens).

**G. Save + publish**
```python
self._save_checkpoint()
self._publish_lora_adapter(path)     # → gRPC PublishPolicyVersion → PolicyRegistry
```

---

## 7. Policy Version Loop — Closing the Circle

**File:** `core/rollout_fabric/policy_registry/server.py`, `fanout.py`

After the trainer calls `PublishPolicyVersion`:

1. **Fanout** (`fanout.py`): POSTs the adapter tarball to all 4 vLLM children concurrently via `/reload_lora`
2. **Abort gate (BC-9):** If ANY child returns non-200 → `success=False` → trainer raises `PublishFailedError`. Manifest is **not updated**. Worker keeps dispatching under the old version.
3. **On full success:** Writes SQLite record + updates `/tmp/prorl_policy_manifest.json`
4. **Worker picks it up:** `FilePollingPolicySubscription` polls manifest at 1 Hz, detects mtime change, calls `cache.update(snap)`
5. **Next group dispatch:** `cache.snapshot()` returns the new version. `created_at_step = new_version`. The staleness filter uses this to age out older groups.

---

## Special Conditions Summary

| Condition | Where enforced | What breaks if violated |
|---|---|---|
| **BC-0: One snapshot per group** | `loop.py` before episode threads | Siblings see different versions → invalid advantages |
| **BC-1: Token IDs as int on every wire** | `episode_builder.py`, `codec.py`, `qwen3.py` (frozen) | KL/entropy NaN within 2 training steps |
| **BC-9: Hard abort on any failed LoRA reload** | `fanout.py`, `policy_registry/server.py` | Mixed-version vLLM pool → IS weights become lies |
| **Zero-variance filter** | `loop.py` step 6 (pre-push) | `std=0` groups enter the store; advantage undefined |
| **Staleness eviction** | `store_core.py:get_batch` | Old off-policy groups accumulate unchecked |
| **Pop-on-sample** | `store_core.py:get_batch` | Groups consumed once; store is a queue not a replay buffer |
| **No-progress detector** | `store_core.py:get_batch` | 1800s with no push → `NoProgressError` (producer wedged) |
| **BC-16: Trainer starts after ≥1 group** | `ops/services/start_all.sh` | No-progress timer burns during warm-up |
| **Concurrent group workers** | `loop.py:_run` | `num_parallel_groups × group_size` concurrent ProRL calls |
| **`_dl_lock`** | `loop.py:_run_one_group` | Multiple group-workers share one dataloader without racing |
| **Group atomic integrity** | `store_core.py:push_group`, `build_group()` | Store or trainer splits a group across calls |
| **Loss mask on tool tokens** | `episode_builder.py:_extract_token_fields` | Agent learns from tool/observation positions |

---

## Key Numbers (current defaults)

| Parameter | Value | Where set |
|---|---|---|
| `num_parallel_groups` | 8 | `rollout_manager/main.py` env `NUM_PARALLEL_GROUPS` |
| `group_size` | 4 (startup cmd) / 16 (default) | `--group-size` arg |
| Concurrent ProRL calls | `8 × 4 = 32` | `num_parallel_groups × group_size` |
| `max_size` (LiveStore) | 256 groups | `live_store/server.py` |
| `staleness_cutoff_k` | 1000 | `live_store/server.py` (recently raised from 4) |
| `no_progress_timeout_s` | 1800 s | `live_store/server.py` |
| Manifest poll interval | 1 Hz | `policy_subscription.py` |
| `get_batch` timeout | 90 min | `fabric_adapter/live_store_batch.py` |
| vLLM slots | 4 (ports 8100-8103) | `inference/vllm/scripts/` |
