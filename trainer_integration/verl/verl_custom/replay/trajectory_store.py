# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""In-process trajectory replay store for Phase 2 fully-async agentic RL.

The store holds variable-length trajectories as raw token tuples and re-pads
them to a sample-local max at ``sample_mini_batch`` time. Padding is deferred
because individual producer batches have different ``max_len_prompt`` and
``max_len_response`` (padding is batch-local in
``_convert_results_to_dataproto_token``); ``DataProto.concat`` requires
matching dim-1, so pre-padded entries from different producer batches cannot
be concatenated.

See ``plans-n-solutions/stages/full_async.md`` §3 for the record shape and
§4 cut 1 for the scope of this module. The downstream caller (Cut 2) wraps
the :class:`SampledMiniBatch` output in a :class:`verl.DataProto`.
"""

from __future__ import annotations

import random
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch


class InsufficientTrajectoriesError(RuntimeError):
    """Raised when ``sample_mini_batch`` is called with fewer stored groups."""


@dataclass(slots=True, frozen=True)
class TrajectoryRecord:
    """One agentic trajectory, tagged with the version that produced it.

    All token sequences are stored unpadded. The store re-pads at sample
    time to a sample-local maximum so that ``DataProto.concat`` sees
    dim-1-matching tensors across producer batches.
    """

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_loss_mask: tuple[int, ...]
    response_log_probs: tuple[float, ...]
    reward: float
    advantage: float
    behavior_policy_version: int
    created_at_step: int
    prompt_uid: str
    group_uid: str
    resolved: bool
    success: bool
    finish: bool
    is_padded: bool
    error: str | None
    instance: dict[str, Any]
    # Per-row non-tensor fields that reward managers and downstream logic
    # depend on (``data_source``, ``ability``, ``reward_model``, ``extra_info``,
    # ``index``, ...). Captured at push time and re-emitted at sample time so
    # the sampled DataProto is semantically equivalent to the pushed one.
    # Stored as a plain dict — frozenness is for field reassignment, not
    # transitive immutability. Do not mutate after construction.
    prompt_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.response_ids) != len(self.response_loss_mask):
            raise ValueError(
                'response_ids and response_loss_mask must have equal length; '
                f'got {len(self.response_ids)} vs {len(self.response_loss_mask)}'
            )
        if len(self.response_ids) != len(self.response_log_probs):
            raise ValueError(
                'response_ids and response_log_probs must have equal length; '
                f'got {len(self.response_ids)} vs {len(self.response_log_probs)}'
            )


@dataclass(slots=True)
class SampledMiniBatch:
    """Tensor/non-tensor payload returned by :meth:`TrajectoryStore.sample_mini_batch`.

    The caller wraps this in a :class:`verl.DataProto` via ``DataProto.from_dict``.
    Separating the pack step from ``DataProto`` keeps this module free of
    the heavy verl dependency so it can be unit-tested on host.
    """

    tensors: dict[str, torch.Tensor]
    non_tensors: dict[str, np.ndarray]
    meta_info: dict[str, Any] = field(default_factory=dict)


class TrajectoryStore:
    """FIFO bounded replay buffer of *groups* of trajectories.

    One "group" is the set of ``n`` sibling rollouts from the same prompt
    (GRPO groups). The store guarantees groups are never split — all ``n``
    siblings arrive together via :meth:`push_group` and stay together in
    the buffer. This is load-bearing for GRPO / DAPO advantage computation
    which groups by ``uid``.

    Parameters
    ----------
    max_size:
        Maximum number of groups held in the buffer. Older groups are
        evicted FIFO when ``push_group`` exceeds capacity.
    staleness_cutoff_k:
        Hard staleness cap measured in trainer steps. At sample time,
        groups whose ``current_step - created_at_step > staleness_cutoff_k``
        are dropped from the buffer.
    pad_token_id:
        The tokenizer pad id used to right-pad responses and left-pad
        prompts at sample time.
    prompt_length_cap, response_length_cap:
        Defensive hard caps mirroring the rollout-side packer contract:
        ``prompt_length_cap`` = rollout's ``max_starting_message_length``
        (seed-slot width); ``response_length_cap`` = rollout's
        ``total_len = max_prompt_length + max_response_length`` (vLLM
        ``max_model_len``). The rollout packer is expected to produce
        records that already satisfy these bounds. ``_pack`` raises
        :class:`RuntimeError` if a record exceeds either cap — that is a
        contract violation, not a routine truncation. Use ``None`` to
        skip the assertion.
    """

    def __init__(
        self,
        max_size: int,
        staleness_cutoff_k: int,
        pad_token_id: int,
        prompt_length_cap: int | None = None,
        response_length_cap: int | None = None,
    ) -> None:
        if max_size <= 0:
            raise ValueError(f'max_size must be > 0, got {max_size}')
        if staleness_cutoff_k < 0:
            raise ValueError(
                f'staleness_cutoff_k must be >= 0, got {staleness_cutoff_k}'
            )
        self._max_size = max_size
        self._staleness_cutoff_k = staleness_cutoff_k
        self._pad_token_id = pad_token_id
        self._prompt_cap = prompt_length_cap
        self._response_cap = response_length_cap
        self._groups: deque[list[TrajectoryRecord]] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._dropped_by_staleness_total = 0
        # Contract-violation counters — incremented inside ``_pack`` just
        # before the RuntimeError raise, surfaced via ``metrics()`` so
        # operators see a non-zero value in the W&B history if the
        # rollout-side packer ever produces an over-cap record (and the
        # trainer crash that follows). Steady state must be 0.
        self._oversize_prompt_total = 0
        self._oversize_response_total = 0
        self._last_sample_ages: list[int] = []
        # Monotonic count of groups ever appended. Used by the trainer's
        # no-progress detector (replaces the brittle 7200 s hard-cap on
        # ``_acquire_training_batch_dapo`` — see ``wait_until_with_progress``
        # in ``continuous_producer``).
        self._pushes_total = 0

    # ---- ingest -------------------------------------------------------------

    def push_group(self, records: Sequence[TrajectoryRecord]) -> None:
        """Append one group of trajectories.

        The group is appended as a unit; FIFO eviction drops the oldest
        *group* (not individual records) when capacity is exceeded.
        """
        if not records:
            raise ValueError('push_group requires at least one record')
        group = list(records)
        group_uid = group[0].group_uid
        for r in group[1:]:
            if r.group_uid != group_uid:
                raise ValueError(
                    f'all records in a group must share group_uid; '
                    f"got '{group_uid}' and '{r.group_uid}'"
                )
        with self._lock:
            self._groups.append(group)
            self._pushes_total += 1

    def push_from_dataproto(
        self,
        dp: Any,
        *,
        behavior_policy_version: int,
        current_step: int,
    ) -> int:
        """Unpack a verl :class:`DataProto`, bin rows by ``uid``, push each bin.

        The DataProto is expected to match the shape emitted by the
        async-rollout pipeline at the seam where the trainer first has a
        batch of completed rollouts: tensor keys ``input_ids``, ``responses``,
        ``attention_mask``, ``loss_mask``, ``rollout_log_probs``,
        ``is_padded``, ``error_mask`` and non-tensor keys ``uid``,
        ``success``, ``error``, ``resolved``, ``finish``, ``instance``.

        Under Option A the producer pushes every group whole (all ``n``
        siblings of a prompt arrive together), so grouping by ``uid``
        recovers the GRPO/DAPO advantage groups intact. Rows whose ``uid``
        appears only once still form a singleton group; the advantage
        computation will treat them as a degenerate group with mean=0 std=1.

        Returns the number of groups appended.
        """
        tensors = dp.batch
        non_tensors = dp.non_tensor_batch
        if 'uid' not in non_tensors:
            raise KeyError(
                "push_from_dataproto: DataProto is missing the 'uid' "
                'non-tensor; ray_trainer.py stamps it at line 1603 before '
                'union — is the seam placed before that?'
            )
        batch_size = int(tensors['responses'].shape[0])
        prompt_len = int(tensors['input_ids'].shape[1] - tensors['responses'].shape[1])

        input_ids = tensors['input_ids'].cpu()
        responses = tensors['responses'].cpu()
        attention_mask = tensors['attention_mask'].cpu()
        loss_mask = tensors['loss_mask'].cpu()
        rollout_log_probs = tensors['rollout_log_probs'].cpu()
        is_padded = tensors['is_padded'].cpu()
        error_mask = tensors['error_mask'].cpu()

        # Optional reward / advantage — populated when the push happens
        # downstream of compute_reward / compute_advantage. For the Cut 2
        # lockstep seam (push right after union) these keys don't exist yet
        # and the record carries the 0.0 defaults; the trainer recomputes
        # them on the sampled batch.
        if 'token_level_rewards' in tensors:
            per_row_reward = tensors['token_level_rewards'].cpu().sum(dim=-1).tolist()
        elif 'reward' in tensors and tensors['reward'].ndim == 1:
            per_row_reward = tensors['reward'].cpu().tolist()
        else:
            per_row_reward = [0.0] * batch_size
        if 'advantages' in tensors:
            # advantages is broadcast over response_mask → recover scalar as
            # the max magnitude across the response (all positions share the
            # same scalar up to masking).
            adv_tensor = tensors['advantages'].cpu()
            if adv_tensor.ndim == 2:
                abs_t = adv_tensor.abs()
                idx = abs_t.argmax(dim=-1, keepdim=True)
                per_row_advantage = adv_tensor.gather(-1, idx).squeeze(-1).tolist()
            else:
                per_row_advantage = adv_tensor.tolist()
        else:
            per_row_advantage = [0.0] * batch_size

        def _bool_scalar(arr: Any, i: int) -> bool:
            v = arr[i]
            if hasattr(v, 'item'):
                return bool(v.item())
            return bool(v)

        def _opt_str(arr: Any, i: int) -> str | None:
            v = arr[i] if arr is not None else None
            if v is None:
                return None
            s = str(v)
            return s if s else None

        uids = non_tensors['uid']
        success_arr = non_tensors.get('success')
        error_arr = non_tensors.get('error')
        resolved_arr = non_tensors.get('resolved')
        finish_arr = non_tensors.get('finish')
        instance_arr = non_tensors.get('instance')

        # Any non_tensor key the record doesn't already have a typed slot for
        # is preserved verbatim in `prompt_extras`. Reward managers depend on
        # ``data_source``, ``ability``, ``reward_model``, ``extra_info``,
        # ``index`` and similar fields that originate from the prompt-side of
        # the dataloader. Capturing them here keeps the Cut-2 lockstep seam
        # bit-identical for the downstream reward / advantage path.
        known_non_tensors = {
            'uid',
            'success',
            'error',
            'resolved',
            'finish',
            'instance',
        }
        extra_keys = [k for k in non_tensors if k not in known_non_tensors]

        groups: dict[str, list[TrajectoryRecord]] = {}
        for i in range(batch_size):
            uid = str(uids[i])
            full_attn_row = attention_mask[i]
            prompt_attn = full_attn_row[:prompt_len]
            response_attn = full_attn_row[prompt_len:]

            prompt_tokens = input_ids[i, :prompt_len][prompt_attn.bool()].tolist()
            response_valid_len = int(response_attn.sum().item())
            response_tokens = responses[i, :response_valid_len].tolist()
            response_lp = rollout_log_probs[i, :response_valid_len].tolist()
            response_lm = loss_mask[i, :response_valid_len].tolist()

            prompt_extras: dict[str, Any] = {}
            for k in extra_keys:
                v = non_tensors[k][i]
                # Shallow-copy dict values (e.g., ``reward_model``) so the
                # store does not alias mutable state with the caller; the
                # DataProto's dict entries may be reused elsewhere and
                # mutation after push would silently corrupt stored records.
                prompt_extras[k] = dict(v) if isinstance(v, dict) else v

            rec = TrajectoryRecord(
                prompt_ids=tuple(int(x) for x in prompt_tokens),
                response_ids=tuple(int(x) for x in response_tokens),
                response_loss_mask=tuple(int(x) for x in response_lm),
                response_log_probs=tuple(float(x) for x in response_lp),
                reward=float(per_row_reward[i]),
                advantage=float(per_row_advantage[i]),
                behavior_policy_version=behavior_policy_version,
                created_at_step=current_step,
                prompt_uid=uid,
                group_uid=uid,
                resolved=_bool_scalar(resolved_arr, i)
                if resolved_arr is not None
                else False,
                success=_bool_scalar(success_arr, i)
                if success_arr is not None
                else True,
                finish=_bool_scalar(finish_arr, i) if finish_arr is not None else True,
                is_padded=bool(is_padded[i].item()),
                error=_opt_str(error_arr, i) if error_arr is not None else None,
                instance=(
                    dict(instance_arr[i])
                    if instance_arr is not None and isinstance(instance_arr[i], dict)
                    else {}
                ),
                prompt_extras=prompt_extras,
            )
            # Gate error_mask so the record's `error` surfaces transport
            # failures; don't double-stamp if the non_tensor already flagged it.
            if not rec.error and bool(error_mask[i].item()):
                rec = replace(rec, error='error_mask_set')
            groups.setdefault(uid, []).append(rec)

        # Atomicity: Cut 4 runs the producer in a daemon thread while the
        # trainer samples from the main thread. If we called ``push_group``
        # once per uid here, a concurrent ``sample_mini_batch`` could
        # observe a half-pushed DataProto batch (e.g., 3 of 8 groups).
        # Acquire the lock once and append all groups as a unit.
        group_list = [
            records for records in groups.values() if records
        ]  # drop empties defensively
        for records in group_list:
            group_uid = records[0].group_uid
            for r in records[1:]:
                if r.group_uid != group_uid:
                    raise ValueError(
                        f'all records in a group must share group_uid; '
                        f"got '{group_uid}' and '{r.group_uid}'"
                    )
        with self._lock:
            for records in group_list:
                self._groups.append(list(records))
            self._pushes_total += len(group_list)
        return len(group_list)

    # ---- eviction -----------------------------------------------------------

    def evict_stale(self, current_step: int) -> int:
        """Drop groups whose age exceeds ``staleness_cutoff_k``.

        Returns the count of evicted groups. Age is measured from the
        first record's ``created_at_step`` (all records in a group share
        a step under the Option A producer contract).
        """
        with self._lock:
            return self._evict_stale_locked(current_step)

    def _evict_stale_locked(self, current_step: int) -> int:
        cutoff = self._staleness_cutoff_k
        surviving: deque[list[TrajectoryRecord]] = deque(maxlen=self._max_size)
        dropped = 0
        for group in self._groups:
            age = current_step - group[0].created_at_step
            if age > cutoff:
                dropped += 1
            else:
                surviving.append(group)
        self._groups = surviving
        self._dropped_by_staleness_total += dropped
        return dropped

    # ---- sampling -----------------------------------------------------------

    def sample_mini_batch(
        self,
        n_groups: int,
        current_step: int,
        rng: random.Random | None = None,
    ) -> SampledMiniBatch:
        """Draw and REMOVE ``n_groups`` groups from the store.

        Consume-on-sample (queue semantics): each group is produced once
        and consumed once. Drawing removes the group from ``self._groups``
        so subsequent calls cannot re-sample it. Rationale: when producer
        throughput < trainer throughput the buffer can shrink to ~1
        group; with-replacement sampling would then re-train on the same
        batch K+1 times, which is overfitting, not replay. The paper's
        with-replacement result (Fig 18) assumes an effective buffer size
        large relative to the mini-batch — not our SWE-Gym regime.
        ``staleness_cutoff_k`` stays as a safety drop for groups that
        sit unused (e.g., producer pushed a group during a validation
        pause that took > K steps to resume).

        Raises :class:`InsufficientTrajectoriesError` if the store holds
        fewer than ``n_groups`` non-stale groups. The caller is expected
        to poll the buffer until it warms up.
        """
        if n_groups <= 0:
            raise ValueError(f'n_groups must be > 0, got {n_groups}')
        # Fall back to a fresh Random() instance rather than the module-level
        # singleton — concurrent .sample() on the shared singleton from two
        # threads is not guaranteed thread-safe in CPython and has been
        # reported to corrupt Mersenne Twister state under contention.
        rng = rng or random.Random()
        with self._lock:
            self._evict_stale_locked(current_step)
            if len(self._groups) < n_groups:
                raise InsufficientTrajectoriesError(
                    f'store has {len(self._groups)} groups, asked for {n_groups} '
                    f'(buffer may still be warming up)'
                )
            chosen_idx = set(rng.sample(range(len(self._groups)), n_groups))
            groups_list = list(self._groups)
            chosen = [groups_list[i] for i in sorted(chosen_idx)]
            remaining = [g for i, g in enumerate(groups_list) if i not in chosen_idx]
            self._groups.clear()
            self._groups.extend(remaining)
            records: list[TrajectoryRecord] = [r for group in chosen for r in group]
            self._last_sample_ages = [current_step - r.created_at_step for r in records]
        return self._pack(records, current_step)

    def _pack(
        self,
        records: Sequence[TrajectoryRecord],
        current_step: int,
    ) -> SampledMiniBatch:
        batch = len(records)
        # When a cap is configured, pad to the cap (not the per-sample max)
        # so dim-1 is stable across every sample_mini_batch call. This is
        # what lets the caller DataProto.concat samples taken at different
        # trainer steps. When the cap is None, fall back to a local max so
        # tests and small configurations don't pay for unnecessary padding.
        if self._prompt_cap is not None:
            max_prompt = self._prompt_cap
        else:
            max_prompt = max(len(r.prompt_ids) for r in records)
        if self._response_cap is not None:
            max_response = self._response_cap
        else:
            max_response = max(len(r.response_ids) for r in records)
        # Guard against all-empty prompts / responses producing a zero-width
        # tensor (which would blow up downstream attention). Min 1.
        max_prompt = max(max_prompt, 1)
        max_response = max(max_response, 1)

        prompt_ids = torch.full(
            (batch, max_prompt), self._pad_token_id, dtype=torch.long
        )
        prompt_attn = torch.zeros((batch, max_prompt), dtype=torch.long)
        responses = torch.full(
            (batch, max_response), self._pad_token_id, dtype=torch.long
        )
        response_attn = torch.zeros((batch, max_response), dtype=torch.long)
        loss_mask = torch.zeros((batch, max_response), dtype=torch.long)
        rollout_log_probs = torch.zeros((batch, max_response), dtype=torch.float)
        is_padded = torch.zeros(batch, dtype=torch.bool)
        error_mask = torch.zeros(batch, dtype=torch.bool)
        rewards = torch.zeros(batch, dtype=torch.float)
        advantages_scalar = torch.zeros(batch, dtype=torch.float)

        uids: list[str] = []
        successes: list[bool] = []
        errors: list[str | None] = []
        resolveds: list[bool] = []
        finishes: list[bool] = []
        instances: list[dict[str, Any]] = []
        behavior_versions: list[int] = []
        created_steps: list[int] = []
        sample_ages: list[int] = []
        # Rebuild per-column lists for every extra non-tensor key that
        # appeared on any record. Keys that are missing on some records are
        # filled with None so dim-0 stays uniform.
        extra_keys: set[str] = set()
        for rec in records:
            extra_keys.update(rec.prompt_extras.keys())
        extras_by_key: dict[str, list[Any]] = {k: [] for k in extra_keys}

        for i, rec in enumerate(records):
            # Defensive cap assertions: the rollout-side packer must produce
            # records that already fit within the configured caps (see the
            # ``prompt_length_cap`` / ``response_length_cap`` docstring).
            # If a record over-shoots, that is a contract violation —
            # ``data.truncation='error'`` semantics applied to the rollout/
            # replay seam — so we count it and raise rather than silently
            # right-trimming a slice the reward was already scored on.
            if self._prompt_cap is not None and len(rec.prompt_ids) > self._prompt_cap:
                self._oversize_prompt_total += 1
                raise RuntimeError(
                    f'replay/_pack: prompt_ids length {len(rec.prompt_ids)} '
                    f'exceeds cap {self._prompt_cap}; rollout-side packer '
                    f'contract violated. group_uid={rec.group_uid}'
                )
            if (
                self._response_cap is not None
                and len(rec.response_ids) > self._response_cap
            ):
                self._oversize_response_total += 1
                raise RuntimeError(
                    f'replay/_pack: response_ids length '
                    f'{len(rec.response_ids)} exceeds cap '
                    f'{self._response_cap}; vLLM max_model_len contract '
                    f'violated. group_uid={rec.group_uid}'
                )
            # Slice is now a no-op for any record that passed the assertions
            # above. Kept for the ``cap is None`` fallback path used in
            # tests and small configurations.
            p_ids = rec.prompt_ids[-max_prompt:]
            offset = max_prompt - len(p_ids)
            prompt_ids[i, offset : offset + len(p_ids)] = torch.tensor(
                p_ids, dtype=torch.long
            )
            prompt_attn[i, offset : offset + len(p_ids)] = 1

            r_ids = rec.response_ids[:max_response]
            r_lm = rec.response_loss_mask[:max_response]
            r_lp = rec.response_log_probs[:max_response]
            r_len = len(r_ids)
            if r_len > 0:
                responses[i, :r_len] = torch.tensor(r_ids, dtype=torch.long)
                response_attn[i, :r_len] = 1
                loss_mask[i, :r_len] = torch.tensor(r_lm, dtype=torch.long)
                rollout_log_probs[i, :r_len] = torch.tensor(r_lp, dtype=torch.float)

            is_padded[i] = bool(rec.is_padded)
            error_mask[i] = bool(rec.error)
            rewards[i] = float(rec.reward)
            advantages_scalar[i] = float(rec.advantage)
            uids.append(rec.prompt_uid)
            successes.append(rec.success)
            errors.append(rec.error)
            resolveds.append(rec.resolved)
            finishes.append(rec.finish)
            instances.append(rec.instance)
            behavior_versions.append(rec.behavior_policy_version)
            created_steps.append(rec.created_at_step)
            sample_ages.append(current_step - rec.created_at_step)
            for k in extra_keys:
                extras_by_key[k].append(rec.prompt_extras.get(k, None))

        input_ids = torch.cat([prompt_ids, responses], dim=1)
        attention_mask = torch.cat([prompt_attn, response_attn], dim=1)
        position_ids = torch.clip(attention_mask.cumsum(dim=-1) - 1, min=0)

        tensors = {
            'input_ids': input_ids,
            'responses': responses,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
            'rollout_log_probs': rollout_log_probs,
            'is_padded': is_padded,
            'error_mask': error_mask,
            'reward': rewards,
            'advantage': advantages_scalar,
        }
        non_tensors = {
            'uid': np.array(uids, dtype=object),
            'success': np.array(successes, dtype=object),
            'error': np.array(errors, dtype=object),
            'resolved': np.array(resolveds, dtype=object),
            'finish': np.array(finishes, dtype=object),
            'instance': np.array(instances, dtype=object),
        }
        for k, values in extras_by_key.items():
            non_tensors[k] = np.array(values, dtype=object)
        meta_info = {
            'behavior_policy_versions': behavior_versions,
            'created_at_steps': created_steps,
            'sample_ages': sample_ages,
        }
        return SampledMiniBatch(
            tensors=tensors, non_tensors=non_tensors, meta_info=meta_info
        )

    # ---- introspection -----------------------------------------------------

    def num_groups(self) -> int:
        with self._lock:
            return len(self._groups)

    def total_pushes(self) -> int:
        """Monotonic count of groups ever appended.

        Resets on store re-construction, never on sample. Used by the
        trainer's no-progress detector to distinguish "producer is slow"
        from "producer is wedged".
        """
        with self._lock:
            return self._pushes_total

    def num_fresh_groups(self, current_step: int) -> int:
        """Count groups whose age is within ``staleness_cutoff_k``.

        Used by the trainer as the waiter predicate so it does not
        unblock on a group that ``sample_mini_batch`` is about to
        evict as stale — the race that crashed run4 at step 11.
        """
        cutoff = self._staleness_cutoff_k
        with self._lock:
            return sum(
                1 for g in self._groups if current_step - g[0].created_at_step <= cutoff
            )

    def num_trajectories(self) -> int:
        with self._lock:
            return sum(len(g) for g in self._groups)

    def metrics(self, current_step: int) -> dict[str, float]:
        """Return ``replay/*`` WandB metrics for this store.

        The keys follow the Phase 2 stage-doc naming (``full_async.md`` §5).
        """
        with self._lock:
            groups = [list(g) for g in self._groups]
            last_sample_ages = list(self._last_sample_ages)
            dropped_total = self._dropped_by_staleness_total
            oversize_prompt_total = self._oversize_prompt_total
            oversize_response_total = self._oversize_response_total
        base: dict[str, float] = {
            'replay/store_size': float(len(groups)),
            'replay/store_fill_ratio': float(len(groups)) / float(self._max_size),
            'replay/store_num_trajectories': float(sum(len(g) for g in groups)),
            'replay/dropped_by_staleness_total': float(dropped_total),
            'replay/oversize_prompt_total': float(oversize_prompt_total),
            'replay/oversize_response_total': float(oversize_response_total),
        }
        if groups:
            ages = np.array(
                [current_step - g[0].created_at_step for g in groups],
                dtype=np.float64,
            )
            base['replay/store_age_p50'] = float(np.percentile(ages, 50))
            base['replay/store_age_p95'] = float(np.percentile(ages, 95))
        else:
            base['replay/store_age_p50'] = 0.0
            base['replay/store_age_p95'] = 0.0
        if last_sample_ages:
            sa = np.array(last_sample_ages, dtype=np.float64)
            base['replay/sample_age_steps_p50'] = float(np.percentile(sa, 50))
            base['replay/sample_age_steps_p95'] = float(np.percentile(sa, 95))
        else:
            base['replay/sample_age_steps_p50'] = 0.0
            base['replay/sample_age_steps_p95'] = 0.0
        return base

    # ---- test helpers -------------------------------------------------------

    def _snapshot_groups(self) -> list[list[TrajectoryRecord]]:
        """Test-only deep-ish snapshot of the current groups list."""
        with self._lock:
            return [list(g) for g in self._groups]
