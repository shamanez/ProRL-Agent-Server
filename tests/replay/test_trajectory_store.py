# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Phase 2 trajectory replay store.

These tests are pure-Python / torch / numpy — no verl, no vLLM. The store
intentionally emits a :class:`SampledMiniBatch` (plain tensor / numpy
dicts) so that the packing logic can be exercised outside the verl
container. The Cut 2 integration that wraps the output in a
``DataProto`` is covered by ``tests/trainer/test_trainer_buffer_integration.py``.
"""

from __future__ import annotations

import random
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

# Put the verl patch-package root on sys.path so we can import
# verl_custom.replay.* without the heavyweight verl install.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / 'trainer_integration' / 'verl')
)

from verl_custom.replay.trajectory_store import (  # noqa: E402
    InsufficientTrajectoriesError,
    SampledMiniBatch,
    TrajectoryRecord,
    TrajectoryStore,
)

PAD_ID = 0


# --- helpers ----------------------------------------------------------------


def _record(
    *,
    seed: int = 0,
    prompt_ids: tuple[int, ...] | None = None,
    response_ids: tuple[int, ...] | None = None,
    loss_mask: tuple[int, ...] | None = None,
    log_probs: tuple[float, ...] | None = None,
    reward: float = 0.0,
    advantage: float = 0.0,
    behavior_policy_version: int = 0,
    created_at_step: int = 0,
    prompt_uid: str | None = None,
    group_uid: str | None = None,
    resolved: bool = False,
    success: bool = True,
    finish: bool = True,
    is_padded: bool = False,
    error: str | None = None,
    instance: dict | None = None,
) -> TrajectoryRecord:
    prompt_ids = prompt_ids if prompt_ids is not None else (seed + 1, seed + 2)
    response_ids = (
        response_ids if response_ids is not None else (seed + 10, seed + 11, seed + 12)
    )
    loss_mask = loss_mask if loss_mask is not None else (1,) * len(response_ids)
    log_probs = log_probs if log_probs is not None else (0.0,) * len(response_ids)
    prompt_uid = prompt_uid or f'puid-{seed}'
    group_uid = group_uid or f'guid-{seed}'
    instance = instance or {'instance_id': f'inst-{seed}'}
    return TrajectoryRecord(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_loss_mask=loss_mask,
        response_log_probs=log_probs,
        reward=reward,
        advantage=advantage,
        behavior_policy_version=behavior_policy_version,
        created_at_step=created_at_step,
        prompt_uid=prompt_uid,
        group_uid=group_uid,
        resolved=resolved,
        success=success,
        finish=finish,
        is_padded=is_padded,
        error=error,
        instance=instance,
    )


# Golden: 3 fixed trajectories that must survive the push/sample round-trip
# with bit-identical token IDs, loss masks, and log probs. This is the
# Phase 2 equivalent of handoff.md §12.5 — the token-in/token-out invariant
# is load-bearing for multi-turn KL / PPO stability (CLAUDE.md "Token-in /
# token-out LLM path (INVARIANT)").
_GOLDEN_SPEC = [
    {
        'prompt': (101, 102, 103, 104),
        'response': (201, 202, 203),
        'loss_mask': (1, 1, 0),
        'log_probs': (-0.11, -0.22, -0.33),
        'prompt_uid': 'g-a',
    },
    {
        'prompt': (105, 106),
        'response': (204, 205),
        'loss_mask': (1, 1),
        'log_probs': (-0.44, -0.55),
        'prompt_uid': 'g-b',
    },
    {
        'prompt': (107, 108, 109),
        'response': (206, 207, 208, 209, 210),
        'loss_mask': (1, 1, 1, 1, 0),
        'log_probs': (-0.66, -0.77, -0.88, -0.99, -1.10),
        'prompt_uid': 'g-c',
    },
]


def _golden_group(created_at_step: int = 0) -> list[TrajectoryRecord]:
    return [
        TrajectoryRecord(
            prompt_ids=spec['prompt'],
            response_ids=spec['response'],
            response_loss_mask=spec['loss_mask'],
            response_log_probs=spec['log_probs'],
            reward=0.0,
            advantage=0.0,
            behavior_policy_version=1,
            created_at_step=created_at_step,
            prompt_uid=spec['prompt_uid'],
            group_uid='grp-golden',
            resolved=False,
            success=True,
            finish=True,
            is_padded=False,
            error=None,
            instance={'instance_id': spec['prompt_uid']},
        )
        for spec in _GOLDEN_SPEC
    ]


def _unpad_prompt(row_prompt: torch.Tensor, row_prompt_attn: torch.Tensor) -> list[int]:
    """Strip left-padding from one prompt row using the attention mask."""
    return row_prompt[row_prompt_attn == 1].tolist()


def _unpad_response(
    row_response: torch.Tensor, row_resp_attn: torch.Tensor
) -> list[int]:
    """Strip right-padding from one response row."""
    return row_response[row_resp_attn == 1].tolist()


# --- TrajectoryRecord contract ----------------------------------------------


def test_record_rejects_length_mismatch_response_vs_loss_mask():
    with pytest.raises(ValueError, match='response_loss_mask'):
        TrajectoryRecord(
            prompt_ids=(1,),
            response_ids=(2, 3),
            response_loss_mask=(1,),  # wrong length
            response_log_probs=(0.0, 0.0),
            reward=0.0,
            advantage=0.0,
            behavior_policy_version=0,
            created_at_step=0,
            prompt_uid='x',
            group_uid='x',
            resolved=False,
            success=True,
            finish=True,
            is_padded=False,
            error=None,
            instance={},
        )


def test_record_rejects_length_mismatch_response_vs_log_probs():
    with pytest.raises(ValueError, match='response_log_probs'):
        TrajectoryRecord(
            prompt_ids=(1,),
            response_ids=(2, 3),
            response_loss_mask=(1, 1),
            response_log_probs=(0.0,),  # wrong length
            reward=0.0,
            advantage=0.0,
            behavior_policy_version=0,
            created_at_step=0,
            prompt_uid='x',
            group_uid='x',
            resolved=False,
            success=True,
            finish=True,
            is_padded=False,
            error=None,
            instance={},
        )


# --- FIFO eviction at capacity ----------------------------------------------


def test_push_evicts_fifo_at_capacity():
    """deque(maxlen=N) drops the oldest group when capacity is exceeded."""
    store = TrajectoryStore(max_size=2, staleness_cutoff_k=1000, pad_token_id=PAD_ID)
    store.push_group([_record(seed=0, group_uid='g0', created_at_step=0)])
    store.push_group([_record(seed=1, group_uid='g1', created_at_step=1)])
    assert store.num_groups() == 2

    store.push_group([_record(seed=2, group_uid='g2', created_at_step=2)])

    assert store.num_groups() == 2
    surviving_uids = {group[0].group_uid for group in store._snapshot_groups()}
    assert surviving_uids == {'g1', 'g2'}  # g0 was the oldest and got dropped


# --- push_group validation ---------------------------------------------------


def test_push_group_rejects_empty():
    store = TrajectoryStore(max_size=4, staleness_cutoff_k=5, pad_token_id=PAD_ID)
    with pytest.raises(ValueError, match='at least one record'):
        store.push_group([])


def test_push_group_rejects_mixed_group_uids():
    store = TrajectoryStore(max_size=4, staleness_cutoff_k=5, pad_token_id=PAD_ID)
    with pytest.raises(ValueError, match='share group_uid'):
        store.push_group(
            [
                _record(seed=0, group_uid='grp-a'),
                _record(seed=1, group_uid='grp-b'),
            ]
        )


# --- sampling without replacement within a call -----------------------------


def test_sample_without_replacement_within_call():
    store = TrajectoryStore(max_size=16, staleness_cutoff_k=1000, pad_token_id=PAD_ID)
    for i in range(4):
        store.push_group(
            [_record(seed=i, group_uid=f'g{i}', prompt_uid=f'g{i}', created_at_step=0)]
        )

    rng = random.Random(42)
    mb = store.sample_mini_batch(n_groups=4, current_step=0, rng=rng)

    uids = list(mb.non_tensors['uid'])
    assert len(uids) == 4
    assert len(set(uids)) == 4


def test_sample_rejects_too_few_groups():
    store = TrajectoryStore(max_size=8, staleness_cutoff_k=1000, pad_token_id=PAD_ID)
    store.push_group([_record(seed=0, group_uid='g0')])
    with pytest.raises(InsufficientTrajectoriesError):
        store.sample_mini_batch(n_groups=4, current_step=0, rng=random.Random(0))


# --- staleness cutoff --------------------------------------------------------


def test_staleness_cutoff_drops_old_groups_on_evict():
    store = TrajectoryStore(max_size=16, staleness_cutoff_k=2, pad_token_id=PAD_ID)
    store.push_group([_record(seed=0, group_uid='old', created_at_step=0)])
    store.push_group([_record(seed=1, group_uid='fresh', created_at_step=4)])

    dropped = store.evict_stale(current_step=5)

    # 'old' age = 5 > K=2 → evicted. 'fresh' age = 1 ≤ K=2 → kept.
    assert dropped == 1
    remaining = {g[0].group_uid for g in store._snapshot_groups()}
    assert remaining == {'fresh'}


def test_staleness_cutoff_applies_at_sample_time():
    """sample_mini_batch evicts stale groups before counting availability."""
    store = TrajectoryStore(max_size=16, staleness_cutoff_k=2, pad_token_id=PAD_ID)
    store.push_group([_record(seed=0, group_uid='stale', created_at_step=0)])
    with pytest.raises(InsufficientTrajectoriesError):
        store.sample_mini_batch(n_groups=1, current_step=10, rng=random.Random(0))
    assert store.num_groups() == 0  # got evicted as a side effect


# --- SampledMiniBatch tensor shape ------------------------------------------


def test_sample_emits_consistent_tensor_shapes():
    store = TrajectoryStore(max_size=8, staleness_cutoff_k=1000, pad_token_id=PAD_ID)
    store.push_group(_golden_group())
    mb = store.sample_mini_batch(n_groups=1, current_step=0, rng=random.Random(0))
    assert isinstance(mb, SampledMiniBatch)
    t = mb.tensors
    batch = 3  # golden group has 3 trajectories
    assert t['input_ids'].shape[0] == batch
    assert t['responses'].shape[0] == batch

    # Response-side tensors share dim-1.
    resp_len = t['responses'].shape[1]
    assert t['loss_mask'].shape[1] == resp_len
    assert t['rollout_log_probs'].shape[1] == resp_len

    # Full-sequence tensors share dim-1.
    full_len = t['input_ids'].shape[1]
    assert t['attention_mask'].shape[1] == full_len
    assert t['position_ids'].shape[1] == full_len

    # Position IDs = clamp(cumsum(attn) - 1, 0)
    expected = torch.clip(t['attention_mask'].cumsum(dim=-1) - 1, min=0)
    assert torch.equal(t['position_ids'], expected)

    # Non-tensors match dim-0.
    assert mb.non_tensors['uid'].shape == (batch,)
    assert mb.non_tensors['resolved'].shape == (batch,)

    # meta_info tracks sample age per trajectory.
    assert mb.meta_info['sample_ages'] == [0, 0, 0]
    assert mb.meta_info['behavior_policy_versions'] == [1, 1, 1]


def test_two_sampled_batches_can_torch_cat_dim0():
    """Two independent sample_mini_batch calls on stores configured with
    the same caps produce tensors with **identical** dim-1, so
    ``torch.cat(..., dim=0)`` — the primitive under ``DataProto.concat``
    at ``/tmp/verl/verl/protocol.py:930`` — succeeds even though the two
    populations carry different raw token lengths. This is the design
    contract that motivates storing raw unpadded tuples and padding to a
    fixed cap at sample time."""
    kwargs = dict(
        max_size=4,
        staleness_cutoff_k=1000,
        pad_token_id=PAD_ID,
        prompt_length_cap=8,
        response_length_cap=8,
    )
    store_a = TrajectoryStore(**kwargs)
    store_b = TrajectoryStore(**kwargs)
    store_a.push_group(
        [
            _record(
                seed=0,
                prompt_ids=(1, 2),
                response_ids=(3,),
                loss_mask=(1,),
                log_probs=(0.1,),
            )
        ]
    )
    store_b.push_group(
        [
            _record(
                seed=1,
                prompt_ids=(1, 2, 3, 4, 5),
                response_ids=(6, 7, 8, 9),
                loss_mask=(1, 1, 1, 1),
                log_probs=(0.1, 0.2, 0.3, 0.4),
            )
        ]
    )
    mb_a = store_a.sample_mini_batch(n_groups=1, current_step=0)
    mb_b = store_b.sample_mini_batch(n_groups=1, current_step=0)

    for key in (
        'input_ids',
        'responses',
        'attention_mask',
        'position_ids',
        'loss_mask',
        'rollout_log_probs',
    ):
        assert mb_a.tensors[key].shape[1] == mb_b.tensors[key].shape[1], (
            f'{key} dim-1 must match across samples: '
            f'{mb_a.tensors[key].shape} vs {mb_b.tensors[key].shape}'
        )
        concatted = torch.cat([mb_a.tensors[key], mb_b.tensors[key]], dim=0)
        assert concatted.shape[0] == 2


def test_num_trajectories_counts_records_across_groups():
    store = TrajectoryStore(max_size=4, staleness_cutoff_k=100, pad_token_id=PAD_ID)
    store.push_group(
        [
            _record(seed=0, group_uid='g', prompt_uid='p0'),
            _record(seed=1, group_uid='g', prompt_uid='p1'),
            _record(seed=2, group_uid='g', prompt_uid='p2'),
        ]
    )
    store.push_group([_record(seed=3, group_uid='h', prompt_uid='p3')])
    assert store.num_groups() == 2
    assert store.num_trajectories() == 4


def test_sample_emits_reward_and_advantage_tensors():
    """Store ≠ opaque blob — reward/advantage must come out as tensors
    so downstream loss code can consume them directly."""
    store = TrajectoryStore(max_size=4, staleness_cutoff_k=100, pad_token_id=PAD_ID)
    store.push_group(
        [
            _record(seed=0, group_uid='g', reward=0.75, advantage=-0.25),
            _record(seed=1, group_uid='g', reward=-0.5, advantage=0.5),
        ]
    )
    mb = store.sample_mini_batch(n_groups=1, current_step=0)
    assert mb.tensors['reward'].dtype == torch.float
    assert mb.tensors['advantage'].dtype == torch.float
    assert mb.tensors['reward'].shape == (2,)
    assert mb.tensors['advantage'].shape == (2,)
    # Exact values round-trip.
    rewards = sorted(mb.tensors['reward'].tolist())
    advantages = sorted(mb.tensors['advantage'].tolist())
    assert rewards == pytest.approx([-0.5, 0.75])
    assert advantages == pytest.approx([-0.25, 0.5])


# --- golden round-trip (CLAUDE.md token-in/token-out INVARIANT) -------------


def test_token_id_preservation_golden():
    """3 fixed trajectories survive push/sample with bit-identical tokens,
    loss masks, and log probs. If this test ever regresses the whole
    pipeline is broken — KL and PPO ratios depend on exact token IDs."""
    store = TrajectoryStore(max_size=4, staleness_cutoff_k=1000, pad_token_id=PAD_ID)
    store.push_group(_golden_group())

    mb = store.sample_mini_batch(n_groups=1, current_step=0, rng=random.Random(0))

    prompt_len = mb.tensors['input_ids'].shape[1] - mb.tensors['responses'].shape[1]
    prompt_part = mb.tensors['input_ids'][:, :prompt_len]
    prompt_attn = mb.tensors['attention_mask'][:, :prompt_len]
    response_attn = mb.tensors['attention_mask'][:, prompt_len:]

    # Build a uid → expected-dict index to sidestep row-order dependence on rng.
    golden_by_uid = {spec['prompt_uid']: spec for spec in _GOLDEN_SPEC}
    uids = list(mb.non_tensors['uid'])

    assert set(uids) == set(golden_by_uid)

    for i, uid in enumerate(uids):
        spec = golden_by_uid[uid]
        got_prompt = _unpad_prompt(prompt_part[i], prompt_attn[i])
        got_response = _unpad_response(mb.tensors['responses'][i], response_attn[i])
        got_loss_mask = mb.tensors['loss_mask'][i][: len(spec['response'])].tolist()
        got_log_probs = mb.tensors['rollout_log_probs'][i][
            : len(spec['response'])
        ].tolist()

        assert tuple(got_prompt) == spec['prompt'], (
            f'prompt tokens drifted for uid {uid}: '
            f'got {got_prompt}, want {spec["prompt"]}'
        )
        assert tuple(got_response) == spec['response'], (
            f'response tokens drifted for uid {uid}'
        )
        assert tuple(got_loss_mask) == spec['loss_mask'], (
            f'loss mask drifted for uid {uid}'
        )
        assert got_log_probs == pytest.approx(list(spec['log_probs'])), (
            f'log probs drifted for uid {uid}'
        )


# --- concurrent push / sample ------------------------------------------------


def test_concurrent_push_and_sample_no_race():
    """Producer pushes on one thread, trainer samples on another. No
    exceptions; the store stays consistent under ``threading.Lock``."""
    import time as _time

    store = TrajectoryStore(max_size=32, staleness_cutoff_k=10000, pad_token_id=PAD_ID)

    # Pre-seed so sampling can immediately succeed.
    for i in range(8):
        store.push_group([_record(seed=i, group_uid=f'seed-{i}', created_at_step=0)])

    stop = threading.Event()
    errors: list[BaseException] = []

    def producer():
        try:
            i = 100
            while not stop.is_set():
                store.push_group(
                    [_record(seed=i, group_uid=f'p-{i}', created_at_step=0)]
                )
                i += 1
                # Yield so we don't monopolize the lock; real producer is
                # rate-limited by the rollout-generation pipeline anyway.
                _time.sleep(0)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def consumer():
        try:
            local_rng = random.Random(7)
            # pop-on-sample: consumer can legitimately outrun producer and
            # hit InsufficientTrajectoriesError. Retry — the point of the
            # test is lock correctness, not starvation handling.
            drawn = 0
            while drawn < 50:
                try:
                    store.sample_mini_batch(n_groups=4, current_step=0, rng=local_rng)
                    drawn += 1
                except InsufficientTrajectoriesError:
                    _time.sleep(0)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    p = threading.Thread(target=producer)
    c = threading.Thread(target=consumer)
    p.start()
    c.start()
    c.join(timeout=10.0)
    stop.set()
    p.join(timeout=10.0)

    assert not c.is_alive(), 'consumer thread did not finish'
    assert not p.is_alive(), 'producer thread did not finish'
    assert errors == [], f'thread errors: {errors}'
    # Bounded by max_size.
    assert store.num_groups() <= 32


# --- metrics -----------------------------------------------------------------


def test_metrics_empty_store():
    store = TrajectoryStore(max_size=4, staleness_cutoff_k=10, pad_token_id=PAD_ID)
    m = store.metrics(current_step=0)
    assert m['replay/store_size'] == 0.0
    assert m['replay/store_fill_ratio'] == 0.0
    assert m['replay/store_age_p50'] == 0.0
    assert m['replay/sample_age_steps_p50'] == 0.0
    assert m['replay/dropped_by_staleness_total'] == 0.0


def test_metrics_report_staleness_drop_counter():
    store = TrajectoryStore(max_size=4, staleness_cutoff_k=1, pad_token_id=PAD_ID)
    store.push_group([_record(seed=0, created_at_step=0, group_uid='s')])
    store.evict_stale(current_step=10)
    m = store.metrics(current_step=10)
    assert m['replay/dropped_by_staleness_total'] == 1.0
    assert m['replay/store_size'] == 0.0


def test_metrics_report_fill_ratio_and_age_percentiles():
    store = TrajectoryStore(max_size=4, staleness_cutoff_k=100, pad_token_id=PAD_ID)
    for i in range(2):
        store.push_group([_record(seed=i, created_at_step=i, group_uid=f'g{i}')])
    m = store.metrics(current_step=5)
    assert m['replay/store_size'] == 2.0
    assert m['replay/store_fill_ratio'] == 0.5
    # Ages are [5 - 0, 5 - 1] = [5, 4]; p50 by numpy default = 4.5
    assert m['replay/store_age_p50'] == pytest.approx(4.5)
    assert m['replay/store_num_trajectories'] == 2.0


# --- __init__ validation -----------------------------------------------------


@pytest.mark.parametrize(
    ('max_size', 'staleness_k'),
    [(0, 4), (-1, 4), (4, -1)],
    ids=['max_size_zero', 'max_size_negative', 'staleness_negative'],
)
def test_init_rejects_bad_args(max_size, staleness_k):
    with pytest.raises(ValueError):
        TrajectoryStore(
            max_size=max_size, staleness_cutoff_k=staleness_k, pad_token_id=0
        )


def test_sample_mini_batch_n_groups_must_be_positive():
    store = TrajectoryStore(max_size=4, staleness_cutoff_k=10, pad_token_id=0)
    store.push_group([_record()])
    with pytest.raises(ValueError):
        store.sample_mini_batch(n_groups=0, current_step=0)


# --- truncation / cap behavior ----------------------------------------------


def test_response_length_cap_truncates_head_keeps_front():
    store = TrajectoryStore(
        max_size=4,
        staleness_cutoff_k=10,
        pad_token_id=PAD_ID,
        response_length_cap=2,
    )
    store.push_group(
        [
            _record(
                seed=0,
                response_ids=(99, 100, 101, 102),
                loss_mask=(1, 1, 1, 1),
                log_probs=(0.1, 0.2, 0.3, 0.4),
            )
        ]
    )
    mb = store.sample_mini_batch(n_groups=1, current_step=0, rng=random.Random(0))
    # Response was truncated to first 2 tokens.
    resp_row = mb.tensors['responses'][0]
    assert resp_row.shape[0] == 2
    assert resp_row.tolist() == [99, 100]


def test_prompt_length_cap_truncates_keeps_tail():
    """Prompt truncation keeps the *tail* (most recent context)."""
    store = TrajectoryStore(
        max_size=4,
        staleness_cutoff_k=10,
        pad_token_id=PAD_ID,
        prompt_length_cap=2,
    )
    store.push_group(
        [
            _record(
                seed=0,
                prompt_ids=(1, 2, 3, 4, 5),
                response_ids=(6,),
                loss_mask=(1,),
                log_probs=(0.0,),
            )
        ]
    )
    mb = store.sample_mini_batch(n_groups=1, current_step=0, rng=random.Random(0))
    # With cap=2 we keep (4, 5) — the most recent tokens.
    prompt_len = mb.tensors['input_ids'].shape[1] - mb.tensors['responses'].shape[1]
    prompt = mb.tensors['input_ids'][0, :prompt_len]
    attn = mb.tensors['attention_mask'][0, :prompt_len]
    unpad = prompt[attn == 1].tolist()
    assert unpad == [4, 5]


# --- push_from_dataproto round-trip (host-runnable via SimpleNamespace) ------


def _stub_dataproto(
    *,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    rollout_log_probs: torch.Tensor | None = None,
    is_padded: torch.Tensor | None = None,
    error_mask: torch.Tensor | None = None,
    uids: list[str] | None = None,
    **extra_non_tensors,
):
    """Build a ``DataProto``-shaped stub with only the attrs the store reads.

    ``push_from_dataproto`` accesses ``dp.batch`` (dict[str, Tensor]) and
    ``dp.non_tensor_batch`` (dict[str, np.ndarray]), so a bare namespace
    is enough.
    """
    from types import SimpleNamespace

    import numpy as np

    batch_size = input_ids.shape[0]
    response_len = responses.shape[1]
    prompt_len = input_ids.shape[1] - response_len

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if loss_mask is None:
        loss_mask = torch.ones((batch_size, response_len), dtype=torch.long)
    if rollout_log_probs is None:
        rollout_log_probs = torch.full((batch_size, response_len), -0.1)
    if is_padded is None:
        is_padded = torch.zeros(batch_size, dtype=torch.bool)
    if error_mask is None:
        error_mask = torch.zeros(batch_size, dtype=torch.bool)
    if uids is None:
        uids = [f'u-{i}' for i in range(batch_size)]

    tensors = {
        'input_ids': input_ids,
        'responses': responses,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
        'rollout_log_probs': rollout_log_probs,
        'is_padded': is_padded,
        'error_mask': error_mask,
    }
    non_tensors = {'uid': np.array(uids, dtype=object)}
    for k, v in extra_non_tensors.items():
        non_tensors[k] = np.array(v, dtype=object)
    # prompt_len is derived from input_ids.shape[1] - responses.shape[1],
    # so this assertion keeps the fixture honest.
    assert prompt_len >= 0
    return SimpleNamespace(batch=tensors, non_tensor_batch=non_tensors)


class TestPushFromDataProtoRoundTrip:
    """Host-runnable parity for ``TrajectoryStore.push_from_dataproto``."""

    def _store(self, **kw) -> TrajectoryStore:
        return TrajectoryStore(
            max_size=kw.pop('max_size', 8),
            staleness_cutoff_k=kw.pop('staleness_cutoff_k', 10),
            pad_token_id=PAD_ID,
            **kw,
        )

    def test_prompt_extras_roundtrip_dict_string_none(self) -> None:
        """Reward-model dict, ability string, None-valued extra all survive."""
        store = self._store()
        # 2 prompts, 2 siblings each.
        batch = 4
        prompt_len, response_len = 3, 2
        input_ids = torch.zeros((batch, prompt_len + response_len), dtype=torch.long)
        responses = torch.zeros((batch, response_len), dtype=torch.long)
        uids = ['p0', 'p0', 'p1', 'p1']
        rewards_model = [{'style': 'rule', 'ground_truth': 'x'}] * batch
        abilities = ['math', 'math', 'code', 'code']
        extras_none = [None, None, None, None]
        dp = _stub_dataproto(
            input_ids=input_ids,
            responses=responses,
            uids=uids,
            reward_model=rewards_model,
            ability=abilities,
            extra_info=extras_none,
        )
        pushed = store.push_from_dataproto(
            dp, behavior_policy_version=5, current_step=0
        )
        assert pushed == 2  # two distinct uids → two groups
        mb = store.sample_mini_batch(n_groups=2, current_step=0, rng=random.Random(0))
        # Extras round-trip to the sampled non_tensors.
        for key in ('reward_model', 'ability', 'extra_info'):
            assert key in mb.non_tensors, f'{key} missing from sampled non_tensors'
        assert set(mb.non_tensors['ability'].tolist()) == {'math', 'code'}
        # Reward model is a dict — copies, not aliases.
        for rm in mb.non_tensors['reward_model']:
            assert isinstance(rm, dict)
            assert rm == {'style': 'rule', 'ground_truth': 'x'}

    def test_reward_model_dict_is_copied_not_aliased(self) -> None:
        """Mutating the caller's dict must NOT mutate the stored record."""
        import numpy as np

        store = self._store()
        shared_rm = {'style': 'rule', 'ground_truth': 'original'}
        dp = _stub_dataproto(
            input_ids=torch.zeros((2, 5), dtype=torch.long),
            responses=torch.zeros((2, 2), dtype=torch.long),
            uids=['p0', 'p0'],
            reward_model=np.array([shared_rm, shared_rm], dtype=object),
        )
        store.push_from_dataproto(dp, behavior_policy_version=0, current_step=0)
        shared_rm['ground_truth'] = 'mutated_after_push'
        mb = store.sample_mini_batch(n_groups=1, current_step=0, rng=random.Random(0))
        for rm in mb.non_tensors['reward_model']:
            assert rm['ground_truth'] == 'original', (
                'caller mutation leaked into stored record'
            )

    def test_error_mask_sets_error_field(self) -> None:
        """When error_mask=1 but non_tensor 'error' is absent, record.error='error_mask_set'."""
        store = self._store()
        em = torch.zeros(2, dtype=torch.bool)
        em[1] = True
        dp = _stub_dataproto(
            input_ids=torch.zeros((2, 5), dtype=torch.long),
            responses=torch.zeros((2, 2), dtype=torch.long),
            uids=['p0', 'p0'],
            error_mask=em,
        )
        store.push_from_dataproto(dp, behavior_policy_version=0, current_step=0)
        groups = store._snapshot_groups()
        assert len(groups) == 1
        errs = [r.error for r in groups[0]]
        assert errs[0] is None
        assert errs[1] == 'error_mask_set'

    def test_singleton_group_uid_kept_whole(self) -> None:
        """A uid that appears only once still forms a valid singleton group."""
        store = self._store()
        dp = _stub_dataproto(
            input_ids=torch.zeros((3, 5), dtype=torch.long),
            responses=torch.zeros((3, 2), dtype=torch.long),
            uids=['solo', 'pair', 'pair'],
        )
        pushed = store.push_from_dataproto(
            dp, behavior_policy_version=0, current_step=0
        )
        assert pushed == 2
        groups = store._snapshot_groups()
        # Grouped by uid — one singleton, one pair.
        sizes = sorted(len(g) for g in groups)
        assert sizes == [1, 2]

    def test_missing_optional_non_tensors_default_gracefully(self) -> None:
        """Absent ``success``/``error``/``resolved``/``finish``/``instance`` → safe defaults."""
        store = self._store()
        dp = _stub_dataproto(
            input_ids=torch.zeros((2, 5), dtype=torch.long),
            responses=torch.zeros((2, 2), dtype=torch.long),
            uids=['p0', 'p0'],
        )
        store.push_from_dataproto(dp, behavior_policy_version=0, current_step=0)
        groups = store._snapshot_groups()
        rec = groups[0][0]
        assert rec.success is True
        assert rec.error is None
        assert rec.resolved is False
        assert rec.finish is True
        assert rec.instance == {}

    def test_raises_on_missing_uid(self) -> None:
        """uid is required — push should raise KeyError pointing at the seam."""
        store = self._store()
        # Stub a DataProto with no uid.
        dp = _stub_dataproto(
            input_ids=torch.zeros((1, 3), dtype=torch.long),
            responses=torch.zeros((1, 1), dtype=torch.long),
            uids=['keep'],  # present
        )
        # Drop uid from the stub to trigger the KeyError branch.
        del dp.non_tensor_batch['uid']
        with pytest.raises(KeyError, match='uid'):
            store.push_from_dataproto(dp, behavior_policy_version=0, current_step=0)
