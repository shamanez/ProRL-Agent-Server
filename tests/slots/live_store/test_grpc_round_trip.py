"""S1 — gRPC round-trip: push then get_batch returns equivalent samples."""

from __future__ import annotations

import pytest

from .conftest import make_sample

pytestmark = pytest.mark.contract


def test_push_then_get_batch_round_trip(live_store_client) -> None:
    """Push 4 groups of size 1 each; get_batch(2) returns 2 disjoint groups."""
    for i in range(4):
        s = make_sample(sample_uid=f's{i}', group_uid=f'g{i}')
        live_store_client.push_group([s])
    assert live_store_client.num_groups() == 4

    samples = live_store_client.get_batch(n_groups=2, current_step=0, timeout_ms=2_000)
    assert len(samples) == 2
    assert {s.group_uid for s in samples}.issubset({f'g{i}' for i in range(4)})
    # Pop-on-sample (§3.6): two of the four are gone.
    assert live_store_client.num_groups() == 2


def test_round_trip_preserves_token_arrays(live_store_client) -> None:
    """Token arrays survive packed-bytes encoding without corruption.

    The codec uses int32-LE packed bytes for token sequences. This is
    the §3.1 wire-shape invariant — zero loss on round-trip.
    """
    pids = (101, 202, 303, 404)
    rids = (505, 606, 707)
    s = make_sample(sample_uid='s0', group_uid='g0')
    s = s.__class__(  # rebuild with custom token arrays
        sample_uid=s.sample_uid,
        group_uid=s.group_uid,
        episode_uid=s.episode_uid,
        prompt_token_ids=pids,
        response_token_ids=rids,
        response_loss_mask=(1,) * len(rids),
        behavior_log_probs=tuple(-0.1 * (i + 1) for i in range(len(rids))),
        reward=s.reward,
        raw_reward=s.raw_reward,
        truncated=s.truncated,
        behavior_policy_version=s.behavior_policy_version,
        created_at_step=s.created_at_step,
        task_id=s.task_id,
        split=s.split,
        policy_id=s.policy_id,
        environment_id=s.environment_id,
        environment_version=s.environment_version,
        verifier_version=s.verifier_version,
        trust_level=s.trust_level,
        sample_indices=s.sample_indices,
        instance=s.instance,
        error=s.error,
        is_padded=s.is_padded,
    )
    live_store_client.push_group([s])
    out = live_store_client.get_batch(n_groups=1, current_step=0, timeout_ms=2_000)
    assert len(out) == 1
    got = out[0]
    assert got.prompt_token_ids == pids
    assert got.response_token_ids == rids
    assert got.behavior_log_probs is not None
    assert len(got.behavior_log_probs) == len(rids)


def test_metrics_endpoint(live_store_client) -> None:
    for i in range(3):
        live_store_client.push_group(
            [make_sample(sample_uid=f's{i}', group_uid=f'g{i}')]
        )
    m = live_store_client.metrics(current_step=0)
    assert m['replay/store_size'] == 3.0
    assert m['replay/dropped_by_staleness_total'] == 0.0


def test_notify_policy_version(live_store_client) -> None:
    live_store_client.notify_policy_version(7, 'file:///tmp/v7')
    # No assert beyond not-raising; the registry-side cache is exercised
    # in tests/slots/policy_registry/.
