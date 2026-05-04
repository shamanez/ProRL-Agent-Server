"""§3.2 invariant — GRPO/DAPO group integrity (BC-2)."""

from __future__ import annotations

import pytest

from schemas.episode_record import TrustLevel
from schemas.training_sample import (
    TrainingGroup,
    TrainingSample,
    assert_group_integrity,
)

pytestmark = pytest.mark.invariant


def _make_sample(group_uid: str, sample_uid: str) -> TrainingSample:
    return TrainingSample(
        sample_uid=sample_uid,
        group_uid=group_uid,
        episode_uid=sample_uid,
        prompt_token_ids=(1,),
        response_token_ids=(2,),
        response_loss_mask=(1,),
        behavior_log_probs=(-0.1,),
        reward=0.0,
        raw_reward=0.0,
        truncated=False,
        behavior_policy_version=1,
        created_at_step=0,
        task_id='t1',
        split='train',
        policy_id='qwen3-4b-skyrl',
        environment_id='swe_agent',
        environment_version='v1',
        verifier_version='v1',
        trust_level=TrustLevel.OWN_FABRIC,
        sample_indices=None,
        instance={},
        error=None,
        is_padded=False,
    )


def test_training_group_requires_matching_group_uid() -> None:
    a = _make_sample('g1', 's1')
    b = _make_sample('g2', 's2')
    with pytest.raises(ValueError, match='group_uid'):
        TrainingGroup(group_uid='g1', samples=(a, b))


def test_training_group_accepts_consistent_siblings() -> None:
    siblings = tuple(_make_sample('g1', f's{i}') for i in range(8))
    g = TrainingGroup(group_uid='g1', samples=siblings)
    assert len(g) == 8
    assert all(s.group_uid == 'g1' for s in g.samples)


def test_assert_group_integrity_catches_mixed_groups() -> None:
    a, b = _make_sample('g1', 's1'), _make_sample('g1', 's2')
    c = _make_sample('g2', 's3')
    assert_group_integrity([a, b])
    with pytest.raises(ValueError, match='integrity violation'):
        assert_group_integrity([a, b, c])


def test_training_group_rejects_empty_samples() -> None:
    with pytest.raises(ValueError):
        TrainingGroup(group_uid='g1', samples=())
