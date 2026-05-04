"""S1 — trainer-adapter pack reconstructs the legacy SampledMiniBatch shape.

The legacy ``_pack`` (in-process trajectory store) is lifted to
``trainer_adapters/verl/pad.pack_unpadded_groups``. This test asserts
the post-pack tensor / non-tensor layout matches today's contract so
``DataProto.from_dict`` keeps working bit-identically.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip('torch')

from trainer_adapters.verl.pad import (  # noqa: E402
    SampledMiniBatch,
    pack_unpadded_groups,
)

from .conftest import make_sample

pytestmark = pytest.mark.contract


def _samples(n: int = 4):
    return [make_sample(sample_uid=f's{i}', group_uid=f'g{i}') for i in range(n)]


def test_pack_returns_legacy_layout() -> None:
    out = pack_unpadded_groups(
        _samples(4), pad_token_id=0, prompt_length_cap=8, response_length_cap=4
    )
    assert isinstance(out, SampledMiniBatch)
    for k in (
        'input_ids',
        'responses',
        'attention_mask',
        'position_ids',
        'loss_mask',
        'rollout_log_probs',
        'is_padded',
        'error_mask',
        'reward',
        'raw_reward',
        'truncated',
    ):
        assert k in out.tensors, f'missing tensor key: {k}'
    for k in ('uid', 'success', 'error', 'resolved', 'finish', 'instance'):
        assert k in out.non_tensors, f'missing non-tensor key: {k}'
    assert out.tensors['input_ids'].shape == (4, 8 + 4)
    assert out.tensors['responses'].shape == (4, 4)
    assert out.meta_info['behavior_policy_versions'] == [1, 1, 1, 1]


def test_pack_raises_on_oversize_prompt() -> None:
    s = make_sample(sample_uid='s0', group_uid='g0')
    big = s.__class__(
        sample_uid=s.sample_uid,
        group_uid=s.group_uid,
        episode_uid=s.episode_uid,
        prompt_token_ids=tuple(range(100)),  # cap will be 4
        response_token_ids=s.response_token_ids,
        response_loss_mask=s.response_loss_mask,
        behavior_log_probs=s.behavior_log_probs,
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
    with pytest.raises(RuntimeError, match='prompt_token_ids length'):
        pack_unpadded_groups(
            [big], pad_token_id=0, prompt_length_cap=4, response_length_cap=4
        )
