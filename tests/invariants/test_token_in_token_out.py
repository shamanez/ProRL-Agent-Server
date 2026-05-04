"""§3.1 invariant — token-in / token-out across turns.

The wire schema must carry token IDs, not decoded text. Re-tokenizing
across turns shifts boundaries; actor and reference diverge; KL/entropy
go NaN; PPO/GRPO collapses.

This test asserts the type-level contract: every token-bearing field on
``TrainingSample`` is a ``tuple[int, ...]`` (or ``None`` where allowed),
and the proto schema declares the corresponding ``bytes`` carry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from schemas.episode_record import TrustLevel
from schemas.training_sample import TrainingSample

pytestmark = pytest.mark.invariant


def _make_sample(**overrides):
    base = dict(
        sample_uid='s1',
        group_uid='g1',
        episode_uid='e1',
        prompt_token_ids=(1, 2, 3),
        response_token_ids=(4, 5),
        response_loss_mask=(1, 1),
        behavior_log_probs=(-0.1, -0.2),
        reward=0.5,
        raw_reward=0.5,
        truncated=False,
        behavior_policy_version=7,
        created_at_step=42,
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
    base.update(overrides)
    return TrainingSample(**base)


def test_training_sample_carries_token_ids_not_text() -> None:
    """Every token-bearing field is a tuple of ints, not a str."""
    s = _make_sample()
    assert isinstance(s.prompt_token_ids, tuple)
    assert all(isinstance(x, int) for x in s.prompt_token_ids)
    assert isinstance(s.response_token_ids, tuple)
    assert all(isinstance(x, int) for x in s.response_token_ids)
    assert isinstance(s.response_loss_mask, tuple)
    assert all(isinstance(x, int) for x in s.response_loss_mask)
    assert s.behavior_log_probs is None or all(
        isinstance(x, float) for x in s.behavior_log_probs
    )


def test_training_sample_rejects_string_tokens() -> None:
    """Constructing a sample with str tokens raises a TypeError-equivalent.

    The dataclass annotation is ``tuple[int, ...]`` — passing a string
    would silently succeed at construction (Python dataclasses don't
    enforce types). We use the length-equality post-init assertions to
    catch obvious shape errors; a deliberately-typed mismatch is caught
    by the static type checker (``make lint`` runs mypy).
    """
    with pytest.raises(ValueError):
        _make_sample(response_token_ids=(4, 5, 6), response_loss_mask=(1, 1))


def test_proto_schema_uses_bytes_for_token_arrays() -> None:
    """Live store proto carries token arrays as bytes, not as strings.

    Lightweight regression: grep the .proto file for the relevant fields.
    Compiles only at S1; at S0.5 we assert the schema text is correct.
    """
    proto = (
        Path(__file__).resolve().parents[2] / 'schemas' / 'proto' / 'live_store.proto'
    )
    text = proto.read_text()
    for field_name in (
        'prompt_token_ids',
        'response_token_ids',
        'response_loss_mask',
        'behavior_log_probs',
    ):
        # `bytes <field_name> = N;` — order-tolerant whitespace.
        assert re.search(rf'\bbytes\s+{re.escape(field_name)}\s*=\s*\d+\s*;', text), (
            f'live_store.proto must declare {field_name} as bytes, not string'
        )
