"""S3 — append + query round-trip + tokenizer mismatch enforcement."""

from __future__ import annotations

import pytest

from replay_archive import ArchiveServer, query
from replay_archive.derive import (
    TokenizerMismatchError,
    derive_training_samples,
)
from schemas.episode_record import TrustLevel
from schemas.protocols.replay_archive import FilterSpec

from .conftest import make_episode

pytestmark = pytest.mark.contract


def test_append_then_query_returns_records(tmp_path) -> None:
    arc = ArchiveServer(tmp_path)
    eps = [make_episode(episode_uid=f'ep-{i}', task_id=f't-{i}') for i in range(5)]
    accepted, dups, _uids = arc.append_episodes(eps)
    assert accepted == 5
    assert dups == 0

    # Query by policy_id (matches all 5).
    rows = list(query(tmp_path, filter_spec=FilterSpec(policy_id='qwen3-4b-skyrl')))
    assert len(rows) == 5
    got = {r.episode_uid for r in rows}
    assert got == {f'ep-{i}' for i in range(5)}

    # Filter by task_id subset.
    rows = list(query(tmp_path, filter_spec=FilterSpec(task_ids=('t-1', 't-3'))))
    assert {r.episode_uid for r in rows} == {'ep-1', 'ep-3'}


def test_dedup_on_replay(tmp_path) -> None:
    arc = ArchiveServer(tmp_path)
    eps = [make_episode(episode_uid='ep-x', task_id='t-x')]
    a1, d1, _ = arc.append_episodes(eps)
    a2, d2, _ = arc.append_episodes(eps)
    assert a1 == 1 and d1 == 0
    assert a2 == 0 and d2 == 1
    rows = list(query(tmp_path, filter_spec=FilterSpec(task_ids=('t-x',))))
    assert len(rows) == 1


def test_filtered_groups_archived_alongside_admitted(tmp_path) -> None:
    """§7 corollary — the live store filters; the archive sees everything.

    We just append both; the archive can't tell admitted from dropped.
    The contract is: the worker writes to the archive INDEPENDENTLY of
    whether the live store's zero-variance filter admits the group.
    """
    arc = ArchiveServer(tmp_path)
    eps = [
        make_episode(episode_uid='ep-admitted', task_id='t-1', total_reward=0.5),
        make_episode(episode_uid='ep-dropped', task_id='t-2', total_reward=0.0),
    ]
    arc.append_episodes(eps)
    # Archive contains both.
    rows = list(query(tmp_path, filter_spec=FilterSpec(policy_id='qwen3-4b-skyrl')))
    assert {r.episode_uid for r in rows} == {'ep-admitted', 'ep-dropped'}


def test_derive_training_samples_round_trip(tmp_path) -> None:
    arc = ArchiveServer(tmp_path)
    eps = [make_episode(episode_uid='ep-1', task_id='t-1')]
    arc.append_episodes(eps)
    rows = list(query(tmp_path, filter_spec=FilterSpec(task_ids=('t-1',))))
    samples = derive_training_samples(
        rows, expected_tokenizer_id='Qwen/Qwen3-4B-Instruct'
    )
    assert len(samples) == 1
    s = samples[0]
    assert s.episode_uid == 'ep-1'
    assert s.prompt_token_ids == (1, 2, 3)
    assert s.response_token_ids == (4, 5, 6)


def test_tokenizer_mismatch_raises_typed_error(tmp_path) -> None:
    """Post-S4 checklist item 16."""
    arc = ArchiveServer(tmp_path)
    arc.append_episodes([make_episode(episode_uid='ep-1', task_id='t-1')])
    rows = list(query(tmp_path, filter_spec=FilterSpec(task_ids=('t-1',))))
    with pytest.raises(TokenizerMismatchError):
        derive_training_samples(rows, expected_tokenizer_id='other/tokenizer')


def test_query_by_policy_version_range(tmp_path) -> None:
    arc = ArchiveServer(tmp_path)
    eps = [
        make_episode(episode_uid=f'ep-v{v}', task_id=f't-v{v}', policy_version=v)
        for v in (1, 5, 10, 15)
    ]
    arc.append_episodes(eps)
    rows = list(
        query(
            tmp_path,
            filter_spec=FilterSpec(policy_version_min=5, policy_version_max=10),
        )
    )
    assert {r.episode_uid for r in rows} == {'ep-v5', 'ep-v10'}


def test_query_by_trust_level(tmp_path) -> None:
    arc = ArchiveServer(tmp_path)
    arc.append_episodes(
        [
            make_episode(
                episode_uid='ep-own',
                task_id='t-own',
                trust_level=TrustLevel.OWN_FABRIC,
            ),
            make_episode(
                episode_uid='ep-partner',
                task_id='t-partner',
                trust_level=TrustLevel.PARTNER_VALIDATED,
            ),
        ]
    )
    rows = list(
        query(
            tmp_path,
            filter_spec=FilterSpec(trust_levels=(TrustLevel.OWN_FABRIC,)),
        )
    )
    assert {r.episode_uid for r in rows} == {'ep-own'}
