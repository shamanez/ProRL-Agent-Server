"""S3 — writer at-least-once + dead-letter spillover under chaos.

Two contracts under test:

1. **At-least-once.** The writer retries on append failure; the
   archive's INSERT-OR-IGNORE makes retries idempotent. Replays do
   not produce duplicate Parquet rows.
2. **Non-blocking on backpressure.** When the archive's queue is
   full, ``submit`` does NOT block — the episode spills to the
   dead-letter file. This is the §7 contract: archive availability
   does not stall the producer.
"""

from __future__ import annotations

import json
import time

import pytest

from replay_archive import ArchiveServer, ReplayArchiveWriter, query
from schemas.protocols.replay_archive import FilterSpec

from .conftest import make_episode

pytestmark = pytest.mark.contract


def test_writer_retries_on_transient_failure(tmp_path) -> None:
    arc = ArchiveServer(tmp_path / 'arc')
    dead_letter = tmp_path / 'dl.jsonl'

    # Inject a transient failure: first call raises, subsequent succeed.
    real_append = arc.append_episodes
    state = {'calls': 0}

    def flaky(records):
        state['calls'] += 1
        if state['calls'] == 1:
            raise RuntimeError('transient archive outage')
        return real_append(records)

    arc.append_episodes = flaky  # type: ignore[method-assign]

    writer = ReplayArchiveWriter(
        server=arc,
        queue_max_size=64,
        dead_letter_path=dead_letter,
        batch_size=4,
        max_retry_attempts=3,
    )
    writer.start()
    try:
        for i in range(3):
            writer.submit(make_episode(episode_uid=f'ep-{i}', task_id=f't-{i}'))
        # Bounded settle: backoff is 1s, then archive succeeds.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if writer.stats()['archived'] >= 3:
                break
            time.sleep(0.1)
    finally:
        writer.stop(timeout=2.0)

    stats = writer.stats()
    assert stats['archived'] == 3, f'expected 3 archived, got {stats}'
    assert stats['dead_lettered'] == 0
    rows = list(
        query(tmp_path / 'arc', filter_spec=FilterSpec(policy_id='qwen3-4b-skyrl'))
    )
    assert len(rows) == 3


def test_dead_letter_when_retries_exhausted(tmp_path) -> None:
    arc = ArchiveServer(tmp_path / 'arc')
    dead_letter = tmp_path / 'dl.jsonl'

    def always_fail(records):
        raise RuntimeError('archive permanently down')

    arc.append_episodes = always_fail  # type: ignore[method-assign]

    writer = ReplayArchiveWriter(
        server=arc,
        queue_max_size=64,
        dead_letter_path=dead_letter,
        batch_size=2,
        max_retry_attempts=2,
    )
    writer.start()
    try:
        writer.submit(make_episode(episode_uid='ep-1', task_id='t-1'))
        writer.submit(make_episode(episode_uid='ep-2', task_id='t-2'))
        # 2 attempts × ~1-2s backoff each = ~3-4s before the dead-letter
        # spill. Allow up to 8s before declaring failure.
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if writer.stats()['dead_lettered'] >= 2:
                break
            time.sleep(0.2)
    finally:
        writer.stop(timeout=2.0)

    assert writer.stats()['dead_lettered'] >= 2
    assert dead_letter.exists()
    lines = dead_letter.read_text().strip().splitlines()
    assert len(lines) >= 2
    parsed = [json.loads(line) for line in lines]
    assert {p['episode_uid'] for p in parsed} == {'ep-1', 'ep-2'}
    for p in parsed:
        assert p['reason'] == 'retry_exhausted'


def test_submit_never_blocks_under_overflow(tmp_path) -> None:
    """§7 — archive backpressure must not stall episode generation.

    Use a queue_max_size of 2 and submit 50 episodes back-to-back; the
    writer is paused (we never start it), so the queue fills and
    overflow goes to the dead-letter. The submit calls must complete
    in well under the wall-clock cost of generating 50 episodes
    sequentially.
    """
    arc = ArchiveServer(tmp_path / 'arc')
    dead_letter = tmp_path / 'dl.jsonl'
    writer = ReplayArchiveWriter(
        server=arc,
        queue_max_size=2,  # tiny queue
        dead_letter_path=dead_letter,
    )
    # Note: writer NOT started — queue will fill, overflow to dead-letter.

    t0 = time.monotonic()
    for i in range(50):
        writer.submit(make_episode(episode_uid=f'ep-{i}', task_id=f't-{i}'))
    elapsed = time.monotonic() - t0
    # 50 non-blocking submits should complete in milliseconds.
    assert elapsed < 1.0, f'submit blocked: {elapsed:.2f}s for 50 episodes'

    stats = writer.stats()
    assert stats['submitted'] == 50
    # 2 fit in the queue, 48 overflow to the dead-letter.
    assert stats['dead_lettered'] >= 48
    assert dead_letter.exists()
