"""§3.5 invariant — per-row ``behavior_policy_version`` + the cleverest primitive.

The producer reads the current ``policy_version`` and stamps it on each
emitted trajectory; the trainer thread writes it inside the publish
path after pool ACK.

Post-migration, the read happens via :class:`PolicyVersionCache` —
immutable snapshot + atomic reference swap (``schemas/policy_version.py``).
This test exercises the four properties that make this primitive
correct:

  1. A single attribute store atomically publishes a new snapshot to
     all readers.
  2. Reading the cache once at group dispatch and stamping every row
     of the group with the same snapshot satisfies §3.2 + §3.5
     simultaneously.
  3. Stale updates (lower version) are rejected.
  4. Cross-namespace updates (wrong ``policy_id``) are rejected.

Plus a stress test that exercises the read/write contention pattern
at production rates (1 publish / minute vs many dispatch reads).
"""

from __future__ import annotations

import threading
import time

import pytest

from schemas.policy_version import PolicyVersionCache, PolicyVersionSnapshot

pytestmark = pytest.mark.invariant


def _snap(
    version: int, uri: str = '', policy_id: str = 'qwen3-4b-skyrl'
) -> PolicyVersionSnapshot:
    return PolicyVersionSnapshot(
        policy_id=policy_id,
        version=version,
        adapter_uri=uri,
        received_at=time.monotonic(),
    )


def test_bootstrap_snapshot_has_zero_version() -> None:
    s = PolicyVersionSnapshot.bootstrap('qwen3-4b-skyrl')
    assert s.version == 0
    assert s.adapter_uri == ''
    assert s.policy_id == 'qwen3-4b-skyrl'


def test_snapshot_is_immutable() -> None:
    s = _snap(7, '/tmp/v7')
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        s.version = 8  # type: ignore[misc]


def test_cache_returns_initial_snapshot() -> None:
    init = _snap(7, '/tmp/v7')
    cache = PolicyVersionCache(init)
    assert cache.snapshot() is init


def test_cache_swaps_on_strictly_greater_version() -> None:
    cache = PolicyVersionCache(_snap(7))
    assert cache.update(_snap(8)) is True
    assert cache.snapshot().version == 8


def test_cache_rejects_equal_or_stale_updates() -> None:
    cache = PolicyVersionCache(_snap(7))
    assert cache.update(_snap(7)) is False
    assert cache.update(_snap(6)) is False
    assert cache.snapshot().version == 7


def test_cache_rejects_cross_namespace_update() -> None:
    cache = PolicyVersionCache(_snap(7, policy_id='qwen3-4b-skyrl'))
    with pytest.raises(ValueError, match='policy_id'):
        cache.update(_snap(8, policy_id='other-policy'))


def test_group_dispatch_snapshot_is_per_group_consistent() -> None:
    """One snapshot read at group start binds every row in the group.

    This is the §3.2 + §3.5 simultaneity guarantee. Even if a publish
    lands mid-group, the local snapshot reference still points to the
    pre-publish version, so all rows of the group are uniformly
    tagged.
    """
    cache = PolicyVersionCache(_snap(7))
    # Group dispatch starts: one snapshot read.
    snap = cache.snapshot()

    # Mid-group, a publish lands.
    assert cache.update(_snap(8)) is True

    # The local `snap` reference is unaffected — it points to the
    # immutable v7 record. Every row of the group stamps v7.
    rows = []
    for i in range(8):
        rows.append({'sample_uid': f's{i}', 'policy_version': snap.version})
    assert all(r['policy_version'] == 7 for r in rows)

    # The next group sees the new snapshot.
    next_snap = cache.snapshot()
    assert next_snap.version == 8


def test_high_contention_reader_writer_no_torn_reads() -> None:
    """Single-writer / many-reader stress: readers never see a torn pair.

    A torn read would be a snapshot whose ``version`` and ``adapter_uri``
    came from different publish rounds. We encode the invariant in the
    snapshot: ``adapter_uri`` is always ``f"/v{version}"``, so any read
    where they disagree exposes the torn pair.
    """
    cache = PolicyVersionCache(_snap(0, '/v0'))
    stop_event = threading.Event()
    mismatches: list[tuple[int, str]] = []

    def reader() -> None:
        while not stop_event.is_set():
            s = cache.snapshot()
            expected = f'/v{s.version}'
            if s.adapter_uri != expected:
                mismatches.append((s.version, s.adapter_uri))

    def writer() -> None:
        v = 1
        while not stop_event.is_set():
            cache.update(_snap(v, f'/v{v}'))
            v += 1

    readers = [threading.Thread(target=reader) for _ in range(8)]
    w = threading.Thread(target=writer)
    for t in readers:
        t.start()
    w.start()

    # Run for a short bounded window. Per the operating-principle, this
    # is the production rate inverted: many reads vs steady writes. No
    # torn read is permissible at any rate; this is the contract.
    time.sleep(0.5)
    stop_event.set()
    for t in readers:
        t.join()
    w.join()

    assert not mismatches, (
        f'observed {len(mismatches)} torn (version, adapter_uri) reads — '
        f'first: {mismatches[0]!r}; the snapshot primitive is broken'
    )
