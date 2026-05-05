"""§3.5 invariant — per-row policy_version stamp via PolicyVersionCache (BC-7)."""

from __future__ import annotations

import dataclasses
import threading
import time

import pytest
from rollout_fabric.schemas.policy_version import (
    PolicyVersionCache,
    PolicyVersionSnapshot,
)

pytestmark = pytest.mark.invariant


def _snap(version: int, uri: str = '', policy_id: str = 'qwen3-4b-skyrl'):
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


def test_snapshot_is_immutable() -> None:
    s = _snap(7, '/tmp/v7')
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        s.version = 8  # type: ignore[misc]


def test_cache_returns_initial_snapshot() -> None:
    init = _snap(7, '/tmp/v7')
    assert PolicyVersionCache(init).snapshot() is init


def test_cache_swaps_on_strictly_greater_version() -> None:
    cache = PolicyVersionCache(_snap(7))
    assert cache.update(_snap(8)) is True
    assert cache.snapshot().version == 8


def test_cache_rejects_equal_or_stale() -> None:
    cache = PolicyVersionCache(_snap(7))
    assert cache.update(_snap(7)) is False
    assert cache.update(_snap(6)) is False
    assert cache.snapshot().version == 7


def test_cache_rejects_cross_namespace() -> None:
    cache = PolicyVersionCache(_snap(7, policy_id='qwen3-4b-skyrl'))
    with pytest.raises(ValueError, match='policy_id'):
        cache.update(_snap(8, policy_id='other-policy'))


def test_group_dispatch_snapshot_is_per_group_consistent() -> None:
    """One snapshot read at group start — mid-group publish does NOT affect rows."""
    cache = PolicyVersionCache(_snap(7))
    snap = cache.snapshot()
    assert cache.update(_snap(8)) is True
    rows = [{'policy_version': snap.version} for _ in range(8)]
    assert all(r['policy_version'] == 7 for r in rows)
    assert cache.snapshot().version == 8


def test_high_contention_no_torn_reads() -> None:
    """Single-writer / 8-reader stress: no torn (version, adapter_uri) pairs."""
    cache = PolicyVersionCache(_snap(0, '/v0'))
    stop = threading.Event()
    mismatches: list = []

    def reader():
        while not stop.is_set():
            s = cache.snapshot()
            if s.adapter_uri != f'/v{s.version}':
                mismatches.append((s.version, s.adapter_uri))

    def writer():
        v = 1
        while not stop.is_set():
            cache.update(_snap(v, f'/v{v}'))
            v += 1

    readers = [threading.Thread(target=reader) for _ in range(8)]
    w = threading.Thread(target=writer)
    for t in readers:
        t.start()
    w.start()
    time.sleep(0.5)
    stop.set()
    for t in readers:
        t.join()
    w.join()
    assert not mismatches, f'torn reads: {mismatches[:3]}'
