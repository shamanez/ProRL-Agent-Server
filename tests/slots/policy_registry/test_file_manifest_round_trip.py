"""S2 — file manifest write + atomic-ref-swap subscription round-trip.

Trainer-side: ``write_manifest(PolicyManifest(...))`` does an atomic
rename. Worker-side: ``FilePollingPolicySubscription`` polls the file
at 1 Hz, feeds the new snapshot into the cleverest
``PolicyVersionCache``, which atomically swaps it into a single
attribute the dispatch threads read lock-free.

This is the §3.5 invariant carried over the S2 boundary. The same
cache + same primitive carries forward to S4 unchanged; only the
populator flips from polling to gRPC streaming.
"""

from __future__ import annotations

import time

import pytest

from rollout_fabric.policy_registry.file_registry import (
    PolicyManifest,
    read_manifest,
    write_manifest,
)
from rollout_fabric.rollout_manager.policy_subscription import FilePollingPolicySubscription
from rollout_fabric.schemas.policy_version import PolicyVersionCache, PolicyVersionSnapshot

pytestmark = pytest.mark.contract


def test_write_then_read_round_trip(tmp_path) -> None:
    path = str(tmp_path / 'manifest.json')
    m = PolicyManifest(
        policy_id='qwen3-4b-skyrl',
        version=7,
        adapter_uri='file:///tmp/v7',
        trainer_id='trainer-0',
        published_at=time.time(),
    )
    write_manifest(m, path=path)
    got = read_manifest(path)
    assert got is not None
    assert got.policy_id == m.policy_id
    assert got.version == m.version
    assert got.adapter_uri == m.adapter_uri


def test_subscription_atomic_swaps_into_cache(tmp_path) -> None:
    """Writing a fresher manifest yields a swap on the next poll."""
    path = str(tmp_path / 'manifest.json')
    cache = PolicyVersionCache(PolicyVersionSnapshot.bootstrap('qwen3-4b-skyrl'))
    sub = FilePollingPolicySubscription(
        cache=cache,
        manifest_path=path,
        poll_interval_s=0.05,
    )
    sub.start()
    try:
        write_manifest(
            PolicyManifest(
                policy_id='qwen3-4b-skyrl',
                version=3,
                adapter_uri='file:///tmp/v3',
                trainer_id='trainer-0',
                published_at=time.time(),
            ),
            path=path,
        )
        # Bounded settle: poller is at 50ms.
        for _ in range(40):  # ~2 s
            if cache.snapshot().version == 3:
                break
            time.sleep(0.05)
        snap = cache.snapshot()
        assert snap.version == 3
        assert snap.adapter_uri == 'file:///tmp/v3'

        # Stale write does not regress the cache.
        write_manifest(
            PolicyManifest(
                policy_id='qwen3-4b-skyrl',
                version=2,
                adapter_uri='file:///tmp/v2',
                trainer_id='trainer-0',
                published_at=time.time(),
            ),
            path=path,
        )
        time.sleep(0.3)
        assert cache.snapshot().version == 3, (
            'stale manifest must not regress cache; cache.update enforces '
            'strictly-greater on version'
        )
    finally:
        sub.stop(timeout=2.0)


def test_missing_file_is_silent(tmp_path) -> None:
    path = str(tmp_path / 'never_written.json')
    assert read_manifest(path) is None
    cache = PolicyVersionCache(PolicyVersionSnapshot.bootstrap('qwen3-4b-skyrl'))
    sub = FilePollingPolicySubscription(
        cache=cache,
        manifest_path=path,
        poll_interval_s=0.05,
    )
    sub.start()
    time.sleep(0.3)
    sub.stop(timeout=1.0)
    assert cache.snapshot().version == 0
