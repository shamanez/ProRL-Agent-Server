"""S1 — no-progress detector raises NoProgressError (BC-5)."""

from __future__ import annotations

import threading
import time

import pytest

from rollout_fabric.schemas.protocols.live_store import NoProgressError

from .conftest import make_sample

pytestmark = pytest.mark.contract


def test_no_progress_raises_after_timeout(live_store_client) -> None:
    """Empty store + no producer → NoProgressError within no_progress_timeout_s=2.0s."""
    t0 = time.monotonic()
    with pytest.raises(NoProgressError):
        live_store_client.get_batch(n_groups=1, current_step=0, timeout_ms=10_000)
    elapsed = time.monotonic() - t0
    assert 1.5 <= elapsed <= 5.0, f'elapsed={elapsed:.2f}s outside expected band'


def test_progress_resets_no_progress_deadline(live_store_client) -> None:
    """A late push wakes get_batch before the no-progress deadline fires."""

    def late_push():
        time.sleep(1.0)
        live_store_client.push_group([make_sample(sample_uid='s0', group_uid='g0')])

    t0 = time.monotonic()
    pusher = threading.Thread(target=late_push)
    pusher.start()
    samples = live_store_client.get_batch(n_groups=1, current_step=0, timeout_ms=10_000)
    elapsed = time.monotonic() - t0
    pusher.join()
    assert len(samples) == 1
    assert elapsed < 2.0, f'elapsed={elapsed:.2f}s — push did not wake the waiter'
