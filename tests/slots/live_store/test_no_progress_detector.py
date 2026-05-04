"""S1 — no-progress detector raises NoProgressError after the timeout window.

The server is configured with ``no_progress_timeout_s=2.0`` for tests;
calling ``get_batch`` on an empty store with no producer pushing must
raise :class:`NoProgressError` (not :class:`KeyboardInterrupt`, not a
silent stall).
"""

from __future__ import annotations

import threading
import time

import pytest

from schemas.protocols.live_store import NoProgressError

from .conftest import make_sample

pytestmark = pytest.mark.contract


def test_no_progress_raises_after_timeout(live_store_client) -> None:
    """Empty store + no producer ⇒ NoProgressError after no_progress_timeout_s."""
    t0 = time.monotonic()
    with pytest.raises(NoProgressError):
        live_store_client.get_batch(n_groups=1, current_step=0, timeout_ms=10_000)
    elapsed = time.monotonic() - t0
    # Server's configured timeout is 2.0s; allow some slack for thread
    # wakeups but reject anything <1.5s (would mean we returned without
    # waiting) or >5s (would mean the deadline didn't fire).
    assert 1.5 <= elapsed <= 5.0, f'elapsed={elapsed:.2f}s outside expected band'


def test_progress_resets_no_progress_deadline(live_store_client) -> None:
    """A late push satisfies get_batch even past the no-progress window.

    The producer pushes 1 group at t=1.0s; the server's no-progress
    deadline (2.0s from start) must reset on that push. The
    get_batch(n_groups=1) wakes up immediately after the push lands.
    """
    pushed_at = []

    def late_push() -> None:
        time.sleep(1.0)
        live_store_client.push_group([make_sample(sample_uid='s0', group_uid='g0')])
        pushed_at.append(time.monotonic())

    t0 = time.monotonic()
    pusher = threading.Thread(target=late_push)
    pusher.start()
    samples = live_store_client.get_batch(n_groups=1, current_step=0, timeout_ms=10_000)
    elapsed = time.monotonic() - t0
    pusher.join()
    assert len(samples) == 1
    # The wake should land shortly after the push, NOT after the
    # original 2.0s no-progress deadline.
    assert elapsed < 2.0, f'elapsed={elapsed:.2f}s — push did not wake the waiter'
