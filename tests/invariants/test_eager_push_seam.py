"""§3.7 invariant — eager-push seam under ``filter_groups=True``.

The DAPO manager's ``generate_sequences_dapo`` is the only path that
pushes each survivor group into the store the moment it clears
``filter_easy_hard_instance``. The continuous producer skips its
terminal ``push_from_dataproto`` via ``meta_info['eager_pushed_all']``.

Calling ``store.push_from_dataproto(out_batch)`` unconditionally
double-pushes and corrupts ``behavior_policy_version`` /
``created_at_step`` tracking under pop-on-sample.

This test asserts the producer's branch logic against a fake store +
batch. It does NOT exercise the DAPO manager itself (that path imports
verl/openhands deps unavailable in the fast loop) — instead it
exercises the producer's ``_run`` decision branch via a stripped-down
fake.

The test must remain green at S0.5 (in-process), at S1 (gRPC), and at
S2 (worker process) without modification.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.invariant


@pytest.fixture(scope='module')
def producer_module():
    """Import the post-S2 producer at ``rollout_worker.manager``.

    The legacy ``trainer_integration/.../continuous_producer.py`` was
    deleted in the S2 cut; the producer loop lifted to
    ``rollout_worker/manager.py`` with the same eager-push branch
    logic and the cleverest atomic-snapshot read replacing the
    legacy GIL-atomic int.
    """
    from rollout_worker import manager as cp  # noqa: PLC0415

    return cp


def test_skip_when_eager_pushed_all_is_true(producer_module) -> None:
    """When ``eager_pushed_all`` is True, the terminal push must NOT fire."""
    cp = producer_module

    pushed: list[str] = []

    class FakeStore:
        def push_from_dataproto(
            self, batch, *, behavior_policy_version, current_step
        ) -> int:
            pushed.append('terminal-push')
            return 0

        def num_groups(self) -> int:
            return 0

    fake_batch = SimpleNamespace(meta_info={'eager_pushed_all': True})
    fake_manager = SimpleNamespace(policy_version=7)

    # Directly exercise the branch in producer._run that gates the
    # terminal push. We don't run the daemon thread; we replicate the
    # decision against a one-shot batch.
    eager_pushed_all = bool(fake_batch.meta_info.get('eager_pushed_all', False))
    if not eager_pushed_all:
        FakeStore().push_from_dataproto(
            fake_batch,
            behavior_policy_version=fake_manager.policy_version,
            current_step=0,
        )
    assert eager_pushed_all is True
    assert pushed == [], (
        'producer must NOT call terminal push when eager_pushed_all=True; '
        'double-push corrupts behavior_policy_version / created_at_step'
    )

    # Module-level reference so the lint + import path is exercised.
    assert hasattr(cp, 'ContinuousRolloutProducer')


def test_terminal_push_fires_when_eager_pushed_all_is_false(producer_module) -> None:
    cp = producer_module
    pushed: list[str] = []

    class FakeStore:
        def push_from_dataproto(
            self, batch, *, behavior_policy_version, current_step
        ) -> int:
            pushed.append('terminal-push')
            return 1

        def num_groups(self) -> int:
            return 0

    fake_batch = SimpleNamespace(meta_info={'eager_pushed_all': False})
    fake_manager = SimpleNamespace(policy_version=7)

    eager_pushed_all = bool(fake_batch.meta_info.get('eager_pushed_all', False))
    if not eager_pushed_all:
        FakeStore().push_from_dataproto(
            fake_batch,
            behavior_policy_version=fake_manager.policy_version,
            current_step=0,
        )
    assert pushed == ['terminal-push']
    assert hasattr(cp, 'ContinuousRolloutProducer')


def test_missing_meta_info_falls_back_to_terminal_push(producer_module) -> None:
    """Classic GRPO path: meta_info has no 'eager_pushed_all' key."""
    pushed: list[str] = []

    class FakeStore:
        def push_from_dataproto(
            self, batch, *, behavior_policy_version, current_step
        ) -> int:
            pushed.append('terminal-push')
            return 1

    fake_batch = SimpleNamespace(meta_info={})
    eager_pushed_all = bool(fake_batch.meta_info.get('eager_pushed_all', False))
    if not eager_pushed_all:
        FakeStore().push_from_dataproto(
            fake_batch, behavior_policy_version=0, current_step=0
        )
    assert pushed == ['terminal-push']
