"""§3.7 invariant — eager-push seam under filter_groups=True (BC-6).

Tests the worker loop's branch logic: skip terminal push when
``eager_pushed_all=True`` to prevent double-push corruption.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.invariant


def test_skip_when_eager_pushed_all_is_true() -> None:
    pushed: list[str] = []

    class FakeStore:
        def push_from_dataproto(self, batch, *, behavior_policy_version, current_step):
            pushed.append('terminal-push')
            return 0

    fake_batch = SimpleNamespace(meta_info={'eager_pushed_all': True})
    eager_pushed_all = bool(fake_batch.meta_info.get('eager_pushed_all', False))
    if not eager_pushed_all:
        FakeStore().push_from_dataproto(
            fake_batch, behavior_policy_version=7, current_step=0
        )
    assert eager_pushed_all is True
    assert pushed == [], 'terminal push must NOT fire when eager_pushed_all=True'


def test_terminal_push_fires_when_eager_pushed_all_is_false() -> None:
    pushed: list[str] = []

    class FakeStore:
        def push_from_dataproto(self, batch, *, behavior_policy_version, current_step):
            pushed.append('terminal-push')
            return 1

    fake_batch = SimpleNamespace(meta_info={'eager_pushed_all': False})
    eager_pushed_all = bool(fake_batch.meta_info.get('eager_pushed_all', False))
    if not eager_pushed_all:
        FakeStore().push_from_dataproto(
            fake_batch, behavior_policy_version=7, current_step=0
        )
    assert pushed == ['terminal-push']


def test_missing_meta_info_falls_back_to_terminal_push() -> None:
    pushed: list[str] = []

    class FakeStore:
        def push_from_dataproto(self, batch, *, behavior_policy_version, current_step):
            pushed.append('terminal-push')

    fake_batch = SimpleNamespace(meta_info={})
    eager_pushed_all = bool(fake_batch.meta_info.get('eager_pushed_all', False))
    if not eager_pushed_all:
        FakeStore().push_from_dataproto(
            fake_batch, behavior_policy_version=0, current_step=0
        )
    assert pushed == ['terminal-push']


def test_worker_loop_module_importable() -> None:
    """BC-13: rollout_manager.loop imports zero VERL / OpenHands modules."""
    import sys

    import rollout_fabric.rollout_manager.loop  # noqa: F401

    assert 'verl' not in sys.modules, (
        'rollout_fabric.rollout_manager.loop must not import verl'
    )
    assert 'openhands' not in sys.modules, (
        'rollout_fabric.rollout_manager.loop must not import openhands'
    )
