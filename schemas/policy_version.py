"""Policy-version snapshot primitive — immutable snapshot + atomic ref swap.

Contract
--------
A producer (rollout worker) generates groups of ``n`` sibling trajectories.
Every row of one group **must** be stamped with the *same*
``behavior_policy_version`` (group integrity, §3.2). Across groups, the
producer must observe the freshest published version available at group
dispatch time (per-row freshness, §3.5).

The primitive
-------------
The cache holds **one immutable snapshot** in a single attribute. The
subscription thread allocates a brand-new ``PolicyVersionSnapshot`` on
each update and assigns it to that attribute via a single ``STORE_ATTR``
bytecode — GIL-atomic in CPython; reference-atomic under nogil via the
per-object guard. Concurrent ``LOAD_ATTR`` readers see either the old or
the new reference, never a partially-mutated record.

Snapshot-per-group semantics
-----------------------------
Group dispatch calls :meth:`PolicyVersionCache.snapshot` exactly once at
group start; the returned reference is captured in a local variable and
its ``.version`` / ``.adapter_uri`` stamped on every row of the group:

    snap = cache.snapshot()          # one LOAD_ATTR
    for row in group:
        row.behavior_policy_version = snap.version

If a publish lands mid-group the cache's reference flips, but ``snap``
(the local) still points to the old immutable snapshot — group is
uniformly tagged with the old version.  This satisfies §3.2 + §3.5 with
zero locks and zero coordination.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PolicyVersionSnapshot:
    """Immutable per-publish record. Constructed once, swapped atomically."""

    policy_id: str
    version: int
    adapter_uri: str
    received_at: float

    @classmethod
    def bootstrap(cls, policy_id: str) -> 'PolicyVersionSnapshot':
        return cls(
            policy_id=policy_id,
            version=0,
            adapter_uri='',
            received_at=time.monotonic(),
        )


class PolicyVersionCache:
    """Single-writer / many-reader cache of the latest snapshot.

    Reads are lock-free: one Python attribute load.
    Writes are GIL-atomic: one attribute store of a freshly constructed
    snapshot (single-writer assumption — only the subscription thread
    calls ``update``).
    """

    __slots__ = ('_current', '_policy_id')

    def __init__(self, initial: PolicyVersionSnapshot) -> None:
        self._policy_id = initial.policy_id
        self._current = initial

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def snapshot(self) -> PolicyVersionSnapshot:
        """Return the current snapshot. Lock-free; one attribute load."""
        return self._current

    def update(self, new_snap: PolicyVersionSnapshot) -> bool:
        """Atomically swap if ``new_snap`` is strictly fresher.

        Returns ``True`` if the swap happened; ``False`` if rejected as stale.
        Raises ``ValueError`` on wrong ``policy_id``.
        """
        if new_snap.policy_id != self._policy_id:
            raise ValueError(
                f'PolicyVersionCache for policy_id={self._policy_id!r} '
                f'rejected update with policy_id={new_snap.policy_id!r}'
            )
        cur = self._current
        if new_snap.version <= cur.version:
            return False
        self._current = new_snap  # GIL-atomic STORE_ATTR
        return True
