"""Policy-version snapshot primitive — immutable snapshot + atomic ref swap.

This is the cleverest available primitive for the §3.5 invariant
("``behavior_policy_version`` stamped per row, GIL-atomic write") in the
post-migration fabric. Its design rationale, contract, and why it strictly
dominates today's GIL-atomic single-int pattern (CLAUDE.md gotcha 8) are
written here in full because every reader of this file should understand
why this is *not* a lock and *not* a ``threading.Event`` and is in fact the
correct primitive — ``rollout_fabric.md`` operating-principle 6 makes this
the load-bearing piece for §3.2 + §3.5 simultaneously.

The contract
------------

A producer (rollout worker) generates groups of ``n`` sibling trajectories
(GRPO/DAPO sense). Every row of one group **must** be stamped with the
*same* ``behavior_policy_version`` (group integrity, §3.2). Across groups,
the producer must observe the freshest published version available at
group dispatch time (per-row freshness, §3.5). A separate subscription
thread (file mtime poll at S2; gRPC stream at S4) is the **only** writer.

The naive solutions and why each is wrong
-----------------------------------------

1. ``self.policy_version: int`` updated cross-thread — OK in CPython for a
   single int (one ``STORE_FAST``), but breaks the moment we widen to
   ``(version, adapter_uri)``: two stores, two loads, torn read possible.

2. ``threading.Lock`` around the tuple — correct, but every dispatch read
   pays a lock acquire. Publish rate is ~1/min, dispatch rate is
   ~1k/min; this is paying contention cost for a write that almost never
   happens. More importantly: it loses *snapshot-per-group* semantics —
   a multi-row dispatch loop has to either lock once at the top (good,
   but indistinguishable from this primitive in effect), or lock per row
   (creates a window where row k stamps v and row k+1 stamps v+1 inside
   the same group, **violating §3.2**).

3. ``threading.Event`` + manual coordination — solves nothing the lock
   doesn't, adds complexity.

The primitive used here
-----------------------

The cache holds **one immutable snapshot** in a single attribute. The
subscription thread allocates a brand-new ``PolicyVersionSnapshot`` on
each update and assigns it to that attribute. CPython compiles
``self._current = new_snap`` to a single ``STORE_ATTR`` bytecode, which
holds the GIL for its duration; concurrent ``LOAD_ATTR`` readers see
either the old or the new reference, never a partially-mutated record.
Because the snapshot is ``frozen=True``, readers cannot observe a
half-updated tuple — the snapshot was constructed in full *before* the
publish thread did the swap.

This generalizes today's "single int is GIL-atomic" pattern to *any*
record shape: every field of the snapshot is consistent because the
swap publishes them as one indivisible reference.

Snapshot-per-group semantics for free
-------------------------------------

Group dispatch calls :meth:`PolicyVersionCache.snapshot` exactly once at
group start; the returned reference is captured in a local variable and
its ``.version`` / ``.adapter_uri`` are stamped on every row of the
group:

    snap = cache.snapshot()                  # one LOAD_ATTR
    for row in group:
        row.behavior_policy_version = snap.version
        row.adapter_uri              = snap.adapter_uri

If a publish lands mid-group, the cache's internal reference flips, but
``snap`` (the local) still points to the *old* immutable snapshot. The
group is uniformly tagged with the old version. This satisfies §3.2
(group integrity) and §3.5 (per-row stamping is the version active when
this group started) with **zero locks** and **zero coordination** — the
benign race the temporal IS correction is designed for.

Forward path: nogil / free-threaded Python
------------------------------------------

In nogil Python (PEP 703), single attribute load/store is no longer
implicitly serialized by the GIL. The CPython implementation of
``STORE_ATTR`` for an instance attribute under nogil uses a per-object
lock (or in some forks, atomic tagged-pointer stores) so that the
guarantee preserved here ("readers see either the old or the new ref,
never an intermediate") still holds. This pattern therefore *survives*
the migration to nogil; an integer-write pattern does not necessarily,
because integer object boxing under nogil can produce torn reads on
larger int values.

Single-writer assumption
------------------------

This primitive assumes **one writer thread** (the subscription thread).
Multiple-writer support would require atomic compare-and-swap, which
Python does not provide; ``self._current = new_snap`` would otherwise
race when two writers concurrently produce snapshots derived from
different baselines. This matches the fabric's design: PolicyRegistry
(§A.7) is the single source of truth and produces a single update
stream per ``policy_id``. If, in S8 federation, multiple registries
fan out to the same worker, the worker must demultiplex onto separate
caches keyed by ``policy_id``.

Per-``policy_id`` namespacing
-----------------------------

When multiple ``policy_id`` namespaces are active (S6 onward), keep one
:class:`PolicyVersionCache` per ``policy_id``. The cache itself is not
keyed because per-key dict updates are not atomic; per-cache attribute
swaps are.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PolicyVersionSnapshot:
    """Immutable per-publish record. Constructed once, swapped atomically.

    All fields are populated at construction. ``frozen=True`` is enforced
    so consumers cannot observe a partially-mutated snapshot; the only
    way to "change" the contents is to allocate a new ``PolicyVersionSnapshot``
    and atomically swap the cache's reference.

    Attributes
    ----------
    policy_id:
        Logical policy name (e.g. ``"qwen3-4b-skyrl"``). Stable across
        versions of the same policy. Required by §6.1 / §6.2 provenance
        fields.
    version:
        Monotonic per ``policy_id``. ``0`` is reserved for the bootstrap
        snapshot (no publish observed yet).
    adapter_uri:
        Storage URI of the LoRA adapter blob for this version. Local FS
        (``file:///...``) at S2/S4, NFS (``nfs://...``) or S3 (``s3://...``)
        at S5+.
    received_at:
        ``time.monotonic()`` at the moment the subscription thread received
        this update. Used to compute publish-to-cache-update latency
        histograms (post-S4 checklist item 18).
    """

    policy_id: str
    version: int
    adapter_uri: str
    received_at: float

    @classmethod
    def bootstrap(cls, policy_id: str) -> 'PolicyVersionSnapshot':
        """Construct the version-0 placeholder snapshot.

        Used at worker startup before any publish has been observed.
        ``adapter_uri`` is empty; consumers should treat ``version == 0``
        as "policy is the base model only, no adapter yet".
        """
        return cls(
            policy_id=policy_id,
            version=0,
            adapter_uri='',
            received_at=time.monotonic(),
        )


class PolicyVersionCache:
    """Single-writer / many-reader cache of the latest snapshot.

    Reads are lock-free: one Python attribute load returning an immutable
    snapshot. Writes are atomic-by-virtue-of-CPython-attribute-store: one
    Python attribute store of a freshly constructed snapshot.

    Concurrency model
    -----------------
    * Exactly **one writer thread** (the subscription thread) calls
      :meth:`update`.
    * **Many reader threads** (group dispatch threads) call
      :meth:`snapshot`.
    * No locks on the read or write path. The single-attribute store
      and load are GIL-atomic in CPython 3.x and remain reference-atomic
      under nogil Python via ``STORE_ATTR``'s per-object guard.

    Parameters
    ----------
    initial:
        The bootstrap snapshot. Must not be ``None``; callers wanting
        "no data yet" should pass :meth:`PolicyVersionSnapshot.bootstrap`.
    """

    __slots__ = ('_current', '_policy_id')

    def __init__(self, initial: PolicyVersionSnapshot) -> None:
        self._policy_id = initial.policy_id
        self._current = initial

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def snapshot(self) -> PolicyVersionSnapshot:
        """Return the current snapshot. Lock-free; one attribute load.

        The returned object is immutable; the caller can hold it for the
        lifetime of a group dispatch and stamp every row of the group with
        its ``version`` / ``adapter_uri`` knowing the whole group will be
        consistent (§3.2 group integrity). Subsequent calls may return a
        newer snapshot if a publish landed in between.
        """
        return self._current

    def update(self, new_snap: PolicyVersionSnapshot) -> bool:
        """Atomically swap the cache to ``new_snap`` if it is fresher.

        Single-writer contract: only the subscription thread calls this.
        The check is:

        * ``new_snap.policy_id`` must match this cache's ``policy_id``;
          otherwise :class:`ValueError`.
        * ``new_snap.version`` must be **strictly greater** than the
          current snapshot's version; out-of-order replays from the
          subscription stream are rejected (idempotent on the wire).

        Returns ``True`` if the swap happened, ``False`` if the update was
        rejected as stale.
        """
        if new_snap.policy_id != self._policy_id:
            raise ValueError(
                f'PolicyVersionCache for policy_id={self._policy_id!r} '
                f'rejected update with policy_id={new_snap.policy_id!r}; '
                f'use one cache per policy_id namespace'
            )
        # Read once into a local; subsequent comparison and store happen
        # against this consistent baseline. The single-writer assumption
        # means no other thread can race the comparison.
        cur = self._current
        if new_snap.version <= cur.version:
            return False
        # The single STORE_ATTR below is the atomic publish point. After
        # this opcode completes, every reader on every dispatch thread
        # observing the cache sees the new snapshot — no torn reads,
        # no half-updated tuple.
        self._current = new_snap
        return True
