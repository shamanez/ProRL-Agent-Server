"""In-memory FIFO bounded buffer of TrainingGroups.

Lifted from ``trainer_integration/verl/verl_custom/replay/trajectory_store.py``
lines 1–465, minus ``_pack`` (which moved to
``trainer_adapters/verl/pad.py``). The data structure is the same:

* ``deque(maxlen=N)`` of groups.
* ``threading.Lock`` over all mutations.
* K-staleness eviction at sample time.
* Pop-on-sample (§3.6 — queue semantics, not with-replacement).
* Whole-group integrity (§3.2 — sampler never splits a group).

The differences from the legacy module:

1. Operates on :class:`schemas.training_sample.TrainingSample` (§6.2 wire
   shape), not the legacy ``TrajectoryRecord``. Field renames:
   ``prompt_ids → prompt_token_ids``, ``response_ids → response_token_ids``,
   ``response_log_probs → behavior_log_probs``, ``prompt_uid → sample_uid``.

2. **Knows nothing about tensor shapes.** The legacy ``_pack`` lifted to
   the trainer adapter; this module returns unpadded ``TrainingSample``
   lists. Per the §6.2 padding stance, every adapter pads to its own
   compute shape.

3. Server-side no-progress detector. ``get_batch`` blocks up to
   ``timeout_ms`` waiting for ``n_groups`` non-stale groups; if
   ``total_pushes`` does not advance within ``no_progress_timeout_s``,
   raises :class:`NoProgressError`. Replaces the trainer-side busy-loop
   from ``continuous_producer.py:357-392``.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from schemas.protocols.live_store import NoProgressError

if TYPE_CHECKING:
    from schemas.training_sample import TrainingSample

logger = logging.getLogger(__name__)


class InsufficientTrajectoriesError(RuntimeError):
    """Raised when the store cannot satisfy a request even after a wait."""


@dataclass(slots=True)
class StoreMetrics:
    """Snapshot of store-level counters for ``GetMetrics``."""

    store_size: int
    fill_ratio: float
    num_trajectories: int
    age_p50: float
    age_p95: float
    dropped_by_staleness_total: int
    total_pushes: int


class StoreCore:
    """Process-local replay buffer of groups.

    The class is transport-agnostic; the gRPC server in
    ``live_store/server.py`` wraps it. Test it without a network.

    Parameters
    ----------
    max_size:
        Maximum number of groups; FIFO eviction on overflow.
    staleness_cutoff_k:
        Groups whose ``current_step - created_at_step > k`` are dropped
        at sample time.
    no_progress_timeout_s:
        Seconds without forward progress on ``total_pushes`` before
        ``get_batch`` raises :class:`NoProgressError`. Replaces the
        trainer-side busy-loop. The semantic is "producer is wedged",
        not "producer is slow"; the deadline resets whenever any push
        lands.
    """

    def __init__(
        self,
        *,
        max_size: int,
        staleness_cutoff_k: int,
        no_progress_timeout_s: float = 1800.0,
    ) -> None:
        if max_size <= 0:
            raise ValueError(f'max_size must be > 0, got {max_size}')
        if staleness_cutoff_k < 0:
            raise ValueError(
                f'staleness_cutoff_k must be >= 0, got {staleness_cutoff_k}'
            )
        if no_progress_timeout_s <= 0:
            raise ValueError(
                f'no_progress_timeout_s must be > 0, got {no_progress_timeout_s}'
            )
        self._max_size = max_size
        self._staleness_cutoff_k = staleness_cutoff_k
        self._no_progress_timeout_s = float(no_progress_timeout_s)
        self._groups: deque[list['TrainingSample']] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        # Condition so push() wakes up a blocked get_batch() without busy-poll.
        self._push_cv = threading.Condition(self._lock)
        self._dropped_by_staleness_total = 0
        self._last_sample_ages: list[int] = []
        self._pushes_total = 0

    # ---- ingest -------------------------------------------------------------

    def push_group(self, samples: Sequence['TrainingSample']) -> int:
        """Append one group. All samples must share ``group_uid`` (§3.2).

        Returns the post-push store size.
        """
        if not samples:
            raise ValueError('push_group requires at least one sample')
        group = list(samples)
        guid = group[0].group_uid
        for s in group[1:]:
            if s.group_uid != guid:
                raise ValueError(
                    f'all samples in a group must share group_uid; got '
                    f"'{guid}' and '{s.group_uid}'"
                )
        with self._push_cv:
            self._groups.append(group)
            self._pushes_total += 1
            store_size = len(self._groups)
            self._push_cv.notify_all()
        return store_size

    # ---- eviction -----------------------------------------------------------

    def _evict_stale_locked(self, current_step: int) -> int:
        cutoff = self._staleness_cutoff_k
        surviving: deque[list['TrainingSample']] = deque(maxlen=self._max_size)
        dropped = 0
        for group in self._groups:
            age = current_step - group[0].created_at_step
            if age > cutoff:
                dropped += 1
            else:
                surviving.append(group)
        self._groups = surviving
        # condition variable is rebound to the lock; recreating is unnecessary
        # since deque was replaced on the SAME lock holder.
        self._dropped_by_staleness_total += dropped
        return dropped

    def evict_stale(self, current_step: int) -> int:
        with self._lock:
            return self._evict_stale_locked(current_step)

    # ---- sampling -----------------------------------------------------------

    def get_batch(
        self,
        *,
        n_groups: int,
        current_step: int,
        timeout_ms: int,
        rng: random.Random | None = None,
    ) -> list['TrainingSample']:
        """Pop ``n_groups`` non-stale groups from the buffer.

        Blocks server-side up to ``timeout_ms`` for the buffer to warm
        up. The "no progress" deadline resets every time any push lands;
        if no push lands for ``no_progress_timeout_s`` while we wait,
        :class:`NoProgressError` is raised.

        Returns an unpadded flat list of :class:`TrainingSample`. Per
        §6.2 the trainer adapter pads to its own shape.
        """
        if n_groups <= 0:
            raise ValueError(f'n_groups must be > 0, got {n_groups}')
        rng = rng or random.Random()
        deadline_overall = time.monotonic() + max(timeout_ms / 1000.0, 0.0)
        deadline_no_progress = time.monotonic() + self._no_progress_timeout_s
        with self._push_cv:
            last_pushes_total = self._pushes_total
            while True:
                self._evict_stale_locked(current_step)
                if len(self._groups) >= n_groups:
                    break
                # Determine how long to wait until a push lands.
                now = time.monotonic()
                if now >= deadline_no_progress:
                    raise NoProgressError(
                        f'no producer push observed for '
                        f'{self._no_progress_timeout_s:.1f}s; the producer '
                        f'is wedged or the inference pool is dead'
                    )
                wait_s = min(
                    deadline_overall - now,
                    deadline_no_progress - now,
                )
                if wait_s <= 0:
                    # Overall timeout fired; surface as InsufficientTrajectoriesError
                    # (callers can decide to retry; not a hard abort).
                    raise InsufficientTrajectoriesError(
                        f'store has {len(self._groups)} groups, asked for '
                        f'{n_groups} (timeout_ms={timeout_ms} elapsed)'
                    )
                self._push_cv.wait(timeout=wait_s)
                # Reset the no-progress deadline if the producer pushed.
                if self._pushes_total > last_pushes_total:
                    last_pushes_total = self._pushes_total
                    deadline_no_progress = (
                        time.monotonic() + self._no_progress_timeout_s
                    )
            # Pick + pop the chosen groups.
            chosen_idx = set(rng.sample(range(len(self._groups)), n_groups))
            groups_list = list(self._groups)
            chosen = [groups_list[i] for i in sorted(chosen_idx)]
            remaining = [g for i, g in enumerate(groups_list) if i not in chosen_idx]
            self._groups.clear()
            self._groups.extend(remaining)
            samples_flat = [s for group in chosen for s in group]
            self._last_sample_ages = [
                current_step - s.created_at_step for s in samples_flat
            ]
        return samples_flat

    # ---- introspection -----------------------------------------------------

    def num_groups(self) -> int:
        with self._lock:
            return len(self._groups)

    def total_pushes(self) -> int:
        with self._lock:
            return self._pushes_total

    def num_fresh_groups(self, current_step: int) -> int:
        cutoff = self._staleness_cutoff_k
        with self._lock:
            return sum(
                1 for g in self._groups if current_step - g[0].created_at_step <= cutoff
            )

    def num_trajectories(self) -> int:
        with self._lock:
            return sum(len(g) for g in self._groups)

    def metrics(self, current_step: int) -> StoreMetrics:
        with self._lock:
            groups = list(self._groups)
            dropped = self._dropped_by_staleness_total
            pushes = self._pushes_total
        size = len(groups)
        n_traj = sum(len(g) for g in groups)
        if groups:
            ages = sorted(current_step - g[0].created_at_step for g in groups)
            p50 = float(ages[len(ages) // 2])
            p95_idx = max(0, int(len(ages) * 0.95) - 1)
            p95 = float(ages[p95_idx])
        else:
            p50 = 0.0
            p95 = 0.0
        return StoreMetrics(
            store_size=size,
            fill_ratio=size / float(self._max_size),
            num_trajectories=n_traj,
            age_p50=p50,
            age_p95=p95,
            dropped_by_staleness_total=dropped,
            total_pushes=pushes,
        )

    # ---- test helpers -------------------------------------------------------

    def _snapshot_groups(self) -> list[list['TrainingSample']]:
        with self._lock:
            return [list(g) for g in self._groups]
