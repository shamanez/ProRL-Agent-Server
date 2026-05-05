"""In-memory FIFO bounded buffer of TrainingGroups.

Lifted from ``trainer_integration/verl/verl_custom/replay/trajectory_store.py``
minus ``_pack`` (moved to ``trainer_adapters/verl/pad.py`` per §6.2).

Key invariants preserved:
- Pop-on-sample (§3.6 — queue semantics).
- Whole-group integrity (§3.2 — sampler never splits a group).
- Server-side no-progress detector replaces trainer-side busy-loop.
- Returns unpadded ``TrainingSample`` lists; knows nothing about tensors.
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

from rollout_fabric.schemas.protocols.live_store import NoProgressError

if TYPE_CHECKING:
    from rollout_fabric.schemas.training_sample import TrainingSample

logger = logging.getLogger(__name__)


class InsufficientTrajectoriesError(RuntimeError):
    """Raised when the store cannot satisfy a request even after waiting."""


@dataclass(slots=True)
class StoreMetrics:
    store_size: int
    fill_ratio: float
    num_trajectories: int
    age_p50: float
    age_p95: float
    dropped_by_staleness_total: int
    total_pushes: int


class StoreCore:
    """Process-local replay buffer. Transport-agnostic; wrapped by gRPC server."""

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
        self._max_size = max_size
        self._staleness_cutoff_k = staleness_cutoff_k
        self._no_progress_timeout_s = float(no_progress_timeout_s)
        self._groups: deque[list[TrainingSample]] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._push_cv = threading.Condition(self._lock)
        self._dropped_by_staleness_total = 0
        self._pushes_total = 0

    # ---- ingest -----------------------------------------------------------

    def push_group(self, samples: Sequence[TrainingSample]) -> int:
        """Append one group atomically. All samples must share ``group_uid`` (§3.2)."""
        if not samples:
            raise ValueError('push_group requires at least one sample')
        group = list(samples)
        guid = group[0].group_uid
        for s in group[1:]:
            if s.group_uid != guid:
                raise ValueError(
                    f'all samples in a group must share group_uid; '
                    f"got '{guid}' and '{s.group_uid}'"
                )
        with self._push_cv:
            self._groups.append(group)
            self._pushes_total += 1
            store_size = len(self._groups)
            self._push_cv.notify_all()
        return store_size

    # ---- eviction ---------------------------------------------------------

    def _evict_stale_locked(self, current_step: int) -> int:
        cutoff = self._staleness_cutoff_k
        surviving: deque[list[TrainingSample]] = deque(maxlen=self._max_size)
        dropped = 0
        for group in self._groups:
            age = current_step - group[0].created_at_step
            if age > cutoff:
                dropped += 1
            else:
                surviving.append(group)
        self._groups = surviving
        self._dropped_by_staleness_total += dropped
        return dropped

    def evict_stale(self, current_step: int) -> int:
        with self._lock:
            return self._evict_stale_locked(current_step)

    # ---- sampling ---------------------------------------------------------

    def get_batch(
        self,
        *,
        n_groups: int,
        current_step: int,
        timeout_ms: int,
        rng: random.Random | None = None,
    ) -> list[TrainingSample]:
        """Pop ``n_groups`` non-stale groups. Blocks server-side (BC-4, BC-5).

        Raises :class:`NoProgressError` if no push lands within
        ``no_progress_timeout_s`` while waiting — the producer is wedged.
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
                now = time.monotonic()
                if now >= deadline_no_progress:
                    raise NoProgressError(
                        f'no producer push observed for '
                        f'{self._no_progress_timeout_s:.1f}s; the producer '
                        f'is wedged or the inference pool is dead'
                    )
                wait_s = min(deadline_overall - now, deadline_no_progress - now)
                if wait_s <= 0:
                    raise InsufficientTrajectoriesError(
                        f'store has {len(self._groups)} groups, asked for '
                        f'{n_groups} (timeout_ms={timeout_ms} elapsed)'
                    )
                self._push_cv.wait(timeout=wait_s)
                if self._pushes_total > last_pushes_total:
                    last_pushes_total = self._pushes_total
                    deadline_no_progress = (
                        time.monotonic() + self._no_progress_timeout_s
                    )
            chosen_idx = set(rng.sample(range(len(self._groups)), n_groups))
            groups_list = list(self._groups)
            chosen = [groups_list[i] for i in sorted(chosen_idx)]
            remaining = [g for i, g in enumerate(groups_list) if i not in chosen_idx]
            self._groups.clear()
            self._groups.extend(remaining)
            return [s for group in chosen for s in group]

    # ---- introspection ----------------------------------------------------

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
            p95 = float(ages[max(0, int(len(ages) * 0.95) - 1)])
        else:
            p50 = p95 = 0.0
        return StoreMetrics(
            store_size=size,
            fill_ratio=size / float(self._max_size),
            num_trajectories=n_traj,
            age_p50=p50,
            age_p95=p95,
            dropped_by_staleness_total=dropped,
            total_pushes=pushes,
        )

    def _snapshot_groups(self) -> list[list[TrainingSample]]:
        with self._lock:
            return [list(g) for g in self._groups]
