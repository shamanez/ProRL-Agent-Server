"""§A.4 — LiveStore protocol.

Bounded, low-latency, hot buffer between RolloutWorker and TrainerAdapter.
Operates in groups (``n`` siblings together; §3.2), pops on sample (§3.6),
evicts by staleness, returns batches sized for the trainer step.

Padding stance (§6.2): ``get_batch`` returns **unpadded** ``TrainingSample``
records. Padding / sequence-packing is the trainer adapter's job.

No-progress detector (§S1 scope): server-side blocking on ``get_batch``
up to ``timeout_ms``; if ``total_pushes`` does not grow within
``no_progress_timeout_s``, raise :class:`NoProgressError`. The trainer is
no longer responsible for the busy-loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from schemas.training_sample import (
    BatchResult,
    PushResult,
    TrainingSample,
)


class NoProgressError(RuntimeError):
    """Raised by the LiveStore when no producer push has landed within
    the configured ``no_progress_timeout_s`` window.

    Distinct from "store empty" — this signals the producer is wedged or
    the pool is dead; trainer should abort, not poll.
    """


@dataclass(slots=True)
class Ack:
    ok: bool
    detail: str = ''


@dataclass(slots=True)
class BackpressureHint:
    """Advisory hint returned on ``push_group``.

    ``hint_ms = 0`` means no backpressure. Positive values are the
    suggested pause before the next push. Multi-producer-friendly
    (S5+); single producer (S1-S4) can ignore.
    """

    hint_ms: int


@dataclass(slots=True)
class StoreMetrics:
    store_size: int
    fill_ratio: float
    num_trajectories: int
    age_p50: float
    age_p95: float
    dropped_by_staleness_total: int
    total_pushes: int


class LiveStore(Protocol):
    def push_group(
        self,
        records: list[TrainingSample],
        group_uid: str,
        producer_id: str,
    ) -> PushResult: ...

    def get_batch(
        self,
        n_groups: int,
        current_step: int,
        staleness_cutoff_k: int,
        timeout_ms: int,
    ) -> BatchResult: ...

    def get_metrics(self, current_step: int) -> StoreMetrics: ...

    def notify_policy_version(
        self,
        version: int,
        adapter_uri: str,
    ) -> Ack: ...
