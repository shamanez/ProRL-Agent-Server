"""§A.4 — LiveStore protocol.

Bounded, low-latency hot buffer between RolloutManager and TrainerAdapter.
Pop-on-sample (§3.6). No-progress detector server-side (§S1).
Unpadded wire (§6.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rollout_fabric.schemas.training_sample import BatchResult, PushResult, TrainingSample


class NoProgressError(RuntimeError):
    """Raised when no producer push landed within ``no_progress_timeout_s``.

    Distinct from "store empty" — signals the producer is wedged.
    Trainer should abort, not poll.
    """


@dataclass(slots=True)
class Ack:
    ok: bool
    detail: str = ''


@dataclass(slots=True)
class BackpressureHint:
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

    def notify_policy_version(self, version: int, adapter_uri: str) -> Ack: ...
