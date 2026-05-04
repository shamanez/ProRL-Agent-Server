"""§A.5 — ReplayArchive protocol.

Append-only, durable, queryable record of every episode the fabric ever
produced. Source of truth for offline RL, distillation, curation, audit,
reproducibility.

This slot does not exist today; lands at S3. The minimum data unit is
:class:`EpisodeRecord` (§6.1). ``TrainingSample`` derivation can run on
demand or be pre-cached as secondary Parquet.

Different product from the LiveStore (§4.3, §7): unbounded durable
storage, append-only, range-and-predicate queries, weeks-to-months
horizon. Tee from the producer, not from the live store.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from schemas.episode_record import EpisodeRecord, TrustLevel
from schemas.training_sample import TrainingSample


@dataclass(slots=True)
class AppendResult:
    accepted: int
    duplicates: int
    episode_uids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FilterSpec:
    """Range and predicate filters for ``query``.

    All fields default to "no filter". Combined as logical AND.

    The minimum predicate set per §S3 Validation:
    ``policy_id``, ``policy_version`` range, ``environment_id``,
    ``split``, time range. ``reward_predicate`` and ``trust_levels`` are
    optional extensions.
    """

    policy_id: str | None = None
    policy_version_min: int | None = None
    policy_version_max: int | None = None
    environment_id: str | None = None
    split: str | None = None
    started_after: str | None = None  # ISO-8601 UTC
    finished_before: str | None = None
    reward_min: float | None = None
    reward_max: float | None = None
    trust_levels: tuple[TrustLevel, ...] | None = None
    task_ids: tuple[str, ...] | None = None


class ReplayArchive(Protocol):
    def append_episodes(
        self,
        records: list[EpisodeRecord],
    ) -> AppendResult: ...

    def query(
        self,
        filter_spec: FilterSpec,
    ) -> Iterable[EpisodeRecord]: ...

    def derive_training_samples(
        self,
        episode_uids: list[str],
        schema_version: str,
    ) -> list[TrainingSample]: ...
