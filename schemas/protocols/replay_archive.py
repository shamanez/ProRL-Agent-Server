"""§A.5 — ReplayArchive protocol. Lands at S3."""

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
    policy_id: str | None = None
    policy_version_min: int | None = None
    policy_version_max: int | None = None
    environment_id: str | None = None
    split: str | None = None
    started_after: str | None = None
    finished_before: str | None = None
    reward_min: float | None = None
    reward_max: float | None = None
    trust_levels: tuple[TrustLevel, ...] | None = None
    task_ids: tuple[str, ...] | None = None


class ReplayArchive(Protocol):
    def append_episodes(self, records: list[EpisodeRecord]) -> AppendResult: ...

    def query(self, filter_spec: FilterSpec) -> Iterable[EpisodeRecord]: ...

    def derive_training_samples(
        self, episode_uids: list[str], schema_version: str
    ) -> list[TrainingSample]: ...
