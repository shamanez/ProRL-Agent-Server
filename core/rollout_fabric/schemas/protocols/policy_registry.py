"""§A.7 — PolicyRegistry protocol.

§3.3 abort gate: ``success`` implies ``endpoints_failed == 0``.
Partial publish is failure, not degraded mode.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from rollout_fabric.schemas.policy_version import PolicyVersionSnapshot
from rollout_fabric.schemas.protocols.live_store import Ack


@dataclass(slots=True)
class VersionInfo:
    policy_id: str
    version: int
    adapter_uri: str
    published_at: str  # ISO-8601 UTC


@dataclass(slots=True)
class PublishResult:
    success: bool
    endpoints_ok: int
    endpoints_failed: int
    latency_s: float
    error: str | None = None


class PolicyRegistry(Protocol):
    def publish_policy_version(
        self, policy_id: str, version: int, adapter_uri: str, trainer_id: str
    ) -> PublishResult: ...

    def get_latest_version(self, policy_id: str) -> VersionInfo: ...

    def subscribe_version_updates(
        self, policy_id: str
    ) -> Iterable[PolicyVersionSnapshot]: ...

    def register_policy_namespace(
        self,
        policy_id: str,
        base_model_id: str,
        tokenizer_id: str,
        adapter_storage_root: str,
    ) -> Ack: ...
