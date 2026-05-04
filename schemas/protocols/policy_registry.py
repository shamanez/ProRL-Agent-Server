"""§A.7 — PolicyRegistry protocol.

Source of truth for the active policy version set. Receives publishes
from TrainerAdapter, fans out to InferenceBackend (``/reload_lora``),
LiveStore (metrics tagging), and RolloutWorkers (subscriptions).

Today: trainer-owned ``policy_version`` int + direct ``/reload_lora``
fanout. S2: file-backed minimal registry (worker polls). S4: full
gRPC registry with streaming subscriptions; abort gate moves into
``fanout``.

Invariant §3.3 (load-bearing): ``success`` implies ``endpoints_failed
== 0``. Partial publish is failure, not degraded mode.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from schemas.policy_version import PolicyVersionSnapshot
from schemas.protocols.live_store import Ack


@dataclass(slots=True)
class VersionInfo:
    policy_id: str
    version: int
    adapter_uri: str
    published_at: str  # ISO-8601 UTC


@dataclass(slots=True)
class PublishResult:
    """§3.3 abort gate: ``success`` implies ``endpoints_failed == 0``.

    The fanout impl raises if any pool child fails; the trainer aborts.
    """

    success: bool
    endpoints_ok: int
    endpoints_failed: int
    latency_s: float
    error: str | None = None


class PolicyRegistry(Protocol):
    def publish_policy_version(
        self,
        policy_id: str,
        version: int,
        adapter_uri: str,
        trainer_id: str,
    ) -> PublishResult: ...

    def get_latest_version(self, policy_id: str) -> VersionInfo: ...

    def subscribe_version_updates(
        self,
        policy_id: str,
    ) -> Iterable[PolicyVersionSnapshot]: ...

    def register_policy_namespace(
        self,
        policy_id: str,
        base_model_id: str,
        tokenizer_id: str,
        adapter_storage_root: str,
    ) -> Ack: ...
