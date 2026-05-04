"""§A.3 — RolloutWorker protocol.

Owns the training task dataset (§3.8). The trainer never holds a parquet
path after S2. Zero VERL/OpenHands imports in the worker process (BC-13).

Validation is NOT on this RPC surface (deferred per Sec.11).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from schemas.policy_version import PolicyVersionSnapshot
from schemas.protocols.live_store import Ack


class TaskSource(Protocol):
    def __iter__(self) -> Iterator[dict[str, Any]]: ...

    def state_dict(self) -> bytes: ...

    def load_state_dict(self, state: bytes) -> None: ...


class FilterStrategy(Protocol):
    def admit(self, group_records: list[Any]) -> bool: ...


class PolicyVersionStream(Protocol):
    def __iter__(self) -> Iterator[PolicyVersionSnapshot]: ...

    def close(self) -> None: ...


class RolloutWorker(Protocol):
    worker_id: str

    def pause_production(self) -> Ack: ...
    def resume_production(self) -> Ack: ...
    def get_dataloader_state(self) -> bytes: ...
    def load_dataloader_state(self, state: bytes) -> Ack: ...
