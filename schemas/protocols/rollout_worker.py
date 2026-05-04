"""§A.3 — RolloutWorker protocol.

Executes the agent loop against EnvironmentProvider and InferenceBackend,
produces episodes, derives training groups, applies producer-side filters
(zero-variance drop), tags provenance and policy version, and pushes
results to LiveStore and ReplayArchive.

Owns the training **task dataset** (§3.8 — load-bearing data ownership
boundary). The trainer never holds a parquet path after S2.

Validation flow (per the operating-principle revision in the plan):
**removed**. ``request_validation`` and ``score_validation`` are NOT in
this Protocol. If validation is wanted later, it lands as a separate
cut.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from schemas.policy_version import PolicyVersionSnapshot
from schemas.protocols.live_store import Ack


class TaskSource(Protocol):
    """Worker-side dataloader / task iterator.

    Owns ``state_dict()`` / ``load_state_dict()`` for resume. Today's
    ``StatefulDataLoader`` wraps this.
    """

    def __iter__(self) -> Iterator[dict[str, Any]]: ...

    def state_dict(self) -> bytes: ...

    def load_state_dict(self, state: bytes) -> None: ...


class FilterStrategy(Protocol):
    """Producer-side filter (§3.7 zero-variance drop is the canonical case).

    Operates on whole groups. Returning ``False`` drops the entire group
    from the live path; the archive still receives the group (S3+).
    """

    def admit(self, group_records: list[Any]) -> bool: ...


class PolicyVersionStream(Protocol):
    """Subscription to policy-version updates.

    Implemented as JSON-mtime polling at S2 and as gRPC server-streaming
    at S4. The stream pushes :class:`PolicyVersionSnapshot` updates that
    the worker installs into its :class:`PolicyVersionCache` via the
    cache's atomic-ref-swap pattern (see ``schemas/policy_version.py``).
    """

    def __iter__(self) -> Iterator[PolicyVersionSnapshot]: ...

    def close(self) -> None: ...


class RolloutWorker(Protocol):
    worker_id: str

    # RPC surface exposed to the trainer.
    def pause_production(self) -> Ack: ...
    def resume_production(self) -> Ack: ...
    def get_dataloader_state(self) -> bytes: ...
    def load_dataloader_state(self, state: bytes) -> Ack: ...
