"""§A.1 — EnvironmentProvider protocol.

Owns: task registry / splits, episode lifecycle, ground truth, reward
function, sandbox runtime, termination logic. The agent-side runtime
(tool execution, browser harness, code sandbox, file system) is internal
to this slot.

Today's adapter: ProRL FastAPI :8006 wrapping OpenHands + Singularity.
Future adapters: ROCK, GEM, ORS/OpenReward, browser/code-exec sandboxes.

Five operations, derived from the convergence of ROCK / GEM /
ORS-OpenReward (§5.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True, frozen=True)
class ContentBlock:
    """One content block in a multimodal observation or prompt.

    ``kind`` is ``'text'`` or ``'image'``; payload depends on kind.
    Text blocks carry ``text``; image blocks carry either an inline
    ``image_bytes`` blob (with ``mime_type``) or a ``image_uri``
    pointer to durable storage.
    """

    kind: str
    text: str | None = None
    image_uri: str | None = None
    image_bytes: bytes | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Typed tool invocation. Actions are NEVER raw strings."""

    name: str
    input: dict[str, Any]


@dataclass(slots=True, frozen=True)
class StepResult:
    """One step's observation, reward delta, done flag, and info bag."""

    observation: list[ContentBlock]
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class EpisodeHandle(Protocol):
    """Opaque, provider-internal episode handle.

    Concrete adapters may use any type so long as it round-trips through
    their own ``act`` / ``close`` methods. Treated as opaque outside the
    adapter.
    """

    @property
    def episode_uid(self) -> str: ...


class EnvironmentProvider(Protocol):
    environment_id: str
    environment_version: str

    def list_tasks(self, split: str) -> list[str]: ...
    def create_episode(self, task_id: str, **opts: Any) -> EpisodeHandle: ...
    def get_prompt(self, h: EpisodeHandle) -> list[ContentBlock]: ...
    def act(self, h: EpisodeHandle, tc: ToolCall) -> StepResult: ...
    def close(self, h: EpisodeHandle) -> None: ...
