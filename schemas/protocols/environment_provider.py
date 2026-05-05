"""§A.1 — EnvironmentProvider protocol.

Today's adapter: ProRL FastAPI :8006 (openhands/nvidia/async_server.py).
The RolloutManager calls this via plain HTTP — no OpenHands imports.

HTTP contract (ProRL's existing API):
  POST /process
  Body: {"instance": {..., "policy_version": N}, "sampling_params": {...}}
  Response: {"messages": [{..., "token_ids": [...], "logprobs": [...]}],
             "resolved": bool, "success": bool, "finish": bool, "error": str|null}

Future adapters: ROCK, GEM, ORS/OpenReward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True, frozen=True)
class ContentBlock:
    """One content block in a multimodal observation or prompt."""

    kind: str  # 'text' | 'image'
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
    observation: list[ContentBlock]
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class EpisodeHandle(Protocol):
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
