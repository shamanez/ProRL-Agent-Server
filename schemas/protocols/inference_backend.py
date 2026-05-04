"""§A.2 — InferenceBackend protocol.

Today's adapter: vLLM child pool :8100-8103 (_vllm_child.py). Frozen
through S4. Pinning swap protocol is the §3.4 correctness invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class PolicyRef(Protocol):
    @property
    def policy_id(self) -> str: ...

    @property
    def version(self) -> int: ...


@dataclass(slots=True)
class GenerationResult:
    token_ids: list[int]
    logprobs: list[float] | None
    finish_reason: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ReloadResult:
    ok: bool
    error: str | None = None


@dataclass(slots=True)
class HealthStatus:
    healthy: bool
    policy_version: int
    pinned_versions: list[int]
    inflight_per_version: dict[int, int]


class InferenceBackend(Protocol):
    backend_id: str

    def generate(
        self,
        policy_ref: PolicyRef,
        tokenized_prompt: list[int],
        sampling_params: dict[str, Any],
    ) -> GenerationResult: ...

    def reload_policy(
        self,
        policy_id: str,
        policy_version: int,
        adapter_uri_or_blob: str | bytes,
    ) -> ReloadResult: ...

    def health(self) -> HealthStatus: ...
