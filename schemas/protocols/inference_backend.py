"""§A.2 — InferenceBackend protocol.

Owns: model weights, LoRA cache, KV cache, serving infrastructure.
Produces tokens and per-token logprobs from a policy.

Today's adapter: vLLM child pool :8100-8103 with the pinning swap
protocol (``scripts/serving/_vllm_child.py``). Pinning is the §3.4
correctness invariant — a single ``generate`` call must see exactly one
policy version. Any future backend (SGLang, TGI, hosted API) must
provide either ``/v{N}/generate``-style path-versioned pinning or an
equivalent guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class PolicyRef(Protocol):
    """Opaque per-call policy pin.

    Concrete adapters may implement this as a versioned URL fragment
    (``/v{N}/generate``), an explicit ``policy_id`` argument, a session
    binding, or any equivalent that produces the §3.4 pinning guarantee.
    """

    @property
    def policy_id(self) -> str: ...

    @property
    def version(self) -> int: ...


@dataclass(slots=True)
class GenerationResult:
    """Tokens + logprobs returned by ``generate``.

    ``logprobs`` is the per-token log-probability of the **selected**
    token under the served policy version, length-matched to ``token_ids``.
    Adapters whose backends cannot produce logprobs return ``None``; the
    consumer must restrict trust accordingly (§6.3).
    """

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
