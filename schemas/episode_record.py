"""§6.1 wire schema — EpisodeRecord, the canonical durable record.

Written to the ReplayArchive (slot 5.5). Captures everything needed to
reconstruct or audit an episode without further reference to the
EnvironmentProvider's internal state.

Token-in / token-out (§3.1): ``prompt_token_ids`` and
``response_token_ids`` are ``tuple[int, ...]``. Decoded text appears
only as tool-call args / observations where the agent ran in text-mode
by definition.

Trust + provenance (§6.3): :class:`TrustLevel` is first-class. External
trajectories are untrusted by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TrustLevel(str, Enum):
    OWN_FABRIC = 'own-fabric'
    PARTNER_VALIDATED = 'partner-validated'
    PARTNER_UNTRUSTED = 'partner-untrusted'
    EXTERNAL_EVAL_ONLY = 'external-eval-only'


@dataclass(slots=True, frozen=True)
class RewardEvent:
    turn_index: int
    delta: float
    verifier_id: str
    judge_model_id: str | None = None


@dataclass(slots=True, frozen=True)
class Event:
    """One step of the episode timeline.

    ``kind`` discriminates the variant:
    ``"agent_turn"`` | ``"tool_call"`` | ``"tool_result"`` |
    ``"reward_update"`` | ``"meta"``
    """

    turn_index: int
    kind: str
    response_token_ids: tuple[int, ...] | None = None
    response_loss_mask: tuple[int, ...] | None = None
    behavior_log_probs: tuple[float, ...] | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    observation: list[dict[str, Any]] | None = None
    reward: RewardEvent | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EpisodeRecord:
    """Canonical, durable, audit-friendly record of one episode (§6.1).

    ``events`` is the load-bearing field; all flat fields are summaries
    for indexing and query.
    """

    episode_uid: str
    task_id: str
    split: str
    environment_provider: str
    environment_id: str
    environment_version: str
    verifier_version: str
    reward_spec_id: str

    policy_id: str
    policy_version: int
    base_model_id: str
    tokenizer_id: str
    inference_backend: str
    sampling_params: dict[str, Any]

    created_at_step: int
    started_at: datetime
    finished_at: datetime
    termination_reason: str  # 'done' | 'truncated' | 'error' | 'timeout'

    events: tuple[Event, ...]

    messages_or_turns: tuple[dict[str, Any], ...] = ()
    total_reward: float = 0.0
    reward_events: tuple[RewardEvent, ...] = ()

    prompt_token_ids: tuple[int, ...] = ()
    response_token_ids: tuple[int, ...] = ()
    response_loss_mask: tuple[int, ...] = ()
    behavior_log_probs: tuple[float, ...] | None = None

    tool_calls: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    trust_level: TrustLevel = TrustLevel.OWN_FABRIC
    schema_version: str = '1.0.0'
