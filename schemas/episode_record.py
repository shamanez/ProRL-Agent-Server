"""§6.1 wire schema — EpisodeRecord, the canonical durable record.

The unit written to the ReplayArchive (slot 5.5). Captures everything
needed to reconstruct or audit an episode without further reference to
the EnvironmentProvider's internal state.

Field inventory derived from §6.1 with semantics matching the
ORS/OpenReward, ROCK/GEM, and current ``TrajectoryRecord`` shapes. The
``events`` field is the load-bearing one: an EpisodeRecord can always
be reconstructed from its event stream. The flat fields above are
summaries and indexes for query.

Trust + provenance (§6.3): :class:`TrustLevel` is first-class. External
trajectories are untrusted by default; per-trainer-adapter routing
rules consume this field to decide whether a sample can drive a gradient
step or only an eval-only / SFT path.

Token-in / token-out (§3.1): ``prompt_token_ids`` and
``response_token_ids`` are ``tuple[int, ...]``; the canonical event
stream also carries token ids on every assistant turn. Decoded text
appears only as auxiliary tool-call args / observations where the
agent ran in text-mode by definition (e.g., shell commands).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TrustLevel(str, Enum):
    """§6.1 / §6.3 trust-level enum.

    String-valued so it survives JSON round-trips on the archive ingest
    path; ``Enum`` so consumers can do exhaustiveness checks.

    Semantics:

    * ``OWN_FABRIC``        — produced by this fabric; full trust.
    * ``PARTNER_VALIDATED`` — produced by a partner whose contract has
                              been validated (matching tokenizer, signed
                              behavior_log_probs, audited verifier).
    * ``PARTNER_UNTRUSTED`` — produced by a partner; consumable only on
                              non-RL paths (SFT, rejection learning) or
                              with explicit IS-correction-disabled flag.
    * ``EXTERNAL_EVAL_ONLY`` — eval / leaderboard data; never trains.
    """

    OWN_FABRIC = 'own-fabric'
    PARTNER_VALIDATED = 'partner-validated'
    PARTNER_UNTRUSTED = 'partner-untrusted'
    EXTERNAL_EVAL_ONLY = 'external-eval-only'


@dataclass(slots=True, frozen=True)
class RewardEvent:
    """Per-event reward delta with provenance.

    Allows offline jobs to attribute the episode-final reward to specific
    turns and verifiers (e.g., an LLM-judge model id, a unit-test runner
    id, a reward-model hash).
    """

    turn_index: int
    delta: float
    verifier_id: str
    judge_model_id: str | None = None


@dataclass(slots=True, frozen=True)
class Event:
    """One step of the episode timeline.

    ``kind`` discriminates the variant; the matching payload field
    is populated and the others are ``None``. This is the load-bearing
    event-stream representation an EpisodeRecord can always be
    reconstructed from.

    Variants:

    * ``"agent_turn"``        — model-emitted token ids + per-token logprobs.
    * ``"tool_call"``         — typed tool call (name + input dict).
    * ``"tool_result"``       — observation block returned by the env.
    * ``"reward_update"``     — verifier emits a reward delta.
    * ``"meta"``              — environment-internal annotations.
    """

    turn_index: int
    kind: str
    # agent_turn payload
    response_token_ids: tuple[int, ...] | None = None
    response_loss_mask: tuple[int, ...] | None = None
    behavior_log_probs: tuple[float, ...] | None = None
    # tool_call payload
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    # tool_result payload
    observation: list[dict[str, Any]] | None = None  # list[ContentBlock]
    # reward_update payload
    reward: RewardEvent | None = None
    # any-variant
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class EpisodeRecord:
    """Canonical, durable, audit-friendly record of one episode.

    Fields per §6.1. ``events`` is the load-bearing representation;
    everything else is a summary or index. The :meth:`from_events` and
    :meth:`recompute_summary` helpers (deferred to S3 implementation)
    rebuild the flat fields from ``events``.

    Required tokenizer constraint (§6.1): every consumer of an
    EpisodeRecord must verify ``tokenizer_id`` matches their own
    expected tokenizer. Mismatched-tokenizer records route to drop /
    re-tokenize-flag / alternate-adapter handling, never silently
    consumed (else §3.1 token-in/token-out invariant breaks).
    """

    # Identity / provenance
    episode_uid: str
    task_id: str
    split: str
    environment_provider: str
    environment_id: str
    environment_version: str
    verifier_version: str
    reward_spec_id: str

    # Policy provenance
    policy_id: str
    policy_version: int
    base_model_id: str
    tokenizer_id: str
    inference_backend: str
    sampling_params: dict[str, Any]

    # Step / wall-clock
    created_at_step: int
    started_at: datetime
    finished_at: datetime
    termination_reason: str  # 'done' | 'truncated' | 'error' | 'timeout'

    # Event stream — the canonical representation
    events: tuple[Event, ...]

    # Optional structured turn-level view (redundant with events but
    # sometimes more convenient for trainers).
    messages_or_turns: tuple[dict[str, Any], ...] = ()

    # Reward summary
    total_reward: float = 0.0
    reward_events: tuple[RewardEvent, ...] = ()

    # Token-level summary (derived from events but materialized for
    # query convenience).
    prompt_token_ids: tuple[int, ...] = ()
    response_token_ids: tuple[int, ...] = ()
    response_loss_mask: tuple[int, ...] = ()
    behavior_log_probs: tuple[float, ...] | None = None

    # Tool calls (denormalized for query).
    tool_calls: tuple[dict[str, Any], ...] = ()

    # Provenance bag and trust.
    provenance: dict[str, Any] = field(default_factory=dict)
    trust_level: TrustLevel = TrustLevel.OWN_FABRIC

    # Schema version. Consumers ignore unknown fields on minor bumps;
    # major bumps trigger explicit migration.
    schema_version: str = '1.0.0'
