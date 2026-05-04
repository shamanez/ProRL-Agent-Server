"""§6.2 wire schema — TrainingSample / TrainingGroup, the live-path record.

Pushed to LiveStore (slot 5.4) and consumed by TrainerAdapter (slot 5.6).

Padding stance (§6.2): the wire is **unpadded everywhere**. Padding is
the trainer adapter's responsibility (``trainer_adapters/verl/pad.py``).

Token-in / token-out (§3.1): every token-bearing field is
``tuple[int, ...]``. The wire never carries decoded text.

What this schema does NOT carry (per §6.2 / principle 4.4):
- ``advantage`` — computed by the trainer adapter.
- ``returns``, ``KL``, ``ref_log_probs`` — recomputed by FSDP actor.
- Raw task description — lives at EnvironmentProvider behind ``task_id``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from schemas.episode_record import TrustLevel


@dataclass(slots=True, frozen=True)
class TrainingSample:
    """One row of training data, derived from one trajectory."""

    sample_uid: str
    group_uid: str
    episode_uid: str

    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_loss_mask: tuple[int, ...]
    behavior_log_probs: tuple[float, ...] | None

    reward: float
    raw_reward: float
    truncated: bool

    behavior_policy_version: int
    created_at_step: int

    task_id: str
    split: str
    policy_id: str
    environment_id: str
    environment_version: str
    verifier_version: str
    trust_level: TrustLevel

    sample_indices: tuple[int, ...] | None
    instance: dict[str, Any]
    error: str | None
    is_padded: bool

    def __post_init__(self) -> None:
        if len(self.response_token_ids) != len(self.response_loss_mask):
            raise ValueError(
                'response_token_ids and response_loss_mask must have equal length; '
                f'got {len(self.response_token_ids)} vs {len(self.response_loss_mask)}'
            )
        if self.behavior_log_probs is not None and len(self.behavior_log_probs) != len(
            self.response_token_ids
        ):
            raise ValueError(
                'behavior_log_probs must match response_token_ids length when present; '
                f'got {len(self.behavior_log_probs)} vs {len(self.response_token_ids)}'
            )


@dataclass(slots=True, frozen=True)
class TrainingGroup:
    """``n`` :class:`TrainingSample` records sharing a ``group_uid``.

    The store operates in groups; sampling is in groups. Whole-group
    integrity is invariant §3.2.
    """

    group_uid: str
    samples: tuple[TrainingSample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError('TrainingGroup must contain at least one sample')
        for s in self.samples:
            if s.group_uid != self.group_uid:
                raise ValueError(
                    f'TrainingGroup.group_uid={self.group_uid!r} but sample '
                    f'has group_uid={s.group_uid!r}'
                )

    def __len__(self) -> int:
        return len(self.samples)


def assert_group_integrity(samples: Sequence[TrainingSample]) -> None:
    """Raise :class:`ValueError` unless all samples share one ``group_uid``."""
    if not samples:
        raise ValueError('group must contain at least one sample')
    expected = samples[0].group_uid
    for s in samples[1:]:
        if s.group_uid != expected:
            raise ValueError(
                f'group integrity violation: expected group_uid={expected!r}, '
                f'got group_uid={s.group_uid!r} on sample_uid={s.sample_uid!r}'
            )


@dataclass(slots=True)
class PushResult:
    accepted: bool
    store_size: int
    backpressure_hint_ms: int = 0


@dataclass(slots=True)
class BatchResult:
    """Return shape for ``LiveStore.get_batch``. Wire is unpadded (§6.2)."""

    samples: list[TrainingSample]
    behavior_policy_versions: list[int]
    created_at_steps: list[int]
    sample_ages: list[int]
    metrics_pre: dict[str, float] = field(default_factory=dict)
    metrics_post: dict[str, float] = field(default_factory=dict)
