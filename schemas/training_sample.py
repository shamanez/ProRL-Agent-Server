"""§6.2 wire schema — TrainingSample / TrainingGroup, the live-path record.

The unit pushed to the LiveStore (slot 5.4) and consumed by the
TrainerAdapter (slot 5.6). Compact, shaped to satisfy ROLL, slime, VERL,
and SFT/distillation trainers simultaneously per the trainer-side
research summarized in ``rollout_fabric.md`` §6.2.

Field inventory mirrors the current ``TrajectoryRecord`` at
``trainer_integration/verl/verl_custom/replay/trajectory_store.py:43-74``
with three additions explicitly called out by §6.2:

  * ``raw_reward``      — pre-normalization reward; ROLL/slime carry this,
                          VERL DataProto today does not. Added.
  * ``truncated``       — True if the rollout was cut by length; useful
                          for reward-shaping decisions. Added.
  * ``sample_indices``  — optional back-pointer to a structured offset
                          within the episode (slime carries this). Added.

What this schema deliberately does NOT carry (per §6.2 / principle 4.4):

  * ``advantage``       — computed by the trainer adapter, not on the wire.
  * ``returns``         — same.
  * ``KL`` / ``ref_log_probs`` — recomputed by the FSDP/Megatron actor.
  * Per-trainer normalization constants — adapter-local.
  * Raw task description — that lives at the EnvironmentProvider behind
    ``task_id`` (invariant §3.8). The trainer never resolves ``task_id``
    back to a problem statement.

Padding stance (§6.2): the wire is **unpadded everywhere**, both on push
and on ``get_batch`` return. Padding / sequence-packing is the trainer
adapter's responsibility (``trainer_adapters/verl/pad.py`` after S1).
The LiveStore does not know tensor shapes.

Token-in / token-out (§3.1): every token-bearing field is
``tuple[int, ...]``. The wire never carries decoded text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from schemas.episode_record import TrustLevel


@dataclass(slots=True, frozen=True)
class TrainingSample:
    """One row of training data, derived from one trajectory.

    Attributes
    ----------
    sample_uid:
        Per-row UUID. Stable across pushes; primary key for the row.
    group_uid:
        Per-group UUID; ``n`` siblings of one GRPO/DAPO group share this.
        The store operates in groups (§3.2); sampling is in groups; the
        wire format groups by this key.
    episode_uid:
        Back-pointer to the canonical :class:`EpisodeRecord` in the
        ReplayArchive. Allows the offline path to recover the full event
        stream from a training row.
    prompt_token_ids, response_token_ids, response_loss_mask:
        Unpadded token sequences (§3.1). ``response_loss_mask`` is 1 on
        assistant tokens and 0 on tool/observation tokens. Re-padding is
        the trainer adapter's job.
    behavior_log_probs:
        Per-token log-probabilities from the inference backend at
        generation time. Default-required (§4.4 + §6.3 algorithm matrix).
        ``None`` only when the inference backend cannot provide them; in
        that case ``trust_level`` is restricted (§6.3). **Never silently
        set IS=1.0 for missing logprobs.**
    reward, raw_reward:
        Episode-final reward (post-normalization) and pre-normalization
        reward. Trainer adapters compute advantages from these — not from
        a pre-computed ``advantage`` field.
    truncated:
        True if the rollout was cut by length (vs done-by-environment).
    behavior_policy_version, created_at_step:
        §3.5 per-row stamp + trainer step at push time. Set by the
        :class:`PolicyVersionCache.snapshot` read at group dispatch start.
    task_id, split, policy_id, environment_id, environment_version,
    verifier_version:
        Provenance only. The trainer **must not** use ``task_id`` to
        re-load the task from a local dataset (§3.8 — data ownership).
    trust_level:
        Inherited from :class:`EpisodeRecord`. Routes per §6.3.
    sample_indices:
        Optional back-pointer to a structured offset inside the episode
        (slime carries this). ``None`` if the episode is single-segment.
    instance:
        Compact instance metadata (``data_source``, ``ability``,
        ``reward_model``, ``extra_info``, ``index``, ...) — preserves
        the current ``prompt_extras`` payload so reward managers and
        downstream logic remain bit-identical to today's seam.
    error:
        Error message if dispatch failed; non-None implies row-padding
        behavior at the trainer adapter (preserves current contract).
    is_padded:
        Padding row flag (preserves current behavior).
    """

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
                'response_token_ids and response_loss_mask must have equal '
                f'length; got {len(self.response_token_ids)} vs '
                f'{len(self.response_loss_mask)}'
            )
        if self.behavior_log_probs is not None and len(self.behavior_log_probs) != len(
            self.response_token_ids
        ):
            raise ValueError(
                'behavior_log_probs must match response_token_ids length when '
                f'present; got {len(self.behavior_log_probs)} vs '
                f'{len(self.response_token_ids)}'
            )


@dataclass(slots=True, frozen=True)
class TrainingGroup:
    """``n`` :class:`TrainingSample` records sharing a ``group_uid``.

    The store operates in groups; the trainer adapter samples in groups;
    the wire format groups them. Whole-group integrity is invariant §3.2 —
    samples are never split across groups, and zero-variance filtering at
    the producer is per-group (§3.7).

    A ``TrainingGroup`` is intentionally a thin wrapper: it carries the
    ``group_uid`` and the list of samples. Group-level metadata (e.g.
    aggregated reward variance) is recomputed by the trainer adapter on
    sample.
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
    """Raise :class:`ValueError` unless all samples share one ``group_uid``.

    Helper for code paths that build groups from arbitrary sample lists
    (push-side packers, archive re-derivation). Defends against the
    §3.2 invariant being violated by an upstream bug — groups must never
    be split.
    """
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
    """Return shape for ``LiveStore.push_group`` (§A.4)."""

    accepted: bool
    store_size: int
    backpressure_hint_ms: int = 0


@dataclass(slots=True)
class BatchResult:
    """Return shape for ``LiveStore.get_batch`` (§A.4).

    Per §6.2 the wire is unpadded; padding is trainer-adapter local.
    """

    samples: list[TrainingSample]
    behavior_policy_versions: list[int]
    created_at_steps: list[int]
    sample_ages: list[int]
    metrics_pre: dict[str, float] = field(default_factory=dict)
    metrics_post: dict[str, float] = field(default_factory=dict)
