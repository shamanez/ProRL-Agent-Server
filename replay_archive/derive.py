"""Re-derive :class:`TrainingSample` records from archived episodes.

Per §A.5: the archive's canonical record is :class:`EpisodeRecord`;
``TrainingSample`` is derived on demand. This is the offline path —
distillation, SFT-on-filtered-trajectories, mid-training data audits
all start here.

The derivation mirrors the live-path codec
(:func:`live_store.codec.dataproto_to_samples`) for fields the live
path computes from the agent state. Token IDs come from the canonical
``response_token_ids`` field on :class:`EpisodeRecord` (already
populated at ingest); the loss mask is the same length.

Tokenizer mismatch (§3.1 corollary): if the caller-supplied
``expected_tokenizer_id`` differs from the record's, raise
:class:`TokenizerMismatchError`. Per the post-S4 checklist item 16:
"`tokenizer_id` mismatch raises typed error" — that's enforced here.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from schemas.episode_record import EpisodeRecord
from schemas.training_sample import TrainingSample


class TokenizerMismatchError(RuntimeError):
    """Raised when an archived record's tokenizer differs from caller's."""


def derive_training_samples(
    records: Iterable[EpisodeRecord],
    *,
    schema_version: str = '1.0.0',
    expected_tokenizer_id: str | None = None,
) -> list[TrainingSample]:
    """Convert episodes to :class:`TrainingSample` rows."""
    out: list[TrainingSample] = []
    for r in records:
        if (
            expected_tokenizer_id is not None
            and r.tokenizer_id != expected_tokenizer_id
        ):
            raise TokenizerMismatchError(
                f'episode {r.episode_uid!r} tokenizer_id={r.tokenizer_id!r} '
                f'!= expected={expected_tokenizer_id!r}'
            )
        if r.schema_version.split('.')[0] != schema_version.split('.')[0]:
            # Major-version mismatch is a hard error per §6.2 versioning.
            raise RuntimeError(
                f'episode {r.episode_uid!r} schema_version={r.schema_version!r} '
                f'incompatible with caller schema_version={schema_version!r}'
            )
        sample = TrainingSample(
            sample_uid=str(uuid.uuid4()),
            group_uid=r.task_id,  # one episode per group at re-derivation
            episode_uid=r.episode_uid,
            prompt_token_ids=r.prompt_token_ids,
            response_token_ids=r.response_token_ids,
            response_loss_mask=r.response_loss_mask,
            behavior_log_probs=r.behavior_log_probs,
            reward=r.total_reward,
            raw_reward=r.total_reward,
            truncated=(r.termination_reason == 'truncated'),
            behavior_policy_version=r.policy_version,
            created_at_step=r.created_at_step,
            task_id=r.task_id,
            split=r.split,
            policy_id=r.policy_id,
            environment_id=r.environment_id,
            environment_version=r.environment_version,
            verifier_version=r.verifier_version,
            trust_level=r.trust_level,
            sample_indices=None,
            instance=r.provenance,
            error=None,
            is_padded=False,
        )
        out.append(sample)
    return out
