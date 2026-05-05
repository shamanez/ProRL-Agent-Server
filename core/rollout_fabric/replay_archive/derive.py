"""Re-derive TrainingSample from archived EpisodeRecord (offline path).

§3.1 corollary: tokenizer mismatch raises :class:`TokenizerMismatchError`.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from rollout_fabric.schemas.episode_record import EpisodeRecord
from rollout_fabric.schemas.training_sample import TrainingSample


class TokenizerMismatchError(RuntimeError):
    pass


def derive_training_samples(
    records: Iterable[EpisodeRecord],
    *,
    schema_version: str = '1.0.0',
    expected_tokenizer_id: str | None = None,
) -> list[TrainingSample]:
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
            raise RuntimeError(
                f'episode {r.episode_uid!r} schema_version={r.schema_version!r} '
                f'incompatible with schema_version={schema_version!r}'
            )
        out.append(
            TrainingSample(
                sample_uid=str(uuid.uuid4()),
                group_uid=r.task_id,
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
        )
    return out
