"""Convert a ProRL HTTP response into a :class:`TrainingSample`.

**No VERL, no OpenHands imports.** Parses the JSON response from
``POST /process`` and builds the §6.2 wire schema directly.

Token-in / token-out (§3.1): token IDs are extracted from the
``token_ids`` field of each message turn. They are never decoded or
re-tokenized — the exact IDs from vLLM are passed through.

Group-policy consistency (BC-0): the caller passes one
``PolicyVersionSnapshot`` that was read **once** before dispatching all N
siblings. Every sample built by this module stamps ``snap.version`` on
``behavior_policy_version``, so the whole group is uniformly tagged (§3.2).

Loss-mask semantics: 1 on assistant tokens, 0 on tool/observation tokens.
"""

from __future__ import annotations

import uuid
from typing import Any

from rollout_fabric.schemas.episode_record import TrustLevel
from rollout_fabric.schemas.policy_version import PolicyVersionSnapshot
from rollout_fabric.schemas.training_sample import TrainingSample


def build_training_sample(
    prorl_result: Any,  # ProRLEpisodeResult
    snap: PolicyVersionSnapshot,
    *,
    created_at_step: int,
    task_id: str,
    split: str = 'train',
    environment_id: str = 'prorl',
    environment_version: str = '',
    verifier_version: str = '',
    trust_level: TrustLevel = TrustLevel.OWN_FABRIC,
    episode_uid: str | None = None,
) -> TrainingSample:
    """Build one ``TrainingSample`` from a ``ProRLEpisodeResult``.

    The ``group_uid`` is set by the caller (shared across all N siblings
    of one GRPO/DAPO group). Here we return a sample with
    ``group_uid = sample_uid``; the caller must override ``group_uid``
    to bind siblings together — see :func:`build_group`.
    """
    prompt_ids, response_ids, loss_mask, logprobs = _extract_token_fields(
        prorl_result.messages
    )
    sample_uid = str(uuid.uuid4())
    return TrainingSample(
        sample_uid=sample_uid,
        group_uid=sample_uid,  # caller overrides to the shared group_uid
        episode_uid=episode_uid or f'ep-{sample_uid}',
        prompt_token_ids=prompt_ids,
        response_token_ids=response_ids,
        response_loss_mask=loss_mask,
        behavior_log_probs=logprobs if logprobs else None,
        reward=float(prorl_result.reward),
        raw_reward=float(prorl_result.reward),
        truncated=not prorl_result.finish,
        behavior_policy_version=int(snap.version),  # stamped from group-start snap
        created_at_step=created_at_step,
        task_id=task_id,
        split=split,
        policy_id=snap.policy_id,
        environment_id=environment_id,
        environment_version=environment_version,
        verifier_version=verifier_version,
        trust_level=trust_level,
        sample_indices=None,
        instance={
            'success': prorl_result.success,
            'resolved': prorl_result.resolved,
            'finish': prorl_result.finish,
            'instance_id': prorl_result.instance_id,
            'policy_version': int(snap.version),
        },
        error=prorl_result.error,
        is_padded=False,
    )


def build_group(
    samples: list[TrainingSample],
    group_uid: str,
) -> list[TrainingSample]:
    """Rebind all samples to a shared ``group_uid`` (§3.2 group integrity).

    Returns new immutable ``TrainingSample`` objects with the shared uid.
    Must be called before pushing to LiveStore.
    """
    result = []
    for s in samples:
        # dataclass is frozen=True; replace via object.__setattr__ workaround
        # is not available. Instead reconstruct (fields are cheap to copy).
        result.append(
            TrainingSample(
                sample_uid=s.sample_uid,
                group_uid=group_uid,
                episode_uid=s.episode_uid,
                prompt_token_ids=s.prompt_token_ids,
                response_token_ids=s.response_token_ids,
                response_loss_mask=s.response_loss_mask,
                behavior_log_probs=s.behavior_log_probs,
                reward=s.reward,
                raw_reward=s.raw_reward,
                truncated=s.truncated,
                behavior_policy_version=s.behavior_policy_version,
                created_at_step=s.created_at_step,
                task_id=s.task_id,
                split=s.split,
                policy_id=s.policy_id,
                environment_id=s.environment_id,
                environment_version=s.environment_version,
                verifier_version=s.verifier_version,
                trust_level=s.trust_level,
                sample_indices=s.sample_indices,
                instance=s.instance,
                error=s.error,
                is_padded=s.is_padded,
            )
        )
    return result


def is_zero_variance_group(samples: list[TrainingSample]) -> bool:
    """Return True if all rewards in the group are identical (zero variance).

    Zero-variance groups provide no learning signal for GRPO/DAPO; the
    advantage is 0 for every sample. Filter them at the producer side
    (§3.7 eager-push seam) so the LiveStore never sees them.
    """
    if len(samples) <= 1:
        return False
    first = samples[0].reward
    return all(abs(s.reward - first) < 1e-9 for s in samples)


# ---- internal -------------------------------------------------------------


def _extract_token_fields(
    messages: list[dict],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[float, ...] | None,
]:
    """Extract prompt/response token arrays from the messages list.

    Returns ``(prompt_ids, response_ids, loss_mask, logprobs)``.

    The first message (role=user or role=system) provides prompt_token_ids.
    Subsequent messages alternate assistant/tool turns:
    - assistant turns: token_ids → response, logprobs → behavior_log_probs,
      loss_mask bit = 1
    - tool/observation turns: token_ids → response (counted but masked out),
      loss_mask bit = 0

    If any assistant turn lacks ``token_ids`` the sample cannot be used for
    RL; return empty tuples and the caller will mark it as is_padded=True or
    error out.
    """
    if not messages:
        return (), (), (), None

    prompt_ids: tuple[int, ...] = ()
    response_ids: list[int] = []
    loss_mask_bits: list[int] = []
    logprob_values: list[float] = []
    has_logprobs = True

    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        token_ids = msg.get('token_ids') or []
        lps = msg.get('logprobs') or []

        if i == 0 and role in ('user', 'system'):
            prompt_ids = tuple(int(t) for t in token_ids)
            continue

        if role == 'assistant':
            response_ids.extend(int(t) for t in token_ids)
            loss_mask_bits.extend(1 for _ in token_ids)
            if lps:
                logprob_values.extend(float(lp) for lp in lps)
            else:
                has_logprobs = False
        else:
            # tool / observation / environment turn — masked out (loss=0)
            response_ids.extend(int(t) for t in token_ids)
            loss_mask_bits.extend(0 for _ in token_ids)
            # no logprobs for tool turns

    final_logprobs: tuple[float, ...] | None = None
    if has_logprobs and logprob_values:
        # logprobs cover only assistant tokens; pad zeros on tool positions
        # so length matches response_ids. We rebuild aligned to response.
        final_logprobs = _align_logprobs(messages, response_ids, logprob_values)

    return (
        prompt_ids,
        tuple(response_ids),
        tuple(loss_mask_bits),
        final_logprobs,
    )


def _align_logprobs(
    messages: list[dict],
    response_ids: list[int],
    raw_logprobs: list[float],
) -> tuple[float, ...]:
    """Build a logprob array aligned to ``response_ids`` (zeros on tool turns)."""
    aligned: list[float] = []
    lp_cursor = 0
    for i, msg in enumerate(messages):
        if i == 0:
            continue
        role = msg.get('role', '')
        tids = msg.get('token_ids') or []
        lps = msg.get('logprobs') or []
        if role == 'assistant':
            for lp in lps:
                aligned.append(float(lp))
            # If assistant turn has fewer logprobs than token_ids, pad zeros.
            for _ in range(len(tids) - len(lps)):
                aligned.append(0.0)
            lp_cursor += len(lps)
        else:
            aligned.extend(0.0 for _ in tids)
    # Truncate or pad to exactly len(response_ids)
    while len(aligned) < len(response_ids):
        aligned.append(0.0)
    return tuple(aligned[: len(response_ids)])
