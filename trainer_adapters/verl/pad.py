"""Lifted ``_pack`` from the legacy in-process replay store.

Source: ``trainer_integration/verl/verl_custom/replay/trajectory_store.py:470-621``.
Per the §6.2 padding stance, the LiveStore returns unpadded
:class:`TrainingSample` records; this module pads to the trainer's
expected ``(B, prompt_cap + response_cap)`` shape so the existing
``DataProto.from_dict`` path is bit-identical to today's seam.

What changed from the legacy version:

* Operates on ``list[TrainingSample]`` (§6.2), not ``TrajectoryRecord``.
  Field renames in the codec (``prompt_token_ids``, ``response_token_ids``,
  ``behavior_log_probs``, ``sample_uid``).
* Reconstructs the legacy non-tensor layout (``uid``, ``success``,
  ``error``, ``resolved``, ``finish``, ``instance``) from the §6.2
  ``instance`` dict so today's reward managers see the same payload
  they do today. New §6.2 fields (``raw_reward``, ``truncated``,
  ``trust_level``) are added to ``tensors`` / ``non_tensors`` for
  consumers that want them; existing consumers ignore unknown keys.

Cap semantics preserved: ``prompt_length_cap`` and ``response_length_cap``
are HARD caps; over-sized records raise :class:`RuntimeError` (a
contract violation, not a routine truncation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from schemas.training_sample import TrainingSample


@dataclass(slots=True)
class SampledMiniBatch:
    """Tensor / non-tensor payload for ``DataProto.from_dict``."""

    tensors: dict[str, torch.Tensor]
    non_tensors: dict[str, np.ndarray]
    meta_info: dict[str, Any] = field(default_factory=dict)


def pack_unpadded_groups(
    samples: Sequence[TrainingSample],
    *,
    pad_token_id: int,
    prompt_length_cap: int | None = None,
    response_length_cap: int | None = None,
    current_step: int = 0,
) -> SampledMiniBatch:
    """Pad ``samples`` to a fixed shape and assemble a ``SampledMiniBatch``.

    See module docstring for cap semantics. ``current_step`` is used only
    to compute ``sample_ages`` for the meta_info payload.
    """
    if not samples:
        raise ValueError('pack_unpadded_groups requires at least one sample')

    batch = len(samples)
    if prompt_length_cap is not None:
        max_prompt = prompt_length_cap
    else:
        max_prompt = max(len(s.prompt_token_ids) for s in samples)
    if response_length_cap is not None:
        max_response = response_length_cap
    else:
        max_response = max(len(s.response_token_ids) for s in samples)
    max_prompt = max(max_prompt, 1)
    max_response = max(max_response, 1)

    prompt_ids = torch.full((batch, max_prompt), pad_token_id, dtype=torch.long)
    prompt_attn = torch.zeros((batch, max_prompt), dtype=torch.long)
    responses = torch.full((batch, max_response), pad_token_id, dtype=torch.long)
    response_attn = torch.zeros((batch, max_response), dtype=torch.long)
    loss_mask = torch.zeros((batch, max_response), dtype=torch.long)
    rollout_log_probs = torch.zeros((batch, max_response), dtype=torch.float)
    is_padded = torch.zeros(batch, dtype=torch.bool)
    error_mask = torch.zeros(batch, dtype=torch.bool)
    rewards = torch.zeros(batch, dtype=torch.float)
    raw_rewards = torch.zeros(batch, dtype=torch.float)
    truncated = torch.zeros(batch, dtype=torch.bool)

    uids: list[str] = []
    successes: list[bool] = []
    errors: list[str | None] = []
    resolveds: list[bool] = []
    finishes: list[bool] = []
    instances: list[dict[str, Any]] = []
    behavior_versions: list[int] = []
    created_steps: list[int] = []
    sample_ages: list[int] = []
    extra_keys: set[str] = set()
    for s in samples:
        extras = s.instance.get('_extras', {})
        if isinstance(extras, dict):
            extra_keys.update(extras.keys())
    extras_by_key: dict[str, list[Any]] = {k: [] for k in extra_keys}

    for i, s in enumerate(samples):
        if (
            prompt_length_cap is not None
            and len(s.prompt_token_ids) > prompt_length_cap
        ):
            raise RuntimeError(
                f'pack_unpadded_groups: prompt_token_ids length '
                f'{len(s.prompt_token_ids)} exceeds cap '
                f'{prompt_length_cap}; rollout-side packer contract '
                f'violated. group_uid={s.group_uid}'
            )
        if (
            response_length_cap is not None
            and len(s.response_token_ids) > response_length_cap
        ):
            raise RuntimeError(
                f'pack_unpadded_groups: response_token_ids length '
                f'{len(s.response_token_ids)} exceeds cap '
                f'{response_length_cap}; vLLM max_model_len contract '
                f'violated. group_uid={s.group_uid}'
            )

        p_ids = s.prompt_token_ids[-max_prompt:]
        offset = max_prompt - len(p_ids)
        if p_ids:
            prompt_ids[i, offset : offset + len(p_ids)] = torch.tensor(
                p_ids, dtype=torch.long
            )
            prompt_attn[i, offset : offset + len(p_ids)] = 1

        r_ids = s.response_token_ids[:max_response]
        r_lm = s.response_loss_mask[:max_response]
        r_lp = (s.behavior_log_probs[:max_response]) if s.behavior_log_probs else ()
        r_len = len(r_ids)
        if r_len > 0:
            responses[i, :r_len] = torch.tensor(r_ids, dtype=torch.long)
            response_attn[i, :r_len] = 1
            loss_mask[i, :r_len] = torch.tensor(r_lm, dtype=torch.long)
            if r_lp:
                rollout_log_probs[i, : len(r_lp)] = torch.tensor(
                    r_lp, dtype=torch.float
                )

        is_padded[i] = bool(s.is_padded)
        error_mask[i] = bool(s.error)
        rewards[i] = float(s.reward)
        raw_rewards[i] = float(s.raw_reward)
        truncated[i] = bool(s.truncated)
        uids.append(s.sample_uid)
        instance = dict(s.instance)
        successes.append(bool(instance.get('success', True)))
        resolveds.append(bool(instance.get('resolved', False)))
        finishes.append(bool(instance.get('finish', True)))
        errors.append(s.error)
        instances.append(instance)
        behavior_versions.append(s.behavior_policy_version)
        created_steps.append(s.created_at_step)
        sample_ages.append(current_step - s.created_at_step)
        extras = instance.get('_extras', {})
        for k in extra_keys:
            extras_by_key[k].append(extras.get(k, None))

    input_ids = torch.cat([prompt_ids, responses], dim=1)
    attention_mask = torch.cat([prompt_attn, response_attn], dim=1)
    position_ids = torch.clip(attention_mask.cumsum(dim=-1) - 1, min=0)

    tensors: dict[str, torch.Tensor] = {
        'input_ids': input_ids,
        'responses': responses,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'loss_mask': loss_mask,
        'rollout_log_probs': rollout_log_probs,
        'is_padded': is_padded,
        'error_mask': error_mask,
        'reward': rewards,
        'raw_reward': raw_rewards,
        'truncated': truncated,
    }
    non_tensors: dict[str, np.ndarray] = {
        'uid': np.array(uids, dtype=object),
        'success': np.array(successes, dtype=object),
        'error': np.array(errors, dtype=object),
        'resolved': np.array(resolveds, dtype=object),
        'finish': np.array(finishes, dtype=object),
        'instance': np.array(instances, dtype=object),
    }
    for k, values in extras_by_key.items():
        non_tensors[k] = np.array(values, dtype=object)
    meta_info = {
        'behavior_policy_versions': behavior_versions,
        'created_at_steps': created_steps,
        'sample_ages': sample_ages,
    }
    return SampledMiniBatch(
        tensors=tensors, non_tensors=non_tensors, meta_info=meta_info
    )
