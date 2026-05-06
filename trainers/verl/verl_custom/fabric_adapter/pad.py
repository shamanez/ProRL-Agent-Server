"""Extracted ``_pack`` from the legacy in-process replay store (§6.2).

Source: ``trainer_integration/verl/verl_custom/replay/trajectory_store.py:470-621``.
The LiveStore returns unpadded :class:`TrainingSample` records; this module
pads to the VERL trainer's expected ``(B, prompt_cap + response_cap)`` shape.

Cap semantics: over-sized records raise :class:`RuntimeError` (not silent
truncation — a contract violation).

Trainer connects to LiveStore and uses this pad helper locally (BC-15):
    samples = live_store_client.get_batch(n_groups=N, current_step=step, timeout_ms=T)
    mini_batch = pack_unpadded_groups(samples, pad_token_id=0,
                                      prompt_length_cap=P, response_length_cap=R)
    # mini_batch is DataProto-ready via DataProto.from_dict(...)

Warm-up: the trainer calls sample_mini_batch which blocks server-side until
N fresh groups arrive (BC-16). timeout_ms is set large; no-progress detector
(1800s) is the abort path, not a short RPC timeout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from rollout_fabric.schemas.training_sample import TrainingSample


@dataclass(slots=True)
class SampledMiniBatch:
    """Tensor / non-tensor payload compatible with ``DataProto.from_dict``."""

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
    """Pad ``samples`` to a fixed shape and assemble a ``SampledMiniBatch``."""
    if not samples:
        raise ValueError('pack_unpadded_groups requires at least one sample')
    B = len(samples)
    max_prompt = prompt_length_cap or max(len(s.prompt_token_ids) for s in samples)
    max_resp = response_length_cap or max(len(s.response_token_ids) for s in samples)
    max_prompt = max(max_prompt, 1)
    max_resp = max(max_resp, 1)

    prompt_ids = torch.full((B, max_prompt), pad_token_id, dtype=torch.long)
    prompt_attn = torch.zeros((B, max_prompt), dtype=torch.long)
    responses = torch.full((B, max_resp), pad_token_id, dtype=torch.long)
    resp_attn = torch.zeros((B, max_resp), dtype=torch.long)
    loss_mask = torch.zeros((B, max_resp), dtype=torch.long)
    rollout_lp = torch.zeros((B, max_resp), dtype=torch.float)
    is_padded = torch.zeros(B, dtype=torch.bool)
    error_mask = torch.zeros(B, dtype=torch.bool)
    rewards = torch.zeros(B, dtype=torch.float)
    raw_rewards = torch.zeros(B, dtype=torch.float)
    truncated = torch.zeros(B, dtype=torch.bool)

    uids: list[str] = []
    successes: list[bool] = []
    errors: list[str | None] = []
    resolveds: list[bool] = []
    finishes: list[bool] = []
    instances: list[dict] = []
    bvs: list[int] = []
    cs: list[int] = []
    ages: list[int] = []
    extra_keys: set[str] = set()
    for s in samples:
        extras = s.instance.get('_extras', {})
        if isinstance(extras, dict):
            extra_keys.update(extras.keys())
    extras_by_key: dict[str, list[Any]] = {k: [] for k in extra_keys}

    for i, s in enumerate(samples):
        if prompt_length_cap and len(s.prompt_token_ids) > prompt_length_cap:
            raise RuntimeError(
                f'pack_unpadded_groups: prompt_token_ids length '
                f'{len(s.prompt_token_ids)} > cap {prompt_length_cap}; '
                f'group_uid={s.group_uid}'
            )
        if response_length_cap and len(s.response_token_ids) > response_length_cap:
            raise RuntimeError(
                f'pack_unpadded_groups: response_token_ids length '
                f'{len(s.response_token_ids)} > cap {response_length_cap}; '
                f'group_uid={s.group_uid}'
            )
        p_ids = s.prompt_token_ids[-max_prompt:]
        off = max_prompt - len(p_ids)
        if p_ids:
            prompt_ids[i, off : off + len(p_ids)] = torch.tensor(
                p_ids, dtype=torch.long
            )
            prompt_attn[i, off : off + len(p_ids)] = 1
        r_ids = s.response_token_ids[:max_resp]
        r_lm = s.response_loss_mask[:max_resp]
        r_lp = (s.behavior_log_probs[:max_resp]) if s.behavior_log_probs else ()
        rlen = len(r_ids)
        if rlen > 0:
            responses[i, :rlen] = torch.tensor(r_ids, dtype=torch.long)
            resp_attn[i, :rlen] = 1
            loss_mask[i, :rlen] = torch.tensor(r_lm, dtype=torch.long)
            if r_lp:
                rollout_lp[i, : len(r_lp)] = torch.tensor(r_lp, dtype=torch.float)
        is_padded[i] = bool(s.is_padded)
        error_mask[i] = bool(s.error)
        rewards[i] = float(s.reward)
        raw_rewards[i] = float(s.raw_reward)
        truncated[i] = bool(s.truncated)
        uids.append(s.group_uid)
        inst = dict(s.instance)
        successes.append(bool(inst.get('success', True)))
        resolveds.append(bool(inst.get('resolved', False)))
        finishes.append(bool(inst.get('finish', True)))
        errors.append(s.error)
        instances.append(inst)
        bvs.append(s.behavior_policy_version)
        cs.append(s.created_at_step)
        ages.append(current_step - s.created_at_step)
        extras = inst.get('_extras', {})
        for k in extra_keys:
            extras_by_key[k].append(extras.get(k))

    input_ids = torch.cat([prompt_ids, responses], dim=1)
    attn = torch.cat([prompt_attn, resp_attn], dim=1)
    pos_ids = torch.clip(attn.cumsum(dim=-1) - 1, min=0)

    tensors: dict[str, torch.Tensor] = {
        'input_ids': input_ids,
        'responses': responses,
        'attention_mask': attn,
        'position_ids': pos_ids,
        'loss_mask': loss_mask,
        'rollout_log_probs': rollout_lp,
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
        # SWEBenchRewardManager groups metrics by ability; default to 'swe_agent'
        # since all samples in this fabric are SWE-Bench tasks.
        'ability': np.array(['swe_agent'] * len(uids), dtype=object),
    }
    for k, vals in extras_by_key.items():
        non_tensors[k] = np.array(vals, dtype=object)
    meta_info = {
        'behavior_policy_versions': bvs,
        'created_at_steps': cs,
        'sample_ages': ages,
    }
    return SampledMiniBatch(
        tensors=tensors, non_tensors=non_tensors, meta_info=meta_info
    )
