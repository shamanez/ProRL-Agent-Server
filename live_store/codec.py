"""Wire codec — :class:`TrainingSample` ↔ protobuf, DataProto ↔ samples.

This module owns three conversions:

1. ``TrainingSample`` (Python dataclass) → ``live_store_pb2.TrainingSample``
   (protobuf wire shape) and back. Token arrays go on the wire as
   packed int32-LE / int8 ``bytes`` to skip the protobuf parser
   overhead on long sequences.

2. ``DataProto`` (verl tensor batch) → ``list[TrainingSample]``. Lifted
   from ``trainer_integration/verl/verl_custom/replay/trajectory_store.py:194-392``
   (``push_from_dataproto``). The §6.2 wire shape adds ``raw_reward``,
   ``truncated``, ``sample_indices``; the legacy DataProto carries no
   distinct ``raw_reward``, so it is set equal to ``reward`` and the
   trainer adapter is responsible for reconstructing pre-normalization
   reward downstream if needed. ``truncated`` is derived from
   ``finish=False`` (today's contract: ``finish == True`` means the
   trajectory ran to ``done``; ``finish == False`` means it was cut by
   length).

3. ``list[TrainingSample]`` → ``SampledMiniBatch`` is **NOT here** —
   that lives in ``trainer_adapters/verl/pad.py`` per the §6.2 padding
   stance.

Token-in / token-out (§3.1): every conversion preserves token IDs as
ints; no decoded text touches the wire.
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING, Any

from schemas._gen import live_store_pb2
from schemas.episode_record import TrustLevel
from schemas.training_sample import TrainingSample

if TYPE_CHECKING:
    pass


# ---- byte packing -----------------------------------------------------------


def _pack_int32(seq: tuple[int, ...] | list[int]) -> bytes:
    return struct.pack(f'<{len(seq)}i', *seq) if seq else b''


def _unpack_int32(buf: bytes) -> tuple[int, ...]:
    if not buf:
        return ()
    n = len(buf) // 4
    return struct.unpack(f'<{n}i', buf)


def _pack_int8(seq: tuple[int, ...] | list[int]) -> bytes:
    if not seq:
        return b''
    return bytes(int(x) & 0xFF for x in seq)


def _unpack_int8(buf: bytes) -> tuple[int, ...]:
    return tuple(int(b) for b in buf)


def _pack_float32(seq: tuple[float, ...] | list[float]) -> bytes:
    return struct.pack(f'<{len(seq)}f', *seq) if seq else b''


def _unpack_float32(buf: bytes) -> tuple[float, ...]:
    if not buf:
        return ()
    n = len(buf) // 4
    return struct.unpack(f'<{n}f', buf)


# ---- TrainingSample <-> proto ----------------------------------------------


def to_proto(s: TrainingSample) -> live_store_pb2.TrainingSample:
    """Serialize a Python :class:`TrainingSample` to its proto form.

    The instance dict is JSON-encoded; this preserves dict-of-dict
    payloads (``reward_model``, ``extra_info``) without introducing a
    protobuf ``Any`` dance. Non-JSON-serializable values raise here at
    push time, NOT at sample time on the trainer.
    """
    has_logprobs = s.behavior_log_probs is not None
    return live_store_pb2.TrainingSample(
        sample_uid=s.sample_uid,
        group_uid=s.group_uid,
        episode_uid=s.episode_uid,
        prompt_token_ids=_pack_int32(s.prompt_token_ids),
        response_token_ids=_pack_int32(s.response_token_ids),
        response_loss_mask=_pack_int8(s.response_loss_mask),
        behavior_log_probs=_pack_float32(s.behavior_log_probs or ()),
        has_behavior_log_probs=has_logprobs,
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
        trust_level=str(s.trust_level.value),
        sample_indices=_pack_int32(s.sample_indices or ()),
        instance_json=json.dumps(s.instance, default=_json_fallback),
        error=s.error or '',
        is_padded=s.is_padded,
    )


def from_proto(p: live_store_pb2.TrainingSample) -> TrainingSample:
    """Deserialize a proto into a Python :class:`TrainingSample`.

    Inverse of :func:`to_proto`. ``error`` empty-string maps back to
    ``None``; ``has_behavior_log_probs=False`` maps to ``None``.
    """
    blp = _unpack_float32(p.behavior_log_probs) if p.has_behavior_log_probs else None
    sidx = _unpack_int32(p.sample_indices) if p.sample_indices else None
    return TrainingSample(
        sample_uid=p.sample_uid,
        group_uid=p.group_uid,
        episode_uid=p.episode_uid,
        prompt_token_ids=_unpack_int32(p.prompt_token_ids),
        response_token_ids=_unpack_int32(p.response_token_ids),
        response_loss_mask=_unpack_int8(p.response_loss_mask),
        behavior_log_probs=blp,
        reward=float(p.reward),
        raw_reward=float(p.raw_reward),
        truncated=bool(p.truncated),
        behavior_policy_version=int(p.behavior_policy_version),
        created_at_step=int(p.created_at_step),
        task_id=p.task_id,
        split=p.split,
        policy_id=p.policy_id,
        environment_id=p.environment_id,
        environment_version=p.environment_version,
        verifier_version=p.verifier_version,
        trust_level=TrustLevel(p.trust_level)
        if p.trust_level
        else TrustLevel.OWN_FABRIC,
        sample_indices=sidx,
        instance=json.loads(p.instance_json) if p.instance_json else {},
        error=p.error or None,
        is_padded=bool(p.is_padded),
    )


def _json_fallback(obj: Any) -> Any:
    """JSON encoder fallback for instance-dict values that aren't JSON-native.

    The legacy producer occasionally puts numpy scalars or sets in
    ``prompt_extras`` / ``instance``. We coerce them lossily but
    auditably (the value's ``repr`` is preserved).
    """
    try:
        return obj.tolist()  # numpy
    except AttributeError:
        pass
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return repr(obj)


# ---- DataProto -> list[TrainingSample] -------------------------------------


def dataproto_to_samples(
    dp: Any,
    *,
    behavior_policy_version: int,
    current_step: int,
    policy_id: str,
    environment_id: str,
    environment_version: str = '',
    verifier_version: str = '',
    split: str = 'train',
    episode_uid_prefix: str = 'ep',
    trust_level: TrustLevel = TrustLevel.OWN_FABRIC,
) -> list[TrainingSample]:
    """Lifted shape of legacy ``push_from_dataproto``; emits §6.2 samples.

    Bin rows by ``uid`` (the producer's group key); whole-group integrity
    is preserved (§3.2). Returns a flat list ordered group-then-row;
    callers reassemble groups by ``group_uid``.

    The per-row ``behavior_policy_version`` stamp prefers the row's
    ``instance['policy_version']`` (set by ``DataProto2Messages`` at
    expansion time) and falls back to the call-level scalar — this is
    the load-bearing §3.5 invariant.
    """
    tensors = dp.batch
    non_tensors = dp.non_tensor_batch
    if 'uid' not in non_tensors:
        raise KeyError(
            "dataproto_to_samples: DataProto missing 'uid' non-tensor; "
            'the trainer stamps it before union — is the seam placed correctly?'
        )
    batch_size = int(tensors['responses'].shape[0])
    prompt_len = int(tensors['input_ids'].shape[1] - tensors['responses'].shape[1])

    input_ids = tensors['input_ids'].cpu()
    responses = tensors['responses'].cpu()
    attention_mask = tensors['attention_mask'].cpu()
    loss_mask = tensors['loss_mask'].cpu()
    rollout_log_probs = tensors['rollout_log_probs'].cpu()
    is_padded_t = tensors['is_padded'].cpu()
    error_mask_t = tensors['error_mask'].cpu()

    # Reward (today's seam may or may not have it; default 0.0).
    if 'token_level_rewards' in tensors:
        per_row_reward = tensors['token_level_rewards'].cpu().sum(dim=-1).tolist()
    elif 'reward' in tensors and tensors['reward'].ndim == 1:
        per_row_reward = tensors['reward'].cpu().tolist()
    else:
        per_row_reward = [0.0] * batch_size
    # ``raw_reward`` semantics: today's verl path doesn't separate it;
    # mirror ``reward`` so downstream consumers can still compute their
    # own normalization. ROLL/slime adapters that DO carry pre-norm
    # reward will populate it from a different field.
    per_row_raw_reward = list(per_row_reward)

    uids = non_tensors['uid']
    success_arr = non_tensors.get('success')
    error_arr = non_tensors.get('error')
    resolved_arr = non_tensors.get('resolved')
    finish_arr = non_tensors.get('finish')
    instance_arr = non_tensors.get('instance')

    # All non-typed non_tensor keys live in instance['_extras'] so
    # downstream reward managers see exactly what they used to see.
    known = {'uid', 'success', 'error', 'resolved', 'finish', 'instance'}
    extra_keys = [k for k in non_tensors if k not in known]

    samples: list[TrainingSample] = []
    for i in range(batch_size):
        uid = str(uids[i])
        full_attn = attention_mask[i]
        prompt_attn = full_attn[:prompt_len]
        response_attn = full_attn[prompt_len:]
        prompt_tokens = tuple(
            int(x) for x in input_ids[i, :prompt_len][prompt_attn.bool()].tolist()
        )
        r_valid_len = int(response_attn.sum().item())
        response_tokens = tuple(int(x) for x in responses[i, :r_valid_len].tolist())
        response_lp = tuple(
            float(x) for x in rollout_log_probs[i, :r_valid_len].tolist()
        )
        response_lm = tuple(int(x) for x in loss_mask[i, :r_valid_len].tolist())

        row_instance: dict[str, Any] = {}
        if instance_arr is not None and isinstance(instance_arr[i], dict):
            row_instance = dict(instance_arr[i])
        # Per-row policy version: prefer the per-row stamp, fall back to
        # the call-level scalar (§3.5).
        row_pv = int(row_instance.get('policy_version', behavior_policy_version) or 0)

        # Carry env-level booleans via the instance bag so the wire
        # schema remains §6.2-clean. Reward managers downstream that
        # used to read ``non_tensors['success']`` etc. now read
        # ``instance['success']``.
        if success_arr is not None:
            row_instance['success'] = _bool_scalar(success_arr, i)
        if resolved_arr is not None:
            row_instance['resolved'] = _bool_scalar(resolved_arr, i)
        if finish_arr is not None:
            row_instance['finish'] = _bool_scalar(finish_arr, i)
        # Truncation: today's contract is finish==False ⇒ truncated.
        truncated = not bool(row_instance.get('finish', True))

        extras: dict[str, Any] = {}
        for k in extra_keys:
            v = non_tensors[k][i]
            extras[k] = dict(v) if isinstance(v, dict) else v
        if extras:
            row_instance.setdefault('_extras', {}).update(extras)

        err: str | None = None
        if error_arr is not None:
            v = error_arr[i]
            s = str(v) if v is not None else ''
            err = s or None
        if not err and bool(error_mask_t[i].item()):
            err = 'error_mask_set'

        sample = TrainingSample(
            sample_uid=uid,
            group_uid=uid,
            episode_uid=f'{episode_uid_prefix}-{uid}',
            prompt_token_ids=prompt_tokens,
            response_token_ids=response_tokens,
            response_loss_mask=response_lm,
            behavior_log_probs=response_lp if response_lp else None,
            reward=float(per_row_reward[i]),
            raw_reward=float(per_row_raw_reward[i]),
            truncated=truncated,
            behavior_policy_version=row_pv,
            created_at_step=current_step,
            task_id=str(row_instance.get('task_id', uid)),
            split=split,
            policy_id=policy_id,
            environment_id=environment_id,
            environment_version=environment_version,
            verifier_version=verifier_version,
            trust_level=trust_level,
            sample_indices=None,
            instance=row_instance,
            error=err,
            is_padded=bool(is_padded_t[i].item()),
        )
        samples.append(sample)
    return samples


def _bool_scalar(arr: Any, i: int) -> bool:
    v = arr[i]
    if hasattr(v, 'item'):
        return bool(v.item())
    return bool(v)
