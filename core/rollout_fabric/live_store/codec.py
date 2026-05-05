"""Wire codec — :class:`TrainingSample` ↔ protobuf.

Token arrays are packed int32-LE / int8 bytes to avoid protobuf
parser overhead on long sequences (BC-1 — token IDs never as strings).
"""

from __future__ import annotations

import json
import struct
from typing import Any

from rollout_fabric.schemas._gen import live_store_pb2
from rollout_fabric.schemas.episode_record import TrustLevel
from rollout_fabric.schemas.training_sample import TrainingSample

# ---- byte packing ---------------------------------------------------------


def _pack_int32(seq: tuple[int, ...] | list[int]) -> bytes:
    return struct.pack(f'<{len(seq)}i', *seq) if seq else b''


def _unpack_int32(buf: bytes) -> tuple[int, ...]:
    if not buf:
        return ()
    return struct.unpack(f'<{len(buf) // 4}i', buf)


def _pack_int8(seq: tuple[int, ...] | list[int]) -> bytes:
    return bytes(int(x) & 0xFF for x in seq) if seq else b''


def _unpack_int8(buf: bytes) -> tuple[int, ...]:
    return tuple(int(b) for b in buf)


def _pack_float32(seq: tuple[float, ...] | list[float]) -> bytes:
    return struct.pack(f'<{len(seq)}f', *seq) if seq else b''


def _unpack_float32(buf: bytes) -> tuple[float, ...]:
    if not buf:
        return ()
    return struct.unpack(f'<{len(buf) // 4}f', buf)


# ---- TrainingSample <-> proto ---------------------------------------------


def to_proto(s: TrainingSample) -> live_store_pb2.TrainingSample:
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
    try:
        return obj.tolist()  # numpy
    except AttributeError:
        pass
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return repr(obj)
