"""gRPC client for the LiveStore (slot 5.4).

Used by the rollout manager (push side) and the trainer (get_batch side).
Drop-in replacement for the legacy in-process ``TrajectoryStore``.

Wire is unpadded (§6.2). Trainer calls ``get_batch()`` then packs locally
via ``trainers/verl/verl_custom/fabric_adapter/live_store_batch.py``.
"""

from __future__ import annotations

import logging

import grpc

from rollout_fabric.live_store.codec import from_proto, to_proto
from rollout_fabric.live_store.store_core import (
    InsufficientTrajectoriesError,
    StoreMetrics,
)
from rollout_fabric.schemas._gen import live_store_pb2, live_store_pb2_grpc
from rollout_fabric.schemas.protocols.live_store import NoProgressError
from rollout_fabric.schemas.training_sample import TrainingSample

logger = logging.getLogger(__name__)

_MAX_MB = 256


class LiveStoreClient:
    """gRPC client wrapping push_group / get_batch / metrics."""

    def __init__(
        self,
        socket_path: str,
        *,
        policy_id: str,
        environment_id: str,
        environment_version: str = '',
        verifier_version: str = '',
        split: str = 'train',
        pad_token_id: int = 0,
        prompt_length_cap: int | None = None,
        response_length_cap: int | None = None,
    ) -> None:
        opts = [
            ('grpc.max_receive_message_length', _MAX_MB * 1024 * 1024),
            ('grpc.max_send_message_length', _MAX_MB * 1024 * 1024),
        ]
        self._channel = grpc.insecure_channel(f'unix:{socket_path}', options=opts)
        self._stub = live_store_pb2_grpc.LiveStoreStub(self._channel)
        self._policy_id = policy_id
        self._environment_id = environment_id
        self._pad_token_id = int(pad_token_id)
        self._prompt_cap = prompt_length_cap
        self._response_cap = response_length_cap
        self._environment_version = environment_version
        self._verifier_version = verifier_version
        self._split = split

    # ---- ingest -----------------------------------------------------------

    def push_group(self, samples: list[TrainingSample]) -> int:
        """Append one group. Returns post-push store size."""
        if not samples:
            raise ValueError('push_group requires at least one sample')
        req = live_store_pb2.PushGroupRequest(
            group_uid=samples[0].group_uid,
            producer_id='rollout_manager',
            records=[to_proto(s) for s in samples],
        )
        resp = self._stub.PushGroup(req)
        return resp.store_size

    # ---- sample / introspect ----------------------------------------------

    def get_batch(
        self,
        *,
        n_groups: int,
        current_step: int,
        timeout_ms: int,
    ) -> list[TrainingSample]:
        """Pop ``n_groups`` non-stale groups. Raises on no-progress (BC-5)."""
        req = live_store_pb2.GetBatchRequest(
            n_groups=n_groups,
            current_step=current_step,
            staleness_cutoff_k=0,
            timeout_ms=timeout_ms,
        )
        try:
            resp = self._stub.GetBatch(req)
        except grpc.RpcError as exc:
            if (
                getattr(exc, 'code', lambda: None)()
                == grpc.StatusCode.DEADLINE_EXCEEDED
            ):
                raise InsufficientTrajectoriesError(exc.details()) from exc
            raise
        if resp.no_progress:
            raise NoProgressError(resp.no_progress_reason or 'no progress')
        return [from_proto(p) for p in resp.samples]

    def metrics(self, current_step: int, suffix: str = '') -> dict[str, float]:
        m = self._raw_metrics(current_step)
        out: dict[str, float] = {
            f'replay/store_size{suffix}': float(m.store_size),
            f'replay/store_fill_ratio{suffix}': float(m.fill_ratio),
            f'replay/store_num_trajectories{suffix}': float(m.num_trajectories),
            f'replay/store_age_p50{suffix}': float(m.age_p50),
            f'replay/store_age_p95{suffix}': float(m.age_p95),
        }
        if not suffix:
            out['replay/dropped_by_staleness_total'] = float(
                m.dropped_by_staleness_total
            )
        return out

    def _raw_metrics(self, current_step: int) -> StoreMetrics:
        resp = self._stub.GetMetrics(
            live_store_pb2.GetMetricsRequest(current_step=current_step)
        )
        return StoreMetrics(
            store_size=resp.store_size,
            fill_ratio=resp.fill_ratio,
            num_trajectories=resp.num_trajectories,
            age_p50=resp.age_p50,
            age_p95=resp.age_p95,
            dropped_by_staleness_total=resp.dropped_by_staleness_total,
            total_pushes=resp.total_pushes,
        )

    def num_groups(self) -> int:
        return self._raw_metrics(0).store_size

    def total_pushes(self) -> int:
        return self._raw_metrics(0).total_pushes

    def num_fresh_groups(self, current_step: int) -> int:
        return self._raw_metrics(current_step).store_size

    def notify_policy_version(self, version: int, adapter_uri: str) -> None:
        self._stub.NotifyPolicyVersion(
            live_store_pb2.NotifyPolicyVersionRequest(
                version=version, adapter_uri=adapter_uri
            )
        )

    def close(self) -> None:
        self._channel.close()
