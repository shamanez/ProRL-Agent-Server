"""gRPC client for the LiveStore (slot 5.4).

Two consumers:

* The **producer** in the rollout worker calls :meth:`push_group` and
  :meth:`push_from_dataproto` (the latter wraps :func:`codec.dataproto_to_samples`
  + :meth:`push_group` so the producer's call sites change minimally).
* The **trainer** calls :meth:`get_batch`, :meth:`metrics`,
  :meth:`num_groups`, :meth:`total_pushes`, :meth:`num_fresh_groups`.

Surface intentionally mirrors today's ``TrajectoryStore`` so the
constructor swap at ``ray_trainer.py:438`` is the only diff in the
trainer; the call sites in
``trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py``
(and ``ray_trainer_dapo.py``) keep their existing names.

Per §6.2 padding stance, :meth:`get_batch` returns unpadded samples;
the trainer adapter's ``trainer_adapters/verl/pad.pack_unpadded_groups``
turns them into a ``SampledMiniBatch``.
"""

from __future__ import annotations

import logging
from typing import Any

import grpc

from live_store.codec import dataproto_to_samples, from_proto, to_proto
from live_store.store_core import (
    InsufficientTrajectoriesError,
    StoreMetrics,
)
from schemas._gen import live_store_pb2, live_store_pb2_grpc
from schemas.protocols.live_store import NoProgressError
from schemas.training_sample import TrainingSample
from trainer_adapters.verl.pad import SampledMiniBatch, pack_unpadded_groups

logger = logging.getLogger(__name__)

DEFAULT_MAX_RECV_MB = 256
DEFAULT_MAX_SEND_MB = 256


class LiveStoreClient:
    """Drop-in replacement for the in-process ``TrajectoryStore``.

    Parameters
    ----------
    socket_path:
        Path of the UDS the server is bound to.
    policy_id, environment_id, environment_version, verifier_version,
    split:
        Provenance fields stamped on every sample produced via
        :meth:`push_from_dataproto`. The trainer's config supplies them
        once at construction; they're constant across pushes for a
        single worker / trainer.
    """

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
            ('grpc.max_receive_message_length', DEFAULT_MAX_RECV_MB * 1024 * 1024),
            ('grpc.max_send_message_length', DEFAULT_MAX_SEND_MB * 1024 * 1024),
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

    # ---- ingest -------------------------------------------------------------

    def push_group(self, samples: list[TrainingSample]) -> int:
        """Append one group via the gRPC ``PushGroup`` RPC.

        Returns the post-push store size. The caller is expected to
        respect §3.2 — all samples share ``group_uid``.
        """
        if not samples:
            raise ValueError('push_group requires at least one sample')
        req = live_store_pb2.PushGroupRequest(
            group_uid=samples[0].group_uid,
            producer_id='rollout_worker',
            records=[to_proto(s) for s in samples],
        )
        resp = self._stub.PushGroup(req)
        return resp.store_size

    def push_from_dataproto(
        self,
        dp: Any,
        *,
        behavior_policy_version: int,
        current_step: int,
    ) -> int:
        """Wraps ``codec.dataproto_to_samples`` + per-group ``push_group``.

        Bin samples by ``group_uid`` and push each group as a unit.
        Mirrors today's atomicity guarantee — under pop-on-sample, a
        partially-pushed batch corrupts ``behavior_policy_version`` /
        ``created_at_step`` tracking, so groups are emitted whole.
        """
        flat = dataproto_to_samples(
            dp,
            behavior_policy_version=behavior_policy_version,
            current_step=current_step,
            policy_id=self._policy_id,
            environment_id=self._environment_id,
            environment_version=self._environment_version,
            verifier_version=self._verifier_version,
            split=self._split,
        )
        # Bin by group_uid preserving insertion order.
        groups: dict[str, list[TrainingSample]] = {}
        for s in flat:
            groups.setdefault(s.group_uid, []).append(s)
        n_pushed = 0
        for guid, samples in groups.items():
            self.push_group(samples)
            n_pushed += 1
        return n_pushed

    # ---- sample / introspect -----------------------------------------------

    def get_batch(
        self,
        *,
        n_groups: int,
        current_step: int,
        timeout_ms: int,
    ) -> list[TrainingSample]:
        """Pop ``n_groups`` non-stale groups; raise on no-progress.

        Blocks server-side; the trainer no longer busy-loops. Per the
        plan, raises :class:`NoProgressError` (not ``KeyboardInterrupt``)
        when the producer is wedged for ``no_progress_timeout_s``.
        """
        req = live_store_pb2.GetBatchRequest(
            n_groups=n_groups,
            current_step=current_step,
            staleness_cutoff_k=0,  # server-side cutoff comes from StoreCore
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

    def sample_mini_batch(
        self,
        n_groups: int,
        current_step: int,
        rng: Any
        | None = None,  # accepted for signature compatibility; ignored server-side
    ) -> SampledMiniBatch:
        """Compatibility shim mirroring today's ``TrajectoryStore.sample_mini_batch``.

        Returns a :class:`SampledMiniBatch` (verl tensor/non-tensor
        layout) so trainer call sites swap only the constructor of the
        store. The wire is unpadded (§6.2); padding happens here in the
        client via :func:`trainer_adapters.verl.pad.pack_unpadded_groups`.
        """
        # Default timeout: wait up to 10 s per call. The server's
        # no-progress detector covers the wedged-producer case; the
        # ``InsufficientTrajectoriesError`` covers buffer-warming cases
        # the trainer wants to retry on.
        samples = self.get_batch(
            n_groups=n_groups,
            current_step=current_step,
            timeout_ms=10_000,
        )
        return pack_unpadded_groups(
            samples,
            pad_token_id=self._pad_token_id,
            prompt_length_cap=self._prompt_cap,
            response_length_cap=self._response_cap,
            current_step=current_step,
        )

    def metrics(self, current_step: int, suffix: str = '') -> dict[str, float]:
        """Mirror today's metrics dict. Suffix appended for pre/post-sample logging."""
        m: StoreMetrics = self._raw_metrics(current_step)
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
        return self._raw_metrics(current_step=0).store_size

    def total_pushes(self) -> int:
        return self._raw_metrics(current_step=0).total_pushes

    def num_fresh_groups(self, current_step: int) -> int:
        # Best-effort: server-side staleness eviction happens at sample
        # time, so this returns the same value as ``num_groups`` modulo
        # the staleness window. The trainer no-progress detector no
        # longer relies on this — the server enforces it.
        return self._raw_metrics(current_step).store_size

    def notify_policy_version(self, version: int, adapter_uri: str) -> None:
        self._stub.NotifyPolicyVersion(
            live_store_pb2.NotifyPolicyVersionRequest(
                version=version,
                adapter_uri=adapter_uri,
            )
        )

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._channel.close()
