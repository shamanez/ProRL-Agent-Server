"""gRPC server for the LiveStore (slot 5.4).

Wraps :class:`live_store.store_core.StoreCore`. Implements the four
RPCs from ``schemas/proto/live_store.proto`` (§A.4):

* ``PushGroup``  — producer push.
* ``GetBatch``   — trainer sample (server-side blocking + no-progress).
* ``GetMetrics`` — counters for WandB.
* ``NotifyPolicyVersion`` — registry → store metrics tag.

The transport is gRPC over Unix domain socket
(``unix:/tmp/prorl_live_store.sock`` by default). Tested via
:mod:`live_store.client` against this server in the same process.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent import futures

import grpc

from live_store.codec import from_proto, to_proto
from live_store.store_core import (
    InsufficientTrajectoriesError,
    StoreCore,
)
from schemas._gen import live_store_pb2, live_store_pb2_grpc
from schemas.protocols.live_store import NoProgressError

logger = logging.getLogger(__name__)


class LiveStoreServicer(live_store_pb2_grpc.LiveStoreServicer):
    """gRPC façade over :class:`StoreCore`."""

    def __init__(self, core: StoreCore) -> None:
        self._core = core
        # Lock-protected (version, adapter_uri) for metrics tagging only.
        # Per the cleverest primitive in ``schemas/policy_version.py``,
        # the source of truth for ``policy_version`` is the
        # PolicyRegistry / worker subscription path; this is a passive
        # cache the registry pushes into for histogram tagging.
        self._policy_lock = threading.Lock()
        self._policy_version = 0
        self._policy_adapter_uri = ''

    # ---- RPCs ---------------------------------------------------------------

    def PushGroup(self, request, context):  # noqa: N802 — gRPC method casing
        try:
            samples = [from_proto(p) for p in request.records]
            store_size = self._core.push_group(samples)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise
        return live_store_pb2.PushGroupResponse(
            accepted=True,
            store_size=store_size,
            backpressure_hint_ms=0,
        )

    def GetBatch(self, request, context):  # noqa: N802
        try:
            samples = self._core.get_batch(
                n_groups=request.n_groups,
                current_step=request.current_step,
                timeout_ms=request.timeout_ms,
            )
        except NoProgressError as exc:
            return live_store_pb2.GetBatchResponse(
                samples=[],
                no_progress=True,
                no_progress_reason=str(exc),
            )
        except InsufficientTrajectoriesError as exc:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc))
            raise
        sample_protos = [to_proto(s) for s in samples]
        bvs = [s.behavior_policy_version for s in samples]
        cas = [s.created_at_step for s in samples]
        ages = [request.current_step - s.created_at_step for s in samples]
        return live_store_pb2.GetBatchResponse(
            samples=sample_protos,
            behavior_policy_versions=bvs,
            created_at_steps=cas,
            sample_ages=ages,
            no_progress=False,
        )

    def GetMetrics(self, request, context):  # noqa: N802
        m = self._core.metrics(request.current_step)
        return live_store_pb2.GetMetricsResponse(
            store_size=m.store_size,
            fill_ratio=m.fill_ratio,
            num_trajectories=m.num_trajectories,
            age_p50=m.age_p50,
            age_p95=m.age_p95,
            dropped_by_staleness_total=m.dropped_by_staleness_total,
            total_pushes=m.total_pushes,
        )

    def NotifyPolicyVersion(self, request, context):  # noqa: N802
        with self._policy_lock:
            self._policy_version = int(request.version)
            self._policy_adapter_uri = request.adapter_uri
        return live_store_pb2.Ack(ok=True, detail=f'v={request.version}')

    # ---- accessors for the metrics tagging path ----------------------------

    def policy_metrics_tag(self) -> tuple[int, str]:
        with self._policy_lock:
            return (self._policy_version, self._policy_adapter_uri)


def serve(
    *,
    socket_path: str,
    max_size: int,
    staleness_cutoff_k: int,
    no_progress_timeout_s: float = 1800.0,
    max_workers: int = 16,
) -> grpc.Server:
    """Start the LiveStore gRPC server on a Unix domain socket.

    Returns the server handle. The caller is expected to call
    ``server.wait_for_termination()`` (in :mod:`live_store.main`) or
    ``server.stop(grace=...)`` for clean shutdown.

    The socket path is rewritten to absolute and any stale socket file
    is unlinked first; gRPC's ``add_insecure_port`` does not clean up
    after a previous crash.
    """
    abs_path = os.path.abspath(socket_path)
    try:
        os.unlink(abs_path)
    except FileNotFoundError:
        pass

    core = StoreCore(
        max_size=max_size,
        staleness_cutoff_k=staleness_cutoff_k,
        no_progress_timeout_s=no_progress_timeout_s,
    )
    servicer = LiveStoreServicer(core)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    live_store_pb2_grpc.add_LiveStoreServicer_to_server(servicer, server)
    server.add_insecure_port(f'unix:{abs_path}')
    server.start()
    logger.info(
        'LiveStore listening on unix:%s (max_size=%d, k=%d)',
        abs_path,
        max_size,
        staleness_cutoff_k,
    )
    # Annotate the server with its servicer for tests that want the inner core.
    server._live_store_servicer = servicer  # type: ignore[attr-defined]
    return server
