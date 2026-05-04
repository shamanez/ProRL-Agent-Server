"""S4 — gRPC PolicyRegistry service.

Single source of truth for the active policy version set. Receives
publishes from TrainerAdapter (slot 5.6); fans out to InferenceBackend
(slot 5.2 — vLLM children) via :mod:`policy_registry.fanout`;
broadcasts to RolloutWorker subscribers (slot 5.3) over server-side
streaming.

Critical invariants this service preserves:

* §3.3 abort gate. ``publish_policy_version`` returns
  ``success=True`` ONLY when every pool child ACKed. The trainer
  raises on ``success=False``; the worker sees no version update for
  a failed publish (so :class:`PolicyVersionCache` stays at the prior
  snapshot).
* §3.5 per-row stamping primitive. The streamed update goes to the
  worker's :class:`PolicyVersionCache.update` (atomic ref-swap).
  Reads remain lock-free.

Storage: SQLite. The publish history is small (~1/min) and queryable
by ``policy_id`` for ``get_latest_version``. Subscribers are tracked
via in-memory ``threading.Event`` per subscription.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent import futures
from pathlib import Path

import grpc

from policy_registry.fanout import fanout_to_pool
from schemas._gen import policy_registry_pb2, policy_registry_pb2_grpc
from schemas.protocols.policy_registry import PublishResult

logger = logging.getLogger(__name__)


_INDEX_DDL = """
CREATE TABLE IF NOT EXISTS namespaces (
    policy_id TEXT PRIMARY KEY,
    base_model_id TEXT NOT NULL,
    tokenizer_id TEXT NOT NULL,
    adapter_storage_root TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS publishes (
    policy_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    adapter_uri TEXT NOT NULL,
    trainer_id TEXT NOT NULL,
    published_at TEXT NOT NULL,
    PRIMARY KEY (policy_id, version)
);
CREATE INDEX IF NOT EXISTS idx_pub_latest ON publishes(policy_id, version DESC);
"""


class PolicyRegistryServicer(policy_registry_pb2_grpc.PolicyRegistryServicer):
    """gRPC implementation of §A.7 PolicyRegistry."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        pool_endpoints: list[str],
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pool_endpoints = list(pool_endpoints)
        self._init_db()
        # Per-policy_id condition variable: subscribers wait on the
        # policy's CV; publishes notify_all() to wake every streamer.
        # The streamer reads the latest publish via SQLite and emits
        # only if it's strictly fresher than what it last sent.
        self._cv_per_policy: dict[str, threading.Condition] = defaultdict(
            lambda: threading.Condition(threading.Lock())
        )

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(_INDEX_DDL)

    # ---- RPCs ---------------------------------------------------------------

    def PublishPolicyVersion(self, request, context):  # noqa: N802
        # Step 1: §3.3 abort-gate fanout. Only on success do we
        # commit to SQLite + wake subscribers.
        try:
            result: PublishResult = fanout_to_pool(
                adapter_uri=request.adapter_uri,
                new_version=request.version,
                endpoints=self._pool_endpoints,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception('publish fanout raised')
            return policy_registry_pb2.PublishPolicyVersionResponse(
                success=False,
                endpoints_ok=0,
                endpoints_failed=len(self._pool_endpoints),
                latency_s=0.0,
                error=f'{type(exc).__name__}: {exc}',
            )

        if not result.success:
            # §3.3: do NOT commit on partial failure. Worker
            # subscribers see no update; trainer aborts.
            return policy_registry_pb2.PublishPolicyVersionResponse(
                success=False,
                endpoints_ok=result.endpoints_ok,
                endpoints_failed=result.endpoints_failed,
                latency_s=result.latency_s,
                error=result.error or '',
            )

        # Step 2: commit to SQLite. Idempotent on (policy_id, version).
        published_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO publishes VALUES (?, ?, ?, ?, ?)',
                    (
                        request.policy_id,
                        int(request.version),
                        request.adapter_uri,
                        request.trainer_id,
                        published_at,
                    ),
                )
                conn.commit()

        # Step 3: wake subscribers.
        cv = self._cv_per_policy[request.policy_id]
        with cv:
            cv.notify_all()

        return policy_registry_pb2.PublishPolicyVersionResponse(
            success=True,
            endpoints_ok=result.endpoints_ok,
            endpoints_failed=0,
            latency_s=result.latency_s,
            error='',
        )

    def GetLatestVersion(self, request, context):  # noqa: N802
        row = self._get_latest_locked(request.policy_id)
        if row is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f'no publishes for policy_id={request.policy_id!r}',
            )
            raise RuntimeError  # unreachable; satisfies type checker
        return policy_registry_pb2.VersionInfo(
            policy_id=row[0],
            version=row[1],
            adapter_uri=row[2],
            published_at=row[4],
        )

    def SubscribeVersionUpdates(self, request, context):  # noqa: N802
        """Server-streaming subscription.

        Sends an immediate snapshot of the current latest, then
        blocks on the per-policy_id CV until each subsequent publish.
        On disconnect (client drops, ``context.is_active()`` flips
        false), the streamer exits cleanly.
        """
        policy_id = request.policy_id
        cv = self._cv_per_policy[policy_id]
        last_sent_version = -1

        # Initial flush.
        row = self._get_latest_locked(policy_id)
        if row is not None and row[1] > last_sent_version:
            yield policy_registry_pb2.PolicyVersionSnapshot(
                policy_id=row[0],
                version=row[1],
                adapter_uri=row[2],
                received_at=time.monotonic(),
            )
            last_sent_version = row[1]

        # Stream subsequent publishes.
        while context.is_active():
            with cv:
                cv.wait(timeout=1.0)
            row = self._get_latest_locked(policy_id)
            if row is None or row[1] <= last_sent_version:
                continue
            yield policy_registry_pb2.PolicyVersionSnapshot(
                policy_id=row[0],
                version=row[1],
                adapter_uri=row[2],
                received_at=time.monotonic(),
            )
            last_sent_version = row[1]

    def RegisterPolicyNamespace(self, request, context):  # noqa: N802
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO namespaces VALUES (?, ?, ?, ?)',
                (
                    request.policy_id,
                    request.base_model_id,
                    request.tokenizer_id,
                    request.adapter_storage_root,
                ),
            )
            conn.commit()
        return policy_registry_pb2.Ack(ok=True, detail=request.policy_id)

    # ---- internals ----------------------------------------------------------

    def _get_latest_locked(self, policy_id: str) -> tuple | None:
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                'SELECT policy_id, version, adapter_uri, trainer_id, published_at '
                'FROM publishes WHERE policy_id = ? '
                'ORDER BY version DESC LIMIT 1',
                (policy_id,),
            )
            return cur.fetchone()


def serve(
    *,
    socket_path: str,
    db_path: str,
    pool_endpoints: list[str],
    max_workers: int = 16,
) -> grpc.Server:
    """Start the registry over a Unix domain socket."""
    abs_path = os.path.abspath(socket_path)
    try:
        os.unlink(abs_path)
    except FileNotFoundError:
        pass

    servicer = PolicyRegistryServicer(db_path=db_path, pool_endpoints=pool_endpoints)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    policy_registry_pb2_grpc.add_PolicyRegistryServicer_to_server(servicer, server)
    server.add_insecure_port(f'unix:{abs_path}')
    server.start()
    logger.info(
        'PolicyRegistry listening on unix:%s (db=%s, pool=%d endpoints)',
        abs_path,
        db_path,
        len(pool_endpoints),
    )
    server._policy_registry_servicer = servicer  # type: ignore[attr-defined]
    return server
