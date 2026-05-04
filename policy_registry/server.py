"""S4 — gRPC PolicyRegistry service (single source of truth).

Invariants:
- §3.3 abort gate: ``publish_policy_version`` returns success ONLY when
  every pool child ACKed. Trainer raises on failure; worker sees no update.
- Streaming subscriptions: worker receives version updates within 5s (BC-7).
- SQLite storage: publish history queryable by policy_id.
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

_DDL = """
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
    def __init__(self, *, db_path: str | Path, pool_endpoints: list[str]) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pool_endpoints = list(pool_endpoints)
        self._init_db()
        # Per-policy CV: subscribers wait; publish notifies all.
        self._cv_per_policy: dict[str, threading.Condition] = defaultdict(
            lambda: threading.Condition(threading.Lock())
        )

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript(_DDL)

    def PublishPolicyVersion(self, request, context):  # noqa: N802
        # Step 1: fanout with abort gate (BC-9 / §3.3)
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
            # §3.3: do NOT commit on partial failure. Worker sees no update.
            return policy_registry_pb2.PublishPolicyVersionResponse(
                success=False,
                endpoints_ok=result.endpoints_ok,
                endpoints_failed=result.endpoints_failed,
                latency_s=result.latency_s,
                error=result.error or '',
            )
        # Step 2: commit to SQLite
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
        # Step 3: wake subscribers
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
        row = self._get_latest(request.policy_id)
        if row is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f'no publishes for policy_id={request.policy_id!r}',
            )
            raise RuntimeError
        return policy_registry_pb2.VersionInfo(
            policy_id=row[0],
            version=row[1],
            adapter_uri=row[2],
            published_at=row[4],
        )

    def SubscribeVersionUpdates(self, request, context):  # noqa: N802
        """Server-streaming: initial flush + CV-based updates (BC-7)."""
        policy_id = request.policy_id
        cv = self._cv_per_policy[policy_id]
        last_sent = -1
        row = self._get_latest(policy_id)
        if row is not None and row[1] > last_sent:
            yield policy_registry_pb2.PolicyVersionSnapshot(
                policy_id=row[0],
                version=row[1],
                adapter_uri=row[2],
                received_at=time.monotonic(),
            )
            last_sent = row[1]
        while context.is_active():
            with cv:
                cv.wait(timeout=1.0)
            row = self._get_latest(policy_id)
            if row is None or row[1] <= last_sent:
                continue
            yield policy_registry_pb2.PolicyVersionSnapshot(
                policy_id=row[0],
                version=row[1],
                adapter_uri=row[2],
                received_at=time.monotonic(),
            )
            last_sent = row[1]

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

    def _get_latest(self, policy_id: str) -> tuple | None:
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                'SELECT policy_id, version, adapter_uri, trainer_id, published_at '
                'FROM publishes WHERE policy_id = ? ORDER BY version DESC LIMIT 1',
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
