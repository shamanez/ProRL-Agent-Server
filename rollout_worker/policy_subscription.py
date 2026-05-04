"""Policy-version subscription — file-mtime poll (S2) and gRPC stream (S4).

Both implementations feed the same :class:`PolicyVersionCache` via
atomic-ref-swap. Only the populator changes across the S2→S4 cut.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Protocol

from schemas.policy_version import PolicyVersionCache, PolicyVersionSnapshot

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = '/tmp/prorl_policy_manifest.json'


class PolicyVersionStream(Protocol):
    def start(self) -> None: ...

    def stop(self, timeout: float | None = None) -> None: ...


class FilePollingPolicySubscription:
    """S2 — poll the JSON manifest the trainer writes after each pool ACK.

    Single-writer: only this daemon thread calls ``cache.update()``.
    Many readers: group dispatch threads call ``cache.snapshot()`` lock-free.

    Poll interval default 1 Hz; the publish rate is ~1/min so 1 Hz is
    well under the <5s latency target for version propagation (BC-7).
    """

    def __init__(
        self,
        *,
        cache: PolicyVersionCache,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        poll_interval_s: float = 1.0,
        on_update: Callable[[PolicyVersionSnapshot], None] | None = None,
    ) -> None:
        self._cache = cache
        self._manifest_path = manifest_path
        self._poll_interval_s = float(poll_interval_s)
        self._on_update = on_update
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_mtime_ns: int = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError('FilePollingPolicySubscription already started')
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name='PolicyVersionPoller', daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001
                logger.exception('PolicyVersionPoller iteration failed')
            self._stop_event.wait(timeout=self._poll_interval_s)

    def _poll_once(self) -> None:
        try:
            st = os.stat(self._manifest_path)
        except FileNotFoundError:
            return
        if st.st_mtime_ns <= self._last_mtime_ns:
            return
        self._last_mtime_ns = st.st_mtime_ns
        manifest = _read_manifest(self._manifest_path)
        if manifest is None:
            return
        snap = PolicyVersionSnapshot(
            policy_id=manifest['policy_id'],
            version=manifest['version'],
            adapter_uri=manifest['adapter_uri'],
            received_at=time.monotonic(),
        )
        if self._cache.update(snap):
            logger.info(
                'PolicyVersion updated: policy_id=%s version=%d uri=%s',
                snap.policy_id,
                snap.version,
                snap.adapter_uri,
            )
            if self._on_update is not None:
                try:
                    self._on_update(snap)
                except Exception:  # noqa: BLE001
                    logger.exception('on_update callback failed')


def _read_manifest(path: str) -> dict | None:
    import json  # noqa: PLC0415

    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class GrpcStreamingPolicySubscription:
    """S4 — gRPC server-streaming subscription. Same cache-populator API.

    Reconnects on disconnect; cache NOT rolled back on disconnect (workers
    continue reading last-known snapshot until a fresh one arrives).
    """

    def __init__(
        self,
        *,
        cache: PolicyVersionCache,
        registry_client: object,
        policy_id: str,
        on_update: Callable[[PolicyVersionSnapshot], None] | None = None,
        reconnect_backoff_s: float = 0.5,
        max_reconnect_backoff_s: float = 30.0,
    ) -> None:
        self._cache = cache
        self._client = registry_client
        self._policy_id = policy_id
        self._on_update = on_update
        self._reconnect_backoff_s = float(reconnect_backoff_s)
        self._max_reconnect_backoff_s = float(max_reconnect_backoff_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError('GrpcStreamingPolicySubscription already started')
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name='PolicyVersionGrpcStream', daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        backoff = self._reconnect_backoff_s
        while not self._stop_event.is_set():
            try:
                stream = self._client.stream_version_updates(self._policy_id)
                backoff = self._reconnect_backoff_s
                for snap in stream:
                    if self._stop_event.is_set():
                        break
                    if self._cache.update(snap):
                        logger.info(
                            'PolicyVersion (gRPC): policy_id=%s version=%d uri=%s',
                            snap.policy_id,
                            snap.version,
                            snap.adapter_uri,
                        )
                        if self._on_update is not None:
                            try:
                                self._on_update(snap)
                            except Exception:  # noqa: BLE001
                                logger.exception('on_update callback failed')
            except Exception:  # noqa: BLE001
                if self._stop_event.is_set():
                    break
                logger.warning(
                    'policy registry stream disconnected; reconnecting in %.1fs',
                    backoff,
                    exc_info=True,
                )
                if self._stop_event.wait(timeout=backoff):
                    break
                backoff = min(backoff * 2.0, self._max_reconnect_backoff_s)
