"""S2 — JSON-mtime poller that populates :class:`PolicyVersionCache`.

The worker subscribes to policy version updates by polling the trainer's
manifest file at 1 Hz (configurable). Each strictly-fresher version
seen is shipped to the cache via :meth:`PolicyVersionCache.update`,
which atomically swaps a new immutable :class:`PolicyVersionSnapshot`
into the cache's single attribute. Reader threads (group dispatch)
observe the new snapshot on their next ``cache.snapshot()`` call —
zero lock contention on the read path.

The poller is a daemon thread. It owns no state beyond the file path
and the cache reference; the cache itself is the canonical home of the
current snapshot. On worker restart, the next poll cycle re-establishes
the latest version from the manifest.

S4 replaces this with a gRPC streaming subscription
(``policy_registry/client.py``). The cache + atomic-ref-swap primitive
is preserved across the cut; only the populator flips. This is the
load-bearing pattern ``rollout_fabric.md`` operating-principle 6
prescribes.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Protocol

from policy_registry.file_registry import (
    DEFAULT_MANIFEST_PATH,
    PolicyManifest,
    read_manifest,
)
from schemas.policy_version import PolicyVersionCache, PolicyVersionSnapshot

logger = logging.getLogger(__name__)


class PolicyVersionStream(Protocol):
    """Subscription source.

    Two concrete implementations: :class:`FilePollingPolicySubscription`
    at S2 and the gRPC streaming client at S4.
    """

    def start(self) -> None: ...
    def stop(self, timeout: float | None = None) -> None: ...


class FilePollingPolicySubscription:
    """Poll the policy manifest file; feed updates into a cache.

    Parameters
    ----------
    cache:
        The :class:`PolicyVersionCache` to populate. Single-writer
        contract: only this poller's thread writes; many reader threads
        do lock-free attribute loads via ``cache.snapshot()``.
    manifest_path:
        Path to the JSON file the trainer atomically renames after
        every successful pool publish.
    poll_interval_s:
        Seconds between mtime polls. Default 1.0; the publish rate is
        ~once per ``save_freq`` (~1/min), so 1 Hz polling is
        well under the publish-to-cache-update latency target (<1 s
        per the post-S4 checklist item 18).
    on_update:
        Optional callback invoked with the new snapshot after a
        successful ``cache.update``. Used by the worker to log or to
        notify the LiveStore for metrics tagging.
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
            target=self._run,
            name='PolicyVersionPoller',
            daemon=True,
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
        manifest = read_manifest(self._manifest_path)
        if manifest is None:
            return
        snap = PolicyVersionSnapshot(
            policy_id=manifest.policy_id,
            version=manifest.version,
            adapter_uri=manifest.adapter_uri,
            received_at=time.monotonic(),
        )
        # ``cache.update`` rejects same/older versions; we feed every
        # mtime change unconditionally and let the cache decide.
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


def _ensure_unused(_x: PolicyManifest | None) -> None:
    """Keep the ``PolicyManifest`` re-export visible to import-time linters."""
    return None


class GrpcStreamingPolicySubscription:
    """S4 — gRPC server-streaming subscription. Same cache populator.

    Replaces :class:`FilePollingPolicySubscription` at S4. The
    :class:`schemas.policy_version.PolicyVersionCache` API and the
    atomic-ref-swap primitive are unchanged across the S2→S4 cut —
    only the populator flips.

    Reconnect contract (post-S4 checklist item 19): on disconnect the
    streamer re-establishes the subscription and resumes; the cache
    is **not** rolled back. The worker dispatch threads continue to
    read the last-known snapshot until a fresh one arrives. Stale
    snapshots are NEVER injected by this code path — only the
    registry is the source of truth, and the cache only swaps on
    strictly-greater versions.

    Parameters
    ----------
    cache:
        :class:`PolicyVersionCache` to populate. Single-writer
        contract: only this thread updates.
    registry_client:
        :class:`PolicyRegistryClient` whose ``stream_version_updates``
        is consumed.
    policy_id:
        The namespace to subscribe to.
    on_update:
        Optional callback invoked on each successful cache.update.
    reconnect_backoff_s:
        Initial reconnect delay; capped at ``max_reconnect_backoff_s``.
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
            target=self._run,
            name='PolicyVersionGrpcStream',
            daemon=True,
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
                # Reset backoff on successful connection.
                backoff = self._reconnect_backoff_s
                for snap in stream:
                    if self._stop_event.is_set():
                        break
                    if self._cache.update(snap):
                        logger.info(
                            'PolicyVersion (gRPC stream): policy_id=%s '
                            'version=%d uri=%s',
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
