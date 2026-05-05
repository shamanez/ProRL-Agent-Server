"""S4 — gRPC subscription feeds atomic-ref-swap cache.

The :class:`PolicyVersionCache` and its atomic-ref-swap primitive are
preserved across the S2→S4 cut; only the populator flips. This test
exercises the new populator (gRPC streaming) against a live registry.
Reads remain lock-free; writes go through the same
``cache.update(snap)`` API.
"""

from __future__ import annotations

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rollout_fabric.policy_registry.client import PolicyRegistryClient
from rollout_fabric.policy_registry.server import serve
from rollout_fabric.rollout_manager.policy_subscription import GrpcStreamingPolicySubscription
from rollout_fabric.schemas.policy_version import PolicyVersionCache, PolicyVersionSnapshot

pytestmark = pytest.mark.contract


class _AlwaysOK(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get('Content-Length', '0'))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')


def _start_pool() -> HTTPServer:
    s = HTTPServer(('127.0.0.1', 0), _AlwaysOK)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def _make_adapter(tmp_path):
    d = tmp_path / 'adapter'
    d.mkdir(exist_ok=True)
    (d / 'adapter_model.safetensors').write_bytes(b'x')
    (d / 'adapter_config.json').write_text('{}')
    return f'file://{d.resolve()}'


def test_streaming_subscription_swaps_cache(tmp_path) -> None:
    pool = _start_pool()
    endpoint = f'http://{pool.server_address[0]}:{pool.server_address[1]}'
    socket = str(tmp_path / 'registry.sock')
    db = str(tmp_path / 'registry.db')
    server = serve(
        socket_path=socket,
        db_path=db,
        pool_endpoints=[endpoint],
        manifest_path=str(tmp_path / 'manifest.json'),
    )
    try:
        for _ in range(50):
            if os.path.exists(socket):
                break
            time.sleep(0.01)

        # Worker side: cleverest cache + gRPC subscription.
        cache = PolicyVersionCache(PolicyVersionSnapshot.bootstrap('qwen3-4b-skyrl'))
        client = PolicyRegistryClient(socket)
        sub = GrpcStreamingPolicySubscription(
            cache=cache,
            registry_client=client,
            policy_id='qwen3-4b-skyrl',
        )
        sub.start()

        # Trainer side: another client; publish three versions.
        publisher = PolicyRegistryClient(socket)
        adapter_uri = _make_adapter(tmp_path)
        for v in (1, 2, 3):
            publisher.publish_policy_version(
                policy_id='qwen3-4b-skyrl',
                version=v,
                adapter_uri=adapter_uri,
            )

        # Bounded settle for the stream to deliver v=3.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if cache.snapshot().version == 3:
                break
            time.sleep(0.05)

        assert cache.snapshot().version == 3, (
            f'cache stuck at version={cache.snapshot().version}'
        )
        # Lock-free read still returns an immutable snapshot.
        snap = cache.snapshot()
        assert snap.policy_id == 'qwen3-4b-skyrl'
        assert snap.adapter_uri.startswith('file://')

        sub.stop(timeout=2.0)
        client.close()
        publisher.close()
    finally:
        server.stop(grace=1.0)
        pool.shutdown()


def test_subscription_reconnect_no_stale_version(tmp_path) -> None:
    """Drop the channel mid-run; reconnect; cache stays fresh.

    The cache's ``update`` rejects strictly-not-greater versions, so
    even if the registry replays the latest snapshot on reconnect,
    the cache does not regress. Post-S4 checklist item 19.
    """
    pool = _start_pool()
    endpoint = f'http://{pool.server_address[0]}:{pool.server_address[1]}'
    socket = str(tmp_path / 'registry.sock')
    db = str(tmp_path / 'registry.db')
    server = serve(
        socket_path=socket,
        db_path=db,
        pool_endpoints=[endpoint],
        manifest_path=str(tmp_path / 'manifest.json'),
    )
    try:
        for _ in range(50):
            if os.path.exists(socket):
                break
            time.sleep(0.01)

        cache = PolicyVersionCache(PolicyVersionSnapshot.bootstrap('qwen3-4b-skyrl'))
        client = PolicyRegistryClient(socket)
        sub = GrpcStreamingPolicySubscription(
            cache=cache,
            registry_client=client,
            policy_id='qwen3-4b-skyrl',
            reconnect_backoff_s=0.1,
        )
        sub.start()

        publisher = PolicyRegistryClient(socket)
        adapter_uri = _make_adapter(tmp_path)
        publisher.publish_policy_version(
            policy_id='qwen3-4b-skyrl',
            version=5,
            adapter_uri=adapter_uri,
        )

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if cache.snapshot().version == 5:
                break
            time.sleep(0.05)
        assert cache.snapshot().version == 5

        # Force-disconnect by closing the subscription's channel.
        client.close()
        time.sleep(0.5)  # let the subscription notice the disconnect

        # Cache must still hold v=5 — disconnect does not regress.
        assert cache.snapshot().version == 5

        sub.stop(timeout=2.0)
        publisher.close()
    finally:
        server.stop(grace=1.0)
        pool.shutdown()
