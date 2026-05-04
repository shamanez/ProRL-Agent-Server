"""S4 — §3.3 abort gate moves into the PolicyRegistry's fanout.

The trainer publishes via :class:`PolicyRegistryClient.publish_policy_version`.
The registry runs ``fanout_to_pool``; if any pool child fails, the
registry returns ``success=False`` and DOES NOT commit to SQLite.
The trainer raises :class:`PublishFailedError`.

Worker subscribers must NOT see a version update for a failed
publish — verified via ``get_latest_version`` returning the prior
version (or NOT_FOUND) after a fanout failure.
"""

from __future__ import annotations

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from policy_registry.client import PolicyRegistryClient, PublishFailedError
from policy_registry.server import serve

pytestmark = pytest.mark.contract


class _PoolStub(BaseHTTPRequestHandler):
    """Per-test pool stub that lets us script per-port responses."""

    statuses: dict[int, int] = {}  # port → status code to return

    def log_message(self, format, *args):  # silence
        pass

    def do_POST(self):  # noqa: N802
        if self.path != '/reload_lora':
            self.send_response(404)
            self.end_headers()
            return
        port = self.server.server_address[1]
        status = _PoolStub.statuses.get(port, 200)
        # Drain the body so the client doesn't block.
        length = int(self.headers.get('Content-Length', '0'))
        if length:
            self.rfile.read(length)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"vllm_load_latency_ms": 10}')


def _start_pool_stub() -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(('127.0.0.1', 0), _PoolStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture()
def two_pool_endpoints():
    p1, t1 = _start_pool_stub()
    p2, t2 = _start_pool_stub()
    endpoints = [
        f'http://{p1.server_address[0]}:{p1.server_address[1]}',
        f'http://{p2.server_address[0]}:{p2.server_address[1]}',
    ]
    yield endpoints, [p1, p2]
    for p in (p1, p2):
        p.shutdown()
    _PoolStub.statuses.clear()


@pytest.fixture()
def registry(two_pool_endpoints, tmp_path):
    endpoints, _ = two_pool_endpoints
    socket = str(tmp_path / 'registry.sock')
    db = str(tmp_path / 'registry.db')
    server = serve(socket_path=socket, db_path=db, pool_endpoints=endpoints)
    # Bounded settle.
    for _ in range(50):
        if os.path.exists(socket):
            break
        time.sleep(0.01)
    yield socket, server
    server.stop(grace=1.0)


def _make_adapter(tmp_path):
    """Write a fake adapter dir the fanout can tar up."""
    d = tmp_path / 'adapter'
    d.mkdir()
    (d / 'adapter_model.safetensors').write_bytes(b'fake-weights')
    (d / 'adapter_config.json').write_text('{}')
    return f'file://{d.resolve()}'


def test_abort_gate_partial_failure_blocks_commit(
    two_pool_endpoints, registry, tmp_path
) -> None:
    """If one pool child returns 500, registry returns success=False
    and the SQLite has no row for this version (worker stays at prior).
    """
    endpoints, [pool1, pool2] = two_pool_endpoints
    # Pool-2 fails.
    _PoolStub.statuses[pool2.server_address[1]] = 500

    socket, _ = registry
    client = PolicyRegistryClient(socket)
    try:
        adapter_uri = _make_adapter(tmp_path)
        with pytest.raises(PublishFailedError, match='ABORT'):
            client.publish_policy_version(
                policy_id='qwen3-4b-skyrl',
                version=1,
                adapter_uri=adapter_uri,
                trainer_id='trainer-0',
            )
        # NOT_FOUND because the failed publish never committed.
        assert client.get_latest_version('qwen3-4b-skyrl') is None
    finally:
        client.close()


def test_full_success_commits_and_makes_queryable(
    two_pool_endpoints, registry, tmp_path
) -> None:
    socket, _ = registry
    client = PolicyRegistryClient(socket)
    try:
        adapter_uri = _make_adapter(tmp_path)
        metrics = client.publish_policy_version(
            policy_id='qwen3-4b-skyrl',
            version=7,
            adapter_uri=adapter_uri,
            trainer_id='trainer-0',
        )
        assert metrics['weight_sync/endpoints_ok'] == 2
        assert metrics['weight_sync/endpoints_failed'] == 0
        info = client.get_latest_version('qwen3-4b-skyrl')
        assert info is not None
        assert info['version'] == 7
        assert info['policy_id'] == 'qwen3-4b-skyrl'
    finally:
        client.close()


def test_409_idempotent_replay_treated_as_success(
    two_pool_endpoints, registry, tmp_path
) -> None:
    """409 from the pool means "already installed" — counts as success.

    Mirrors the pre-S4 contract; pool-side dedup must not fail-the-run.
    """
    endpoints, [pool1, pool2] = two_pool_endpoints
    _PoolStub.statuses[pool1.server_address[1]] = 409
    _PoolStub.statuses[pool2.server_address[1]] = 200

    socket, _ = registry
    client = PolicyRegistryClient(socket)
    try:
        adapter_uri = _make_adapter(tmp_path)
        metrics = client.publish_policy_version(
            policy_id='qwen3-4b-skyrl',
            version=1,
            adapter_uri=adapter_uri,
        )
        assert metrics['weight_sync/endpoints_ok'] == 2
    finally:
        client.close()
