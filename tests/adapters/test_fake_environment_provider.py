"""Fake EnvironmentProvider contract test.

Proves the architecture accepts any server implementing POST /process without
OpenHands or Singularity. A minimal in-process HTTP server proves that
ProRLClient succeeds against any compliant implementation.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from rollout_fabric.rollout_manager.prorl_client import ProRLClient


def _fake_process_response() -> dict:
    # Minimal valid ProRL /process response (see prorl_client.ProRLEpisodeResult).
    return {
        'messages': [
            {
                'role': 'assistant',
                'content': 'fake response',
                'token_ids': [1, 2, 3],  # BC-1: must be int
                'logprobs': [-0.1, -0.2, -0.3],
            }
        ],
        'resolved': True,
        'success': True,
        'finish': True,
        'reward': 1.0,
    }


class _FakeEnvHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get('Content-Length', 0))
        self.rfile.read(length)
        body = json.dumps(_fake_process_response()).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # noqa: ANN002
        pass


@pytest.fixture()
def fake_env_url():
    server = HTTPServer(('127.0.0.1', 0), _FakeEnvHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f'http://127.0.0.1:{port}'
    server.shutdown()


def test_prorl_client_works_with_fake_provider(fake_env_url: str) -> None:
    """ProRLClient must succeed against any server that implements POST /process."""
    client = ProRLClient(base_url=fake_env_url)
    result = client.run_episode(
        instance={'instance_id': 'fake-001'},
        policy_version=1,
        sampling_params={'temperature': 0.6},
    )
    assert result.resolved is True
    assert result.reward == 1.0

    # BC-1: every token_id in every message must be int
    for msg in result.messages:
        for tid in msg.get('token_ids', []):
            assert isinstance(tid, int), f'BC-1 violation: token_id {tid!r} is not int'

    client.close()
