# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pool child's POST /reload_lora endpoint.

These run in the host venv which does not have vLLM installed (vLLM lives
in the trainer Docker image and in the pool's own venv on EC2), so the
`vllm.*` modules touched by `_vllm_child` are stubbed in `sys.modules`
before the module is loaded.

Scope: endpoint behaviour only — happy path, version monotonicity, tarball
validation, engine-failure propagation. The vLLM engine itself is replaced
with an AsyncMock so `add_lora`/`remove_lora` side effects are asserted
directly, not inferred from a live engine.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_PATH = _REPO_ROOT / 'scripts' / 'serving' / '_vllm_child.py'


def _install_vllm_stubs() -> None:
    """Register stub modules for `vllm.*` so `_vllm_child` imports without vllm.

    Idempotent — safe to call once per test-session.
    """
    if 'vllm' in sys.modules:
        return

    vllm_pkg = types.ModuleType('vllm')
    vllm_pkg.__path__ = []  # type: ignore[attr-defined]  # mark as package

    def _submod(name: str, attrs: dict[str, Any] | None = None) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    class _AsyncEngineArgs:
        @staticmethod
        def add_cli_args(parser: Any) -> Any:
            return parser

        @classmethod
        def from_cli_args(cls, args: Any) -> Any:
            return cls()

    class _AsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, engine_args: Any) -> Any:
            return cls()

    class _TokensPrompt(dict):
        def __init__(self, prompt_token_ids: list[int]) -> None:
            super().__init__(prompt_token_ids=prompt_token_ids)

    class _LoRARequest:
        def __init__(self, lora_name: str, lora_int_id: int, lora_path: str) -> None:
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    class _SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    sys.modules['vllm'] = vllm_pkg
    _submod('vllm.engine', {})
    _submod('vllm.engine.arg_utils', {'AsyncEngineArgs': _AsyncEngineArgs})
    _submod('vllm.engine.async_llm_engine', {'AsyncLLMEngine': _AsyncLLMEngine})
    _submod('vllm.inputs', {'TokensPrompt': _TokensPrompt})
    _submod('vllm.lora', {})
    _submod('vllm.lora.request', {'LoRARequest': _LoRARequest})
    _submod('vllm.sampling_params', {'SamplingParams': _SamplingParams})
    _submod('vllm.utils', {'random_uuid': lambda: 'test-request-id'})


def _load_child_module() -> Any:
    """Load `_vllm_child` fresh from disk. Each call returns a new module object
    so test-to-test module-global state (active_lora, active_policy_version)
    does not leak.
    """
    _install_vllm_stubs()
    spec = importlib.util.spec_from_file_location(
        f'_vllm_child_test_{id(object())}', _CHILD_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_adapter_tarball(
    *,
    include_safetensors: bool = True,
    include_config: bool = True,
    extra_members: dict[str, bytes] | None = None,
) -> bytes:
    """Build a gzipped tarball in memory matching the PEFT layout verl emits.

    The safetensors payload is a fixed 64-byte placeholder — the endpoint
    validates file presence, not content (vLLM's add_lora would parse it, and
    we mock add_lora).
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:

        def _add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

        if include_safetensors:
            _add('adapter_model.safetensors', b'\x00' * 64)
        if include_config:
            cfg = json.dumps(
                {
                    'peft_type': 'LORA',
                    'r': 16,
                    'lora_alpha': 32,
                    'target_modules': ['q_proj'],
                }
            ).encode()
            _add('adapter_config.json', cfg)
        for name, data in (extra_members or {}).items():
            _add(name, data)
    return buf.getvalue()


@pytest.fixture()
def child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Fresh `_vllm_child` module with an AsyncMock engine and a tmp staging root."""
    module = _load_child_module()

    fake_engine = MagicMock()
    fake_engine.add_lora = AsyncMock()
    fake_engine.remove_lora = AsyncMock()
    fake_engine.generate = MagicMock()  # not exercised in this file
    module.engine = fake_engine

    monkeypatch.setattr(module, 'ADAPTER_STAGING_ROOT', tmp_path)
    module.CHILD_PORT = 8100
    return module


@pytest.fixture()
def client(child: Any) -> TestClient:
    """FastAPI TestClient triggers startup events (creates `_swap_lock`)."""
    return TestClient(child.app)


def test_health_reports_policy_version_zero_before_first_publish(
    child: Any, client: TestClient
) -> None:
    with client:
        response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'policy_version': 0}


def test_reload_lora_happy_path_installs_adapter_and_returns_200(
    child: Any, client: TestClient
) -> None:
    tarball = _make_adapter_tarball()
    with client:
        response = client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
            data={'policy_version': '1'},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['policy_version'] == 1
    assert body['adapter_bytes'] == len(tarball)
    assert body['vllm_load_latency_ms'] >= 0

    # add_lora called with a LoRARequest whose int_id matches version.
    assert child.engine.add_lora.await_count == 1
    (lora_request,) = child.engine.add_lora.await_args.args
    assert lora_request.lora_int_id == 1
    assert lora_request.lora_name == 'pv1'
    # Extracted files exist on disk at the promised path.
    assert (Path(lora_request.lora_path) / 'adapter_model.safetensors').exists()
    assert (Path(lora_request.lora_path) / 'adapter_config.json').exists()

    # No prior adapter → remove_lora not called.
    assert child.engine.remove_lora.await_count == 0

    # Module state updated.
    assert child.active_policy_version == 1
    assert child.active_lora is lora_request


def test_reload_lora_non_monotonic_returns_409_without_swap(
    child: Any, client: TestClient
) -> None:
    tarball = _make_adapter_tarball()
    with client:
        r1 = client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
            data={'policy_version': '3'},
        )
        assert r1.status_code == 200

        # Replay same version: reject.
        r2 = client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
            data={'policy_version': '3'},
        )
        assert r2.status_code == 409
        assert r2.json()['policy_version'] == 3

        # Lower version: reject.
        r3 = client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
            data={'policy_version': '2'},
        )
        assert r3.status_code == 409

    # add_lora was only called for the first (successful) publish.
    assert child.engine.add_lora.await_count == 1
    assert child.active_policy_version == 3


def test_reload_lora_retires_prior_adapter_on_success(
    child: Any, client: TestClient
) -> None:
    tarball = _make_adapter_tarball()
    with client:
        client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
            data={'policy_version': '1'},
        )
        client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
            data={'policy_version': '2'},
        )

    assert child.engine.add_lora.await_count == 2
    assert child.engine.remove_lora.await_count == 1
    # remove_lora called with the prior int_id (= 1).
    (prior_id,) = child.engine.remove_lora.await_args.args
    assert prior_id == 1
    assert child.active_policy_version == 2


def test_reload_lora_missing_required_files_returns_400(
    child: Any, client: TestClient
) -> None:
    # Tarball with only the config, no safetensors.
    tarball = _make_adapter_tarball(include_safetensors=False)
    with client:
        response = client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
            data={'policy_version': '1'},
        )
    assert response.status_code == 400
    assert 'missing required files' in response.json()['detail']
    assert child.engine.add_lora.await_count == 0
    assert child.active_policy_version == 0


def test_reload_lora_invalid_tarball_returns_400(
    child: Any, client: TestClient
) -> None:
    with client:
        response = client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', b'not-a-tarball', 'application/gzip')},
            data={'policy_version': '1'},
        )
    assert response.status_code == 400
    assert child.engine.add_lora.await_count == 0
    assert child.active_policy_version == 0


def test_reload_lora_engine_failure_returns_500_and_preserves_state(
    child: Any, client: TestClient
) -> None:
    child.engine.add_lora.side_effect = RuntimeError('shape mismatch')
    tarball = _make_adapter_tarball()
    with client:
        response = client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
            data={'policy_version': '1'},
        )
    assert response.status_code == 500
    body = response.json()
    assert body['policy_version_before'] == 0
    assert body['policy_version_after'] == 0
    assert 'shape mismatch' in body['detail']
    # State did not advance.
    assert child.active_policy_version == 0
    assert child.active_lora is None


@pytest.mark.asyncio
async def test_reload_lora_drains_prior_inflight_before_remove_lora(
    child: Any,
) -> None:
    """Regression guard for the /generate vs remove_lora race.

    Simulates one in-flight generate on the prior adapter while /reload_lora
    is publishing a new version. Asserts remove_lora(prior) is called only
    AFTER the in-flight generate has decremented _inflight.
    """
    import asyncio as _asyncio

    # Manually run the startup hook since we're not going through TestClient.
    await child._init_async_primitives()

    # Pre-install prior adapter (version 1).
    prior = child.LoRARequest(lora_name='pv1', lora_int_id=1, lora_path='/tmp/ignored')
    child.active_lora = prior
    child.active_policy_version = 1

    remove_order: list[str] = []

    async def _slow_remove(_int_id: int) -> None:
        remove_order.append(f'remove_called_{_int_id}')

    child.engine.remove_lora.side_effect = _slow_remove

    # Register an in-flight generate against the prior adapter.
    async with child._inflight_cond:
        child._inflight[prior.lora_int_id] = 1

    tarball = _make_adapter_tarball()

    # Kick off the reload: it must block at the drain wait.
    from fastapi.testclient import TestClient as _TC  # noqa: PLC0415

    def _publish() -> None:
        with _TC(child.app) as c:
            resp = c.post(
                '/reload_lora',
                files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
                data={'policy_version': '2'},
            )
            assert resp.status_code == 200

    # We can't cleanly drive the drain wait + notify across two sync TestClient
    # invocations, so exercise the internal machinery directly: simulate the
    # generate-finally path releasing the count, then assert the draining
    # reload proceeds.
    async def _simulate_generate_finish() -> None:
        # Give the reload task a chance to enter the drain wait first.
        await _asyncio.sleep(0.05)
        remove_order.append('generate_decremented')
        async with child._inflight_cond:
            child._inflight[prior.lora_int_id] = 0
            child._inflight_cond.notify_all()

    async def _run_reload_via_handler() -> None:
        # Build a tiny in-process request to avoid threads.

        class _FakeUpload:
            async def read(self) -> bytes:
                return tarball

            filename = 'adapter.tgz'

        resp = await child.reload_lora(adapter=_FakeUpload(), policy_version=2)
        assert resp.status_code == 200, resp.body

    reload_task = _asyncio.create_task(_run_reload_via_handler())
    await _simulate_generate_finish()
    await _asyncio.wait_for(reload_task, timeout=5.0)

    # The decrement must have been observed before remove_lora fired.
    assert remove_order == ['generate_decremented', 'remove_called_1'], remove_order
    assert child.active_policy_version == 2
    assert child.engine.add_lora.await_count == 1
    assert child.engine.remove_lora.await_count == 1
    _ = _publish  # silence unused-var lint; kept for intent doc


@pytest.mark.asyncio
async def test_reload_lora_drain_timeout_returns_502_and_skips_remove(
    child: Any,
) -> None:
    """On drain timeout, the endpoint must NOT call remove_lora (would corrupt
    in-flight generates) and MUST return 5xx so the trainer aborts the run.
    """
    import asyncio as _asyncio

    await child._init_async_primitives()
    # Force the drain to time out immediately so the test doesn't wait 120s.
    child._DRAIN_TIMEOUT_S = 0.01

    prior = child.LoRARequest(lora_name='pv1', lora_int_id=1, lora_path='/tmp/ignored')
    child.active_lora = prior
    child.active_policy_version = 1

    # Register an in-flight generate that will never decrement — simulates a
    # stuck request that blocks the drain.
    async with child._inflight_cond:
        child._inflight[prior.lora_int_id] = 1

    class _FakeUpload:
        async def read(self) -> bytes:
            return _make_adapter_tarball()

        filename = 'adapter.tgz'

    resp = await child.reload_lora(adapter=_FakeUpload(), policy_version=2)
    assert resp.status_code == 502, resp.body
    # The swap commit still happened (new adapter is active), but the prior
    # slot is intentionally leaked rather than corrupting in-flight tokens.
    assert child.active_policy_version == 2
    assert child.engine.add_lora.await_count == 1
    assert child.engine.remove_lora.await_count == 0
    _ = _asyncio  # silence unused-import lint


@pytest.mark.asyncio
async def test_reload_lora_remove_failure_returns_502(child: Any) -> None:
    """If engine.remove_lora raises after a successful add_lora, the endpoint
    must return 5xx so the trainer aborts (prior adapter is leaked in vLLM
    and will exhaust max-loras=2 on the next publish otherwise).
    """
    await child._init_async_primitives()

    prior = child.LoRARequest(lora_name='pv1', lora_int_id=1, lora_path='/tmp/ignored')
    child.active_lora = prior
    child.active_policy_version = 1
    child.engine.remove_lora.side_effect = RuntimeError('engine internal error')

    class _FakeUpload:
        async def read(self) -> bytes:
            return _make_adapter_tarball()

        filename = 'adapter.tgz'

    resp = await child.reload_lora(adapter=_FakeUpload(), policy_version=2)
    assert resp.status_code == 502, resp.body
    assert child.active_policy_version == 2  # swap committed before remove


def test_reload_lora_rejects_tar_member_escape(child: Any, client: TestClient) -> None:
    tarball = _make_adapter_tarball(
        extra_members={'../escape.txt': b'oops'},
    )
    with client:
        response = client.post(
            '/reload_lora',
            files={'adapter': ('adapter.tgz', tarball, 'application/gzip')},
            data={'policy_version': '1'},
        )
    assert response.status_code == 400
    assert child.active_policy_version == 0
