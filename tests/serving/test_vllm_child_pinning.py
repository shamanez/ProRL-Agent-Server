# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Host-runnable unit tests for path-versioned LoRA pinning in the vLLM child.

The production module ``scripts/serving/_vllm_child.py`` imports vLLM at
module level (it runs inside the pool venv on the GPU box). These tests stub
out ``vllm.*`` via ``sys.modules`` so the FastAPI handlers can be exercised on
the host without GPUs.

Coverage:
  * ``/v{N}/generate`` against an unknown N with no disk dir → HTTP 410.
  * ``/v{N}/generate`` against an N that had a /reload_lora call → dispatches
    with the cached LoRARequest.
  * ``/v{N}/generate`` against an N evicted from cache but with disk dir →
    re-adds via engine.add_lora and dispatches.
  * ``/health`` exposes ``pinned_versions`` and ``inflight_per_version``.
  * Pinning-mode ``/reload_lora`` does NOT call engine.remove_lora and keeps
    the prior pv{N} dir on disk.
  * Quiesce-mode ``/reload_lora`` calls engine.remove_lora after drain.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import pytest

# --- vLLM stubs (must be installed before importing _vllm_child) ------------


class _StubLoRARequest:
    """Stand-in for vllm.lora.request.LoRARequest.

    Mirrors the constructor and the int_id attribute the production code
    relies on. No msgspec import.
    """

    def __init__(
        self,
        *,
        lora_name: str,
        lora_int_id: int,
        lora_path: str,
    ) -> None:
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return f'StubLoRARequest(name={self.lora_name}, int_id={self.lora_int_id})'


class _StubAsyncLLMEngine:
    """Records add_lora / remove_lora calls; ``generate`` yields one fake out."""

    def __init__(self) -> None:
        self.add_lora_calls: list[int] = []
        self.remove_lora_calls: list[int] = []
        self.generate_calls: list[dict[str, Any]] = []

    @classmethod
    def from_engine_args(cls, engine_args):  # pragma: no cover — not exercised
        return cls()

    async def add_lora(self, lora_request) -> None:
        self.add_lora_calls.append(lora_request.lora_int_id)

    async def remove_lora(self, lora_int_id: int) -> None:
        self.remove_lora_calls.append(int(lora_int_id))

    async def abort(self, request_id: str) -> None:
        return None

    async def generate(
        self,
        prompt,
        sampling_params,
        request_id,
        lora_request=None,
    ):
        # Record per-call lora int_id so tests can assert what the engine saw.
        self.generate_calls.append(
            {
                'request_id': request_id,
                'lora_int_id': (
                    lora_request.lora_int_id if lora_request is not None else 0
                ),
                'prompt_token_ids': list(prompt.prompt_token_ids),
            }
        )
        # vLLM contract: an async iterator of outputs. Yield one record whose
        # last value carries the response_ids and a per-token logprobs dict.
        out = types.SimpleNamespace(
            outputs=[
                types.SimpleNamespace(
                    token_ids=[42, 43, 44],
                    logprobs=[
                        {42: types.SimpleNamespace(logprob=-0.1)},
                        {43: types.SimpleNamespace(logprob=-0.2)},
                        {44: types.SimpleNamespace(logprob=-0.3)},
                    ],
                )
            ]
        )
        yield out


def _install_vllm_stubs() -> None:
    """Install vllm.* modules in sys.modules so _vllm_child imports cleanly."""
    if 'vllm' in sys.modules:
        return

    vllm = types.ModuleType('vllm')
    sys.modules['vllm'] = vllm

    # vllm.lora.request.LoRARequest
    lora_pkg = types.ModuleType('vllm.lora')
    sys.modules['vllm.lora'] = lora_pkg
    lora_request_mod = types.ModuleType('vllm.lora.request')
    lora_request_mod.LoRARequest = _StubLoRARequest
    sys.modules['vllm.lora.request'] = lora_request_mod

    # vllm.engine.{arg_utils, async_llm_engine}
    engine_pkg = types.ModuleType('vllm.engine')
    sys.modules['vllm.engine'] = engine_pkg
    arg_utils_mod = types.ModuleType('vllm.engine.arg_utils')

    class _StubAsyncEngineArgs:
        @staticmethod
        def add_cli_args(parser):
            return parser

        @classmethod
        def from_cli_args(cls, args):
            return cls()

    arg_utils_mod.AsyncEngineArgs = _StubAsyncEngineArgs
    sys.modules['vllm.engine.arg_utils'] = arg_utils_mod

    async_llm_engine_mod = types.ModuleType('vllm.engine.async_llm_engine')
    async_llm_engine_mod.AsyncLLMEngine = _StubAsyncLLMEngine
    sys.modules['vllm.engine.async_llm_engine'] = async_llm_engine_mod

    # vllm.inputs.TokensPrompt
    inputs_mod = types.ModuleType('vllm.inputs')

    class _StubTokensPrompt:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = list(prompt_token_ids)

    inputs_mod.TokensPrompt = _StubTokensPrompt
    sys.modules['vllm.inputs'] = inputs_mod

    # vllm.sampling_params.SamplingParams
    sp_mod = types.ModuleType('vllm.sampling_params')

    class _StubSamplingParams:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    sp_mod.SamplingParams = _StubSamplingParams
    sys.modules['vllm.sampling_params'] = sp_mod

    # vllm.utils.random_uuid
    utils_mod = types.ModuleType('vllm.utils')
    utils_mod.random_uuid = lambda: 'test-uuid'
    sys.modules['vllm.utils'] = utils_mod


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def vllm_child(tmp_path, monkeypatch):
    """Import _vllm_child against vLLM stubs; isolate adapter staging dir.

    Re-imports the module each call so module-level state (active_lora,
    _resident_loras, _inflight) is fresh per test.
    """
    _install_vllm_stubs()

    # Make sure the script dir is importable.
    repo_root = Path(__file__).resolve().parents[2]
    serving_dir = repo_root / 'scripts' / 'serving'
    sys.path.insert(0, str(serving_dir))
    try:
        # Force a fresh import: prior tests may have populated module globals.
        sys.modules.pop('_vllm_child', None)
        mod = importlib.import_module('_vllm_child')
    finally:
        sys.path.pop(0)

    # Isolate adapter staging dir per test so we don't read a sibling test's
    # leftover pv{N} directories.
    staging = tmp_path / 'lora_adapters'
    staging.mkdir()
    monkeypatch.setattr(mod, 'ADAPTER_STAGING_ROOT', staging)

    # Plant a stub engine + the asyncio primitives the module would normally
    # build inside its FastAPI startup hook.
    engine = _StubAsyncLLMEngine()
    monkeypatch.setattr(mod, 'engine', engine)
    monkeypatch.setattr(mod, '_swap_lock', asyncio.Lock())
    monkeypatch.setattr(mod, '_inflight_cond', asyncio.Condition())
    monkeypatch.setattr(mod, '_inflight', {}, raising=False)
    monkeypatch.setattr(mod, '_resident_loras', {}, raising=False)
    monkeypatch.setattr(mod, 'active_lora', None, raising=False)
    monkeypatch.setattr(mod, 'active_policy_version', 0, raising=False)
    monkeypatch.setattr(mod, 'CHILD_PORT', 8100, raising=False)
    monkeypatch.setattr(mod, 'swap_protocol', 'pinning', raising=False)

    return mod


def _make_adapter_tarball(payload_text: str = 'fake') -> bytes:
    """Build a minimal tar.gz with the two files _vllm_child expects."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            data = payload_text.encode('utf-8')
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _client(mod):
    from fastapi.testclient import TestClient

    return TestClient(mod.app)


# --- tests ------------------------------------------------------------------


def test_health_exposes_pinned_versions_and_inflight(vllm_child):
    client = _client(vllm_child)
    resp = client.get('/health')
    assert resp.status_code == 200
    body = resp.json()
    assert body['policy_version'] == 0
    assert body['swap_protocol'] == 'pinning'
    assert body['pinned_versions'] == []
    assert body['inflight_per_version'] == {}
    assert body['inflight_versions_count'] == 0


def test_pinned_generate_410_when_version_unknown_and_no_disk(vllm_child):
    """No prior /reload_lora and no pv{N} dir → HTTP 410, never silent fallback."""
    client = _client(vllm_child)
    resp = client.post('/v7/generate', json={'prompt_ids': [1, 2, 3]})
    assert resp.status_code == 410
    body = resp.json()
    assert body['requested_policy_version'] == 7
    assert body['active_policy_version'] == 0
    # Engine should NOT have been asked to generate against an unknown int_id.
    assert vllm_child.engine.generate_calls == []


def test_reload_lora_pinning_keeps_prior_resident(vllm_child):
    """Pinning mode: two reloads → both versions resident, no remove_lora."""
    client = _client(vllm_child)
    payload_v1 = _make_adapter_tarball('v1')
    payload_v2 = _make_adapter_tarball('v2')

    r1 = client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', payload_v1, 'application/gzip')},
        data={'policy_version': '1'},
    )
    assert r1.status_code == 200, r1.text

    r2 = client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', payload_v2, 'application/gzip')},
        data={'policy_version': '2'},
    )
    assert r2.status_code == 200, r2.text

    # Both versions resident; remove_lora never called in pinning mode.
    assert sorted(vllm_child._resident_loras.keys()) == [1, 2]
    assert vllm_child.engine.remove_lora_calls == []
    # Both pv dirs persist on disk so /v{N}/generate can re-page if evicted.
    assert (vllm_child.ADAPTER_STAGING_ROOT / 'pv1').exists()
    assert (vllm_child.ADAPTER_STAGING_ROOT / 'pv2').exists()

    # /health reflects the resident set.
    h = client.get('/health').json()
    assert h['policy_version'] == 2
    assert h['pinned_versions'] == [1, 2]


def test_pinned_generate_dispatches_against_requested_version(vllm_child):
    """/v{N}/generate with N resident → engine sees lora_int_id = N."""
    client = _client(vllm_child)
    payload = _make_adapter_tarball('v1')
    r = client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', payload, 'application/gzip')},
        data={'policy_version': '1'},
    )
    assert r.status_code == 200

    payload2 = _make_adapter_tarball('v2')
    r2 = client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', payload2, 'application/gzip')},
        data={'policy_version': '2'},
    )
    assert r2.status_code == 200

    # Pin to version 1 even though active is 2.
    g = client.post('/v1/generate', json={'prompt_ids': [10, 11]})
    assert g.status_code == 200, g.text
    assert g.json()['response_ids'] == [42, 43, 44]

    # Engine call must have used int_id == 1 (NOT the active 2).
    last = vllm_child.engine.generate_calls[-1]
    assert last['lora_int_id'] == 1, (
        f'expected pinned dispatch to lora_int_id=1, got {last["lora_int_id"]}'
    )


def test_pinned_generate_repages_from_disk_when_evicted_from_cache(vllm_child):
    """If _resident_loras was wiped but the pv{N} dir still exists, re-add."""
    client = _client(vllm_child)
    payload = _make_adapter_tarball('v3')
    r = client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', payload, 'application/gzip')},
        data={'policy_version': '3'},
    )
    assert r.status_code == 200
    assert vllm_child.engine.add_lora_calls == [3]

    # Simulate an eviction from our local _resident_loras dict (e.g., a
    # process restart where vLLM dropped the slot but disk persists).
    vllm_child._resident_loras.clear()

    g = client.post('/v3/generate', json={'prompt_ids': [1]})
    assert g.status_code == 200
    # Engine.add_lora should have been called a second time on the re-page.
    assert vllm_child.engine.add_lora_calls == [3, 3]
    assert 3 in vllm_child._resident_loras


def test_pinned_generate_v0_dispatches_against_base_model(vllm_child):
    """/v0/generate is the explicit 'no adapter' pin (validation/warmup)."""
    client = _client(vllm_child)
    g = client.post('/v0/generate', json={'prompt_ids': [9]})
    assert g.status_code == 200
    last = vllm_child.engine.generate_calls[-1]
    assert last['lora_int_id'] == 0


def test_quiesce_mode_calls_remove_lora_on_reload(vllm_child, monkeypatch):
    """In quiesce mode the prior version is drained + removed; pv dir cleared."""
    monkeypatch.setattr(vllm_child, 'swap_protocol', 'quiesce')

    client = _client(vllm_child)
    p1 = _make_adapter_tarball('q1')
    r1 = client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', p1, 'application/gzip')},
        data={'policy_version': '1'},
    )
    assert r1.status_code == 200
    p2 = _make_adapter_tarball('q2')
    r2 = client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', p2, 'application/gzip')},
        data={'policy_version': '2'},
    )
    assert r2.status_code == 200

    # Quiesce mode evicts the prior version (no in-flight to drain in this
    # test → drain wait completes immediately).
    assert vllm_child.engine.remove_lora_calls == [1]
    assert 1 not in vllm_child._resident_loras
    # Disk dir for the prior version is cleaned up in quiesce mode.
    assert not (vllm_child.ADAPTER_STAGING_ROOT / 'pv1').exists()


def test_legacy_unpinned_generate_uses_active_lora(vllm_child):
    """The bare /generate route preserves backward-compat snapshot semantics."""
    client = _client(vllm_child)
    p = _make_adapter_tarball('vX')
    client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', p, 'application/gzip')},
        data={'policy_version': '5'},
    )
    g = client.post('/generate', json={'prompt_ids': [7]})
    assert g.status_code == 200
    last = vllm_child.engine.generate_calls[-1]
    assert last['lora_int_id'] == 5


def test_reload_lora_rejects_non_monotonic(vllm_child):
    """A repeat or backwards version must be rejected with HTTP 409."""
    client = _client(vllm_child)
    p = _make_adapter_tarball('first')
    r1 = client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', p, 'application/gzip')},
        data={'policy_version': '5'},
    )
    assert r1.status_code == 200
    r2 = client.post(
        '/reload_lora',
        files={'adapter': ('a.tgz', p, 'application/gzip')},
        data={'policy_version': '5'},
    )
    assert r2.status_code == 409
