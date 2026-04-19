# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for `RayPPOTrainer._publish_lora_adapter`.

The method lives on the trainer class which pulls in `ray` + `verl`. Those
are only installed inside the verl Docker image, so on host these tests
skip cleanly via `importorskip`. Inside the container they execute in full.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

# Must be placed BEFORE we import RayPPOTrainer so sys.path is wired.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / 'trainer_integration' / 'verl')
)

ray = pytest.importorskip('ray', reason='ray only installed inside verl container')
_verl = pytest.importorskip('verl', reason='verl only installed inside verl container')

from verl_custom.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: E402


def _write_fixture_adapter(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / 'global_step_5' / 'actor' / 'lora_adapter'
    adapter_dir.mkdir(parents=True)
    save_file({'x': torch.zeros(1)}, str(adapter_dir / 'adapter_model.safetensors'))
    (adapter_dir / 'adapter_config.json').write_text(
        json.dumps(
            {
                'peft_type': 'LORA',
                'r': 16,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'v_proj'],
            }
        )
    )
    return tmp_path / 'global_step_5'


def _bare_trainer(endpoints: list[str]) -> RayPPOTrainer:
    """Build a RayPPOTrainer shell without running __init__ (which needs Ray)."""
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.policy_version = 0
    trainer._last_publish_metrics = {}
    trainer.async_rollout_manager = SimpleNamespace(policy_version=0)
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(external_llm_endpoints=list(endpoints))
        )
    )
    return trainer


def _mk_response(status: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def test_publish_happy_path_four_of_four(tmp_path: Path) -> None:
    step_folder = _write_fixture_adapter(tmp_path)
    endpoints = [f'http://vllm-{i}:8100' for i in range(4)]
    trainer = _bare_trainer(endpoints)

    with patch(
        'requests.post',
        return_value=_mk_response(
            200, {'policy_version': 1, 'vllm_load_latency_ms': 42.0}
        ),
    ) as post:
        trainer._publish_lora_adapter(str(step_folder))

    assert post.call_count == 4
    assert trainer.policy_version == 1
    assert trainer.async_rollout_manager.policy_version == 1
    m = trainer._last_publish_metrics
    assert m['weight_sync/policy_version'] == 1
    assert m['weight_sync/endpoints_ok'] == 4
    assert m['weight_sync/endpoints_failed'] == 0
    assert m['weight_sync/adapter_mib'] > 0
    assert m['weight_sync/vllm_load_latency_s'] == pytest.approx(0.042)
    assert m['weight_sync/transfer_latency_s'] >= 0.0


def test_publish_partial_failure_raises(tmp_path: Path) -> None:
    step_folder = _write_fixture_adapter(tmp_path)
    endpoints = [f'http://vllm-{i}:8100' for i in range(4)]
    trainer = _bare_trainer(endpoints)

    responses_by_url = {
        endpoints[0]: _mk_response(200, {'vllm_load_latency_ms': 30.0}),
        endpoints[1]: _mk_response(200, {'vllm_load_latency_ms': 32.0}),
        endpoints[2]: _mk_response(200, {'vllm_load_latency_ms': 28.0}),
        endpoints[3]: _mk_response(500, {'detail': 'add_lora failed'}),
    }

    def fake_post(url: str, **_kwargs):
        for ep, r in responses_by_url.items():
            if url.startswith(ep):
                return r
        raise AssertionError(f'unexpected url {url}')

    with patch('requests.post', side_effect=fake_post):
        with pytest.raises(RuntimeError, match='1/4 endpoints failed'):
            trainer._publish_lora_adapter(str(step_folder))

    # Version must NOT advance on partial failure — mixed-version would be bug.
    assert trainer.policy_version == 0
    assert trainer.async_rollout_manager.policy_version == 0


def test_publish_network_timeout_counted_as_failure(tmp_path: Path) -> None:
    import requests as _requests  # noqa: PLC0415

    step_folder = _write_fixture_adapter(tmp_path)
    endpoints = ['http://vllm-0:8100']
    trainer = _bare_trainer(endpoints)

    with patch('requests.post', side_effect=_requests.Timeout('slow')):
        with pytest.raises(RuntimeError, match='1/1 endpoints failed'):
            trainer._publish_lora_adapter(str(step_folder))


def test_publish_missing_adapter_file_raises(tmp_path: Path) -> None:
    # Create actor/lora_adapter dir but omit adapter_model.safetensors.
    step_folder = tmp_path / 'global_step_5'
    (step_folder / 'actor' / 'lora_adapter').mkdir(parents=True)
    (step_folder / 'actor' / 'lora_adapter' / 'adapter_config.json').write_text('{}')
    trainer = _bare_trainer(['http://vllm-0:8100'])

    with pytest.raises(RuntimeError, match='PEFT adapter missing'):
        trainer._publish_lora_adapter(str(step_folder))


def test_publish_empty_endpoints_raises(tmp_path: Path) -> None:
    step_folder = _write_fixture_adapter(tmp_path)
    trainer = _bare_trainer([])
    with pytest.raises(RuntimeError, match='external_llm_endpoints is empty'):
        trainer._publish_lora_adapter(str(step_folder))


def test_publish_is_monotonic_across_calls(tmp_path: Path) -> None:
    step_folder = _write_fixture_adapter(tmp_path)
    trainer = _bare_trainer(['http://vllm-0:8100'])

    with patch(
        'requests.post', return_value=_mk_response(200, {'vllm_load_latency_ms': 5.0})
    ):
        trainer._publish_lora_adapter(str(step_folder))
        assert trainer.policy_version == 1
        trainer._publish_lora_adapter(str(step_folder))
        assert trainer.policy_version == 2


def test_publish_treats_409_as_success(tmp_path: Path) -> None:
    """409 = pool already has this version installed (idempotent replay after
    a partial-failure retry). Must count toward `ok`, not `failed`.
    """
    step_folder = _write_fixture_adapter(tmp_path)
    trainer = _bare_trainer(['http://vllm-0:8100', 'http://vllm-1:8100'])

    responses_by_url = {
        'http://vllm-0:8100': _mk_response(200, {'vllm_load_latency_ms': 10.0}),
        'http://vllm-1:8100': _mk_response(409, {'detail': 'already at pv>=1'}),
    }

    def fake_post(url: str, **_kwargs):
        for ep, r in responses_by_url.items():
            if url.startswith(ep):
                return r
        raise AssertionError(f'unexpected url {url}')

    with patch('requests.post', side_effect=fake_post):
        trainer._publish_lora_adapter(str(step_folder))

    assert trainer.policy_version == 1
    assert trainer._last_publish_metrics['weight_sync/endpoints_ok'] == 2
    assert trainer._last_publish_metrics['weight_sync/endpoints_failed'] == 0
