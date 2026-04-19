# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Asserts that every message produced by `DataProto2Messages` carries the
trainer-authoritative `policy_version` stamp. Pool-side consumers and ProRL
log-analysis key off this field to detect mixed-version batches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / 'trainer_integration' / 'verl')
)

ray = pytest.importorskip('ray', reason='ray only installed inside verl container')
_verl = pytest.importorskip('verl', reason='verl only installed inside verl container')
pytest.importorskip('transformers')


def _build_manager(num_trajectories: int = 2) -> object:
    """Construct an AsyncLLMServerManager shell without running __init__ (which
    wires Ray, tokenizers, datasets, and sampling configs)."""
    from verl_custom.nvidia.rollout.async_server import (  # noqa: PLC0415
        AsyncLLMServerManager,
    )

    manager = AsyncLLMServerManager.__new__(AsyncLLMServerManager)
    manager.policy_version = 0
    manager.num_trajectories = num_trajectories
    manager.num_val_trajectories = 1
    return manager


def test_datum_messages_stamp_policy_version_on_every_entry() -> None:
    from verl_custom.nvidia.rollout.async_server import (  # noqa: PLC0415
        AsyncLLMServerManager,
    )

    manager = _build_manager(num_trajectories=3)
    base_messages = [
        {'instance_id': 'a', 'data_source': 'test'},
        {'instance_id': 'b', 'data_source': 'test'},
    ]

    out = AsyncLLMServerManager.DataProto2Messages(
        manager, base_messages, val_mode=False
    )

    assert len(out) == len(base_messages) * manager.num_trajectories
    for m in out:
        assert 'policy_version' in m
        assert m['policy_version'] == 0
        assert 'trajectory_id' in m


def test_datum_messages_stamp_advances_with_manager_state() -> None:
    from verl_custom.nvidia.rollout.async_server import (  # noqa: PLC0415
        AsyncLLMServerManager,
    )

    manager = _build_manager(num_trajectories=2)
    manager.policy_version = 7

    out = AsyncLLMServerManager.DataProto2Messages(
        manager, [{'instance_id': 'x', 'data_source': 't'}], val_mode=False
    )
    assert [m['policy_version'] for m in out] == [7, 7]


def test_val_mode_uses_num_val_trajectories_but_still_stamps() -> None:
    from verl_custom.nvidia.rollout.async_server import (  # noqa: PLC0415
        AsyncLLMServerManager,
    )

    manager = _build_manager(num_trajectories=4)
    manager.num_val_trajectories = 1
    manager.policy_version = 3

    out = AsyncLLMServerManager.DataProto2Messages(
        manager, [{'instance_id': 'x', 'data_source': 't'}], val_mode=True
    )
    assert len(out) == 1
    assert out[0]['policy_version'] == 3
