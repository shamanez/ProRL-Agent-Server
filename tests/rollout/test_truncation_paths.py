# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nvidia/rollout/utils.py`` length-contract assertions.

These helpers used to silently right-trim sequences that overflowed the
configured cap. The fullasync replay/rollout contract now requires both
helpers to raise :class:`RuntimeError` instead, surfacing any drift in the
upstream length bounds (``max_starting_message_length`` for the seed slot
and vLLM's ``max_model_len`` for the packed prompt+response).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

# Put the verl patch-package root on sys.path so we can import
# verl_custom.nvidia.* without the heavyweight verl install.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / 'trainer_integration' / 'verl')
)

from verl_custom.nvidia.rollout.utils import (  # noqa: E402
    convert_right_padding_to_left,
    pad_to_max_length_right,
)


def _tokenizer(pad_id: int = 0) -> SimpleNamespace:
    return SimpleNamespace(pad_token_id=pad_id)


def test_convert_right_padding_to_left_packs_at_cap():
    """A sequence whose non-pad length exactly equals ``max_len`` packs."""
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    out_ids, out_attn = convert_right_padding_to_left(
        _tokenizer(), input_ids, attention_mask, device=torch.device('cpu'), max_len=4
    )
    assert out_ids.tolist() == [[1, 2, 3, 4]]
    assert out_attn.tolist() == [[1, 1, 1, 1]]


def test_convert_right_padding_to_left_raises_on_overflow():
    """Non-pad length > ``max_len`` raises RuntimeError instead of trimming."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
    with pytest.raises(RuntimeError, match='convert_right_padding_to_left'):
        convert_right_padding_to_left(
            _tokenizer(),
            input_ids,
            attention_mask,
            device=torch.device('cpu'),
            max_len=3,
        )


def test_pad_to_max_length_right_packs_at_cap():
    """An encoding whose attention sum equals ``max_length`` packs cleanly."""
    encodings = {
        'input_ids': [[10, 11, 12]],
        'attention_mask': [[1, 1, 1]],
        'assistant_masks': [[0, 1, 1]],
        'log_probs': [[-0.1, -0.2, -0.3]],
    }
    pid, pam, pasm, plp = pad_to_max_length_right(
        _tokenizer(), encodings, max_length=3, device=torch.device('cpu')
    )
    assert pid.tolist() == [[10, 11, 12]]
    assert pam.tolist() == [[1, 1, 1]]
    assert pasm.tolist() == [[0, 1, 1]]
    assert plp[0].tolist() == pytest.approx([-0.1, -0.2, -0.3])


def test_pad_to_max_length_right_raises_on_overflow():
    """Attention sum > ``max_length`` raises RuntimeError instead of trimming."""
    encodings = {
        'input_ids': [[10, 11, 12, 13]],
        'attention_mask': [[1, 1, 1, 1]],
        'assistant_masks': [[0, 1, 1, 1]],
        'log_probs': [[-0.1, -0.2, -0.3, -0.4]],
    }
    with pytest.raises(RuntimeError, match='pad_to_max_length_right'):
        pad_to_max_length_right(
            _tokenizer(), encodings, max_length=3, device=torch.device('cpu')
        )
