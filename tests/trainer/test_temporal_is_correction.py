# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Cut 3 tests: temporal IS correction + ``is_weight/*`` metrics.

With Phase 2's replay buffer wired at the ingest seam (Cut 2), the
``rollout_log_probs`` fed into :func:`compute_policy_loss` is the
behavior-policy logprobs captured at generation time — ``old_log_prob``
is the current-policy recompute. The existing ``tis_imp_ratio`` math at
``core_algos.py:586-590`` is unchanged; only the data source shifts.

These tests exercise :func:`compute_policy_loss` directly with a stub
``config`` so they are host-runnable (no verl / ray / FSDP involved).
See ``plans-n-solutions/stages/full_async.md`` §4 (Cut 3).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / 'trainer_integration' / 'verl')
)

# compute_policy_loss depends on `verl_F = verl.utils.torch_functional`, which
# is in the verl package. Skip on host when verl isn't installed.
pytest.importorskip('verl', reason='verl only installed inside verl container')

from verl_custom.trainer.ppo.core_algos import compute_policy_loss  # noqa: E402


def _make_batch(
    *,
    batch: int,
    seq: int,
    delta: float,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Return inputs to compute_policy_loss with a fixed ``old-rollout`` delta.

    ``log_prob`` is random (current-policy recompute), ``old_log_prob``
    equals ``log_prob`` (so ``ratio`` stays 1 and ``pg_clipfrac=0`` — we
    want to isolate the TIS ratio). ``rollout_log_probs = old_log_prob -
    delta`` so the TIS ratio is ``exp(delta)`` at every token.
    """
    g = torch.Generator().manual_seed(seed)
    log_prob = torch.randn(batch, seq, generator=g) * 0.1
    old_log_prob = log_prob.clone()
    rollout_log_probs = old_log_prob - delta
    advantages = torch.randn(batch, seq, generator=g)
    response_mask = torch.ones(batch, seq, dtype=torch.long)
    return {
        'old_log_prob': old_log_prob,
        'log_prob': log_prob,
        'advantages': advantages,
        'response_mask': response_mask,
        'rollout_log_probs': rollout_log_probs,
    }


def _stub_cfg(cap: float = 2.0) -> SimpleNamespace:
    return SimpleNamespace(tis_imp_ratio_cap=cap)


class TestTISMetricsShape:
    def test_returns_five_tuple(self) -> None:
        inp = _make_batch(batch=2, seq=4, delta=0.0)
        out = compute_policy_loss(
            config=_stub_cfg(),
            cliprange=0.2,
            loss_agg_mode='token-mean',
            **inp,
        )
        assert len(out) == 5, 'compute_policy_loss must return 5-tuple'
        _, _, _, _, tis_metrics = out
        assert isinstance(tis_metrics, dict)
        assert set(tis_metrics) == {
            'is_weight/mean',
            'is_weight/p99',
            'is_weight/clip_fraction',
        }

    def test_metrics_empty_when_tis_disabled(self) -> None:
        inp = _make_batch(batch=2, seq=4, delta=0.5)
        # cap=0 disables TIS path entirely.
        out = compute_policy_loss(
            config=_stub_cfg(cap=0.0),
            cliprange=0.2,
            loss_agg_mode='token-mean',
            **inp,
        )
        _, _, _, _, tis_metrics = out
        assert tis_metrics == {}

    def test_metrics_empty_when_rollout_log_probs_none(self) -> None:
        inp = _make_batch(batch=2, seq=4, delta=0.5)
        inp['rollout_log_probs'] = None
        out = compute_policy_loss(
            config=_stub_cfg(),
            cliprange=0.2,
            loss_agg_mode='token-mean',
            **inp,
        )
        _, _, _, _, tis_metrics = out
        assert tis_metrics == {}

    def test_metrics_empty_when_config_none(self) -> None:
        inp = _make_batch(batch=2, seq=4, delta=0.5)
        out = compute_policy_loss(
            config=None,
            cliprange=0.2,
            loss_agg_mode='token-mean',
            **inp,
        )
        _, _, _, _, tis_metrics = out
        assert tis_metrics == {}


class TestIsWeightValues:
    def test_mean_one_when_old_matches_rollout(self) -> None:
        """When behavior PV == current PV, is_weight ≈ 1 and clip = 0."""
        inp = _make_batch(batch=4, seq=8, delta=0.0)
        _, _, _, _, tis = compute_policy_loss(
            config=_stub_cfg(cap=2.0),
            cliprange=0.2,
            loss_agg_mode='token-mean',
            **inp,
        )
        assert abs(tis['is_weight/mean'] - 1.0) < 1e-5
        assert tis['is_weight/clip_fraction'] == 0.0

    def test_mean_increases_with_staleness(self) -> None:
        """Bigger |old_log_prob − rollout_log_probs| → bigger IS weight."""
        small = compute_policy_loss(
            config=_stub_cfg(cap=10.0),
            cliprange=0.2,
            loss_agg_mode='token-mean',
            **_make_batch(batch=4, seq=8, delta=0.1),
        )[4]
        large = compute_policy_loss(
            config=_stub_cfg(cap=10.0),
            cliprange=0.2,
            loss_agg_mode='token-mean',
            **_make_batch(batch=4, seq=8, delta=0.8),
        )[4]
        assert small['is_weight/mean'] < large['is_weight/mean']
        # Uniform delta → mean ≈ exp(delta) exactly (no variance).
        assert abs(small['is_weight/mean'] - float(torch.exp(torch.tensor(0.1)))) < 1e-5
        assert abs(large['is_weight/mean'] - float(torch.exp(torch.tensor(0.8)))) < 1e-5

    def test_clip_fraction_when_pathologically_stale(self) -> None:
        """delta >> cap → every token exceeds cap → clip_fraction=1."""
        _, _, _, _, tis = compute_policy_loss(
            config=_stub_cfg(cap=2.0),
            cliprange=0.2,
            loss_agg_mode='token-mean',
            # exp(3) ≈ 20 >> cap=2
            **_make_batch(batch=4, seq=8, delta=3.0),
        )
        assert tis['is_weight/clip_fraction'] == 1.0
        # p99 ≈ exp(3) — reflects the *raw* (unclamped) ratio.
        assert abs(tis['is_weight/p99'] - float(torch.exp(torch.tensor(3.0)))) < 1e-3

    def test_clip_fraction_partial(self) -> None:
        """Mix of masked and unmasked stale tokens → fractional clip."""
        g = torch.Generator().manual_seed(0)
        log_prob = torch.randn(2, 4, generator=g) * 0.01
        old_log_prob = log_prob.clone()
        rollout_log_probs = old_log_prob.clone()
        # Half the tokens severely stale, half aligned.
        rollout_log_probs[:, :2] -= 3.0  # cap=2 → exp(3) > 2 → clip
        advantages = torch.randn(2, 4, generator=g)
        response_mask = torch.ones(2, 4, dtype=torch.long)
        _, _, _, _, tis = compute_policy_loss(
            config=_stub_cfg(cap=2.0),
            cliprange=0.2,
            loss_agg_mode='token-mean',
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            rollout_log_probs=rollout_log_probs,
        )
        assert 0.4 < tis['is_weight/clip_fraction'] < 0.6

    def test_response_mask_excludes_padding(self) -> None:
        """Masked positions don't contribute to is_weight/mean."""
        inp = _make_batch(batch=2, seq=6, delta=0.0)
        # Make positions 3..6 severely stale, then mask them out.
        inp['rollout_log_probs'][:, 3:] = inp['old_log_prob'][:, 3:] - 5.0
        mask = torch.ones(2, 6, dtype=torch.long)
        mask[:, 3:] = 0
        inp['response_mask'] = mask

        _, _, _, _, tis = compute_policy_loss(
            config=_stub_cfg(cap=2.0),
            cliprange=0.2,
            loss_agg_mode='token-mean',
            **inp,
        )
        # Unmasked positions all have delta=0 → ratio=1.
        assert abs(tis['is_weight/mean'] - 1.0) < 1e-5
        assert tis['is_weight/clip_fraction'] == 0.0
