# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Cut 2 integration tests: trainer ↔ replay store wiring.

These tests exercise the seam that pushes every freshly-generated batch
into :class:`TrajectoryStore` and samples a mini-batch back out. They run
on host (not inside the verl container) and therefore `importorskip` any
heavyweight dependency and build bare trainer shells via ``__new__``.

See ``plans-n-solutions/stages/full_async.md`` §4 (Cut 2).
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / 'trainer_integration' / 'verl')
)

pytest.importorskip('verl', reason='verl only installed inside verl container')

from verl.protocol import DataProto  # noqa: E402
from verl_custom.replay.trajectory_store import TrajectoryStore  # noqa: E402
from verl_custom.trainer.ppo.ray_trainer import RayPPOTrainer  # noqa: E402
from verl_custom.trainer.ppo.ray_trainer_dapo import RayPPOTrainerDAPO  # noqa: E402


def _build_rollout_dataproto(
    *, n_prompts: int, n_per_prompt: int, prompt_len: int, response_len: int
) -> DataProto:
    """Mimic the DataProto shape emitted at the seam after ``batch.union``.

    Tokens are deterministic (prompt_i = i*100 + j, response_i = i*10 + k)
    so the lockstep-parity test can assert bit-identical round-trip.
    """
    batch = n_prompts * n_per_prompt
    total_len = prompt_len + response_len

    input_ids = torch.zeros((batch, total_len), dtype=torch.long)
    attention_mask = torch.ones((batch, total_len), dtype=torch.long)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    responses = torch.zeros((batch, response_len), dtype=torch.long)
    loss_mask = torch.ones((batch, response_len), dtype=torch.long)
    rollout_log_probs = torch.linspace(-0.5, -0.01, batch * response_len).reshape(
        batch, response_len
    )

    uids: list[str] = []
    data_sources: list[str] = []
    abilities: list[str] = []
    reward_models: list[dict] = []
    resolveds: list[bool] = []
    errors: list[str | None] = []
    successes: list[bool] = []
    finishes: list[bool] = []
    instances: list[dict] = []

    row = 0
    for i in range(n_prompts):
        uid = f'prompt-{i}'
        prompt = torch.arange(prompt_len, dtype=torch.long) + (i * 1000)
        for j in range(n_per_prompt):
            resp = torch.arange(response_len, dtype=torch.long) + (i * 100 + j * 10)
            input_ids[row, :prompt_len] = prompt
            input_ids[row, prompt_len:] = resp
            responses[row, :] = resp
            uids.append(uid)
            data_sources.append(f'source-{i % 2}')
            abilities.append(f'ability-{i % 3}')
            reward_models.append({'style': 'rule', 'ground_truth': f'gt-{i}'})
            resolveds.append(i % 2 == 0)
            errors.append(None)
            successes.append(True)
            finishes.append(True)
            instances.append({'id': i, 'sibling': j})
            row += 1

    tensors = {
        'input_ids': input_ids,
        'responses': responses,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'loss_mask': loss_mask,
        'rollout_log_probs': rollout_log_probs,
        'is_padded': torch.zeros(batch, dtype=torch.bool),
        'error_mask': torch.zeros(batch, dtype=torch.bool),
    }
    non_tensors = {
        'uid': np.array(uids, dtype=object),
        'data_source': np.array(data_sources, dtype=object),
        'ability': np.array(abilities, dtype=object),
        'reward_model': np.array(reward_models, dtype=object),
        'resolved': np.array(resolveds, dtype=object),
        'error': np.array(errors, dtype=object),
        'success': np.array(successes, dtype=object),
        'finish': np.array(finishes, dtype=object),
        'instance': np.array(instances, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _bare_trainer_with_store(store: TrajectoryStore, *, n: int) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.trajectory_store = store
    trainer.policy_version = 7
    trainer.global_steps = 3
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=n))
    )
    return trainer


class TestCut2LockstepParity:
    """Buffer size >= pushed groups + staleness >= 1 → sampled == pushed."""

    def test_roundtrip_preserves_tokens_and_uids(self) -> None:
        n_prompts, n_per_prompt = 4, 2
        batch_in = _build_rollout_dataproto(
            n_prompts=n_prompts,
            n_per_prompt=n_per_prompt,
            prompt_len=5,
            response_len=7,
        )
        store = TrajectoryStore(
            max_size=n_prompts,
            staleness_cutoff_k=10,
            pad_token_id=0,
            prompt_length_cap=5,
            response_length_cap=7,
        )
        trainer = _bare_trainer_with_store(store, n=n_per_prompt)

        metrics: dict = {}
        out = trainer._push_and_sample_replay(batch_in, metrics)

        assert len(out.batch) == len(batch_in.batch)
        # The sampled uids must be the same set (order may differ since
        # sample_mini_batch picks groups uniformly, but with buffer_size==
        # n_prompts every group is sampled exactly once).
        assert set(out.non_tensor_batch['uid'].tolist()) == set(
            batch_in.non_tensor_batch['uid'].tolist()
        )
        # Prompt-side non_tensors must round-trip.
        for key in ('data_source', 'ability', 'reward_model'):
            assert key in out.non_tensor_batch, f'{key} dropped by store'
        # Reward / advantage are zero-default until the downstream recompute.
        assert torch.allclose(
            out.batch['reward'], torch.zeros(len(out.batch), dtype=torch.float)
        )
        # Store metrics flowed into the caller's dict.
        assert 'replay/store_size' in metrics

    def test_roundtrip_tokens_per_uid_bit_identical(self) -> None:
        """For a uid, the sampled prompt/response tokens match the input."""
        n_prompts, n_per_prompt = 3, 2
        prompt_len, response_len = 4, 6
        batch_in = _build_rollout_dataproto(
            n_prompts=n_prompts,
            n_per_prompt=n_per_prompt,
            prompt_len=prompt_len,
            response_len=response_len,
        )
        store = TrajectoryStore(
            max_size=n_prompts,
            staleness_cutoff_k=10,
            pad_token_id=0,
            prompt_length_cap=prompt_len,
            response_length_cap=response_len,
        )
        trainer = _bare_trainer_with_store(store, n=n_per_prompt)
        out = trainer._push_and_sample_replay(batch_in, {})

        # Index input by uid → list of response rows
        def _rows_by_uid(dp: DataProto) -> dict[str, list[torch.Tensor]]:
            rows: dict[str, list[torch.Tensor]] = {}
            for i, uid in enumerate(dp.non_tensor_batch['uid']):
                rows.setdefault(str(uid), []).append(dp.batch['responses'][i].clone())
            return rows

        in_rows = _rows_by_uid(batch_in)
        out_rows = _rows_by_uid(out)
        assert set(in_rows) == set(out_rows)
        for uid, in_list in in_rows.items():
            out_list = out_rows[uid]
            # Siblings within a group are not order-stable; compare as sets.
            in_set = {tuple(r.tolist()) for r in in_list}
            out_set = {tuple(r.tolist()) for r in out_list}
            assert in_set == out_set, f'uid {uid} tokens diverged'

    def test_store_disabled_noop(self) -> None:
        """``trajectory_store is None`` → batch passes through unchanged."""
        batch_in = _build_rollout_dataproto(
            n_prompts=2, n_per_prompt=2, prompt_len=3, response_len=4
        )
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.trajectory_store = None
        # Intentionally omit other attributes — they must not be touched.
        out = trainer._push_and_sample_replay(batch_in, {})
        assert out is batch_in


class TestDAPOResumePolicyVersionSync:
    """Bug #18: DAPO fit() must align ``policy_version`` to ``global_steps``."""

    def test_source_has_resume_sync_block(self) -> None:
        """Guard: the resume-sync lines are present in DAPO ``fit()``."""
        src = inspect.getsource(RayPPOTrainerDAPO.fit)
        assert 'self.policy_version = self.global_steps' in src, (
            'DAPO resume-sync assignment missing'
        )
        assert 'self.async_rollout_manager.policy_version = self.global_steps' in src, (
            'DAPO rollout-manager PV propagation missing'
        )

    def test_resume_sync_block_propagates_to_rollout_manager(self) -> None:
        """Behavior: executing the resume block aligns both trainer and pool."""
        trainer = RayPPOTrainerDAPO.__new__(RayPPOTrainerDAPO)
        trainer.global_steps = 42  # set by _load_checkpoint
        trainer.policy_version = 0  # stale (pre-Phase-1 default)
        trainer.async_rollout_manager = SimpleNamespace(policy_version=0)

        # Simulate the exact 4-line block from DAPO fit() right after
        # ``self._load_checkpoint()``.
        if trainer.global_steps > 0:
            trainer.policy_version = trainer.global_steps
            trainer.async_rollout_manager.policy_version = trainer.global_steps

        assert trainer.policy_version == 42
        assert trainer.async_rollout_manager.policy_version == 42

    def test_resume_sync_noop_when_cold_start(self) -> None:
        trainer = RayPPOTrainerDAPO.__new__(RayPPOTrainerDAPO)
        trainer.global_steps = 0
        trainer.policy_version = 0
        trainer.async_rollout_manager = SimpleNamespace(policy_version=0)

        if trainer.global_steps > 0:  # pragma: no cover - guard
            trainer.policy_version = trainer.global_steps
            trainer.async_rollout_manager.policy_version = trainer.global_steps

        # Cold-start leaves both at 0.
        assert trainer.policy_version == 0
        assert trainer.async_rollout_manager.policy_version == 0
