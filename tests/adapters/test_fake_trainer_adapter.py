"""Fake TrainerAdapter contract test.

Proves the LiveStore → TrainerAdapter data flow works without VERL or torch.
A fake consumer calls get_batch() and verifies the wire format:
  - Records are unpadded (BC-11)
  - Token IDs are ints (BC-1)
  - Pop-on-sample semantics hold (BC-3)

This test is framework-agnostic: no VERL, no torch, no Docker required.
"""

from __future__ import annotations

import os
import time

import pytest
from rollout_fabric.live_store.client import LiveStoreClient
from rollout_fabric.live_store.server import serve
from rollout_fabric.schemas.episode_record import TrustLevel
from rollout_fabric.schemas.training_sample import TrainingSample

# ---------------------------------------------------------------------------
# Fixtures (in-process LiveStore — same approach as tests/slots/live_store/)
# ---------------------------------------------------------------------------


@pytest.fixture()
def ls_server(tmp_path):
    socket = str(tmp_path / 'fake_trainer_ls.sock')
    server = serve(
        socket_path=socket,
        max_size=8,
        staleness_cutoff_k=4,
        no_progress_timeout_s=2.0,
        max_workers=2,
    )
    for _ in range(50):
        if os.path.exists(socket):
            break
        time.sleep(0.01)
    yield server, socket
    server.stop(grace=1.0)
    try:
        os.unlink(socket)
    except FileNotFoundError:
        pass


@pytest.fixture()
def ls_client(ls_server):
    _, socket = ls_server
    client = LiveStoreClient(
        socket,
        policy_id='qwen3-4b-skyrl',
        environment_id='swe_agent',
        environment_version='v1',
        verifier_version='v1',
        split='train',
    )
    yield client
    client.close()


def _make_sample(*, group_uid: str, sample_uid: str) -> TrainingSample:
    return TrainingSample(
        sample_uid=sample_uid,
        group_uid=group_uid,
        episode_uid=f'ep-{sample_uid}',
        prompt_token_ids=(1, 2, 3),
        response_token_ids=(4, 5),
        response_loss_mask=(1, 1),
        behavior_log_probs=(-0.1, -0.2),
        reward=1.0,
        raw_reward=1.0,
        truncated=False,
        behavior_policy_version=1,
        created_at_step=0,
        task_id='task-abc',
        split='train',
        policy_id='qwen3-4b-skyrl',
        environment_id='swe_agent',
        environment_version='v1',
        verifier_version='v1',
        trust_level=TrustLevel.OWN_FABRIC,
        sample_indices=None,
        instance={'success': True, 'resolved': True, 'finish': True},
        error=None,
        is_padded=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fake_trainer_can_get_batch_from_live_store(ls_client: LiveStoreClient) -> None:
    """A trainer that calls get_batch() receives correctly-typed unpadded records."""
    # Push two groups (fake RolloutManager)
    ls_client.push_group([_make_sample(group_uid='g0', sample_uid='s0')])
    ls_client.push_group([_make_sample(group_uid='g1', sample_uid='s1')])
    assert ls_client.num_groups() == 2

    # Pull one group (fake TrainerAdapter)
    batch = ls_client.get_batch(n_groups=1, current_step=0, timeout_ms=2_000)
    assert len(batch) == 1

    # BC-3: pop-on-sample — one group consumed
    assert ls_client.num_groups() == 1

    sample = batch[0]

    # BC-1: token IDs must be integers on the wire
    for tid in sample.prompt_token_ids:
        assert isinstance(tid, int), (
            f'BC-1 violation: prompt token_id {tid!r} is not int'
        )
    for tid in sample.response_token_ids:
        assert isinstance(tid, int), (
            f'BC-1 violation: response token_id {tid!r} is not int'
        )

    # BC-11: wire is unpadded — sequences should have original lengths, not padded
    assert len(sample.prompt_token_ids) == 3
    assert len(sample.response_token_ids) == 2


def test_trainer_adapter_boundary_no_torch_required() -> None:
    """trainer_adapters/verl/ imports check: pad.py must NOT be importable without torch.

    This test does NOT import pad.py (torch may be absent in the fast test env).
    It verifies the file exists and has the right function signature via AST inspection.
    """
    import ast
    from pathlib import Path

    pad_file = (
        Path(__file__).parent.parent.parent
        / 'trainers'
        / 'verl'
        / 'verl_custom'
        / 'fabric_adapter'
        / 'pad.py'
    )
    assert pad_file.exists(), (
        'trainers/verl/verl_custom/fabric_adapter/pad.py must exist'
    )

    tree = ast.parse(pad_file.read_text())
    func_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert 'pack_unpadded_groups' in func_names, (
        'trainers/verl/verl_custom/fabric_adapter/pad.py must define pack_unpadded_groups()'
    )
