"""Fixtures for LiveStore slot tests."""

from __future__ import annotations

import os
import time

import pytest

from rollout_fabric.live_store.client import LiveStoreClient
from rollout_fabric.live_store.server import serve
from rollout_fabric.schemas.episode_record import TrustLevel
from rollout_fabric.schemas.training_sample import TrainingSample


@pytest.fixture()
def live_store_server(tmp_path):
    socket = str(tmp_path / 'live_store.sock')
    server = serve(
        socket_path=socket,
        max_size=8,
        staleness_cutoff_k=4,
        no_progress_timeout_s=2.0,
        max_workers=4,
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
def live_store_client(live_store_server):
    _, socket = live_store_server
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


def make_sample(
    *,
    sample_uid: str,
    group_uid: str,
    behavior_policy_version: int = 1,
    created_at_step: int = 0,
) -> TrainingSample:
    return TrainingSample(
        sample_uid=sample_uid,
        group_uid=group_uid,
        episode_uid=f'ep-{sample_uid}',
        prompt_token_ids=(1, 2, 3),
        response_token_ids=(4, 5),
        response_loss_mask=(1, 1),
        behavior_log_probs=(-0.1, -0.2),
        reward=0.5,
        raw_reward=0.5,
        truncated=False,
        behavior_policy_version=behavior_policy_version,
        created_at_step=created_at_step,
        task_id='t1',
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
