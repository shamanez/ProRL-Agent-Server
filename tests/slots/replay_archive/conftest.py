"""Fixtures for ReplayArchive slot tests."""

from __future__ import annotations

from datetime import datetime, timezone

from schemas.episode_record import EpisodeRecord, Event, RewardEvent, TrustLevel


def make_episode(
    *,
    episode_uid: str = 'ep-1',
    task_id: str = 't-1',
    policy_id: str = 'qwen3-4b-skyrl',
    policy_version: int = 1,
    environment_id: str = 'swe_agent',
    split: str = 'train',
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    termination_reason: str = 'done',
    total_reward: float = 0.5,
    trust_level: TrustLevel = TrustLevel.OWN_FABRIC,
) -> EpisodeRecord:
    if started_at is None:
        started_at = datetime(2026, 5, 4, 10, 0, 0, tzinfo=timezone.utc)
    if finished_at is None:
        finished_at = datetime(2026, 5, 4, 10, 1, 0, tzinfo=timezone.utc)
    return EpisodeRecord(
        episode_uid=episode_uid,
        task_id=task_id,
        split=split,
        environment_provider='prorl',
        environment_id=environment_id,
        environment_version='v1',
        verifier_version='v1',
        reward_spec_id='rs-1',
        policy_id=policy_id,
        policy_version=policy_version,
        base_model_id='Qwen/Qwen3-4B-Instruct',
        tokenizer_id='Qwen/Qwen3-4B-Instruct',
        inference_backend='vllm-pinning',
        sampling_params={'temperature': 1.0},
        created_at_step=0,
        started_at=started_at,
        finished_at=finished_at,
        termination_reason=termination_reason,
        events=(
            Event(
                turn_index=0,
                kind='agent_turn',
                response_token_ids=(4, 5, 6),
                response_loss_mask=(1, 1, 1),
                behavior_log_probs=(-0.1, -0.2, -0.3),
            ),
        ),
        total_reward=total_reward,
        reward_events=(
            RewardEvent(turn_index=0, delta=total_reward, verifier_id='v1'),
        ),
        prompt_token_ids=(1, 2, 3),
        response_token_ids=(4, 5, 6),
        response_loss_mask=(1, 1, 1),
        behavior_log_probs=(-0.1, -0.2, -0.3),
        trust_level=trust_level,
    )
