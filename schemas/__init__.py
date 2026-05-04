"""Wire-schema and slot-contract single source of truth for the rollout fabric.

Contract: token-in / token-out everywhere (§3.1). Every wire field that
carries model output is ``tuple[int, ...]`` or ``bytes`` with int32-packed
token ids. Decoded text never leaves the EnvironmentProvider boundary.
"""

from schemas.episode_record import EpisodeRecord, Event, RewardEvent, TrustLevel
from schemas.policy_version import PolicyVersionCache, PolicyVersionSnapshot
from schemas.training_sample import TrainingGroup, TrainingSample

SCHEMA_VERSION = '1.0.0'

__all__ = [
    'SCHEMA_VERSION',
    'EpisodeRecord',
    'Event',
    'PolicyVersionCache',
    'PolicyVersionSnapshot',
    'RewardEvent',
    'TrainingGroup',
    'TrainingSample',
    'TrustLevel',
]
