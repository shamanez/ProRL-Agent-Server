"""Wire-schema and slot-contract single source of truth for the rollout fabric.

This package is the typed boundary between the seven slots defined in
``plans-n-solutions/rollout_fabric.md`` (§5):

  EnvironmentProvider | InferenceBackend | RolloutWorker | LiveStore
  ReplayArchive | TrainerAdapter | PolicyRegistry

Submodules:

  * ``policy_version``   — :class:`PolicyVersionSnapshot` and
                           :class:`PolicyVersionCache` (the cleverest primitive
                           for §3.5 per-row stamping; see module docstring).
  * ``training_sample``  — §6.2 ``TrainingSample`` / ``TrainingGroup`` (live path).
  * ``episode_record``   — §6.1 ``EpisodeRecord`` (durable archive).
  * ``protocols/*``      — Appendix A ``Protocol``-shaped slot interfaces.
  * ``proto/*.proto``    — gRPC schema sketches, compiled at S1.

The contract is **token-in / token-out everywhere** (§3.1): every wire field
that carries model output is ``tuple[int, ...]`` or ``bytes`` with int32-packed
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
