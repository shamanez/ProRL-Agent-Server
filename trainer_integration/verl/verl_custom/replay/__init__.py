"""Phase 2 replay buffer + continuous producer (decoupled clocks).

This package holds the in-process replay buffer that backs Phase 2's
fully-async agentic RL loop. The buffer stores variable-length trajectories
tagged with the behavior policy version that generated them; the trainer
samples mini-batches from it at its own cadence and applies a temporal
importance-sampling correction at the loss.

See `plans-n-solutions/stages/full_async.md` for the design.
"""

from verl_custom.replay.trajectory_store import (
    InsufficientTrajectoriesError,
    SampledMiniBatch,
    TrajectoryRecord,
    TrajectoryStore,
)

__all__ = [
    'InsufficientTrajectoriesError',
    'SampledMiniBatch',
    'TrajectoryRecord',
    'TrajectoryStore',
]
