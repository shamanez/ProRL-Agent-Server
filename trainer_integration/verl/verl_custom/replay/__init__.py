"""DEPRECATED package — see ``live_store/`` and ``trainer_adapters/verl/``.

S1 hard-cutover landed:

* ``trajectory_store.py`` (in-process replay buffer) → deleted; lifted
  to :mod:`live_store.store_core` (the data structure) and
  :mod:`trainer_adapters.verl.pad` (the pack helper).
* ``InsufficientTrajectoriesError`` → re-exported from
  :mod:`live_store`.
* ``SampledMiniBatch`` → re-exported from :mod:`trainer_adapters.verl`.
* ``TrajectoryRecord`` → removed entirely; the wire shape is
  :class:`schemas.training_sample.TrainingSample`.

``continuous_producer.py`` stays at this path through S1 (it still
runs in the trainer process and pushes into the LiveStoreClient).
S2 lifts it into ``rollout_worker/``.

These re-exports exist so any stray import of the legacy names
surfaces at the new home rather than ``ImportError``-ing the trainer
on the cut.
"""

from live_store import InsufficientTrajectoriesError
from trainer_adapters.verl import SampledMiniBatch

__all__ = [
    'InsufficientTrajectoriesError',
    'SampledMiniBatch',
]
