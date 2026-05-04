"""VERL trainer adapter — current concrete adapter for slot 5.6.

Wraps ``trainer_integration/verl/verl_custom/...`` for the post-S1 wire.
At S1 we expose only ``pad.pack_unpadded_groups`` — the lifted ``_pack``
that turns unpadded :class:`schemas.training_sample.TrainingSample`
records into the legacy ``SampledMiniBatch`` tensor/non-tensor layout
the trainer's ``DataProto.from_dict`` path expects.
"""

from trainer_adapters.verl.pad import (
    SampledMiniBatch,
    pack_unpadded_groups,
)

__all__ = ['SampledMiniBatch', 'pack_unpadded_groups']
