"""VERL trainer adapter helpers.

``pad.pack_unpadded_groups``: pads unpadded TrainingSample list received from
LiveStore.get_batch into a SampledMiniBatch (DataProto-ready tensors + non-tensors).
"""

from trainer_adapters.verl.pad import SampledMiniBatch, pack_unpadded_groups

__all__ = ['SampledMiniBatch', 'pack_unpadded_groups']
