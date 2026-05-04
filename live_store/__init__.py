"""Slot 5.4 — LiveStore.

A bounded, low-latency, hot buffer between RolloutWorker (slot 5.3) and
TrainerAdapter (slot 5.6). Operates in groups (``n`` siblings together;
§3.2), pops on sample (§3.6), evicts by staleness, returns batches
sized for the trainer step.

The wire is **unpadded everywhere** (§6.2). Padding / sequence-packing
is the trainer adapter's job; see ``trainer_adapters/verl/pad.py``.

This package provides:

* :mod:`live_store.store_core` — process-local FIFO bounded buffer.
* :mod:`live_store.server`     — gRPC service over UDS wrapping the core.
* :mod:`live_store.client`     — gRPC client (trainer + producer side).
* :mod:`live_store.codec`      — DataProto ↔ ``list[TrainingSample]``
                                  conversion + protobuf marshalling.
* :mod:`live_store.main`       — service entry point.
"""

from live_store.client import LiveStoreClient
from live_store.store_core import (
    InsufficientTrajectoriesError,
    StoreCore,
    StoreMetrics,
)

__all__ = [
    'InsufficientTrajectoriesError',
    'LiveStoreClient',
    'StoreCore',
    'StoreMetrics',
]
