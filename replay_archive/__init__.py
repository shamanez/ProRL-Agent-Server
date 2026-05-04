"""Slot 5.5 — ReplayArchive (durable canonical record).

Lands at S3. Every episode the rollout worker produces is teed to the
archive in canonical :class:`schemas.episode_record.EpisodeRecord` form,
including filtered groups (zero-variance drop). The live store sees only
filter survivors; the archive sees everything (§7).

Writes go through :class:`ReplayArchiveWriter` (worker-side, async
fire-and-forget with at-least-once retry + dead-letter spillover).
Reads go through :func:`query` (offline jobs).

Different product from the LiveStore (§4.3): unbounded durable
storage, append-only, range-and-predicate queries. No latency target
on reads; ingest never blocks the worker's hot path.
"""

from replay_archive.derive import derive_training_samples
from replay_archive.query import query
from replay_archive.server import ArchiveServer
from replay_archive.writer import ReplayArchiveWriter

__all__ = [
    'ArchiveServer',
    'ReplayArchiveWriter',
    'derive_training_samples',
    'query',
]
