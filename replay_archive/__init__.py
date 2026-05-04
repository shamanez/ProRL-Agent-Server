"""Slot 5.5 — ReplayArchive (S3). Append-only Parquet + SQLite index.

Archive sees ALL episodes (pre-filter). LiveStore sees only filter survivors
(BC-12). Different product from the LiveStore (§4.3, §7).
"""

from replay_archive.derive import TokenizerMismatchError, derive_training_samples
from replay_archive.query import count, query
from replay_archive.server import ArchiveServer
from replay_archive.writer import ReplayArchiveWriter

__all__ = [
    'ArchiveServer',
    'ReplayArchiveWriter',
    'TokenizerMismatchError',
    'count',
    'derive_training_samples',
    'query',
]
