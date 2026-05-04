"""Simple parquet dataloader owned by the RolloutWorker (BC-14).

No VERL, no OpenHands. Reads parquet files directly with pyarrow,
cycles infinitely, and supports ``state_dict`` / ``load_state_dict``
for checkpoint resume.

The worker owns the dataset (§3.8 data ownership boundary). After S2
the trainer has no parquet files, no ``data.train_files`` Hydra key,
and no knowledge of task IDs.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetDataLoader:
    """Infinitely cycling iterator over a list of parquet files.

    One ``next()`` call returns one row as a plain Python dict. The loader
    shuffles within each file on each pass and advances through files in
    round-robin order.

    Parameters
    ----------
    data_files:
        Paths to parquet files. At least one required.
    seed:
        RNG seed for within-file shuffling. Reproducible across restarts
        when loaded from ``state_dict``.
    """

    def __init__(
        self,
        data_files: list[str],
        *,
        seed: int = 42,
    ) -> None:
        if not data_files:
            raise ValueError('ParquetDataLoader requires at least one data file')
        self._files = [str(Path(f).resolve()) for f in data_files]
        self._rng = random.Random(seed)
        self._file_idx: int = 0
        self._row_idx: int = 0
        self._current_rows: list[dict[str, Any]] = []
        self._total_yielded: int = 0
        self._load_file(self._file_idx)

    # ---- public API -------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._row_idx >= len(self._current_rows):
            self._advance_file()
        row = self._current_rows[self._row_idx]
        self._row_idx += 1
        self._total_yielded += 1
        return row

    def state_dict(self) -> bytes:
        """Serialise loader position for checkpoint resume."""
        state = {
            'files': self._files,
            'file_idx': self._file_idx,
            'row_idx': self._row_idx,
            'total_yielded': self._total_yielded,
            'rng_state': self._rng.getstate(),
        }
        return json.dumps(state).encode()

    def load_state_dict(self, state: bytes) -> None:
        """Restore loader position after a worker restart."""
        d = json.loads(state.decode())
        self._files = d['files']
        self._file_idx = int(d['file_idx'])
        self._row_idx = int(d['row_idx'])
        self._total_yielded = int(d['total_yielded'])
        self._rng.setstate(d['rng_state'])
        self._load_file(self._file_idx)
        # Advance the shuffled rows to the saved row_idx so the next
        # call to ``__next__`` returns the same row as before the restart.

    # ---- internals --------------------------------------------------------

    def _load_file(self, file_idx: int) -> None:
        path = self._files[file_idx % len(self._files)]
        table = pq.read_table(path)
        rows = table.to_pylist()
        self._rng.shuffle(rows)
        self._current_rows = rows
        logger.debug('ParquetDataLoader loaded %d rows from %s', len(rows), path)

    def _advance_file(self) -> None:
        self._file_idx = (self._file_idx + 1) % len(self._files)
        self._row_idx = 0
        self._load_file(self._file_idx)
