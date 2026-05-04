"""Worker-side archive writer — bounded queue + retry + dead-letter.

Per §7 (rollout_fabric.md): the producer tees every episode to BOTH
the LiveStore (filtered) and the ReplayArchive (unfiltered). The
archive write is async-fire-and-forget from the producer's
perspective: it MUST NOT block episode generation.

This writer enforces that contract:

* Bounded in-memory queue. ``submit`` is non-blocking — the producer
  enqueues and returns; a daemon thread drains the queue.
* On queue overflow, the episode is spilled to a JSONL **dead-letter
  file** so generation never stalls; an offline replay job can ingest
  the dead-letter file later.
* On archive ingest failure, the writer retries with exponential
  backoff (initial 1s, max 30s, capped attempts). After max attempts
  the episode is also spilled to the dead-letter.
* At-least-once delivery: the archive's ``append_episodes`` is
  idempotent on ``episode_uid``, so retries are safe.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import timezone
from pathlib import Path

from replay_archive.server import ArchiveServer
from schemas.episode_record import EpisodeRecord

logger = logging.getLogger(__name__)


class ReplayArchiveWriter:
    """Worker-side async writer.

    Parameters
    ----------
    server:
        :class:`ArchiveServer` instance the writer drains into. Owned
        by the same process here at S3 (in-process tee); future
        deployments may swap this for a network client.
    queue_max_size:
        Max episodes the in-memory queue holds. Overflow goes to the
        dead-letter file.
    dead_letter_path:
        File where overflow / retry-exhausted episodes are appended
        as one JSONL record per line.
    batch_size:
        How many episodes are bundled into one ``append_episodes``
        call. Larger batches amortize SQLite + Parquet overhead at
        the cost of latency-to-archive.
    """

    def __init__(
        self,
        *,
        server: ArchiveServer,
        queue_max_size: int = 1024,
        dead_letter_path: str | Path = '/tmp/replay_archive_deadletter.jsonl',
        batch_size: int = 8,
        max_retry_attempts: int = 5,
    ) -> None:
        self._server = server
        self._queue: queue.Queue[EpisodeRecord] = queue.Queue(maxsize=queue_max_size)
        self._dead_letter_path = Path(dead_letter_path)
        self._dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
        self._dead_letter_lock = threading.Lock()
        self._batch_size = int(batch_size)
        self._max_retry_attempts = int(max_retry_attempts)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Counters for the post-S4 checklist signals.
        self._submitted = 0
        self._archived = 0
        self._dead_lettered = 0
        self._duplicates = 0

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError('ReplayArchiveWriter already started')
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name='ReplayArchiveWriter', daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ---- ingest -------------------------------------------------------------

    def submit(self, record: EpisodeRecord) -> None:
        """Async-fire-and-forget. Non-blocking; overflow → dead-letter."""
        self._submitted += 1
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dead_letter([record], reason='queue_overflow')

    # ---- counters -----------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            'submitted': self._submitted,
            'archived': self._archived,
            'dead_lettered': self._dead_lettered,
            'duplicates': self._duplicates,
            'queue_depth': self._queue.qsize(),
        }

    # ---- worker -------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            batch = self._drain_batch()
            if not batch:
                # Block briefly on the queue so we wake up promptly on
                # the next submit.
                try:
                    head = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                batch = [head] + self._drain_batch()
            self._flush_with_retry(batch)
        # Final drain on shutdown.
        remaining = self._drain_batch()
        if remaining:
            self._flush_with_retry(remaining)

    def _drain_batch(self) -> list[EpisodeRecord]:
        out: list[EpisodeRecord] = []
        while len(out) < self._batch_size:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    def _flush_with_retry(self, batch: list[EpisodeRecord]) -> None:
        delay = 1.0
        for attempt in range(self._max_retry_attempts):
            try:
                accepted, duplicates, _uids = self._server.append_episodes(batch)
                self._archived += accepted
                self._duplicates += duplicates
                return
            except Exception:  # noqa: BLE001
                logger.warning(
                    'replay_archive append failed (attempt %d/%d); retrying',
                    attempt + 1,
                    self._max_retry_attempts,
                    exc_info=True,
                )
                if self._stop_event.wait(timeout=delay):
                    break
                delay = min(delay * 2.0, 30.0)
        # Out of retries: dead-letter.
        self._dead_letter(batch, reason='retry_exhausted')

    def _dead_letter(self, records: Iterable[EpisodeRecord], *, reason: str) -> None:
        with self._dead_letter_lock:
            with self._dead_letter_path.open('a', encoding='utf-8') as fh:
                for r in records:
                    self._dead_lettered += 1
                    fh.write(
                        json.dumps(
                            {
                                'reason': reason,
                                'episode_uid': r.episode_uid,
                                'record': _serialize_for_dead_letter(r),
                                'spilled_at': time.time(),
                            }
                        )
                        + '\n'
                    )
        logger.warning(
            'dead-lettered %d episode(s) reason=%s path=%s',
            sum(1 for _ in records),
            reason,
            self._dead_letter_path,
        )


def _serialize_for_dead_letter(r: EpisodeRecord) -> dict:
    """JSON-friendly view used only by the dead-letter file.

    Lossy on datetimes (ISO-8601) and TrustLevel (string). Inverse is
    handled by an offline ingestion script — not on this hot path.
    """
    d = asdict(r)
    d['started_at'] = r.started_at.astimezone(timezone.utc).isoformat()
    d['finished_at'] = r.finished_at.astimezone(timezone.utc).isoformat()
    d['trust_level'] = r.trust_level.value
    return d
