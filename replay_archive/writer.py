"""Worker-side async archive writer — bounded queue + retry + dead-letter.

The tee is async-fire-and-forget (BC-12): the worker's hot path (episode
generation) is never blocked by archive latency or failures.

At-least-once delivery: the archive's ``append_episodes`` is idempotent
on ``episode_uid``; retries are safe.
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

from schemas.episode_record import EpisodeRecord

logger = logging.getLogger(__name__)


class ReplayArchiveWriter:
    def __init__(
        self,
        *,
        server: object,  # ArchiveServer
        queue_max_size: int = 1024,
        dead_letter_path: str | Path = '/tmp/replay_archive_deadletter.jsonl',
        batch_size: int = 8,
        max_retry_attempts: int = 5,
    ) -> None:
        self._server = server
        self._queue: queue.Queue[EpisodeRecord] = queue.Queue(maxsize=queue_max_size)
        self._dead_letter_path = Path(dead_letter_path)
        self._dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
        self._dl_lock = threading.Lock()
        self._batch_size = int(batch_size)
        self._max_retries = int(max_retry_attempts)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._submitted = 0
        self._archived = 0
        self._dead_lettered = 0
        self._duplicates = 0

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

    def submit(self, record: EpisodeRecord) -> None:
        """Non-blocking enqueue. Overflow → dead-letter (never blocks generation)."""
        self._submitted += 1
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dead_letter([record], reason='queue_overflow')

    def stats(self) -> dict[str, int]:
        return {
            'submitted': self._submitted,
            'archived': self._archived,
            'dead_lettered': self._dead_lettered,
            'duplicates': self._duplicates,
            'queue_depth': self._queue.qsize(),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            batch = self._drain_batch()
            if not batch:
                try:
                    head = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                batch = [head] + self._drain_batch()
            self._flush_with_retry(batch)
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
        for attempt in range(self._max_retries):
            try:
                accepted, duplicates, _uids = self._server.append_episodes(batch)
                self._archived += accepted
                self._duplicates += duplicates
                return
            except Exception:  # noqa: BLE001
                logger.warning(
                    'replay_archive append failed (attempt %d/%d); retrying',
                    attempt + 1,
                    self._max_retries,
                    exc_info=True,
                )
                if self._stop_event.wait(timeout=delay):
                    break
                delay = min(delay * 2.0, 30.0)
        self._dead_letter(batch, reason='retry_exhausted')

    def _dead_letter(self, records: Iterable[EpisodeRecord], *, reason: str) -> None:
        with self._dl_lock:
            with self._dead_letter_path.open('a', encoding='utf-8') as fh:
                for r in records:
                    self._dead_lettered += 1
                    d = asdict(r)
                    d['started_at'] = r.started_at.astimezone(timezone.utc).isoformat()
                    d['finished_at'] = r.finished_at.astimezone(
                        timezone.utc
                    ).isoformat()
                    d['trust_level'] = r.trust_level.value
                    fh.write(
                        json.dumps(
                            {'reason': reason, 'spilled_at': time.time(), 'record': d}
                        )
                        + '\n'
                    )
