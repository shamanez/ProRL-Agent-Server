"""Append-only Parquet archive with a SQLite index.

Storage layout::

    {root}/
        index.db                     # SQLite index, single file
        {policy_id}/{YYYY-MM-DD}/    # partition: (policy_id, date)
            {environment_id}/        # subpartition by env
                {N}.parquet          # append-only segment files

Every ``append_episodes`` call appends one Parquet segment AND inserts
one row per :class:`EpisodeRecord` into the SQLite index. The index is
the query path; the Parquet files are the bulk storage. This keeps
``query`` fast (B-tree on indexed columns) and ``append`` lightweight
(no per-row index rebuild).

Episode dedup: the SQLite ``episodes(episode_uid PRIMARY KEY)`` table
makes ``INSERT OR IGNORE`` semantically idempotent. At-least-once
delivery from the writer is therefore safe — replays land as no-ops.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from schemas.episode_record import EpisodeRecord, TrustLevel

logger = logging.getLogger(__name__)

_INDEX_DDL = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_uid TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    split TEXT NOT NULL,
    environment_provider TEXT NOT NULL,
    environment_id TEXT NOT NULL,
    environment_version TEXT NOT NULL,
    verifier_version TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    inference_backend TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    termination_reason TEXT NOT NULL,
    total_reward REAL NOT NULL,
    trust_level TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    parquet_path TEXT NOT NULL,
    parquet_row INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_policy_version ON episodes(policy_id, policy_version);
CREATE INDEX IF NOT EXISTS idx_env ON episodes(environment_id, started_at);
CREATE INDEX IF NOT EXISTS idx_split ON episodes(split);
"""


class ArchiveServer:
    """In-process archive backend. Co-located with the rollout worker.

    The class is thread-safe via a single internal lock for index
    writes; Parquet writes are append-only via fresh segment files
    (one per ``append_episodes`` batch), so they don't need
    coordination beyond the per-call lock.

    Parameters
    ----------
    root:
        Filesystem root for the archive. Created if absent.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / 'index.db'
        self._lock = threading.Lock()
        self._segment_counter: dict[str, int] = {}
        self._init_index()

    def _init_index(self) -> None:
        with sqlite3.connect(self._index_path) as conn:
            conn.executescript(_INDEX_DDL)

    # ---- ingest -------------------------------------------------------------

    def append_episodes(
        self,
        records: list[EpisodeRecord],
    ) -> tuple[int, int, list[str]]:
        """Append episodes; return ``(accepted, duplicates, episode_uids)``.

        Idempotent on ``episode_uid`` — replays of an already-archived
        episode count as duplicates and do not produce a new Parquet
        row.
        """
        if not records:
            return 0, 0, []
        # Group by partition key for segment placement.
        with self._lock:
            accepted = 0
            duplicates = 0
            uids: list[str] = []
            with sqlite3.connect(self._index_path) as conn:
                # Pre-check duplicates so we don't write Parquet rows
                # for episodes the index already has.
                cursor = conn.execute(
                    f'SELECT episode_uid FROM episodes '
                    f'WHERE episode_uid IN ({",".join("?" for _ in records)})',  # noqa: S608
                    [r.episode_uid for r in records],
                )
                seen = {row[0] for row in cursor.fetchall()}
                fresh = [r for r in records if r.episode_uid not in seen]
                duplicates = len(records) - len(fresh)
                for r in fresh:
                    uids.append(r.episode_uid)
                if not fresh:
                    return 0, duplicates, uids
                # Bin by partition.
                bins: dict[Path, list[EpisodeRecord]] = {}
                for r in fresh:
                    bins.setdefault(self._partition_dir(r), []).append(r)
                for part_dir, recs in bins.items():
                    seg_path, base_row = self._next_segment(part_dir)
                    self._write_parquet_segment(seg_path, recs)
                    rows = [
                        self._index_row(rec, str(seg_path), base_row + i)
                        for i, rec in enumerate(recs)
                    ]
                    conn.executemany(
                        'INSERT OR IGNORE INTO episodes VALUES ('
                        ' ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?'
                        ')',
                        rows,
                    )
                    accepted += len(recs)
                conn.commit()
        return accepted, duplicates, uids

    def _partition_dir(self, r: EpisodeRecord) -> Path:
        date = r.started_at.astimezone(timezone.utc).strftime('%Y-%m-%d')
        return self._root / r.policy_id / date / r.environment_id

    def _next_segment(self, part_dir: Path) -> tuple[Path, int]:
        part_dir.mkdir(parents=True, exist_ok=True)
        key = str(part_dir)
        n = self._segment_counter.get(key, 0)
        # Find max segment id on disk to survive process restart.
        if n == 0:
            existing = sorted(part_dir.glob('*.parquet'))
            if existing:
                tail = existing[-1].stem
                try:
                    n = int(tail) + 1
                except ValueError:
                    n = len(existing)
        self._segment_counter[key] = n + 1
        return part_dir / f'{n}.parquet', 0

    def _write_parquet_segment(self, path: Path, records: list[EpisodeRecord]) -> None:
        """Serialize episodes to a single Parquet segment.

        We store the canonical record as JSON in a ``record_json``
        column rather than expanding every field into Parquet schema.
        Two reasons:

        1. The ``events`` field is variable-shape (list of typed
           variants); shoehorning it into a flat columnar shape
           loses information.
        2. The query path uses the SQLite index for the predicates
           we care about (policy_id / version / split / env / time);
           the Parquet load is a follow-up by ``episode_uid``.
        """
        rows = []
        for r in records:
            rows.append(
                {
                    'episode_uid': r.episode_uid,
                    'record_json': json.dumps(_record_to_dict(r)),
                }
            )
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path, compression='zstd')

    def _index_row(
        self, r: EpisodeRecord, parquet_path: str, parquet_row: int
    ) -> tuple:
        return (
            r.episode_uid,
            r.task_id,
            r.split,
            r.environment_provider,
            r.environment_id,
            r.environment_version,
            r.verifier_version,
            r.policy_id,
            r.policy_version,
            r.inference_backend,
            r.started_at.astimezone(timezone.utc).isoformat(),
            r.finished_at.astimezone(timezone.utc).isoformat(),
            r.termination_reason,
            r.total_reward,
            r.trust_level.value,
            r.schema_version,
            parquet_path,
            parquet_row,
        )

    # ---- read ---------------------------------------------------------------

    def fetch_records(self, episode_uids: list[str]) -> Iterable[EpisodeRecord]:
        """Hydrate full :class:`EpisodeRecord` objects by uid.

        Used by ``query()`` after the SQLite index narrows the result
        set; the Parquet segment is opened only for the matching rows.
        """
        if not episode_uids:
            return []
        with sqlite3.connect(self._index_path) as conn:
            cur = conn.execute(
                f'SELECT episode_uid, parquet_path FROM episodes '
                f'WHERE episode_uid IN ({",".join("?" for _ in episode_uids)})',  # noqa: S608
                episode_uids,
            )
            located: dict[str, str] = {row[0]: row[1] for row in cur.fetchall()}
        # Group by parquet path so we open each segment once.
        by_segment: dict[str, list[str]] = {}
        for uid, path in located.items():
            by_segment.setdefault(path, []).append(uid)
        out: list[EpisodeRecord] = []
        for path, uids in by_segment.items():
            table = pq.read_table(path, columns=['episode_uid', 'record_json'])
            uid_set = set(uids)
            for batch in table.to_pylist():
                if batch['episode_uid'] in uid_set:
                    out.append(_dict_to_record(json.loads(batch['record_json'])))
        return out

    def index_path(self) -> Path:
        return self._index_path

    def root(self) -> Path:
        return self._root


# ---- (de)serialization -----------------------------------------------------


def _record_to_dict(r: EpisodeRecord) -> dict:
    """Lossy-but-auditable JSON dict.

    Datetimes go to ISO-8601 UTC; ``Event`` and ``RewardEvent`` are
    flattened via ``asdict``. Token tuples become lists. Inverse is
    :func:`_dict_to_record`.
    """
    d = asdict(r)
    d['started_at'] = r.started_at.astimezone(timezone.utc).isoformat()
    d['finished_at'] = r.finished_at.astimezone(timezone.utc).isoformat()
    d['trust_level'] = r.trust_level.value
    return d


def _dict_to_record(d: dict) -> EpisodeRecord:
    from schemas.episode_record import Event, RewardEvent  # noqa: PLC0415

    def _parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s)

    events = tuple(
        Event(
            turn_index=int(e['turn_index']),
            kind=str(e['kind']),
            response_token_ids=tuple(e['response_token_ids'])
            if e.get('response_token_ids')
            else None,
            response_loss_mask=tuple(e['response_loss_mask'])
            if e.get('response_loss_mask')
            else None,
            behavior_log_probs=tuple(e['behavior_log_probs'])
            if e.get('behavior_log_probs')
            else None,
            tool_name=e.get('tool_name'),
            tool_input=e.get('tool_input'),
            observation=e.get('observation'),
            reward=RewardEvent(**e['reward']) if e.get('reward') else None,
            metadata=e.get('metadata') or {},
        )
        for e in d.get('events', [])
    )
    reward_events = tuple(RewardEvent(**re) for re in d.get('reward_events', []))
    return EpisodeRecord(
        episode_uid=d['episode_uid'],
        task_id=d['task_id'],
        split=d['split'],
        environment_provider=d['environment_provider'],
        environment_id=d['environment_id'],
        environment_version=d['environment_version'],
        verifier_version=d['verifier_version'],
        reward_spec_id=d['reward_spec_id'],
        policy_id=d['policy_id'],
        policy_version=int(d['policy_version']),
        base_model_id=d['base_model_id'],
        tokenizer_id=d['tokenizer_id'],
        inference_backend=d['inference_backend'],
        sampling_params=d.get('sampling_params') or {},
        created_at_step=int(d.get('created_at_step', 0)),
        started_at=_parse_dt(d['started_at']),
        finished_at=_parse_dt(d['finished_at']),
        termination_reason=d['termination_reason'],
        events=events,
        messages_or_turns=tuple(d.get('messages_or_turns', ())),
        total_reward=float(d.get('total_reward', 0.0)),
        reward_events=reward_events,
        prompt_token_ids=tuple(d.get('prompt_token_ids', ())),
        response_token_ids=tuple(d.get('response_token_ids', ())),
        response_loss_mask=tuple(d.get('response_loss_mask', ())),
        behavior_log_probs=tuple(d['behavior_log_probs'])
        if d.get('behavior_log_probs')
        else None,
        tool_calls=tuple(d.get('tool_calls', ())),
        provenance=d.get('provenance') or {},
        trust_level=TrustLevel(d.get('trust_level', 'own-fabric')),
        schema_version=d.get('schema_version', '1.0.0'),
    )
