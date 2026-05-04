"""Append-only Parquet archive with a SQLite index (slot 5.5, S3).

Layout:
    {root}/index.db
    {root}/{policy_id}/{YYYY-MM-DD}/{environment_id}/{N}.parquet

Dedup: ``INSERT OR IGNORE`` on ``episode_uid PRIMARY KEY`` — at-least-once
delivery is safe (BC-13 of the archive). Record JSON stored as a single
``record_json`` column; the index is the query path.
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

_DDL = """
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
    """In-process archive backend, co-located with the rollout worker."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / 'index.db'
        self._lock = threading.Lock()
        self._seg_counter: dict[str, int] = {}
        self._init_index()

    def _init_index(self) -> None:
        with sqlite3.connect(self._index_path) as conn:
            conn.executescript(_DDL)

    def append_episodes(
        self, records: list[EpisodeRecord]
    ) -> tuple[int, int, list[str]]:
        """Append; return (accepted, duplicates, episode_uids)."""
        if not records:
            return 0, 0, []
        with self._lock:
            with sqlite3.connect(self._index_path) as conn:
                cur = conn.execute(
                    f'SELECT episode_uid FROM episodes '  # noqa: S608
                    f'WHERE episode_uid IN ({",".join("?" for _ in records)})',
                    [r.episode_uid for r in records],
                )
                seen = {row[0] for row in cur.fetchall()}
                fresh = [r for r in records if r.episode_uid not in seen]
                duplicates = len(records) - len(fresh)
                uids: list[str] = []
                if not fresh:
                    return 0, duplicates, uids
                bins: dict[Path, list[EpisodeRecord]] = {}
                for r in fresh:
                    bins.setdefault(self._partition_dir(r), []).append(r)
                for part_dir, recs in bins.items():
                    seg_path, _ = self._next_segment(part_dir)
                    self._write_segment(seg_path, recs)
                    rows = [
                        self._index_row(rec, str(seg_path), i)
                        for i, rec in enumerate(recs)
                    ]
                    conn.executemany(
                        'INSERT OR IGNORE INTO episodes VALUES '
                        '( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        rows,
                    )
                    uids.extend(r.episode_uid for r in recs)
                conn.commit()
                accepted = len(fresh)
        return accepted, duplicates, uids

    def fetch_records(self, episode_uids: list[str]) -> Iterable[EpisodeRecord]:
        if not episode_uids:
            return []
        with sqlite3.connect(self._index_path) as conn:
            cur = conn.execute(
                f'SELECT episode_uid, parquet_path FROM episodes '  # noqa: S608
                f'WHERE episode_uid IN ({",".join("?" for _ in episode_uids)})',
                episode_uids,
            )
            located = {row[0]: row[1] for row in cur.fetchall()}
        by_seg: dict[str, list[str]] = {}
        for uid, path in located.items():
            by_seg.setdefault(path, []).append(uid)
        out: list[EpisodeRecord] = []
        for path, uids in by_seg.items():
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

    # ---- internals -------------------------------------------------------

    def _partition_dir(self, r: EpisodeRecord) -> Path:
        date = r.started_at.astimezone(timezone.utc).strftime('%Y-%m-%d')
        return self._root / r.policy_id / date / r.environment_id

    def _next_segment(self, part_dir: Path) -> tuple[Path, int]:
        part_dir.mkdir(parents=True, exist_ok=True)
        key = str(part_dir)
        n = self._seg_counter.get(key, 0)
        if n == 0:
            existing = sorted(part_dir.glob('*.parquet'))
            if existing:
                try:
                    n = int(existing[-1].stem) + 1
                except ValueError:
                    n = len(existing)
        self._seg_counter[key] = n + 1
        return part_dir / f'{n}.parquet', 0

    def _write_segment(self, path: Path, records: list[EpisodeRecord]) -> None:
        rows = [
            {
                'episode_uid': r.episode_uid,
                'record_json': json.dumps(_record_to_dict(r)),
            }
            for r in records
        ]
        pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')

    def _index_row(self, r: EpisodeRecord, parquet_path: str, row: int) -> tuple:
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
            row,
        )


def _record_to_dict(r: EpisodeRecord) -> dict:
    d = asdict(r)
    d['started_at'] = r.started_at.astimezone(timezone.utc).isoformat()
    d['finished_at'] = r.finished_at.astimezone(timezone.utc).isoformat()
    d['trust_level'] = r.trust_level.value
    return d


def _dict_to_record(d: dict) -> EpisodeRecord:
    from schemas.episode_record import Event, RewardEvent  # noqa: PLC0415

    def _dt(s: str) -> datetime:
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
        reward_spec_id=d.get('reward_spec_id', ''),
        policy_id=d['policy_id'],
        policy_version=int(d['policy_version']),
        base_model_id=d.get('base_model_id', ''),
        tokenizer_id=d.get('tokenizer_id', ''),
        inference_backend=d['inference_backend'],
        sampling_params=d.get('sampling_params') or {},
        created_at_step=int(d.get('created_at_step', 0)),
        started_at=_dt(d['started_at']),
        finished_at=_dt(d['finished_at']),
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
