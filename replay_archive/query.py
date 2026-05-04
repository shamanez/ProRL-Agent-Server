"""Read-side filter against the SQLite index + Parquet payload.

Used by offline jobs (reward-distribution histograms, distillation
training, audits). Not on any hot path.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from replay_archive.server import ArchiveServer
from schemas.episode_record import EpisodeRecord
from schemas.protocols.replay_archive import FilterSpec


def query(
    archive_root: str | Path,
    *,
    filter_spec: FilterSpec,
    limit: int | None = None,
) -> Iterable[EpisodeRecord]:
    """Run ``filter_spec`` against the archive at ``archive_root``.

    Matches the §A.5 ``ReplayArchive.query`` contract: returns an
    iterable of full :class:`EpisodeRecord` objects. The SQLite index
    narrows the result set; the Parquet segments are loaded only for
    matching rows.
    """
    server = ArchiveServer(archive_root)
    where, params = _build_where_clause(filter_spec)
    sql = f'SELECT episode_uid FROM episodes {where} ORDER BY started_at'  # noqa: S608
    if limit is not None:
        sql += f' LIMIT {int(limit)}'
    with sqlite3.connect(server.index_path()) as conn:
        cur = conn.execute(sql, params)
        uids = [row[0] for row in cur.fetchall()]
    return list(server.fetch_records(uids))


def count(
    archive_root: str | Path,
    *,
    filter_spec: FilterSpec,
) -> int:
    """Cheap row count via the index, without hydrating Parquet."""
    server = ArchiveServer(archive_root)
    where, params = _build_where_clause(filter_spec)
    sql = f'SELECT COUNT(*) FROM episodes {where}'  # noqa: S608
    with sqlite3.connect(server.index_path()) as conn:
        cur = conn.execute(sql, params)
        return int(cur.fetchone()[0])


def _build_where_clause(spec: FilterSpec) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if spec.policy_id is not None:
        clauses.append('policy_id = ?')
        params.append(spec.policy_id)
    if spec.policy_version_min is not None:
        clauses.append('policy_version >= ?')
        params.append(spec.policy_version_min)
    if spec.policy_version_max is not None:
        clauses.append('policy_version <= ?')
        params.append(spec.policy_version_max)
    if spec.environment_id is not None:
        clauses.append('environment_id = ?')
        params.append(spec.environment_id)
    if spec.split is not None:
        clauses.append('split = ?')
        params.append(spec.split)
    if spec.started_after is not None:
        clauses.append('started_at >= ?')
        params.append(spec.started_after)
    if spec.finished_before is not None:
        clauses.append('finished_at <= ?')
        params.append(spec.finished_before)
    if spec.reward_min is not None:
        clauses.append('total_reward >= ?')
        params.append(spec.reward_min)
    if spec.reward_max is not None:
        clauses.append('total_reward <= ?')
        params.append(spec.reward_max)
    if spec.trust_levels:
        placeholders = ','.join('?' for _ in spec.trust_levels)
        clauses.append(f'trust_level IN ({placeholders})')
        params.extend(t.value for t in spec.trust_levels)
    if spec.task_ids:
        placeholders = ','.join('?' for _ in spec.task_ids)
        clauses.append(f'task_id IN ({placeholders})')
        params.extend(spec.task_ids)
    if not clauses:
        return '', []
    return 'WHERE ' + ' AND '.join(clauses), params
