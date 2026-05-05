"""Read-side filter against the SQLite index + Parquet payload (offline path)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from rollout_fabric.replay_archive.server import ArchiveServer
from rollout_fabric.schemas.episode_record import EpisodeRecord
from rollout_fabric.schemas.protocols.replay_archive import FilterSpec


def query(
    archive_root: str | Path,
    *,
    filter_spec: FilterSpec,
    limit: int | None = None,
) -> Iterable[EpisodeRecord]:
    server = ArchiveServer(archive_root)
    where, params = _build_where(filter_spec)
    sql = f'SELECT episode_uid FROM episodes {where} ORDER BY started_at'  # noqa: S608
    if limit is not None:
        sql += f' LIMIT {int(limit)}'
    with sqlite3.connect(server.index_path()) as conn:
        uids = [row[0] for row in conn.execute(sql, params).fetchall()]
    return list(server.fetch_records(uids))


def count(archive_root: str | Path, *, filter_spec: FilterSpec) -> int:
    server = ArchiveServer(archive_root)
    where, params = _build_where(filter_spec)
    with sqlite3.connect(server.index_path()) as conn:
        return int(
            conn.execute(
                f'SELECT COUNT(*) FROM episodes {where}',
                params,  # noqa: S608
            ).fetchone()[0]
        )


def _build_where(spec: FilterSpec) -> tuple[str, list]:
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
        ph = ','.join('?' for _ in spec.trust_levels)
        clauses.append(f'trust_level IN ({ph})')
        params.extend(t.value for t in spec.trust_levels)
    if spec.task_ids:
        ph = ','.join('?' for _ in spec.task_ids)
        clauses.append(f'task_id IN ({ph})')
        params.extend(spec.task_ids)
    return (('WHERE ' + ' AND '.join(clauses)), params) if clauses else ('', [])
