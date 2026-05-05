"""Training Monitor Agent — autonomous health watcher + progress reporter.

Architecture:
    TrainingMonitor (configurable interval, default 30s)
        ├─► MetricsAgent     — parse PRODUCER_ITER JSON from rollout_manager.log
        ├─► WeightSyncAgent  — check PolicyRegistry log for reload_lora events
        ├─► RolloutAgent     — check vLLM pool /health + response latency
        ├─► LiveStoreAgent   — check socket exists + num_groups via LiveStoreClient
        ├─► TrainerAgent     — check trainer log for loss values, NaN detection
        ├─► RescueCoordinator — imported from rescue_team, triggered on failures
        └─► ProgressWriter   — write current_training_progress.md

Usage:
    PYTHONPATH=./core python ops/services/training_monitor.py --interval 30
    PYTHONPATH=./core python ops/services/training_monitor.py --once
    PYTHONPATH=./core python ops/services/training_monitor.py --no-rescue
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── path bootstrap ──────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]  # ops/services/ → ops/ → ProRL-Agent-Server/

# Make rollout_fabric importable if ./core is on sys.path.
_CORE_DIR = REPO_ROOT / 'core'
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

# Make rescue_team importable from the same ops/services/ directory.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ── logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s training_monitor: %(message)s',
)
logger = logging.getLogger('training_monitor')

# ── constants ────────────────────────────────────────────────────────────────

ROLLOUT_LOG = Path('/tmp/rollout_manager.log')
TRAINER_LOG = Path('/tmp/trainer.log')
POLICY_REGISTRY_LOG = Path('/tmp/policy_registry.log')

LIVE_STORE_SOCKET = '/tmp/prorl_live_store.sock'
POLICY_REGISTRY_SOCKET = '/tmp/prorl_policy_registry.sock'

ENV_PROVIDER_URL = 'http://localhost:8006/status'

REMOTE_DNS = os.environ.get('REMOTE_DNS', 'vllm-instance')
VLLM_PORTS = [8100, 8101, 8102, 8103]

STALL_THRESHOLD_S = 300.0  # BC-5: no-progress threshold
DEFAULT_INTERVAL_S = 30
TAIL_LINES_METRICS = 500
TAIL_LINES_TRAINER = 100
TAIL_LINES_WEIGHT = 200
MAX_RECENT_WEIGHT_EVENTS = 5
MAX_RECENT_RESCUE_EVENTS = 5
MAX_RECENT_PRODUCER_EVENTS = 5


# ── shared data types ────────────────────────────────────────────────────────


@dataclass
class ProducerEvent:
    """Parsed PRODUCER_ITER log line."""

    wall_s: float
    policy_id: str
    policy_version: int
    created_at_step: int
    group_uid: str
    group_size: int
    groups_pushed: int
    groups_filtered: int
    parsed_at: float = field(default_factory=time.monotonic)


@dataclass
class MetricsSnapshot:
    """Aggregated rollout-manager metrics."""

    latest_event: ProducerEvent | None
    policy_version: int
    groups_pushed: int
    groups_filtered: int
    filter_rate_pct: float
    throughput_groups_per_hr: float
    last_push_wall: float | None  # monotonic seconds of last parse
    stall_risk: bool
    bc0_ok: bool  # all siblings same version (best effort)
    bc5_ok: bool  # no stall
    bc16_ok: bool  # trainer should not start before first push
    recent_events: list[ProducerEvent]


@dataclass
class WeightSyncSnapshot:
    endpoints_failed_detected: bool  # BC-9
    bc9_ok: bool
    last_events: list[str]  # human-readable summary lines


@dataclass
class VllmPortStatus:
    port: int
    healthy: bool
    detail: str
    latency_ms: float | None


@dataclass
class RolloutSnapshot:
    port_statuses: list[VllmPortStatus]
    all_healthy: bool


@dataclass
class LiveStoreSnapshot:
    socket_exists: bool
    num_groups: int | None
    total_pushes: int | None
    detail: str


@dataclass
class TrainerSnapshot:
    last_loss: float | None
    nan_detected: bool
    last_lines: list[str]
    detail: str


@dataclass
class MonitorSnapshot:
    timestamp: datetime
    metrics: MetricsSnapshot
    weight_sync: WeightSyncSnapshot
    rollout: RolloutSnapshot
    live_store: LiveStoreSnapshot
    trainer: TrainerSnapshot
    overall_status: str  # HEALTHY | DEGRADED | RESCUED | STALLED
    rescue_log: list[str]


# ── MetricsAgent ─────────────────────────────────────────────────────────────


class MetricsAgent:
    """Parse PRODUCER_ITER JSON events from rollout_manager.log."""

    def __init__(self, log_path: Path = ROLLOUT_LOG) -> None:
        self._log_path = log_path

    def collect(self) -> MetricsSnapshot:
        events = self._parse_events()
        if not events:
            return MetricsSnapshot(
                latest_event=None,
                policy_version=0,
                groups_pushed=0,
                groups_filtered=0,
                filter_rate_pct=0.0,
                throughput_groups_per_hr=0.0,
                last_push_wall=None,
                stall_risk=False,
                bc0_ok=True,
                bc5_ok=True,
                bc16_ok=True,
                recent_events=[],
            )

        latest = events[-1]
        groups_pushed = latest.groups_pushed
        groups_filtered = latest.groups_filtered
        total_attempts = groups_pushed + groups_filtered

        filter_rate_pct = (
            (groups_filtered / total_attempts * 100.0) if total_attempts > 0 else 0.0
        )

        # Throughput: use first and last event wall_s delta if ≥2 events.
        if len(events) >= 2:
            dt_s = events[-1].parsed_at - events[0].parsed_at
            n_span = events[-1].groups_pushed - events[0].groups_pushed
            throughput = (n_span / dt_s * 3600.0) if dt_s > 0 else 0.0
        else:
            throughput = 0.0

        # Stall detection (BC-5): last event arrived > STALL_THRESHOLD_S ago.
        now_mono = time.monotonic()
        age_s = now_mono - latest.parsed_at
        stall_risk = age_s > STALL_THRESHOLD_S

        # BC-0: within recent events, check all group_uids from the same push
        # share the same policy_version (best-effort heuristic using events list).
        bc0_ok = self._check_bc0(events)

        # BC-5 ok if not stalled.
        bc5_ok = not stall_risk

        # BC-16: trainer must start only after ≥1 group pushed.
        bc16_ok = groups_pushed >= 1

        recent = events[-MAX_RECENT_PRODUCER_EVENTS:]

        return MetricsSnapshot(
            latest_event=latest,
            policy_version=latest.policy_version,
            groups_pushed=groups_pushed,
            groups_filtered=groups_filtered,
            filter_rate_pct=filter_rate_pct,
            throughput_groups_per_hr=throughput,
            last_push_wall=latest.parsed_at,
            stall_risk=stall_risk,
            bc0_ok=bc0_ok,
            bc5_ok=bc5_ok,
            bc16_ok=bc16_ok,
            recent_events=recent,
        )

    def _parse_events(self) -> list[ProducerEvent]:
        if not self._log_path.exists():
            logger.debug('rollout_manager.log not found: %s', self._log_path)
            return []
        try:
            lines = self._tail(self._log_path, TAIL_LINES_METRICS)
        except Exception as exc:  # noqa: BLE001
            logger.warning('Failed to read %s: %s', self._log_path, exc)
            return []

        events: list[ProducerEvent] = []
        for line in lines:
            # Log lines look like: "... INFO rollout_manager.loop: PRODUCER_ITER {...}"
            # Extract JSON blob from the line.
            idx = line.find('PRODUCER_ITER')
            if idx == -1:
                continue
            json_part = line[idx + len('PRODUCER_ITER') :].strip()
            # Sometimes the JSON is preceded by a space.
            json_part = json_part.lstrip()
            try:
                obj = json.loads(json_part)
            except json.JSONDecodeError:
                # Try to find the first '{' in case there's extra prefix text.
                brace = line.find('{', idx)
                if brace == -1:
                    continue
                try:
                    obj = json.loads(line[brace:])
                except json.JSONDecodeError:
                    continue

            if obj.get('event') != 'producer_iter':
                continue
            try:
                events.append(
                    ProducerEvent(
                        wall_s=float(obj.get('wall_s', 0.0)),
                        policy_id=str(obj.get('policy_id', '')),
                        policy_version=int(obj.get('policy_version', 0)),
                        created_at_step=int(obj.get('created_at_step', 0)),
                        group_uid=str(obj.get('group_uid', '')),
                        group_size=int(obj.get('group_size', 0)),
                        groups_pushed=int(obj.get('groups_pushed', 0)),
                        groups_filtered=int(obj.get('groups_filtered', 0)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.debug('Skipping malformed producer_iter event: %s', exc)

        return events

    @staticmethod
    def _check_bc0(events: list[ProducerEvent]) -> bool:
        """
        BC-0: all siblings in a group share the same policy_version.
        We can't verify within a single-log-line event, but we can check
        that consecutive events within the same groups_pushed counter increment
        share matching policy_versions. Any jump with version mismatch within
        a tight window would be suspicious — best effort check.
        """
        if len(events) < 2:
            return True
        # Flag if policy_version went backwards (definitely wrong).
        for a, b in zip(events, events[1:]):
            if b.policy_version < a.policy_version:
                return False
        return True

    @staticmethod
    def _tail(path: Path, n: int) -> list[str]:
        result = subprocess.run(
            ['tail', '-n', str(n), str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.splitlines()


# ── WeightSyncAgent ──────────────────────────────────────────────────────────


class WeightSyncAgent:
    """Scan PolicyRegistry log for reload_lora events and endpoints_failed alerts."""

    def __init__(self, log_path: Path = POLICY_REGISTRY_LOG) -> None:
        self._log_path = log_path

    def collect(self) -> WeightSyncSnapshot:
        if not self._log_path.exists():
            return WeightSyncSnapshot(
                endpoints_failed_detected=False,
                bc9_ok=True,
                last_events=['(no policy_registry.log found)'],
            )
        try:
            lines = self._tail(self._log_path, TAIL_LINES_WEIGHT)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                'WeightSyncAgent: failed to read %s: %s', self._log_path, exc
            )
            return WeightSyncSnapshot(
                endpoints_failed_detected=False,
                bc9_ok=True,
                last_events=[f'(read error: {exc})'],
            )

        endpoints_failed = False
        event_lines: list[str] = []

        for line in lines:
            lo = line.lower()
            # Detect BC-9 violation: endpoints_failed > 0 in a fanout-FAILED line.
            if 'endpoints_failed' in lo and 'pool fanout failed' in lo:
                endpoints_failed = True
                event_lines.append(f'[BC-9 VIOLATION] {line.strip()}')
                continue
            # Capture weight-sync events: successful fanout OK lines.
            if 'pool fanout ok' in lo or 'reload_lora' in lo or 'policy_version' in lo:
                event_lines.append(line.strip())

        # Keep only the last N relevant events.
        recent = event_lines[-MAX_RECENT_WEIGHT_EVENTS:]

        return WeightSyncSnapshot(
            endpoints_failed_detected=endpoints_failed,
            bc9_ok=not endpoints_failed,
            last_events=recent if recent else ['(no weight sync events yet)'],
        )

    @staticmethod
    def _tail(path: Path, n: int) -> list[str]:
        result = subprocess.run(
            ['tail', '-n', str(n), str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.splitlines()


# ── RolloutAgent ─────────────────────────────────────────────────────────────


class RolloutAgent:
    """HTTP health-check the vLLM pool endpoints."""

    def __init__(
        self,
        remote_dns: str = REMOTE_DNS,
        ports: list[int] | None = None,
    ) -> None:
        self._remote_dns = remote_dns
        self._ports = ports if ports is not None else list(VLLM_PORTS)

    def collect(self) -> RolloutSnapshot:
        statuses: list[VllmPortStatus] = []
        for port in self._ports:
            statuses.append(self._check_port(port))
        all_healthy = all(s.healthy for s in statuses)
        return RolloutSnapshot(port_statuses=statuses, all_healthy=all_healthy)

    def _check_port(self, port: int) -> VllmPortStatus:
        url = f'http://{self._remote_dns}:{port}/health'
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                latency_ms = (time.monotonic() - t0) * 1000.0
                if resp.status == 200:
                    return VllmPortStatus(
                        port=port,
                        healthy=True,
                        detail='OK',
                        latency_ms=round(latency_ms, 1),
                    )
                return VllmPortStatus(
                    port=port,
                    healthy=False,
                    detail=f'HTTP {resp.status}',
                    latency_ms=round(latency_ms, 1),
                )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - t0) * 1000.0
            return VllmPortStatus(
                port=port,
                healthy=False,
                detail=f'{type(exc).__name__}: {exc}',
                latency_ms=round(latency_ms, 1),
            )


# ── LiveStoreAgent ───────────────────────────────────────────────────────────


class LiveStoreAgent:
    """Check LiveStore socket and query metrics via gRPC client."""

    def __init__(self, socket_path: str = LIVE_STORE_SOCKET) -> None:
        self._socket_path = socket_path

    def collect(self) -> LiveStoreSnapshot:
        socket_exists = Path(self._socket_path).exists()
        if not socket_exists:
            return LiveStoreSnapshot(
                socket_exists=False,
                num_groups=None,
                total_pushes=None,
                detail=f'socket not found: {self._socket_path}',
            )

        # Try to query the store metrics via the gRPC client.
        try:
            from rollout_fabric.live_store.client import (
                LiveStoreClient,  # noqa: PLC0415
            )

            cli = LiveStoreClient(
                self._socket_path,
                policy_id=os.environ.get('POLICY_ID', 'qwen3-4b-skyrl'),
                environment_id=os.environ.get('ENVIRONMENT_ID', 'prorl_default'),
            )
            try:
                num_groups = cli.num_groups()
                total_pushes = cli.total_pushes()
            finally:
                cli.close()

            return LiveStoreSnapshot(
                socket_exists=True,
                num_groups=num_groups,
                total_pushes=total_pushes,
                detail=f'store_size={num_groups} total_pushes={total_pushes}',
            )
        except ImportError as exc:
            logger.debug('LiveStoreClient import failed (PYTHONPATH?): %s', exc)
            return LiveStoreSnapshot(
                socket_exists=True,
                num_groups=None,
                total_pushes=None,
                detail=f'socket exists; client unavailable ({exc})',
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning('LiveStoreAgent: gRPC query failed: %s', exc)
            return LiveStoreSnapshot(
                socket_exists=True,
                num_groups=None,
                total_pushes=None,
                detail=f'socket exists; query error: {exc}',
            )


# ── TrainerAgent ─────────────────────────────────────────────────────────────


_LOSS_RE = re.compile(
    r'(?:loss[:\s=]+)([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)',
    re.IGNORECASE,
)
_NAN_RE = re.compile(r'\bnan\b', re.IGNORECASE)


class TrainerAgent:
    """Read trainer log, detect loss values and NaN."""

    def __init__(self, log_path: Path = TRAINER_LOG) -> None:
        self._log_path = log_path

    def collect(self) -> TrainerSnapshot:
        if not self._log_path.exists():
            return TrainerSnapshot(
                last_loss=None,
                nan_detected=False,
                last_lines=[],
                detail='trainer.log not found',
            )
        try:
            lines = self._tail(self._log_path, TAIL_LINES_TRAINER)
        except Exception as exc:  # noqa: BLE001
            return TrainerSnapshot(
                last_loss=None,
                nan_detected=False,
                last_lines=[],
                detail=f'read error: {exc}',
            )

        last_loss: float | None = None
        nan_detected = False

        for line in lines:
            if _NAN_RE.search(line):
                nan_detected = True
            m = _LOSS_RE.search(line)
            if m:
                try:
                    last_loss = float(m.group(1))
                except ValueError:
                    pass

        detail = ''
        if nan_detected:
            detail = 'NaN detected in trainer log!'
        elif last_loss is not None:
            detail = f'last_loss={last_loss:.6f}'
        else:
            detail = 'no loss values found in recent log'

        return TrainerSnapshot(
            last_loss=last_loss,
            nan_detected=nan_detected,
            last_lines=lines[-10:],
            detail=detail,
        )

    @staticmethod
    def _tail(path: Path, n: int) -> list[str]:
        result = subprocess.run(
            ['tail', '-n', str(n), str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.splitlines()


# ── ProgressWriter ────────────────────────────────────────────────────────────


def _tz_now() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_ts(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def _fmt_time(dt: datetime) -> str:
    return dt.strftime('%H:%M:%S')


class ProgressWriter:
    """Write current_training_progress.md in the specified format."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def write(self, snap: MonitorSnapshot) -> None:
        md = self._render(snap)
        self._output_path.write_text(md, encoding='utf-8')
        logger.debug('Wrote progress to %s', self._output_path)

    def _render(self, snap: MonitorSnapshot) -> str:
        ts_str = _fmt_ts(snap.timestamp)
        now_hms = _fmt_time(snap.timestamp)
        m = snap.metrics
        ws = snap.weight_sync
        ro = snap.rollout
        ls = snap.live_store
        tr = snap.trainer

        # ── last push time ─────────────────────────────────────────────────
        if m.last_push_wall is not None:
            age_s = time.monotonic() - m.last_push_wall
            last_push_str = _fmt_ts(
                datetime.fromtimestamp(
                    snap.timestamp.timestamp() - age_s, tz=timezone.utc
                )
            )
        else:
            last_push_str = 'N/A'

        stall_risk_str = 'YES — stall detected' if m.stall_risk else 'No'

        # ── BC checks ──────────────────────────────────────────────────────
        bc0_status = _bc_status(m.bc0_ok)
        bc5_status = _bc_status(m.bc5_ok)
        bc9_status = _bc_status(ws.bc9_ok)
        bc16_status = _bc_status(m.bc16_ok)

        # ── service health ─────────────────────────────────────────────────
        live_store_status = _service_status(ls.socket_exists)
        pr_sock_exists = Path(POLICY_REGISTRY_SOCKET).exists()
        policy_reg_status = _service_status(pr_sock_exists)

        ep_status = _http_status(ENV_PROVIDER_URL)
        ep_icon = '✓ Healthy' if ep_status else '✗ Unhealthy'

        vllm_rows = ''
        for ps in ro.port_statuses:
            icon = '✓ Healthy' if ps.healthy else '✗ Unhealthy'
            lat_str = f'{ps.latency_ms:.0f}ms' if ps.latency_ms is not None else 'N/A'
            detail = '' if ps.healthy else f' ({ps.detail})'
            vllm_rows += (
                f'| vLLM :{ps.port} | {icon}{detail} ({lat_str}) | {now_hms} |\n'
            )

        # ── weight sync log ────────────────────────────────────────────────
        if ws.last_events:
            weight_log_lines = '\n'.join(f'- {e}' for e in ws.last_events)
        else:
            weight_log_lines = '_No weight sync events._'

        # ── rescue log ────────────────────────────────────────────────────
        if snap.rescue_log:
            rescue_log_lines = '\n'.join(
                f'- {e}' for e in snap.rescue_log[-MAX_RECENT_RESCUE_EVENTS:]
            )
        else:
            rescue_log_lines = '_No rescue events._'

        # ── recent events ──────────────────────────────────────────────────
        recent_lines: list[str] = []
        for ev in reversed(m.recent_events):
            age = time.monotonic() - ev.parsed_at
            ev_ts = datetime.fromtimestamp(
                snap.timestamp.timestamp() - age, tz=timezone.utc
            )
            recent_lines.append(
                f'- {_fmt_time(ev_ts)} — groups_pushed={ev.groups_pushed},'
                f' policy_version={ev.policy_version},'
                f' group_size={ev.group_size}'
            )
        if not recent_lines:
            recent_lines = ['- (no events yet)']
        recent_block = '\n'.join(recent_lines)

        # ── trainer section ────────────────────────────────────────────────
        if tr.nan_detected:
            trainer_note = '**WARNING: NaN detected in trainer log!**'
        elif tr.last_loss is not None:
            trainer_note = f'last_loss = {tr.last_loss:.6f}'
        else:
            trainer_note = '(no loss values seen yet)'

        # ── live store section ─────────────────────────────────────────────
        ls_groups = str(ls.num_groups) if ls.num_groups is not None else 'N/A'
        ls_pushes = str(ls.total_pushes) if ls.total_pushes is not None else 'N/A'

        md = f"""# Training Progress — {ts_str}

## Status: {snap.overall_status}  <!-- HEALTHY | DEGRADED | RESCUED | STALLED -->

| Field | Value |
|---|---|
| Policy version | {m.policy_version} |
| Groups pushed | {m.groups_pushed} |
| Groups filtered | {m.groups_filtered} ({m.filter_rate_pct:.1f}%) |
| Throughput | {m.throughput_groups_per_hr:.1f} groups/hr |
| Last push | {last_push_str} |
| Stall risk | {stall_risk_str} |
| Trainer | {trainer_note} |
| LiveStore groups | {ls_groups} |
| LiveStore total pushes | {ls_pushes} |

## Boundary Condition Checks

| BC | Description | Status |
|---|---|---|
| BC-0 | All siblings same policy version | {bc0_status} |
| BC-5 | No-progress detector (stall <300s) | {bc5_status} |
| BC-9 | Weight sync (endpoints_failed=0) | {bc9_status} |
| BC-16 | Trainer started after worker warmup | {bc16_status} |

## Service Health

| Service | Status | Last checked |
|---|---|---|
| LiveStore (UDS) | {live_store_status} | {now_hms} |
| PolicyRegistry (UDS) | {policy_reg_status} | {now_hms} |
| EnvironmentProvider (:8006) | {ep_icon} | {now_hms} |
{vllm_rows}
## Weight Sync Log (last {MAX_RECENT_WEIGHT_EVENTS})
{weight_log_lines}

## Auto-rescue Log (last {MAX_RECENT_RESCUE_EVENTS})
{rescue_log_lines}

## Recent Events
{recent_block}
"""
        return md


def _bc_status(ok: bool) -> str:
    return '✓ OK' if ok else '✗ FAIL'


def _service_status(healthy: bool) -> str:
    return '✓ Healthy' if healthy else '✗ Unhealthy'


def _http_status(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


# ── TrainingMonitor (orchestrator) ───────────────────────────────────────────


class TrainingMonitor:
    """
    Central orchestrator: collects snapshots from all sub-agents, optionally
    runs rescue on unhealthy services, then writes progress.md.
    """

    def __init__(
        self,
        *,
        interval_s: int = DEFAULT_INTERVAL_S,
        no_rescue: bool = False,
        progress_file: Path | None = None,
        remote_dns: str = REMOTE_DNS,
    ) -> None:
        self._interval_s = interval_s
        self._no_rescue = no_rescue
        self._progress_file = progress_file or (
            REPO_ROOT / 'current_training_progress.md'
        )
        self._remote_dns = remote_dns

        self._metrics_agent = MetricsAgent()
        self._weight_agent = WeightSyncAgent()
        self._rollout_agent = RolloutAgent(remote_dns=remote_dns)
        self._live_store_agent = LiveStoreAgent()
        self._trainer_agent = TrainerAgent()
        self._progress_writer = ProgressWriter(self._progress_file)

        # Lazy-import RescueCoordinator to avoid hard failure if rescue_team
        # has missing optional deps on this machine.
        self._rescue_coordinator: Any | None = None
        if not no_rescue:
            self._rescue_coordinator = self._load_rescue_coordinator()

        self._rescue_log: deque[str] = deque(maxlen=100)

    @staticmethod
    def _load_rescue_coordinator() -> Any | None:
        try:
            from rescue_team import RescueCoordinator  # noqa: PLC0415

            logger.info('RescueCoordinator loaded from rescue_team')
            return RescueCoordinator()
        except ImportError as exc:
            logger.warning('rescue_team not importable: %s — auto-rescue disabled', exc)
            return None

    def run_once(self) -> MonitorSnapshot:
        """Collect all metrics, optionally rescue, write progress. Return snapshot."""
        now = _tz_now()
        logger.info('--- monitor tick %s ---', _fmt_ts(now))

        metrics = self._metrics_agent.collect()
        weight_sync = self._weight_agent.collect()
        rollout = self._rollout_agent.collect()
        live_store = self._live_store_agent.collect()
        trainer = self._trainer_agent.collect()

        # Determine overall status before rescue.
        overall_status = self._determine_status(
            metrics, weight_sync, rollout, live_store, trainer
        )

        # Auto-rescue: attempt to fix unhealthy services.
        if not self._no_rescue and self._rescue_coordinator is not None:
            overall_status = self._run_rescue(
                overall_status, rollout, live_store, weight_sync, trainer
            )

        snap = MonitorSnapshot(
            timestamp=now,
            metrics=metrics,
            weight_sync=weight_sync,
            rollout=rollout,
            live_store=live_store,
            trainer=trainer,
            overall_status=overall_status,
            rescue_log=list(self._rescue_log)[-MAX_RECENT_RESCUE_EVENTS:],
        )

        self._progress_writer.write(snap)
        self._log_summary(snap)
        return snap

    def _determine_status(
        self,
        metrics: MetricsSnapshot,
        weight_sync: WeightSyncSnapshot,
        rollout: RolloutSnapshot,
        live_store: LiveStoreSnapshot,
        trainer: TrainerSnapshot,
    ) -> str:
        if metrics.stall_risk:
            return 'STALLED'
        if trainer.nan_detected:
            return 'DEGRADED'
        if not weight_sync.bc9_ok:
            return 'DEGRADED'
        if not rollout.all_healthy:
            return 'DEGRADED'
        if not live_store.socket_exists:
            return 'DEGRADED'
        return 'HEALTHY'

    def _run_rescue(
        self,
        current_status: str,
        rollout: RolloutSnapshot,
        live_store: LiveStoreSnapshot,
        weight_sync: WeightSyncSnapshot,
        trainer: TrainerSnapshot,
    ) -> str:
        coord = self._rescue_coordinator
        rescued_any = False

        def _attempt_rescue(service_name: str, reason: str) -> None:
            nonlocal rescued_any
            logger.warning('Auto-rescuing %s: %s', service_name, reason)
            try:
                report = coord.rescue(service_name)
                ts = _fmt_time(_tz_now())
                outcome = 'resolved' if report.resolved else 'unresolved'
                entry = f'{ts} — rescue {service_name} ({reason}): {outcome}'
                self._rescue_log.append(entry)
                logger.info('[rescue] %s', entry)
                if report.resolved:
                    rescued_any = True
            except Exception as exc:  # noqa: BLE001
                logger.error('rescue(%s) raised: %s', service_name, exc)
                self._rescue_log.append(
                    f'{_fmt_time(_tz_now())} — rescue {service_name} EXCEPTION: {exc}'
                )

        # Check each condition and attempt rescue.
        if not live_store.socket_exists:
            _attempt_rescue('live_store', 'socket missing')

        if not Path(POLICY_REGISTRY_SOCKET).exists():
            _attempt_rescue('policy_registry', 'socket missing')

        if not _http_status(ENV_PROVIDER_URL):
            _attempt_rescue('env_provider', '/status check failed')

        if not rollout.all_healthy:
            failed_ports = [
                str(ps.port) for ps in rollout.port_statuses if not ps.healthy
            ]
            logger.warning('vLLM pool ports down: %s', ', '.join(failed_ports))
            _attempt_rescue('vllm_pool', f'ports down: {", ".join(failed_ports)}')

        if not weight_sync.bc9_ok:
            logger.error('BC-9 violation detected — endpoints_failed > 0')
            # BC-9 is a hard abort; log but do not auto-restart policy_registry
            # blindly — the trainer is expected to abort. Just record it.
            self._rescue_log.append(
                f'{_fmt_time(_tz_now())} — BC-9 VIOLATION: endpoints_failed > 0'
                ' (trainer should have aborted; check trainer + vLLM logs)'
            )

        if rescued_any and current_status != 'HEALTHY':
            return 'RESCUED'
        return current_status

    def _log_summary(self, snap: MonitorSnapshot) -> None:
        m = snap.metrics
        logger.info(
            'status=%s pv=%d pushed=%d filtered=%d(%.1f%%) stall=%s nan=%s vllm=%s',
            snap.overall_status,
            m.policy_version,
            m.groups_pushed,
            m.groups_filtered,
            m.filter_rate_pct,
            m.stall_risk,
            snap.trainer.nan_detected,
            'OK' if snap.rollout.all_healthy else 'DEGRADED',
        )

    def run_loop(self) -> None:
        """Run indefinitely until Ctrl-C."""
        logger.info(
            'Starting monitor loop (interval=%ds, no_rescue=%s, output=%s)',
            self._interval_s,
            self._no_rescue,
            self._progress_file,
        )
        import signal  # noqa: PLC0415

        _running = [True]

        def _stop(signum: int, frame: Any) -> None:
            logger.info('Caught signal %d — stopping monitor loop', signum)
            _running[0] = False

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        while _running[0]:
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception('Monitor tick raised unexpectedly: %s', exc)
            if not _running[0]:
                break
            # Sleep in small increments so SIGINT is responsive.
            deadline = time.monotonic() + self._interval_s
            while _running[0] and time.monotonic() < deadline:
                time.sleep(1.0)

        logger.info('Monitor loop exited.')


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Training Monitor — autonomous health watcher + progress reporter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--interval',
        type=int,
        default=DEFAULT_INTERVAL_S,
        metavar='N',
        help='Poll interval in seconds (ignored with --once)',
    )
    p.add_argument(
        '--once',
        action='store_true',
        help='Run a single monitoring pass then exit',
    )
    p.add_argument(
        '--no-rescue',
        action='store_true',
        help='Metrics-only mode — skip auto-rescue',
    )
    p.add_argument(
        '--progress-file',
        type=Path,
        default=None,
        metavar='PATH',
        help=(
            f'Output file for progress markdown '
            f'(default: {REPO_ROOT / "current_training_progress.md"})'
        ),
    )
    p.add_argument(
        '--remote-dns',
        default=REMOTE_DNS,
        metavar='HOST',
        help='Hostname/IP for vLLM pool health checks',
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    progress_file: Path | None = args.progress_file

    monitor = TrainingMonitor(
        interval_s=args.interval,
        no_rescue=args.no_rescue,
        progress_file=progress_file,
        remote_dns=args.remote_dns,
    )

    if args.once:
        snap = monitor.run_once()
        return 0 if snap.overall_status in ('HEALTHY', 'RESCUED') else 1

    monitor.run_loop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
