"""Rescue Agent Team — startup and runtime failure handler.

Four specialist roles that work in a coordinator-driven loop:

  Coordinator
      │
      ├─► ProbeAgent    — health-checks all services, returns structured report
      ├─► LogAgent      — reads logs, extracts error signatures and root causes
      └─► FixAgent      — applies targeted remedies from a decision table

Protocol (per failure event):
  1. Coordinator detects unhealthy service (via ProbeAgent output or direct call).
  2. Coordinator sends service name + last log tail to LogAgent.
  3. LogAgent returns {error_class, root_cause, suggested_fix}.
  4. Coordinator calls FixAgent with the diagnosis.
  5. FixAgent applies the fix and reports outcome.
  6. Coordinator waits 5 s then re-runs ProbeAgent.
  7. If still failing after MAX_RETRIES → escalate (log to file, exit non-zero).

This module is both the team definition AND a runnable rescue CLI:
    python scripts/services/rescue_team.py --check
    python scripts/services/rescue_team.py --watch   # continuous loop
    python scripts/services/rescue_team.py --rescue <service>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger('rescue_team')

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_RETRIES = 3
WATCH_INTERVAL_S = 15.0

# ── service registry ────────────────────────────────────────────────────────

SERVICES: list[dict[str, Any]] = [
    {
        'name': 'vllm_pool',
        'health_urls': [
            f'http://{os.environ.get("REMOTE_DNS", "vllm-instance")}:{p}/health'
            for p in [8100, 8101, 8102, 8103]
        ],
        'log_path': '/tmp/vllm-child-8100.log',
        'restart_cmd': f'bash {REPO_ROOT}/scripts/serving/launch_remote_vllm_pool.sh start',
    },
    {
        'name': 'env_provider',
        'health_urls': ['http://localhost:8006/status'],
        'log_path': '/tmp/s0-prorl.log',
        'restart_cmd': f'bash {REPO_ROOT}/scripts/services/start_env_provider.sh',
    },
    {
        'name': 'live_store',
        'socket_path': '/tmp/prorl_live_store.sock',
        'log_path': '/tmp/live_store.log',
        'restart_cmd': (
            f'bash {REPO_ROOT}/scripts/services/start_live_store.sh '
            f'> /tmp/live_store.log 2>&1 &'
        ),
    },
    {
        'name': 'policy_registry',
        'socket_path': '/tmp/prorl_policy_registry.sock',
        'log_path': '/tmp/policy_registry.log',
        'restart_cmd': (
            f'bash {REPO_ROOT}/scripts/services/start_policy_registry.sh '
            f'> /tmp/policy_registry.log 2>&1 &'
        ),
    },
    {
        'name': 'rollout_worker',
        'log_path': '/tmp/rollout_worker.log',
        'pid_file': '/tmp/rollout_worker.pid',
        'health_fn': '_check_worker_producing',
        'restart_cmd': (
            f"DATA_FILES='{os.environ.get('DATA_FILES', '/home/ubuntu/data/SkyRL-v0-293/train.parquet')}' "
            f'bash {REPO_ROOT}/scripts/services/start_rollout_worker.sh '
            f'> /tmp/rollout_worker.log 2>&1 &'
        ),
    },
]

# ── data types ──────────────────────────────────────────────────────────────


@dataclass
class ServiceHealth:
    name: str
    healthy: bool
    detail: str = ''
    log_tail: str = ''


@dataclass
class Diagnosis:
    error_class: str  # e.g. "import_error", "port_conflict", "oom"
    root_cause: str
    suggested_fix: str
    confidence: float = 0.8


@dataclass
class FixResult:
    applied: bool
    action: str
    outcome: str


@dataclass
class RescueReport:
    timestamp: str
    service: str
    health: ServiceHealth
    diagnosis: Diagnosis | None = None
    fix_result: FixResult | None = None
    resolved: bool = False


# ── ProbeAgent ──────────────────────────────────────────────────────────────


class ProbeAgent:
    """Checks health of all services. Returns structured report."""

    def check_all(self) -> list[ServiceHealth]:
        return [self._check(svc) for svc in SERVICES]

    def check_one(self, name: str) -> ServiceHealth:
        for svc in SERVICES:
            if svc['name'] == name:
                return self._check(svc)
        return ServiceHealth(name=name, healthy=False, detail='unknown service')

    def _check(self, svc: dict) -> ServiceHealth:
        name = svc['name']
        # HTTP health endpoints
        if 'health_urls' in svc:
            import urllib.request  # noqa: PLC0415

            failures = []
            for url in svc['health_urls']:
                try:
                    r = urllib.request.urlopen(url, timeout=3)
                    if r.status != 200:
                        failures.append(f'{url} → {r.status}')
                except Exception as exc:  # noqa: BLE001
                    failures.append(f'{url} → {type(exc).__name__}: {exc}')
            healthy = len(failures) == 0
            return ServiceHealth(
                name=name,
                healthy=healthy,
                detail='' if healthy else '; '.join(failures),
                log_tail=self._tail(svc.get('log_path')),
            )
        # UDS socket
        if 'socket_path' in svc:
            path = svc['socket_path']
            exists = Path(path).exists()
            return ServiceHealth(
                name=name,
                healthy=exists,
                detail='' if exists else f'socket not found: {path}',
                log_tail=self._tail(svc.get('log_path')),
            )
        # Custom health fn
        if 'health_fn' in svc:
            fn = getattr(self, svc['health_fn'], None)
            if fn:
                return fn(svc)
        return ServiceHealth(name=name, healthy=False, detail='no health check defined')

    def _check_worker_producing(self, svc: dict) -> ServiceHealth:
        """Worker is healthy if it has pushed ≥1 group to the live store."""
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from live_store.client import LiveStoreClient  # noqa: PLC0415

            cli = LiveStoreClient(
                '/tmp/prorl_live_store.sock',
                policy_id=os.environ.get('POLICY_ID', 'qwen3-4b-skyrl'),
                environment_id=os.environ.get('ENVIRONMENT_ID', 'prorl_default'),
            )
            n = cli.num_groups()
            pushes = cli.total_pushes()
            cli.close()
            healthy = pushes > 0
            return ServiceHealth(
                name=svc['name'],
                healthy=healthy,
                detail=f'store groups={n} total_pushes={pushes}',
                log_tail=self._tail(svc.get('log_path')),
            )
        except Exception as exc:  # noqa: BLE001
            return ServiceHealth(
                name=svc['name'],
                healthy=False,
                detail=f'LiveStore unreachable: {exc}',
                log_tail=self._tail(svc.get('log_path')),
            )

    @staticmethod
    def _tail(path: str | None, lines: int = 30) -> str:
        if not path or not Path(path).exists():
            return ''
        try:
            result = subprocess.run(
                ['tail', '-n', str(lines), path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout
        except Exception:  # noqa: BLE001
            return ''


# ── LogAgent ────────────────────────────────────────────────────────────────


class LogAgent:
    """Reads logs and returns structured diagnosis."""

    # Error signature → (error_class, root_cause, fix)
    SIGNATURES: list[tuple[str, str, str, str]] = [
        (
            'ModuleNotFoundError',
            'import_error',
            'Python dependency missing from poetry venv',
            'poetry add <missing_package>',
        ),
        (
            'Address already in use',
            'port_conflict',
            'Another process is bound to the required port',
            'kill -9 $(lsof -ti:<port>) or change port',
        ),
        (
            'Connection refused',
            'service_not_ready',
            'Upstream service not yet accepting connections',
            'Wait for upstream health check to pass, or restart upstream',
        ),
        (
            'CUDA out of memory',
            'oom',
            'GPU memory exhausted — too many vLLM workers or batch too large',
            'Reduce GPU_MEM_UTIL or max_model_len in vLLM child config',
        ),
        (
            'FileNotFoundError',
            'missing_file',
            'Required file (adapter, parquet, config) not found',
            'Check DATA_FILES path and adapter_uri in policy registry',
        ),
        (
            'Permission denied',
            'permissions',
            'File or socket permission error',
            'chmod o+rw on the socket path, or run as the correct user',
        ),
        (
            'grpc._channel._InactiveRpcError',
            'grpc_dead',
            'gRPC channel broken — service crashed or socket stale',
            'Remove stale socket and restart the service',
        ),
        (
            'NaN',
            'nan_loss',
            'Loss went NaN — likely token mismatch, LR too high, or bad checkpoint',
            'Check behavior_policy_version alignment; reduce LR; restart from last checkpoint',
        ),
        (
            'endpoints_failed',
            'pool_publish_fail',
            'vLLM pool child rejected /reload_lora — partial publish (BC-9)',
            'Check vLLM child logs; restart the failing child; re-publish',
        ),
        (
            'no_progress',
            'producer_wedged',
            'Producer stopped pushing to live store — vLLM or ProRL may be down',
            'Check ProRL and vLLM health; restart rollout worker',
        ),
    ]

    def diagnose(self, health: ServiceHealth) -> Diagnosis:
        combined = f'{health.detail}\n{health.log_tail}'
        for sig, err_class, root_cause, fix in self.SIGNATURES:
            if sig.lower() in combined.lower():
                return Diagnosis(
                    error_class=err_class,
                    root_cause=root_cause,
                    suggested_fix=fix,
                    confidence=0.85,
                )
        return Diagnosis(
            error_class='unknown',
            root_cause=f'Unrecognized failure in {health.name}: {health.detail[:120]}',
            suggested_fix='Inspect log manually: '
            + (health.log_tail[:200] or 'no log'),
            confidence=0.3,
        )


# ── FixAgent ─────────────────────────────────────────────────────────────────


class FixAgent:
    """Applies targeted remedies. Each fix is idempotent where possible."""

    def fix(self, service_name: str, diagnosis: Diagnosis) -> FixResult:
        method = f'_fix_{diagnosis.error_class}'
        fn = getattr(self, method, self._fix_generic)
        return fn(service_name, diagnosis)

    def _fix_generic(self, service_name: str, diagnosis: Diagnosis) -> FixResult:
        """Fallback: restart the service."""
        for svc in SERVICES:
            if svc['name'] == service_name:
                return self._restart_service(svc, diagnosis.suggested_fix)
        return FixResult(
            applied=False,
            action='no_restart_cmd',
            outcome=f'No restart command found for {service_name}',
        )

    def _fix_service_not_ready(
        self, service_name: str, diagnosis: Diagnosis
    ) -> FixResult:
        return FixResult(
            applied=True,
            action='wait_5s',
            outcome='Waiting 5s for upstream to become ready',
        )

    def _fix_grpc_dead(self, service_name: str, diagnosis: Diagnosis) -> FixResult:
        # Clean stale socket then restart
        for svc in SERVICES:
            if svc['name'] == service_name and 'socket_path' in svc:
                sock = svc['socket_path']
                try:
                    Path(sock).unlink(missing_ok=True)
                    logger.info('Removed stale socket %s', sock)
                except Exception as exc:  # noqa: BLE001
                    logger.warning('Could not remove socket %s: %s', sock, exc)
                return self._restart_service(svc, 'stale socket removed')
        return self._fix_generic(service_name, diagnosis)

    def _fix_port_conflict(self, service_name: str, diagnosis: Diagnosis) -> FixResult:
        return FixResult(
            applied=False,
            action='manual_required',
            outcome='Port conflict requires manual intervention — '
            'identify and kill the conflicting process',
        )

    def _fix_nan_loss(self, service_name: str, diagnosis: Diagnosis) -> FixResult:
        return FixResult(
            applied=False,
            action='manual_required',
            outcome='NaN loss requires human inspection of the model + LR',
        )

    @staticmethod
    def _restart_service(svc: dict, reason: str) -> FixResult:
        cmd = svc.get('restart_cmd', '')
        if not cmd:
            return FixResult(
                applied=False,
                action='no_cmd',
                outcome=f'No restart command for {svc["name"]}',
            )
        try:
            subprocess.Popen(cmd, shell=True, start_new_session=True)
            logger.info('Restarted %s: %s', svc['name'], reason)
            return FixResult(
                applied=True,
                action=f'restart:{svc["name"]}',
                outcome=f'Restart issued. Reason: {reason}',
            )
        except Exception as exc:  # noqa: BLE001
            return FixResult(applied=False, action='restart_failed', outcome=str(exc))


# ── Coordinator ──────────────────────────────────────────────────────────────


class RescueCoordinator:
    """Orchestrates the probe → diagnose → fix → verify loop."""

    def __init__(self) -> None:
        self.probe = ProbeAgent()
        self.log = LogAgent()
        self.fix = FixAgent()
        self.reports: list[RescueReport] = []

    def check_all(self) -> list[ServiceHealth]:
        return self.probe.check_all()

    def rescue(self, service_name: str) -> RescueReport:
        import datetime  # noqa: PLC0415

        ts = datetime.datetime.utcnow().isoformat() + 'Z'
        health = self.probe.check_one(service_name)
        report = RescueReport(timestamp=ts, service=service_name, health=health)

        if health.healthy:
            report.resolved = True
            logger.info('[rescue] %s is already healthy', service_name)
            return report

        logger.warning('[rescue] %s is unhealthy: %s', service_name, health.detail)
        diagnosis = self.log.diagnose(health)
        report.diagnosis = diagnosis
        logger.info(
            '[rescue] diagnosis: %s — %s', diagnosis.error_class, diagnosis.root_cause
        )
        logger.info('[rescue] suggested fix: %s', diagnosis.suggested_fix)

        for attempt in range(1, MAX_RETRIES + 1):
            fix_result = self.fix.fix(service_name, diagnosis)
            report.fix_result = fix_result
            logger.info(
                '[rescue] fix attempt %d: %s → %s',
                attempt,
                fix_result.action,
                fix_result.outcome,
            )

            if not fix_result.applied:
                logger.error(
                    '[rescue] fix could not be applied — manual intervention needed'
                )
                break

            time.sleep(5.0)
            re_health = self.probe.check_one(service_name)
            if re_health.healthy:
                report.resolved = True
                logger.info(
                    '[rescue] %s recovered after attempt %d', service_name, attempt
                )
                break
            logger.warning(
                '[rescue] %s still unhealthy after fix attempt %d',
                service_name,
                attempt,
            )
            diagnosis = self.log.diagnose(re_health)

        self.reports.append(report)
        return report

    def watch(self, interval_s: float = WATCH_INTERVAL_S) -> None:
        """Continuous watch loop — runs until Ctrl-C."""
        logger.info('[watch] monitoring all services every %.0fs', interval_s)
        while True:
            healths = self.probe.check_all()
            for h in healths:
                icon = '✓' if h.healthy else '✗'
                logger.info('[watch] %s %s %s', icon, h.name, h.detail or 'OK')
                if not h.healthy:
                    self.rescue(h.name)
            time.sleep(interval_s)

    def summary_json(self) -> str:
        return json.dumps([asdict(r) for r in self.reports], indent=2, default=str)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s rescue_team: %(message)s',
    )
    p = argparse.ArgumentParser(description='Rollout Fabric Rescue Agent Team')
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        '--check', action='store_true', help='One-shot health check of all services'
    )
    grp.add_argument(
        '--watch', action='store_true', help='Continuous watch + auto-rescue loop'
    )
    grp.add_argument(
        '--rescue', metavar='SERVICE', help='Rescue a specific service by name'
    )
    args = p.parse_args()

    coordinator = RescueCoordinator()
    sys.path.insert(0, str(REPO_ROOT))

    if args.check:
        healths = coordinator.check_all()
        all_ok = True
        for h in healths:
            icon = '✓' if h.healthy else '✗'
            print(f'  {icon}  {h.name:20s}  {h.detail or "OK"}')
            if not h.healthy:
                all_ok = False
        return 0 if all_ok else 1

    if args.watch:
        coordinator.watch()
        return 0

    if args.rescue:
        report = coordinator.rescue(args.rescue)
        print(json.dumps(asdict(report), indent=2, default=str))
        return 0 if report.resolved else 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
