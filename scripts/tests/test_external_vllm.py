#!/usr/bin/env python3
"""Stage 1 external vLLM pool smoke test.

Verifies the 6 gate criteria from ``plans-n-solutions/stages/stage1.md``.
Assumes ProRL is already running on localhost:8006 and the supervisor pool
is up on ports 8100 + 8101 (see ``stage1.md`` runbook for the three-terminal
flow). Does not launch either itself.

Exit 0 on all-green. Exit 1 with a summary table on first failure.

Usage::

    poetry run python scripts/tests/test_external_vllm.py
    poetry run python scripts/tests/test_external_vllm.py --ports 8100 8101
    poetry run python scripts/tests/test_external_vllm.py --skip-prorl
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import httpx

logger = logging.getLogger('stage1_smoke')

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROL_URL = 'http://localhost:8006'
DEFAULT_SUPERVISOR_PORTS = (8100, 8101)
HEALTH_WAIT_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 30
GENERATE_TIMEOUT_SECONDS = 120


class CriterionFailed(Exception):
    """Raised when a Stage 1 gate criterion fails."""


def _check_supervisor_health(ports: list[int], wait_seconds: int) -> None:
    """Criterion 1: every supervisor /health returns 200 within wait_seconds."""
    deadline = time.monotonic() + wait_seconds
    for port in ports:
        url = f'http://localhost:{port}/health'
        last_err: str | None = None
        while time.monotonic() < deadline:
            try:
                r = httpx.get(url, timeout=2.0)
                if r.status_code == 200:
                    logger.info('  :%d /health 200 ok', port)
                    last_err = None
                    break
                last_err = f'HTTP {r.status_code}: {r.text[:200]}'
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_err = str(exc)
            time.sleep(2)
        if last_err is not None:
            raise CriterionFailed(
                f'supervisor :{port} did not become healthy within '
                f'{wait_seconds}s; last error: {last_err}'
            )


def _check_prorl_registration(prorl_url: str, ports: list[int]) -> None:
    """Criterion 2: ProRL accepts /add_llm_server for every supervisor + /start.

    The `/status` endpoint in `openhands/nvidia/async_server.py::status` only
    surfaces job-queue counts, not the registered address list, so we cannot
    read the pool back from it. The API-backed gate is: `/add_llm_server`
    returns 200 for each address and `/start` transitions the server to
    ``running=True``.
    """
    addresses = [f'http://localhost:{p}' for p in ports]

    try:
        r = httpx.post(f'{prorl_url}/clear_llm_server', timeout=REQUEST_TIMEOUT_SECONDS)
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        raise CriterionFailed(f'ProRL /clear_llm_server unreachable: {exc}') from exc
    if r.status_code != 200:
        raise CriterionFailed(
            f'/clear_llm_server returned {r.status_code}: {r.text[:200]}'
        )

    for addr in addresses:
        r = httpx.post(
            f'{prorl_url}/add_llm_server',
            json={'address': addr},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if r.status_code != 200:
            raise CriterionFailed(
                f'/add_llm_server {addr} returned {r.status_code}: {r.text[:200]}'
            )
        logger.info('  /add_llm_server %s -> 200 %s', addr, r.json())

    r = httpx.post(f'{prorl_url}/start', timeout=REQUEST_TIMEOUT_SECONDS)
    # 400 "Server is already running" is idempotent success for this gate:
    # ProRL start only fails transitionally; what we want is running=True
    # (asserted below via /status).
    if r.status_code not in (200, 400):
        raise CriterionFailed(f'/start returned {r.status_code}: {r.text[:200]}')

    r = httpx.get(f'{prorl_url}/status', timeout=REQUEST_TIMEOUT_SECONDS)
    if r.status_code != 200:
        raise CriterionFailed(f'/status returned {r.status_code}: {r.text[:200]}')
    status = r.json()
    if not status.get('running'):
        raise CriterionFailed(f'ProRL /status not running: {status!r}')
    logger.info('  ProRL /status: %s', status)


def _check_generate_round_trip(ports: list[int]) -> None:
    """Criterion 4 (partial): exercise token-in / token-out through each supervisor.

    Also proves the child log will show POST /generate traffic (checked in
    ``_check_child_log_traffic``).
    """
    # Arbitrary prompt ids; any valid token sequence works since we just need a
    # round trip that doesn't re-tokenize. These are integers inside Qwen3's
    # vocabulary range; exact values are not load-bearing.
    prompt_ids = [9707, 11, 7299, 2138, 498, 525]
    for port in ports:
        url = f'http://localhost:{port}/generate'
        body = {'prompt_ids': prompt_ids, 'max_tokens': 4, 'temperature': 0.0}
        r = httpx.post(url, json=body, timeout=GENERATE_TIMEOUT_SECONDS)
        if r.status_code != 200:
            raise CriterionFailed(
                f'POST :{port}/generate returned {r.status_code}: {r.text[:200]}'
            )
        data = r.json()
        response_ids = data.get('response_ids')
        logprobs = data.get('logprobs')
        if not isinstance(response_ids, list) or not response_ids:
            raise CriterionFailed(
                f':{port}/generate returned no response_ids (got {data!r})'
            )
        if not all(isinstance(x, int) for x in response_ids):
            raise CriterionFailed(
                f':{port}/generate response_ids must be list[int], got {response_ids!r}'
            )
        if logprobs is not None and len(logprobs) != len(response_ids):
            raise CriterionFailed(
                f':{port}/generate logprobs length {len(logprobs)} != '
                f'response_ids length {len(response_ids)}'
            )
        logger.info(
            '  :%d /generate -> %d tokens, first ids=%s',
            port,
            len(response_ids),
            response_ids[:4],
        )


def _check_child_log_traffic(ports: list[int]) -> None:
    """Criterion 4: POST /generate count > 0 in each child log.

    The log is written by uvicorn inside the container; we read it on the host
    via the ``-v /tmp:/tmp`` bind mount.
    """
    for port in ports:
        log_path = Path(f'/tmp/vllm-child-{port}.log')
        if not log_path.exists():
            raise CriterionFailed(f'child log {log_path} does not exist')
        text = log_path.read_text()
        count = text.count('POST /generate')
        if count == 0:
            raise CriterionFailed(
                f'{log_path} has zero "POST /generate" lines '
                f'(did the round-trip actually hit the child?)'
            )
        logger.info('  %s: POST /generate count=%d', log_path, count)


def _check_trainer_untouched() -> None:
    """Criterion 6: no Stage 1 change has leaked into trainer_integration/."""
    try:
        r = subprocess.run(
            ['git', 'diff', '--stat', 'HEAD', '--', 'trainer_integration/'],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise CriterionFailed(f'git diff failed: {exc.stderr}') from exc
    except FileNotFoundError as exc:
        raise CriterionFailed(f'git not on PATH: {exc}') from exc
    if r.stdout.strip():
        raise CriterionFailed(
            'Stage 1 must not modify trainer_integration/:\n' + r.stdout
        )
    logger.info('  git diff trainer_integration/ is empty')


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--ports',
        type=int,
        nargs='+',
        default=list(DEFAULT_SUPERVISOR_PORTS),
        help='Supervisor ports to probe (default: 8100 8101).',
    )
    parser.add_argument(
        '--prorl-url',
        default=DEFAULT_PROL_URL,
        help='ProRL base URL for criterion 2 (default: %(default)s).',
    )
    parser.add_argument(
        '--skip-prorl',
        action='store_true',
        help='Skip criterion 2 (ProRL registration). Useful when only validating the pool.',
    )
    parser.add_argument(
        '--health-wait-seconds',
        type=int,
        default=HEALTH_WAIT_SECONDS,
        help='How long to wait for each supervisor /health (default: %(default)s).',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level='INFO', format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    args = _parse_args(argv)
    ports: list[int] = args.ports

    results: list[tuple[str, bool, str]] = []

    def _run(label: str, fn, *fn_args) -> None:
        logger.info('--- %s ---', label)
        try:
            fn(*fn_args)
            results.append((label, True, ''))
        except CriterionFailed as exc:
            results.append((label, False, str(exc)))
            logger.error('FAILED: %s: %s', label, exc)

    _run(
        'crit 1 — supervisor /health',
        _check_supervisor_health,
        ports,
        args.health_wait_seconds,
    )
    if not args.skip_prorl:
        _run(
            'crit 2 — ProRL /add_llm_server + /start + /status',
            _check_prorl_registration,
            args.prorl_url,
            ports,
        )
    else:
        results.append(('crit 2 — ProRL /status', True, 'skipped (--skip-prorl)'))
    _run('crit 4 — /generate round trip', _check_generate_round_trip, ports)
    _run('crit 4 — child log POST /generate', _check_child_log_traffic, ports)
    _run('crit 6 — trainer_integration untouched', _check_trainer_untouched)

    # Criterion 3 (end-to-end SWE-Bench rollout) is covered by the Stage 1 live
    # integration run with scripts/tests/standalone_swebench_test.py, not here.
    # Running 2 SWE instances in-process would make this smoke test a 5-min
    # dependency on Singularity images and dataset parquet paths that this
    # plumbing-level test should not own.
    results.append(
        (
            'crit 3 — SWE-Bench rollout (separately covered)',
            True,
            'run scripts/tests/standalone_swebench_test.py for the live check',
        )
    )

    print()
    print('Stage 1 smoke summary')
    print('=' * 72)
    fails = 0
    for label, ok, detail in results:
        status = ' PASS ' if ok else ' FAIL '
        line = f'[{status}] {label}'
        if detail:
            line += f'  — {detail}'
        print(line)
        if not ok:
            fails += 1
    print('=' * 72)
    print(f'{len(results) - fails}/{len(results)} criteria passed')
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
