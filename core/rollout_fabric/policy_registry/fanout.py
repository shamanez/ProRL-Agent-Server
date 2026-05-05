"""§3.3 abort-gate fanout — POST /reload_lora to every pool child.

Lifted from ``ray_trainer.py:_publish_lora_adapter``. The contract is
load-bearing: ``success`` implies ``endpoints_failed == 0``. Partial
publish = failure, not degraded mode (BC-9).

S4 moves this from the trainer process into the PolicyRegistry service.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

from rollout_fabric.schemas.protocols.policy_registry import PublishResult

logger = logging.getLogger(__name__)

REQUIRED_ADAPTER_FILES = ('adapter_model.safetensors', 'adapter_config.json')


class FanoutError(RuntimeError):
    pass


def fanout_to_pool(
    *,
    adapter_uri: str,
    new_version: int,
    endpoints: list[str],
    timeout_s: int = 60,
) -> PublishResult:
    """POST adapter tarball to every pool child; abort on partial failure (BC-9)."""
    if not endpoints:
        raise FanoutError('fanout_to_pool: endpoints list is empty')
    started = time.monotonic()
    payload = _read_adapter_tarball(adapter_uri)

    def _post(endpoint: str) -> dict:
        url = endpoint.rstrip('/') + '/reload_lora'
        t0 = time.monotonic()
        try:
            resp = requests.post(
                url,
                files={'adapter': ('adapter.tgz', payload, 'application/gzip')},
                data={'policy_version': str(new_version)},
                timeout=timeout_s,
            )
            try:
                body = resp.json()
            except ValueError:
                body = {'detail': resp.text[:512]}
            return {
                'endpoint': endpoint,
                'status': resp.status_code,
                'wall_s': time.monotonic() - t0,
                'body': body,
            }
        except requests.RequestException as exc:
            return {
                'endpoint': endpoint,
                'status': -1,
                'wall_s': time.monotonic() - t0,
                'body': {'detail': f'{type(exc).__name__}: {exc}'},
            }

    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        responses = list(pool.map(_post, endpoints))

    ok = [r for r in responses if r['status'] in (200, 409)]
    failed = [r for r in responses if r['status'] not in (200, 409)]
    elapsed = time.monotonic() - started

    if failed:
        logger.error(
            'pool fanout FAILED pv=%d ok=%d/%d failed=%s',
            new_version,
            len(ok),
            len(endpoints),
            [{k: v for k, v in r.items() if k != 'body'} for r in failed],
        )
        return PublishResult(
            success=False,
            endpoints_ok=len(ok),
            endpoints_failed=len(failed),
            latency_s=elapsed,
            error=json.dumps(failed)[:1024],
        )
    logger.info(
        'pool fanout OK pv=%d endpoints_ok=%d wall_s=%.3f',
        new_version,
        len(ok),
        elapsed,
    )
    return PublishResult(
        success=True,
        endpoints_ok=len(ok),
        endpoints_failed=0,
        latency_s=elapsed,
        error=None,
    )


def _read_adapter_tarball(adapter_uri: str) -> bytes:
    parsed = urlparse(adapter_uri)
    if parsed.scheme not in ('', 'file'):
        raise FanoutError(f'unsupported adapter URI scheme: {adapter_uri!r}')
    adapter_dir = Path(parsed.path)
    missing = [f for f in REQUIRED_ADAPTER_FILES if not (adapter_dir / f).is_file()]
    if missing:
        raise FanoutError(f'adapter dir {adapter_dir} missing files: {missing}')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for fname in REQUIRED_ADAPTER_FILES:
            tar.add(adapter_dir / fname, arcname=fname)
    return buf.getvalue()
