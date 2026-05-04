"""S2 — file-backed minimal PolicyRegistry.

Trainer writes a JSON manifest atomically after every successful pool ACK.
Worker polls the manifest via mtime at 1 Hz and feeds updates into its
``PolicyVersionCache`` via atomic ref-swap.

§3.3 abort gate: manifest written ONLY after all pool children ACK
``/reload_lora``. Worker will not observe a failed publish (BC-9).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = '/tmp/prorl_policy_manifest.json'


@dataclass(slots=True, frozen=True)
class PolicyManifest:
    policy_id: str
    version: int
    adapter_uri: str
    trainer_id: str
    published_at: float  # POSIX timestamp


def write_manifest(
    manifest: PolicyManifest,
    *,
    path: str = DEFAULT_MANIFEST_PATH,
) -> None:
    """Atomic-rename write so a concurrent reader never sees a partial file."""
    parent = os.path.dirname(path) or '.'
    os.makedirs(parent, exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(asdict(manifest), fh, sort_keys=True)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    logger.info(
        'policy manifest written: policy_id=%s version=%d uri=%s',
        manifest.policy_id,
        manifest.version,
        manifest.adapter_uri,
    )


def read_manifest(path: str = DEFAULT_MANIFEST_PATH) -> PolicyManifest | None:
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    return PolicyManifest(
        policy_id=str(data['policy_id']),
        version=int(data['version']),
        adapter_uri=str(data['adapter_uri']),
        trainer_id=str(data['trainer_id']),
        published_at=float(data.get('published_at', time.time())),
    )
