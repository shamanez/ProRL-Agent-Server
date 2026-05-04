"""S2 — file-backed minimal PolicyRegistry.

The trainer publishes a new policy version with two artifacts:

1. The pool-fanout (POST ``/reload_lora`` to every child) — that stays
   inside the trainer at S2 and moves to ``policy_registry/fanout.py``
   at S4.
2. The JSON manifest at ``${POLICY_MANIFEST_PATH}``, written
   atomically with ``os.replace`` so a concurrent reader never
   observes a half-written file.

The worker reads the manifest via mtime-poll at 1 Hz (see
``rollout_worker/policy_subscription.py``) and feeds each strictly-
fresher version into its :class:`PolicyVersionCache` via
``cache.update(snapshot)``. The cache's atomic ref-swap publishes the
new snapshot to all reader threads with no lock on the read path.

§3.3 abort gate: trainer writes the manifest **only after** every pool
child ACKs. The worker, polling, will not observe a publish that
failed at any pool member. This preserves invariant 3.3 through S2;
S4 elevates it to the registry server's fanout method.
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
    """JSON-serializable manifest entry. One per ``policy_id``."""

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
    """Atomic-rename write of the policy manifest.

    The temp file is created in the same directory as the target so
    ``os.replace`` is genuinely atomic (POSIX rename within a single
    filesystem). A concurrent reader either sees the old file or the
    new one — never a partial write.
    """
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
        'policy manifest written: policy_id=%s version=%d uri=%s path=%s',
        manifest.policy_id,
        manifest.version,
        manifest.adapter_uri,
        path,
    )


def read_manifest(path: str = DEFAULT_MANIFEST_PATH) -> PolicyManifest | None:
    """Read the current manifest. Returns ``None`` if the file is absent.

    Raises :class:`json.JSONDecodeError` if the file is corrupt; the
    caller is expected to ignore that round and retry on the next mtime
    change.
    """
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
