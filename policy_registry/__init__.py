"""Slot 5.7 — PolicyRegistry.

S2: minimal-form registry. The trainer writes a JSON manifest after every
pool ACK; the rollout worker mtime-polls the file at 1 Hz and feeds
updates into its :class:`schemas.policy_version.PolicyVersionCache`.

S4: this file-backed shim is replaced by a gRPC service with streaming
subscriptions and the §3.3 abort-gate moves into the registry's fanout.
The atomic-ref-swap cache primitive on the worker side is unchanged
across the cut — only the populator flips.
"""

from policy_registry.client import PolicyRegistryClient, PublishFailedError
from policy_registry.fanout import FanoutError, fanout_to_pool
from policy_registry.file_registry import (
    PolicyManifest,
    read_manifest,
    write_manifest,
)

__all__ = [
    'FanoutError',
    'PolicyManifest',
    'PolicyRegistryClient',
    'PublishFailedError',
    'fanout_to_pool',
    'read_manifest',
    'write_manifest',
]
