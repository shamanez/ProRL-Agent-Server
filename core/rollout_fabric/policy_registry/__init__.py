"""Slot 5.7 — PolicyRegistry.

S2: file-backed minimal registry (trainer writes manifest; worker polls).
S4: full gRPC service with streaming subscriptions and §3.3 abort gate.
"""

from rollout_fabric.policy_registry.client import PolicyRegistryClient, PublishFailedError
from rollout_fabric.policy_registry.fanout import FanoutError, fanout_to_pool
from rollout_fabric.policy_registry.file_registry import PolicyManifest, read_manifest, write_manifest

__all__ = [
    'FanoutError',
    'PolicyManifest',
    'PolicyRegistryClient',
    'PublishFailedError',
    'fanout_to_pool',
    'read_manifest',
    'write_manifest',
]
