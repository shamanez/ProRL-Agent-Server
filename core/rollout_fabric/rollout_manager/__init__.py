"""Slot 5.3 — RolloutManager.

Owns the dataloader (§3.8), drives rollout against ProRL via HTTP (BC-13),
applies zero-variance filter (§3.7), tees to archive (BC-12), pushes to
LiveStore. Zero VERL / OpenHands imports.

Group-policy consistency: one ``PolicyVersionSnapshot`` per group dispatch;
all N sibling episodes stamp the same ``behavior_policy_version``.
"""

from rollout_fabric.rollout_manager.policy_subscription import (
    FilePollingPolicySubscription,
    GrpcStreamingPolicySubscription,
    PolicyVersionStream,
)

__all__ = [
    'FilePollingPolicySubscription',
    'GrpcStreamingPolicySubscription',
    'PolicyVersionStream',
]
