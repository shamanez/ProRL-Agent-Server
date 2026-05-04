"""Slot 5.3 — RolloutWorker.

Owns the dataloader (§3.8 — load-bearing data ownership boundary), the
agent dispatch loop (calls ProRL :8006 per turn), the DAPO eager-push
seam (§3.7), and the producer-side filters (zero-variance drop). After
S2 the trainer no longer holds any of these.

Exposes a small RPC surface to the trainer:

* ``pause_production`` / ``resume_production`` — used during checkpoint
  publishes so a publish boundary aligns with end-of-batch.
* ``get_dataloader_state`` / ``load_dataloader_state`` — for trainer-
  driven checkpoint resume.

Validation is **not** an RPC here. Per the operating-principle revision
in the plan, the validation flow is removed; trainer never validates.
If validation comes back later, it lands as its own cut.
"""

from rollout_worker.policy_subscription import (
    FilePollingPolicySubscription,
    PolicyVersionStream,
)

__all__ = ['FilePollingPolicySubscription', 'PolicyVersionStream']
