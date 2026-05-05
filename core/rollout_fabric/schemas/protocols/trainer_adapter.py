"""§A.6 — TrainerAdapter protocol.

Per §4.4: adapters compute their own algorithm-specific fields.
Per §3.8: no adapter owns a task dataset.
Validation is removed; trainer never validates (Sec.11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rollout_fabric.schemas.protocols.policy_registry import PublishResult
from rollout_fabric.schemas.training_sample import TrainingGroup


@dataclass(slots=True)
class StepMetrics:
    loss: float
    gradient_norm: float
    extra: dict[str, float] = field(default_factory=dict)


class TrainerAdapter(Protocol):
    trainer_id: str
    policy_id: str

    def request_batch(self, n_groups: int, step: int) -> TrainingGroup: ...

    def step(self, group: TrainingGroup) -> StepMetrics: ...

    def save_checkpoint(self, step: int, dir: str) -> str: ...

    def publish_policy_version(self, step: int, adapter_uri: str) -> PublishResult: ...
