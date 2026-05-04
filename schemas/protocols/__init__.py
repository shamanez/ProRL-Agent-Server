"""Appendix A — slot interface ``Protocol`` classes.

These are the typed boundaries between the seven slots in
``rollout_fabric.md`` §5. Implementing a slot adapter means satisfying
the corresponding ``Protocol`` here; the contract is what makes the slots
genuinely pluggable rather than just architecturally separated.

Read the ``rollout_fabric.md`` §5 + Appendix A sections for the full
narrative behind each method shape.
"""

from schemas.protocols.environment_provider import (
    ContentBlock,
    EnvironmentProvider,
    EpisodeHandle,
    StepResult,
    ToolCall,
)
from schemas.protocols.inference_backend import (
    GenerationResult,
    HealthStatus,
    InferenceBackend,
    PolicyRef,
    ReloadResult,
)
from schemas.protocols.live_store import (
    Ack,
    BackpressureHint,
    LiveStore,
    NoProgressError,
    StoreMetrics,
)
from schemas.protocols.policy_registry import (
    PolicyRegistry,
    PublishResult,
    VersionInfo,
)
from schemas.protocols.replay_archive import (
    AppendResult,
    FilterSpec,
    ReplayArchive,
)
from schemas.protocols.rollout_worker import (
    FilterStrategy,
    PolicyVersionStream,
    RolloutWorker,
    TaskSource,
)
from schemas.protocols.trainer_adapter import StepMetrics, TrainerAdapter

__all__ = [
    'Ack',
    'AppendResult',
    'BackpressureHint',
    'ContentBlock',
    'EnvironmentProvider',
    'EpisodeHandle',
    'FilterSpec',
    'FilterStrategy',
    'GenerationResult',
    'HealthStatus',
    'InferenceBackend',
    'LiveStore',
    'NoProgressError',
    'PolicyRef',
    'PolicyRegistry',
    'PolicyVersionStream',
    'PublishResult',
    'ReloadResult',
    'ReplayArchive',
    'RolloutWorker',
    'StepMetrics',
    'StepResult',
    'StoreMetrics',
    'TaskSource',
    'ToolCall',
    'TrainerAdapter',
    'VersionInfo',
]
