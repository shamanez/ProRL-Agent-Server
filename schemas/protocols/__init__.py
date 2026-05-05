"""Appendix A — slot interface Protocol classes."""

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
from schemas.protocols.rollout_manager import (
    FilterStrategy,
    PolicyVersionStream,
    RolloutManager,
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
    'RolloutManager',
    'StepMetrics',
    'StepResult',
    'StoreMetrics',
    'TaskSource',
    'ToolCall',
    'TrainerAdapter',
    'VersionInfo',
]
