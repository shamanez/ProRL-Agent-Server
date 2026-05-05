"""Appendix A — slot interface Protocol classes."""

from rollout_fabric.schemas.protocols.environment_provider import (
    ContentBlock,
    EnvironmentProvider,
    EpisodeHandle,
    StepResult,
    ToolCall,
)
from rollout_fabric.schemas.protocols.inference_backend import (
    GenerationResult,
    HealthStatus,
    InferenceBackend,
    PolicyRef,
    ReloadResult,
)
from rollout_fabric.schemas.protocols.live_store import (
    Ack,
    BackpressureHint,
    LiveStore,
    NoProgressError,
    StoreMetrics,
)
from rollout_fabric.schemas.protocols.policy_registry import (
    PolicyRegistry,
    PublishResult,
    VersionInfo,
)
from rollout_fabric.schemas.protocols.replay_archive import (
    AppendResult,
    FilterSpec,
    ReplayArchive,
)
from rollout_fabric.schemas.protocols.rollout_manager import (
    FilterStrategy,
    PolicyVersionStream,
    RolloutManager,
    TaskSource,
)
from rollout_fabric.schemas.protocols.trainer_adapter import StepMetrics, TrainerAdapter

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
