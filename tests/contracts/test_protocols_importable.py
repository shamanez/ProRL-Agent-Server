"""Contract sanity — Appendix A protocols are importable, named, and
runtime-checkable for ``isinstance`` purposes.

Concrete adapters land per stage (S1: LiveStore impl; S2: RolloutWorker
impl; S3: ReplayArchive impl; S4: PolicyRegistry impl). At S0.5 the
substrate is the type surface and the import path; this test pins
both.
"""

from __future__ import annotations

import inspect

import pytest

from schemas import protocols
from schemas.protocols import (
    EnvironmentProvider,
    InferenceBackend,
    LiveStore,
    PolicyRegistry,
    ReplayArchive,
    RolloutWorker,
    TrainerAdapter,
)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    'cls,methods',
    [
        (
            EnvironmentProvider,
            ('list_tasks', 'create_episode', 'get_prompt', 'act', 'close'),
        ),
        (InferenceBackend, ('generate', 'reload_policy', 'health')),
        (
            RolloutWorker,
            (
                'pause_production',
                'resume_production',
                'get_dataloader_state',
                'load_dataloader_state',
            ),
        ),
        (
            LiveStore,
            ('push_group', 'get_batch', 'get_metrics', 'notify_policy_version'),
        ),
        (
            ReplayArchive,
            ('append_episodes', 'query', 'derive_training_samples'),
        ),
        (
            TrainerAdapter,
            ('request_batch', 'step', 'save_checkpoint', 'publish_policy_version'),
        ),
        (
            PolicyRegistry,
            (
                'publish_policy_version',
                'get_latest_version',
                'subscribe_version_updates',
                'register_policy_namespace',
            ),
        ),
    ],
    ids=[
        'env-provider',
        'inference-backend',
        'rollout-worker',
        'live-store',
        'replay-archive',
        'trainer-adapter',
        'policy-registry',
    ],
)
def test_protocol_declares_required_methods(cls, methods) -> None:
    members = {name for name, _ in inspect.getmembers(cls) if not name.startswith('_')}
    missing = [m for m in methods if m not in members]
    assert not missing, f'{cls.__name__} missing methods: {missing}'


def test_protocol_re_exports() -> None:
    """Public names from ``schemas.protocols`` import cleanly."""
    for name in (
        'ContentBlock',
        'ToolCall',
        'StepResult',
        'EpisodeHandle',
        'GenerationResult',
        'ReloadResult',
        'HealthStatus',
        'PolicyRef',
        'Ack',
        'BackpressureHint',
        'NoProgressError',
        'StoreMetrics',
        'AppendResult',
        'FilterSpec',
        'PolicyVersionStream',
        'TaskSource',
        'FilterStrategy',
        'StepMetrics',
        'PublishResult',
        'VersionInfo',
    ):
        assert hasattr(protocols, name), f'schemas.protocols missing {name}'
