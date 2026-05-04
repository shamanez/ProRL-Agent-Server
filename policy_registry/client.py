"""gRPC client for PolicyRegistry — trainer-side publish + worker-side stream."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import grpc

from schemas._gen import policy_registry_pb2, policy_registry_pb2_grpc
from schemas.policy_version import PolicyVersionSnapshot

logger = logging.getLogger(__name__)


class PublishFailedError(RuntimeError):
    """§3.3 abort gate: trainer aborts on partial pool publish (BC-9)."""


class PolicyRegistryClient:
    def __init__(self, socket_path: str) -> None:
        self._channel = grpc.insecure_channel(f'unix:{socket_path}')
        self._stub = policy_registry_pb2_grpc.PolicyRegistryStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    # ---- trainer-side ---------------------------------------------------

    def publish_policy_version(
        self,
        *,
        policy_id: str,
        version: int,
        adapter_uri: str,
        trainer_id: str = 'trainer-0',
    ) -> dict:
        """Fanout via registry; raise :class:`PublishFailedError` on §3.3 abort."""
        req = policy_registry_pb2.PublishPolicyVersionRequest(
            policy_id=policy_id,
            version=int(version),
            adapter_uri=adapter_uri,
            trainer_id=trainer_id,
        )
        resp = self._stub.PublishPolicyVersion(req)
        if not resp.success:
            raise PublishFailedError(
                f'ABORT: registry publish failed policy_id={policy_id!r} '
                f'version={version}: endpoints_ok={resp.endpoints_ok} '
                f'endpoints_failed={resp.endpoints_failed} error={resp.error}'
            )
        return {
            'weight_sync/policy_version': version,
            'weight_sync/publish_latency_s': resp.latency_s,
            'weight_sync/endpoints_ok': resp.endpoints_ok,
            'weight_sync/endpoints_failed': 0,
        }

    def get_latest_version(self, policy_id: str) -> dict | None:
        try:
            info = self._stub.GetLatestVersion(
                policy_registry_pb2.GetLatestVersionRequest(policy_id=policy_id)
            )
        except grpc.RpcError as exc:
            if getattr(exc, 'code', lambda: None)() == grpc.StatusCode.NOT_FOUND:
                return None
            raise
        return {
            'policy_id': info.policy_id,
            'version': info.version,
            'adapter_uri': info.adapter_uri,
            'published_at': info.published_at,
        }

    def register_policy_namespace(
        self,
        *,
        policy_id: str,
        base_model_id: str,
        tokenizer_id: str,
        adapter_storage_root: str,
    ) -> None:
        self._stub.RegisterPolicyNamespace(
            policy_registry_pb2.RegisterPolicyNamespaceRequest(
                policy_id=policy_id,
                base_model_id=base_model_id,
                tokenizer_id=tokenizer_id,
                adapter_storage_root=adapter_storage_root,
            )
        )

    # ---- worker-side (S4) -----------------------------------------------

    def stream_version_updates(self, policy_id: str) -> Iterator[PolicyVersionSnapshot]:
        """Server-streaming subscription. Feeds into ``PolicyVersionCache.update``."""
        req = policy_registry_pb2.SubscribeRequest(policy_id=policy_id)
        for proto_snap in self._stub.SubscribeVersionUpdates(req):
            yield PolicyVersionSnapshot(
                policy_id=proto_snap.policy_id,
                version=int(proto_snap.version),
                adapter_uri=proto_snap.adapter_uri,
                received_at=time.monotonic(),
            )
