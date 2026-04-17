# Copyright 2024 Pluralis AI
# Licensed under the Apache License, Version 2.0
"""Custom AsyncActorRolloutRefWorker that restores methods removed in verl v0.8.

verl v0.8 switched to a RolloutReplica + ServerAdapter architecture and removed
execute_method / chat_completion / wake_up / sleep from AsyncActorRolloutRefWorker.
Our ExternalRayDistributedExecutor still needs the execute_method dispatch to talk
directly to the vLLM WorkerWrapperBase that lives inside each Ray worker process.

Design
------
1.  ``VLLMAsyncRolloutCompat`` is a lightweight stand-in for the old
    ``vLLMAsyncRollout``.  It wraps vLLM's ``WorkerWrapperBase`` and dispatches
    ``execute_method`` calls (init_worker / load_model / sleep / wake_up / …).
    The real vLLM engine is created lazily when ``init_worker`` is called by the
    ``ExternalRayDistributedExecutor`` during ``AsyncvLLMServer.init_engine()``.

2.  ``AsyncActorRolloutRefWorker`` subclasses the upstream worker, overrides
    ``_build_rollout`` to install ``VLLMAsyncRolloutCompat`` instead of the
    ``ServerAdapter``, and adds back the four ``@register``-decorated methods.

3.  ``update_weights`` is overridden to be a no-op because verl_custom syncs
    weights through the ``AsyncvLLMServer`` engine lifecycle (wake_up / sleep),
    not through the new ``ServerAdapter.update_weights → BucketedWeightSender``
    path that the upstream ``rollout_mode`` uses.

4.  ``init_model`` is overridden to pass the raw OmegaConf config (not a
    dataclass) to ``DataParallelPPOActor``.  The new verl v0.8 calls
    ``omega_conf_to_dataclass(self.config.actor)`` which fails on custom
    fields (``policy_loss_type``, ``tis_imp_ratio_cap``, etc.) that exist in
    the verl_custom YAML but not in ``FSDPActorConfig``.  Passing the raw
    OmegaConf dict preserves the old v0.4 behavior.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed
from torch.distributed.fsdp import (
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.single_controller.base.decorator import (
    Dispatch,
    register,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fsdp_utils import (
    fsdp_version,
)
from verl.workers.config import HFModelConfig
from verl.workers.fsdp_workers import (
    AsyncActorRolloutRefWorker as _UpstreamAsyncWorker,
)

logger = logging.getLogger(__name__)


class VLLMAsyncRolloutCompat:
    """Thin wrapper around vLLM's ``WorkerWrapperBase``.

    Mirrors the old ``verl.workers.rollout.vllm_rollout.vLLMAsyncRollout``
    from verl v0.4. The inference engine is created lazily when
    ``execute_method("init_worker", …)`` is called by the
    ``ExternalRayDistributedExecutor``.
    """

    def __init__(self) -> None:
        self.inference_engine = None  # set in init_worker
        self.is_sleep = False

    # ------------------------------------------------------------------
    # Methods dispatched via execute_method
    # ------------------------------------------------------------------

    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """Initialize the vLLM worker wrapper (called once per worker)."""
        from vllm.v1.worker.worker_base import WorkerWrapperBase

        all_kwargs[0]['rank'] = int(os.environ['RANK'])
        all_kwargs[0]['local_rank'] = 0

        self.vllm_config = all_kwargs[0]['vllm_config']
        # vLLM 0.18+: WorkerWrapperBase takes (rpc_rank, global_rank),
        # not vllm_config. Config is extracted inside init_worker().
        self.inference_engine = WorkerWrapperBase(rpc_rank=0)
        self.inference_engine.init_worker(all_kwargs)

    def init_device(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the device, ensuring the Ray executor check is skipped.

        Each Ray worker process sees 1 GPU, but vllm_config.parallel_config
        may have local_world_size=TP (e.g. 2).  vLLM 0.18's init_device
        asserts local_world_size <= visible_device_count, but only when
        distributed_executor_backend not in ("ray", "external_launcher").
        Our ExternalRayDistributedExecutor is a class, not a string, so the
        check fires.  We temporarily set it to "ray" to skip the assertion.
        """
        pc = self.vllm_config.parallel_config
        original_backend = pc.distributed_executor_backend
        pc.distributed_executor_backend = 'ray'
        try:
            self.inference_engine.init_device(*args, **kwargs)
        finally:
            pc.distributed_executor_backend = original_backend

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        self.inference_engine.load_model(*args, **kwargs)

    def sleep(self, *args: Any, **kwargs: Any) -> None:
        """Offload model weights / discard kv-cache."""
        if self.is_sleep:
            return
        if self.inference_engine is not None:
            # vLLM 0.18: sleep/wake_up are on the concrete worker, not wrapper.
            self.inference_engine.worker.sleep(*args, **kwargs)
        self.is_sleep = True

    def wake_up(self, *args: Any, **kwargs: Any) -> None:
        """Load model weights / rebuild kv-cache."""
        if not self.is_sleep:
            return
        if self.inference_engine is not None:
            self.inference_engine.worker.wake_up(*args, **kwargs)
        self.is_sleep = False

    # ------------------------------------------------------------------
    # Central dispatch (called from the worker's @register method)
    # ------------------------------------------------------------------

    def execute_method(self, method: str | bytes, *args: Any, **kwargs: Any) -> Any:
        """Route ``method`` to the appropriate handler or to the engine."""
        if method == 'init_worker':
            return self.init_worker(*args, **kwargs)
        if method == 'init_device':
            return self.init_device(*args, **kwargs)
        if method == 'load_model':
            return self.load_model(*args, **kwargs)
        if method == 'sleep':
            return self.sleep(*args, **kwargs)
        if method == 'wake_up':
            return self.wake_up(*args, **kwargs)
        # Everything else (e.g. init_device, execute_model, …) forwards to
        # the underlying vLLM worker.  vLLM 0.18 removed execute_method()
        # from WorkerWrapperBase, so we call the method directly.
        if isinstance(method, str):
            return getattr(self.inference_engine, method)(*args, **kwargs)
        # Callable (cloudpickle bytes) — deserialize and invoke.
        import cloudpickle

        fn = cloudpickle.loads(method)
        return fn(self.inference_engine, *args, **kwargs)


class AsyncActorRolloutRefWorker(_UpstreamAsyncWorker):
    """Extends upstream ``AsyncActorRolloutRefWorker`` with dispatch methods
    needed by verl_custom's ``ExternalRayDistributedExecutor``.

    The upstream v0.8 worker only has ``update_weights``.  We add back
    ``execute_method``, ``chat_completion``, ``wake_up``, and ``sleep``,
    all decorated with ``Dispatch.DIRECT_ROLLOUT_METHOD`` so the
    ``ExternalRayDistributedExecutor.collective_rpc`` can call them via
    Ray actor RPC.
    """

    def _build_rollout(self, trust_remote_code: bool = False) -> None:
        """Build rollout without calling super() — avoids omega_conf_to_dataclass
        on the rollout config (which has extra custom fields that RolloutConfig
        rejects) and avoids creating the ServerAdapter (replaced by
        VLLMAsyncRolloutCompat).

        We replicate only the parts of the upstream _build_rollout that our
        code path needs: device mesh setup, dispatch/collect registration, and
        FSDP state-dict config.
        """
        from torch.distributed.device_mesh import init_device_mesh
        from verl.utils.device import get_device_name

        device_name = get_device_name()

        # Parse model config (this one has no extra custom fields, safe).
        model_config: HFModelConfig = omega_conf_to_dataclass(
            self.config.model, dataclass_type=HFModelConfig
        )
        self.model_config = model_config

        # Build rollout device mesh (same logic as upstream).
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp_size = getattr(self.config.rollout, 'data_parallel_size', 1)
        pp_size = getattr(self.config.rollout, 'pipeline_model_parallel_size', 1)
        infer_world_size = infer_tp * dp_size * pp_size
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f'rollout world_size: {self.world_size} is not divisible by '
            f'infer_world_size: {infer_world_size}'
        )
        rollout_device_mesh = init_device_mesh(
            device_name,
            mesh_shape=(dp, infer_tp * dp_size, pp_size),
            mesh_dim_names=['dp', 'infer_tp', 'infer_pp'],
        )
        self.rollout_device_mesh = rollout_device_mesh

        # Register dispatch/collect info (same as upstream for non-hf rollout).
        is_collect = (
            rollout_device_mesh['infer_tp'].get_local_rank() == 0
            and rollout_device_mesh['infer_pp'].get_local_rank() == 0
        )
        self._register_dispatch_collect_info(
            'rollout',
            dp_rank=rollout_device_mesh['dp'].get_local_rank(),
            is_collect=is_collect,
        )

        # Install VLLMAsyncRolloutCompat instead of ServerAdapter.
        self.rollout = VLLMAsyncRolloutCompat()

        # Set FSDP state dict type (same as upstream).
        if (
            torch.distributed.get_world_size() == 1
            and fsdp_version(self.actor_module_fsdp) == 1
        ):
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # Store TP/DP rank info for logging.
        self.vllm_tp_size = infer_tp
        self.vllm_dp_rank = int(os.environ['RANK']) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ['RANK']) % self.vllm_tp_size

    # ------------------------------------------------------------------
    # Override init_model: pass raw OmegaConf config to DataParallelPPOActor
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        """Override upstream init_model to handle verl_custom config compat.

        Two issues with the new verl v0.8 ``init_model``:

        1. ``omega_conf_to_dataclass(self.config.actor)`` fails on custom YAML
           fields (``policy_loss_type``, ``tis_imp_ratio_cap``, etc.).  We
           patch it to return the raw OmegaConf config when ``_target_`` is
           absent.

        2. OmegaConf struct mode on the config tree rejects access to fields
           that exist in the new verl dataclasses but not in our YAML (e.g.
           ``optimizer``, ``optimizer_impl`` in ``FSDPOptimizerConfig``).
           We disable struct mode so missing keys return defaults or None
           via ``.get()`` instead of crashing.
        """
        import verl.utils.config as _cfg
        import verl.workers.fsdp_workers as _fw
        from omegaconf import OmegaConf

        original_fn = _cfg.omega_conf_to_dataclass

        def _patched_omega_conf_to_dataclass(config, dataclass_type=None):
            """If the config lacks ``_target_`` and no ``dataclass_type`` is
            given, return the raw config instead of crashing."""
            if dataclass_type is None and '_target_' not in config:
                return config
            return original_fn(config, dataclass_type)

        # Disable struct mode on the entire config tree so that accessing
        # new verl v0.8 fields that are absent from our YAML doesn't crash.
        OmegaConf.set_struct(self.config, False)

        _fw.omega_conf_to_dataclass = _patched_omega_conf_to_dataclass
        _cfg.omega_conf_to_dataclass = _patched_omega_conf_to_dataclass
        try:
            super().init_model()
        finally:
            _fw.omega_conf_to_dataclass = original_fn
            _cfg.omega_conf_to_dataclass = original_fn

    # ------------------------------------------------------------------
    # Override compute_log_prob: force calculate_entropy=False to avoid OOM
    # ------------------------------------------------------------------
    #
    # The upstream method (verl/workers/fsdp_workers.py:1125) hardcodes
    # ``calculate_entropy = not is_lora`` (line 1145), which forces
    # ``DataParallelPPOActor.compute_log_prob`` to allocate a
    # [tokens × vocab × 4B] = ~10.9 GiB tensor on Qwen3-4B
    # (vocab=151936, total_len=17920). On 8 × A100-40GB with FSDP +
    # ulysses=2 + vLLM colocated this OOMs every step.
    #
    # We override here to:
    #   1. Force ``calculate_entropy=False`` which lets dp_actor.py:264 use
    #      ``inplace_backward=True`` and free the logits in-place after
    #      ``logprobs_from_logits`` — saves the 10.9 GiB allocation.
    #   2. Still inject a zero ``entropys`` tensor in the output, because
    #      ``ray_trainer.py:1499`` accesses ``old_log_prob.batch['entropys']``
    #      directly.  ``entropy_coeff=0`` in our config so the value is
    #      multiplied out of the loss anyway.
    #
    # Imports are inline so autoflake doesn't strip them as unused at module
    # scope (the rest of the file does not need them).

    def _build_compute_log_prob():
        from verl import DataProto
        from verl.single_controller.base.decorator import (
            make_nd_compute_dataproto_dispatch_fn,
        )
        from verl.utils.fsdp_utils import (
            load_fsdp_model_to_gpu,
            offload_fsdp_model_to_cpu,
        )
        from verl.utils.profiler import DistProfiler, log_gpu_memory_usage

        @register(
            dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='actor')
        )
        @DistProfiler.annotate(color='blue', role='actor_compute_log_prob')
        def compute_log_prob(self, data: DataProto):
            from contextlib import nullcontext

            assert self._is_actor
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)

            is_lora = data.meta_info.pop('is_lora', False)
            adapter_ctx = (
                self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
            )
            config_source = self.config.ref if is_lora else self.config.rollout
            data.meta_info['micro_batch_size'] = (
                config_source.log_prob_micro_batch_size_per_gpu
            )
            data.meta_info['max_token_len'] = (
                config_source.log_prob_max_token_len_per_gpu
            )
            data.meta_info['use_dynamic_bsz'] = config_source.log_prob_use_dynamic_bsz
            data.meta_info['temperature'] = self.config.rollout.temperature
            data.meta_info.setdefault('pad_token_id', self.tokenizer.pad_token_id)

            # KEY CHANGE: force False, override upstream `not is_lora`.
            calculate_entropy = False
            with self.ulysses_sharding_manager:
                with adapter_ctx:
                    outputs = self.actor.compute_log_prob(
                        data=data, calculate_entropy=calculate_entropy
                    )
                if not is_lora:
                    tensors = {'old_log_probs': outputs['log_probs']}
                else:
                    tensors = {'ref_log_prob': outputs['log_probs']}
                # KEY CHANGE: inject zero entropys so trainer access doesn't
                # KeyError (entropy_coeff=0 zeroes its loss contribution).
                tensors['entropys'] = torch.zeros_like(outputs['log_probs'])
                if 'sum_pi_squared' in outputs:
                    tensors['sum_pi_squared'] = outputs['sum_pi_squared']
                output = DataProto.from_dict(
                    tensors=tensors,
                    meta_info={'temperature': self.config.rollout.temperature},
                )

            output = output.to('cpu')

            if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
                self.actor.actor_module._handle.reshard(True)

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage(
                    'After offload actor model during compute_log_prob',
                    logger=logger,
                )

            return output

        return compute_log_prob

    compute_log_prob = _build_compute_log_prob()
    del _build_compute_log_prob

    # ------------------------------------------------------------------
    # Dispatch methods for ExternalRayDistributedExecutor
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: str | bytes, *args: Any, **kwargs: Any) -> Any:
        """Called by ``ExternalRayDistributedExecutor.collective_rpc``."""
        if self.vllm_tp_rank == 0 and method != 'execute_model':
            logger.info(
                '[DP=%d,TP=%d] execute_method: %s',
                self.vllm_dp_rank,
                self.vllm_tp_rank,
                method if isinstance(method, str) else 'Callable',
            )
        return self.rollout.execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request: Any) -> Any:
        """Kept for API compatibility; production uses ``AsyncvLLMServer``."""
        return await self.rollout.chat_completion(json_request)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self) -> bool:
        """Wake up the vLLM engine (load weights / rebuild kv-cache)."""
        self.rollout.wake_up()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self) -> bool:
        """Sleep the vLLM engine (offload weights / discard kv-cache)."""
        self.rollout.sleep()
        return True

    # ------------------------------------------------------------------
    # Override update_weights: weight sync handled by AsyncvLLMServer
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None) -> bool:
        """No-op: verl_custom syncs weights via ``AsyncvLLMServer``
        engine lifecycle (wake_up / sleep), not the upstream
        ``ServerAdapter.update_weights → BucketedWeightSender`` path."""
        return True
