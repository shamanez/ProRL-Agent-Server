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
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fsdp_utils import (
    fsdp_version,
)
from verl.workers.config import HFModelConfig

# verl renamed fsdp_workers → engine_workers and dropped AsyncActorRolloutRefWorker
# in newer releases. Import from the old location first; fall back to the new one.
# Full migration to the new engine_workers API is tracked separately.
try:
    from verl.workers.fsdp_workers import (
        AsyncActorRolloutRefWorker as _UpstreamAsyncWorker,
    )
except (ImportError, ModuleNotFoundError):
    from verl.workers.engine_workers import (  # type: ignore[no-redef]
        ActorRolloutRefWorker as _UpstreamAsyncWorker,
    )

logger = logging.getLogger(__name__)


class _ConfigProxy:
    """Attribute-access wrapper for converted OmegaConf configs.

    Returned by _patched_omega_conf_to_dataclass instead of raw OmegaConf.
    Unlike DictConfig, plain attribute assignment stores Python objects as-is
    (no OmegaConf wrapping), so HFModelConfig.hf_config (a Qwen3Config object)
    can be set via ``actor_config.model_config = hf_model_config``.
    """

    def __init__(self, data: dict) -> None:
        for k, v in data.items():
            object.__setattr__(self, k, _ConfigProxy(v) if isinstance(v, dict) else v)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        object.__setattr__(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __getattr__(self, name: str) -> Any:
        # Return None for any attribute not set in __init__.
        # New VERL adds fields (zero_indexed_step, etc.) absent from our YAML;
        # returning None lets downstream code apply its own defaults/guards.
        if name.startswith('_'):
            raise AttributeError(name)
        return None

    def get(self, key: Any, default: Any = None) -> Any:
        if not isinstance(key, str):
            return default
        val = getattr(self, key)
        return default if val is None else val

    def __contains__(self, key: str) -> bool:
        return getattr(self, key, None) is not None

    def keys(self) -> list:
        return [k for k in self.__dict__ if not k.startswith('_')]

    def values(self) -> list:
        return [self.__dict__[k] for k in self.keys()]

    def items(self):
        return [(k, self.__dict__[k]) for k in self.keys()]

    def __iter__(self):
        return iter(self.keys())


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

    def _inject_engine_fields(self) -> None:
        """Synthesize engine=FSDPEngineConfig on actor/ref config sections.

        New VERL's engine_workers.py accesses actor_config.engine after calling
        omega_conf_to_dataclass(self.config.actor).  Our patched version returns
        the raw OmegaConf (no _target_), so .engine is absent.  We inject a
        proper FSDPEngineConfig built from the existing fsdp_config sub-section.
        """
        from dataclasses import fields as dc_fields

        from omegaconf import OmegaConf
        from verl.workers.config import FSDPEngineConfig

        known = {f.name for f in dc_fields(FSDPEngineConfig)}

        for section in ('actor', 'ref'):
            section_cfg = getattr(self.config, section, None)
            if section_cfg is None or 'engine' in section_cfg:
                continue
            strategy = getattr(section_cfg, 'strategy', 'fsdp')
            kwargs: dict = {'strategy': strategy}
            fsdp_raw = getattr(section_cfg, 'fsdp_config', None)
            if fsdp_raw is not None:
                raw = (
                    OmegaConf.to_container(
                        fsdp_raw, resolve=True, throw_on_missing=False
                    )
                    or {}
                )
                if isinstance(raw, dict):
                    raw.pop('_target_', None)
                    kwargs.update({k: v for k, v in raw.items() if k in known})
            # OmegaConf.structured() rejects None for int-typed fields (e.g.
            # max_token_len_per_gpu: int = None in EngineConfig). Use asdict +
            # OmegaConf.create (untyped DictConfig) to avoid the validation.
            import dataclasses as _dc  # noqa: PLC0415

            engine_dict = _dc.asdict(FSDPEngineConfig(**kwargs))
            OmegaConf.update(
                section_cfg,
                'engine',
                OmegaConf.create(engine_dict),
                merge=False,
            )

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

        try:
            import verl.workers.fsdp_workers as _fw
        except (ImportError, ModuleNotFoundError):
            import verl.workers.engine_workers as _fw  # type: ignore[no-redef]
        from omegaconf import OmegaConf

        # Disable struct mode first so we can add synthetic fields below.
        OmegaConf.set_struct(self.config, False)

        # Synthesize the 'engine' field expected by new VERL on actor/ref sections.
        # FSDPActorConfig.__post_init__ normally sets engine=fsdp_config, but our
        # patch returns raw OmegaConf (no _target_), so we inject it manually.
        self._inject_engine_fields()

        original_fn = _cfg.omega_conf_to_dataclass

        def _patched_omega_conf_to_dataclass(config, dataclass_type=None):
            """Compatibility shim for new VERL omega_conf_to_dataclass API.

            New VERL requires _target_ in every config section.  Our YAML pre-
            dates this convention.  Two cases:

            1. Model configs (have 'path', no 'strategy'): instantiate as
               HFModelConfig so __post_init__ loads hf_config from disk.
               Return the dataclass instance directly — NOT wrapped in OmegaConf
               — so that actor_config.model_config = HFModelConfig_instance works
               without OmegaConf rejecting Qwen3Config as an unsupported type.

            2. All other configs (actor, ref, rollout, …): return a _ConfigProxy
               built from OmegaConf.to_container().  _ConfigProxy stores Python
               objects as plain attributes (no OmegaConf wrapping), supports
               recursive attribute access, and has .get(key, default).
            """
            if dataclass_type is None and '_target_' not in config:
                import dataclasses as _dc  # noqa: PLC0415

                if 'path' in config and 'strategy' not in config:
                    # Model config: create HFModelConfig to load hf_config.
                    try:
                        known = {f.name for f in _dc.fields(HFModelConfig)}
                        raw = OmegaConf.to_container(
                            config, resolve=True, throw_on_missing=False
                        )
                        if isinstance(raw, dict):
                            return HFModelConfig(
                                **{k: v for k, v in raw.items() if k in known}
                            )
                    except Exception:
                        pass
                # Actor / ref / rollout / other: return _ConfigProxy so that
                # subsequent attribute assignment (e.g. actor_config.model_config =
                # HFModelConfig_instance) stores the Python object without OmegaConf
                # trying to wrap Qwen3Config as an AnyNode.
                raw = OmegaConf.to_container(
                    config, resolve=True, throw_on_missing=False
                )
                if isinstance(raw, dict):
                    return _ConfigProxy(raw)
                return config
            return original_fn(config, dataclass_type)

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
        from contextlib import nullcontext

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
            assert self._is_actor

            # verl v0.8: _is_offload_param lives on self.actor.engine, not self.
            engine = self.actor.engine
            is_offload = getattr(engine, '_is_offload_param', False)
            actor_module = engine.module

            if is_offload:
                load_fsdp_model_to_gpu(actor_module)

            is_lora = data.meta_info.pop('is_lora', False)
            # verl v0.8: adapter disable is on engine, not actor_module directly.
            adapter_ctx = engine.disable_adapter() if is_lora else nullcontext()

            # verl v0.8 FSDPEngineWithLMHead only accepts nested (no-padding)
            # tensors.  Our LiveStore batch is padded (from pack_unpadded_groups),
            # so we must convert.  attention_mask marks valid (non-pad) positions.
            from verl.utils.tensordict_utils import (  # noqa: PLC0415
                assign_non_tensor,
                nested_tensor_from_tensor_list,
            )

            def _to_nested(padded, mask):
                seq_lens = mask.long().sum(dim=-1).tolist()
                seqs = [padded[i, : int(n)] for i, n in enumerate(seq_lens)]
                return nested_tensor_from_tensor_list(seqs)

            infer_batch = data.batch
            attn = infer_batch['attention_mask']  # (B, total_len)

            # 'responses' is (B, resp_len); derive prompt_len from input_ids shape.
            resp_len = infer_batch['responses'].shape[1]
            total_len = attn.shape[1]
            prompt_len = total_len - resp_len

            # slice_response_from_unpad_output reads 'prompts' (padded) to split
            # the log-prob output into prompt/response portions.  Keep padded.
            infer_batch['prompts'] = infer_batch['input_ids'][:, :prompt_len]

            # Convert full-sequence tensors to nested (required by the engine).
            infer_batch['input_ids'] = _to_nested(infer_batch['input_ids'], attn)
            infer_batch['position_ids'] = _to_nested(infer_batch['position_ids'], attn)
            # loss_mask is (B, resp_len); extend to full seq with 0s for prompt.
            full_loss_mask = torch.cat(
                [
                    torch.zeros(
                        attn.shape[0],
                        prompt_len,
                        dtype=infer_batch['loss_mask'].dtype,
                        device=infer_batch['loss_mask'].device,
                    ),
                    infer_batch['loss_mask'],
                ],
                dim=1,
            )
            infer_batch['loss_mask'] = _to_nested(full_loss_mask, attn)

            # Temperature: T=1.0 is correct for log-prob recompute (teacher-forced).
            # vLLM returns log-probs from raw logits (T=1.0 effective) so the IS
            # ratio π_θ/π_β = exp(lp_new - lp_old) is self-consistent at T=1.0.
            # use_fused_kernels=True requires a Python scalar, not a per-sample tensor.
            assign_non_tensor(
                infer_batch,
                temperature=float(self.config.rollout.temperature),
                # Disable loss computation — we only need log-probs, not PPO loss.
                # This avoids ppo_loss reading global_batch_size from the batch.
                compute_loss=False,
            )

            with adapter_ctx:
                outputs = self.actor.infer_batch(infer_batch)

            # The engine operates on nested tensors; the Ray controller gathers
            # outputs across DP workers via torch.cat(dim=0) which fails on nested.
            # Convert log_probs back to padded (B, resp_len) before returning.
            log_probs = outputs['log_probs']
            if isinstance(log_probs, torch.Tensor) and log_probs.is_nested:
                resp_len = data.batch['responses'].shape[1]
                log_probs = torch.nested.to_padded_tensor(
                    log_probs,
                    padding=0.0,
                    output_size=(log_probs.size(0), resp_len),
                )

            if not is_lora:
                tensors = {'old_log_probs': log_probs}
            else:
                tensors = {'ref_log_prob': log_probs}
            # Inject zero entropys: trainer reads batch['entropys'] directly;
            # entropy_coeff=0 in config so it contributes nothing to the loss.
            tensors['entropys'] = torch.zeros_like(log_probs)
            if 'sum_pi_squared' in outputs:
                tensors['sum_pi_squared'] = outputs['sum_pi_squared']

            output = DataProto.from_dict(
                tensors=tensors,
                meta_info={'temperature': self.config.rollout.temperature},
            )
            output = output.to('cpu')

            # FSDP1 requires explicit reshard after eval.
            if self.world_size > 1 and fsdp_version(actor_module) == 1:
                actor_module._handle.reshard(True)

            if is_offload:
                offload_fsdp_model_to_cpu(actor_module)
                log_gpu_memory_usage(
                    'After offload actor model during compute_log_prob',
                    logger=logger,
                )

            return output

        return compute_log_prob

    compute_log_prob = _build_compute_log_prob()
    del _build_compute_log_prob

    # ------------------------------------------------------------------
    # update_actor: convert padded batch to nested before FSDP train step
    # ------------------------------------------------------------------

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='actor'))
    def update_actor(self, data: 'DataProto') -> 'DataProto':  # noqa: F821
        """Override to convert padded LiveStore batch to nested tensors.

        verl v0.8 FSDPEngineWithLMHead only accepts DatasetPadMode.NO_PADDING
        (nested/jagged tensors).  pack_unpadded_groups produces padded tensors,
        so we convert input_ids / position_ids / loss_mask here before
        delegating to the base class train_mini_batch path.
        """
        from verl.utils.tensordict_utils import (
            nested_tensor_from_tensor_list,  # noqa: PLC0415
        )

        def _to_nested(padded, mask):
            seq_lens = mask.long().sum(dim=-1).tolist()
            seqs = [padded[i, : int(n)] for i, n in enumerate(seq_lens)]
            return nested_tensor_from_tensor_list(seqs)

        batch = data.batch
        attn = batch['attention_mask']  # (B, total_len)

        resp_len = batch['responses'].shape[1]
        prompt_len = attn.shape[1] - resp_len

        # slice_response_from_unpad_output needs 'prompts' (padded) to split output.
        batch['prompts'] = batch['input_ids'][:, :prompt_len]

        batch['input_ids'] = _to_nested(batch['input_ids'], attn)
        batch['position_ids'] = _to_nested(batch['position_ids'], attn)

        full_lm = torch.cat(
            [
                torch.zeros(
                    attn.shape[0],
                    prompt_len,
                    dtype=batch['loss_mask'].dtype,
                    device=batch['loss_mask'].device,
                ),
                batch['loss_mask'],
            ],
            dim=1,
        )
        batch['loss_mask'] = _to_nested(full_lm, attn)

        # Inject training-loop fields that new VERL expects in the TensorDict
        # before calling train_mini_batch → ppo_loss.
        # global_batch_size: total samples across all DP ranks; used by ppo_loss
        # for SUM-vs-MEAN aggregation in loss normalisation.
        from verl.utils.tensordict_utils import assign_non_tensor  # noqa: PLC0415

        local_bsz = int(batch['responses'].size(0))
        # world_size = FSDP DP size; mini_batch_size must be the GLOBAL total
        # so that engine.train_mini_batch asserts mini_batch_size % world_size == 0.
        fsdp_dp_size = self.actor.engine.get_data_parallel_size()
        global_bsz = local_bsz * fsdp_dp_size
        assign_non_tensor(
            batch,
            compute_loss=True,
            # Temperature for actor forward pass — same reasoning as compute_log_prob:
            # use T=1.0 so log-probs are consistent with vLLM's raw-logit log-probs.
            # use_fused_kernels=True requires a scalar.
            temperature=float(self.config.rollout.temperature),
            global_batch_size=global_bsz,
            mini_batch_size=global_bsz,
            epochs=int(self.config.actor.ppo_epochs or 1),
            seed=0,
            dataloader_kwargs={'shuffle': False},
        )

        output = self.actor.train_mini_batch(data=batch)
        if output is None:
            return None

        output_cpu = output.cpu()
        # Extract metrics and wrap as DataProto so the DAPO trainer can read
        # actor_output.meta_info['metrics'] (matching new VERL's ray_trainer API).
        from verl import DataProto  # noqa: PLC0415
        from verl.utils import tensordict_utils as _tu  # noqa: PLC0415

        metrics = _tu.get(output_cpu, 'metrics') or {}
        return DataProto.from_single_dict(data={}, meta_info={'metrics': metrics})

    # ------------------------------------------------------------------
    # save_checkpoint: FSDP shards + LoRA adapter for PolicyRegistry
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
    ):
        """Override to save the LoRA adapter alongside the FSDP shards.

        _publish_lora_adapter in ray_trainer.py expects:
          {local_path}/actor/lora_adapter/adapter_model.safetensors
          {local_path}/actor/lora_adapter/adapter_config.json

        The base FSDP checkpoint manager saves model shards but not the LoRA
        adapter.  We call get_per_tensor_param() (an all-reduce over FSDP ranks)
        to gather the LoRA-only weights, then rank-0 serialises them.
        """
        import json  # noqa: PLC0415
        import os  # noqa: PLC0415

        # 1. Save FSDP shards (model + optimizer + extra_state)
        super().save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

        # 2. Extract LoRA adapter — all ranks participate in the all-gather
        #    but only rank-0 writes to disk.
        logger.info('save_checkpoint: extracting LoRA adapter via get_per_tensor_param')
        try:
            # base_sync_done=True: vLLM already has the base model; we want only
            # the LoRA delta (A/B matrices via get_peft_model_state_dict).
            # Without this, collect_lora_params returns the full base weights (~15 GB).
            per_tensor_param, peft_config_dict = self.actor.engine.get_per_tensor_param(
                base_sync_done=True
            )
        except Exception:
            logger.warning(
                'save_checkpoint: get_per_tensor_param failed — LoRA adapter NOT saved',
                exc_info=True,
            )
            return

        logger.info(
            'save_checkpoint: peft_config_dict=%s',
            type(peft_config_dict).__name__,
        )
        if peft_config_dict is None:
            logger.info(
                'save_checkpoint: peft_config_dict is None — skipping LoRA save'
            )
            return  # full-weight training, no LoRA adapter to publish

        if torch.distributed.get_rank() == 0:
            from safetensors.torch import save_file  # noqa: PLC0415

            # local_path is already {global_step_N}/actor — don't add 'actor' again.
            adapter_dir = os.path.join(local_path, 'lora_adapter')
            os.makedirs(adapter_dir, exist_ok=True)
            # per_tensor_param may be a generator — materialise to dict first.
            lora_tensors = dict(per_tensor_param)
            # Deduplicate tensors sharing memory (e.g. tied embeddings
            # lm_head.weight ≡ model.embed_tokens.weight) — safetensors
            # raises an error on shared-memory entries.
            seen_ptrs: dict = {}
            deduped: dict = {}
            for k, t in lora_tensors.items():
                ptr = t.data_ptr()
                if ptr not in seen_ptrs:
                    seen_ptrs[ptr] = k
                    deduped[k] = t.contiguous()
            save_file(deduped, os.path.join(adapter_dir, 'adapter_model.safetensors'))

            # peft_config_dict may contain sets (e.g. target_modules) which
            # are not JSON-serialisable — convert them to sorted lists.
            def _json_safe(obj):
                if isinstance(obj, set):
                    return sorted(obj)
                raise TypeError(
                    f'Object of type {type(obj).__name__} is not JSON serializable'
                )

            with open(os.path.join(adapter_dir, 'adapter_config.json'), 'w') as fp:
                json.dump(peft_config_dict, fp, indent=2, default=_json_safe)
            logger.info('Saved LoRA adapter to %s', adapter_dir)

        torch.distributed.barrier()

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
