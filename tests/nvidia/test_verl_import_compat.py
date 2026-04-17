# Copyright 2024 Pluralis AI
# Licensed under the Apache License, Version 2.0
"""Import compatibility tests for verl v0.8.0.dev upgrade (Stage 0 baseline).

These tests verify that all verl imports used by verl_custom resolve
correctly after upgrading from verl v0.4 (commit 60138ebd) to v0.8.0.dev.

Tests marked @pytest.mark.integration require the full verl + torch stack
(run inside the Docker container). Tests without that marker only check
import paths and can run on the host if verl is installed.
"""

import pytest

# ---------------------------------------------------------------------------
# Category C: Imports that should still work (no changes needed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCoreProtocolImports:
    """verl.protocol — DataProto and helpers."""

    def test_dataproto_from_verl(self):
        from verl import DataProto

        assert DataProto is not None

    def test_dataproto_from_protocol(self):
        from verl.protocol import DataProto

        assert DataProto is not None

    def test_pad_unpad(self):
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        assert callable(pad_dataproto_to_divisor)
        assert callable(unpad_dataproto)


@pytest.mark.integration
class TestSingleControllerImports:
    """verl.single_controller — Worker, Ray classes."""

    def test_worker(self):
        from verl.single_controller.base import Worker

        assert Worker is not None

    def test_ray_classes(self):
        from verl.single_controller.ray import (
            RayClassWithInitArgs,
            RayResourcePool,
            RayWorkerGroup,
        )

        assert RayClassWithInitArgs is not None
        assert RayResourcePool is not None
        assert RayWorkerGroup is not None

    def test_create_colocated_worker_cls(self):
        from verl.single_controller.ray.base import create_colocated_worker_cls

        assert callable(create_colocated_worker_cls)


@pytest.mark.integration
class TestFSDPWorkerImports:
    """verl.workers.fsdp_workers — actor/critic worker classes."""

    def test_actor_rollout_ref_worker(self):
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        assert ActorRolloutRefWorker is not None

    def test_async_actor_rollout_ref_worker(self):
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

        assert AsyncActorRolloutRefWorker is not None

    def test_critic_worker(self):
        from verl.workers.fsdp_workers import CriticWorker

        assert CriticWorker is not None

    def test_reward_model_worker(self):
        from verl.workers.fsdp_workers import RewardModelWorker

        assert RewardModelWorker is not None


@pytest.mark.integration
class TestRewardManagerImports:
    """verl.workers.reward_manager — reward manager classes."""

    def test_naive_reward_manager(self):
        from verl.workers.reward_manager import NaiveRewardManager

        assert NaiveRewardManager is not None

    def test_dapo_reward_manager(self):
        from verl.workers.reward_manager import DAPORewardManager

        assert DAPORewardManager is not None

    def test_get_reward_manager_cls(self):
        from verl.workers.reward_manager import get_reward_manager_cls

        assert callable(get_reward_manager_cls)


@pytest.mark.integration
class TestActorImports:
    """verl.workers.actor — BasePPOActor."""

    def test_base_ppo_actor(self):
        from verl.workers.actor import BasePPOActor

        assert BasePPOActor is not None


@pytest.mark.integration
class TestUtilsImports:
    """verl.utils — all utility imports used by verl_custom."""

    def test_torch_functional(self):
        import verl.utils.torch_functional as verl_F

        assert callable(verl_F.masked_mean)
        assert callable(verl_F.logprobs_from_logits)

    def test_model_utils(self):
        from verl.utils.model import compute_position_id_with_mask

        assert callable(compute_position_id_with_mask)

    def test_fs_utils(self):
        from verl.utils.fs import copy_local_path_from_hdfs, copy_to_local

        assert callable(copy_to_local)
        assert callable(copy_local_path_from_hdfs)

    def test_checkpoint_manager(self):
        from verl.utils.checkpoint.checkpoint_manager import (
            BaseCheckpointManager,
            find_latest_ckpt_path,
        )

        assert BaseCheckpointManager is not None
        assert callable(find_latest_ckpt_path)

    def test_debug_gpu_memory_logger(self):
        from verl.utils.debug import GPUMemoryLogger

        assert GPUMemoryLogger is not None

    def test_metric_reduce(self):
        from verl.utils.metric import reduce_metrics

        assert callable(reduce_metrics)

    def test_seqlen_balancing(self):
        from verl.utils.seqlen_balancing import (
            get_reverse_idx,
            get_seqlen_balanced_partitions,
            log_seqlen_unbalance,
            rearrange_micro_batches,
        )

        assert callable(get_seqlen_balanced_partitions)
        assert callable(log_seqlen_unbalance)
        assert callable(get_reverse_idx)
        assert callable(rearrange_micro_batches)

    def test_tracking(self):
        from verl.utils.tracking import Tracking, ValidationGenerationsLogger

        assert Tracking is not None
        assert ValidationGenerationsLogger is not None

    def test_reward_score(self):
        from verl.utils.reward_score import (
            _default_compute_score,
            default_compute_score,
        )

        assert callable(default_compute_score)
        assert callable(_default_compute_score)

    def test_import_utils(self):
        from verl.utils.import_utils import deprecated, load_extern_type

        assert callable(deprecated)
        assert callable(load_extern_type)

    def test_fsdp_utils(self):
        from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_

        # FSDPModule may be None if torch version doesn't support it
        assert callable(fsdp2_clip_grad_norm_)

    def test_py_functional(self):
        from verl.utils.py_functional import append_to_dict

        assert callable(append_to_dict)

    def test_ulysses(self):
        from verl.utils.ulysses import (
            gather_outpus_and_unpad,
            ulysses_pad,
            ulysses_pad_and_slice_inputs,
        )

        assert callable(gather_outpus_and_unpad)
        assert callable(ulysses_pad)
        assert callable(ulysses_pad_and_slice_inputs)

    def test_device_utils(self):
        from verl.utils.device import (
            get_device_name,
            get_torch_device,
        )

        assert callable(get_device_name)
        assert callable(get_torch_device)

    def test_dataset_vision_utils(self):
        from verl.utils.dataset.vision_utils import process_image, process_video

        assert callable(process_image)
        assert callable(process_video)

    def test_hf_tokenizer_via_utils(self):
        """hf_tokenizer and hf_processor re-exported from verl.utils."""
        from verl.utils import hf_processor, hf_tokenizer

        assert callable(hf_tokenizer)
        assert callable(hf_processor)


@pytest.mark.integration
class TestModelImports:
    """verl.models — transformer model utilities."""

    def test_qwen2_vl(self):
        from verl.models.transformers.qwen2_vl import get_rope_index

        assert callable(get_rope_index)


# ---------------------------------------------------------------------------
# Category A: Imports that changed path (verify NEW paths work)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNewImportPaths:
    """Verify the NEW import paths work (after our fixes)."""

    def test_timer_new_path(self):
        """A1: _timer moved from debug.performance to profiler.performance."""
        from verl.utils.profiler.performance import _timer

        assert callable(_timer)

    def test_is_version_ge_new_path(self):
        """A2: is_version_ge moved from vllm_utils to vllm package."""
        from verl.utils.vllm import is_version_ge

        assert callable(is_version_ge)


# ---------------------------------------------------------------------------
# Category A: Removed imports (verify they raise ImportError)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRemovedImports:
    """Verify REMOVED imports correctly raise ImportError."""

    def test_megatron_ray_worker_group_removed(self):
        """A3: NVMegatronRayWorkerGroup removed in verl v0.8."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from verl.single_controller.ray.megatron import (
                NVMegatronRayWorkerGroup,  # noqa: F401
            )

    def test_async_server_class_removed(self):
        """A4: async_server_class removed — module no longer exists."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from verl.workers.rollout.async_server import (
                async_server_class,  # noqa: F401
            )

    def test_old_timer_path_broken(self):
        """A1: old _timer path no longer re-exports _timer."""
        with pytest.raises(ImportError):
            from verl.utils.debug.performance import _timer  # noqa: F401

    def test_old_vllm_utils_path_broken(self):
        """A2: old vllm_utils module no longer exists."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from verl.utils.vllm_utils import is_version_ge  # noqa: F401


# ---------------------------------------------------------------------------
# Category B: Custom worker (verify our subclass has needed methods)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCustomAsyncWorker:
    """B1: Verify our custom AsyncActorRolloutRefWorker has dispatch methods."""

    def test_import_custom_worker(self):
        from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker

        assert AsyncActorRolloutRefWorker is not None

    def test_has_execute_method(self):
        from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker

        assert hasattr(AsyncActorRolloutRefWorker, 'execute_method')

    def test_has_chat_completion(self):
        from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker

        assert hasattr(AsyncActorRolloutRefWorker, 'chat_completion')

    def test_has_wake_up(self):
        from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker

        assert hasattr(AsyncActorRolloutRefWorker, 'wake_up')

    def test_has_sleep(self):
        from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker

        assert hasattr(AsyncActorRolloutRefWorker, 'sleep')

    def test_inherits_from_upstream(self):
        from verl.workers.fsdp_workers import (
            AsyncActorRolloutRefWorker as UpstreamWorker,
        )
        from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker

        assert issubclass(AsyncActorRolloutRefWorker, UpstreamWorker)

    def test_has_update_weights(self):
        """Inherited from upstream — should still exist."""
        from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker

        assert hasattr(AsyncActorRolloutRefWorker, 'update_weights')
