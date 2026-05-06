# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.checkpoint.checkpoint_manager import (
    find_latest_ckpt_path,
)
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.profiler.performance import simple_timer as _timer
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from verl_custom.nvidia.reward_manager.length_penalty import LengthPenalty
from verl_custom.nvidia.utils.timer import TimeoutChecker

# S2 cut: trajectory_store and continuous_producer removed; LiveStoreClient is the only path.
TrajectoryStore = None  # type: ignore[assignment,misc]
InsufficientTrajectoriesError = Exception  # type: ignore[assignment,misc]
from verl_custom.trainer.ppo import core_algos
from verl_custom.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl_custom.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_pass_k_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)

_logger = logging.getLogger(__name__)

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get('GPU', 0)
            if 'GPU' in node_info
            else node_info.get('NPU', 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [
                n_gpus
                for process_on_nodes in self.resource_pool_spec.values()
                for n_gpus in process_on_nodes
            ]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f'Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}'
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f'Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}'
                    + 'cannot be satisfied in this ray cluster'
                )


def apply_kl_penalty(
    data: DataProto,
    kl_ctrl: core_algos.AdaptiveKLController,
    kl_penalty='kl',
    multi_turn=False,
):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_rewards']
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch['loss_mask']
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch['old_log_probs'], data.batch['ref_log_prob'], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {
        'actor/reward_kl_penalty': current_kl,
        'actor/reward_kl_penalty_coeff': beta,
    }

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if 'response_mask' not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch['token_level_rewards'],
            values=data.batch['values'],
            response_mask=data.batch['response_mask'],
            gamma=gamma,
            lam=lam,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch['response_mask']
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch['loss_mask'][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch['uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            'token_level_rewards': data.batch['token_level_rewards'],
            'response_mask': data.batch['response_mask'],
            'config': config,
        }
        if 'uid' in data.non_tensor_batch:  # optional
            adv_kwargs['index'] = data.non_tensor_batch['uid']
        if 'reward_baselines' in data.batch:  # optional
            adv_kwargs['reward_baselines'] = data.batch['reward_baselines']

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name='cuda',
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.timeout = TimeoutChecker()

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.length_penalty = LengthPenalty(config.get('length_penalty', None))
        ray.get(self.reward_fn.set_length_penalty.remote(self.length_penalty))
        ray.get(
            self.reward_fn.set_stop_properly_penalty.remote(
                self.config.stop_properly_penalty.penalty_coef
            )
        )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f'{role_worker_mapping.keys()=}'
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # LoRA weight-sync bookkeeping. `policy_version` is trainer-authoritative
        # and only advances after every endpoint in
        # `actor_rollout_ref.rollout.external_llm_endpoints` ACKs /reload_lora
        # with 200. `_last_publish_step` tracks the trainer step at which the
        # last successful publish happened, used for `rollout/staleness_steps`.
        # See plans-n-solutions/stages/weight_sync_lora.md §5.
        self.policy_version = 0
        self._last_publish_step = 0
        self._last_publish_metrics: dict = {}

        # Phase 2 replay buffer. None when `config.replay.enable=False`, so
        # Cut 2 is a strict no-op for existing Phase 1 runs. When enabled,
        # the store holds full GRPO groups keyed by `uid` and re-emits a
        # sampled mini-batch at the push+sample seam in `fit()`. See
        # `plans-n-solutions/stages/full_async.md`.
        replay_cfg = config.get('replay', None)
        if replay_cfg is not None and replay_cfg.get('enable', False):
            pad_token_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            # Replay-store length caps mirror the rollout-side packer
            # contract, NOT the dataset-side data.max_*_length values:
            #   prompt_length_cap   = rollout's max_starting_message_length
            #       (width of the seed slot the rollout left-pads into;
            #       empirical SWE-Gym cap on system + dataset_instance prompt).
            #   response_length_cap = rollout's total_len
            #       = max_prompt_length + max_response_length
            #       (vLLM-bounded ceiling for accumulated response_ids across
            #       turns — vLLM enforces seed + body <= max_model_len, which
            #       is total_len).
            # data.max_prompt_length is a dataset filter + addend in total_len;
            # it never reaches the rollout or the store directly. Wiring the
            # caps to data.max_*_length silently right-truncates trajectories
            # whose body exceeds max_response_length, breaking reward<->loss
            # alignment.
            max_starting_message_length = int(
                config.actor_rollout_ref.rollout.get('max_starting_message_length', 0)
            )
            total_len = int(config.data.get('max_prompt_length', 0)) + int(
                config.data.get('max_response_length', 0)
            )
            # Prefer env var; fall back to Hydra config value if present.
            _live_store_socket = os.environ.get('LIVE_STORE_SOCKET', '') or str(
                replay_cfg.get('live_store_socket', '')
            )
            if _live_store_socket:
                # S2 migration: use external gRPC LiveStore instead of
                # in-process TrajectoryStore (BC-15).
                from rollout_fabric.live_store.client import (
                    LiveStoreClient,  # noqa: PLC0415
                )

                _policy_id = str(
                    config.actor_rollout_ref.model.get('policy_id', 'qwen3-4b-skyrl')
                )
                _env_id = str(
                    config.actor_rollout_ref.rollout.get('environment_id', 'swe_agent')
                )
                _logger.info(
                    '[trainer] LIVE_STORE_SOCKET=%s — using LiveStoreClient '
                    '(policy_id=%s, environment_id=%s)',
                    _live_store_socket,
                    _policy_id,
                    _env_id,
                )
                self.trajectory_store: TrajectoryStore | None = LiveStoreClient(  # type: ignore[assignment]
                    socket_path=_live_store_socket,
                    policy_id=_policy_id,
                    environment_id=_env_id,
                    pad_token_id=int(pad_token_id),
                    prompt_length_cap=max_starting_message_length or None,
                    response_length_cap=total_len or None,
                )
            else:
                self.trajectory_store = TrajectoryStore(
                    max_size=int(replay_cfg.buffer_size),
                    staleness_cutoff_k=int(replay_cfg.staleness_cutoff_k),
                    pad_token_id=int(pad_token_id),
                    prompt_length_cap=max_starting_message_length or None,
                    response_length_cap=total_len or None,
                )
        else:
            self.trajectory_store = None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                config.algorithm.kl_ctrl
            )

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == 'megatron':
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus
                % (
                    model_parallel_size
                    * config.actor_rollout_ref.actor.megatron.context_parallel_size
                )
                == 0
            ), (
                f'n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})'
            )
            megatron_dp = n_gpus // (
                model_parallel_size
                * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = (
                megatron_dp
                * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
            )
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = (
            config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        )
        assert real_train_batch_size % minimal_bsz == 0, (
            f'real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})'
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                'actor_rollout_ref.actor': 'micro_batch_size',
                'critic': 'micro_batch_size',
                'reward_model': 'micro_batch_size',
                'actor_rollout_ref.ref': 'log_prob_micro_batch_size',
                'actor_rollout_ref.rollout': 'log_prob_micro_batch_size',
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f'{param}_per_gpu'

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + 'is supported (the former is deprecated).'
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                'actor_rollout_ref.actor',
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    'actor_rollout_ref.ref',
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                'actor_rollout_ref.rollout',
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu,
                'critic',
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu,
                'reward_model',
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert (
                config.data.train_batch_size
                >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            )
            sp_size = config.actor_rollout_ref.actor.get(
                'ulysses_sequence_parallel_size', 1
            )
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert (
                    config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size
                    >= n_gpus
                )

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            'token-mean',
            'seq-mean-token-sum',
            'seq-mean-token-mean',
            'seq-mean-token-sum-norm',
        ], f'Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}'

        if (
            config.algorithm.use_kl_in_reward
            and config.actor_rollout_ref.actor.use_kl_loss
        ):
            print('NOTICE: You have both enabled in-reward kl and kl loss.')

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert (
                    config.critic.ppo_mini_batch_size
                    % config.critic.ppo_micro_batch_size
                    == 0
                )
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp' and (
            config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1
            or config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                'When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`.'
            )

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    'When using sequence parallelism for critic, you must enable `use_remove_padding`.'
                )

        if config.data.get('val_batch_size', None) is not None:
            print(
                'WARNING: val_batch_size is deprecated.'
                + ' Validation datasets are sent to inference engines as a whole batch,'
                + ' which will schedule the memory themselves.'
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                'validation gen temperature should be greater than 0 when enabling do_sample'
            )

        # check multi_turn with tool config
        if (
            config.actor_rollout_ref.rollout.multi_turn.enable
            and not config.actor_rollout_ref.rollout.multi_turn.get('agent', False)
        ):
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
            ), (
                'tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support'
            )
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                'only GRPO is tested for multi-turn with tool'
            )

        print('[validate_config] All configuration checks passed successfully!')

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """BC-14 + BC-15: trainer is a pure LiveStore consumer.

        All training data arrives via LiveStoreClient.get_batch(); no parquet
        DataLoader is created here. RolloutManager owns the dataset (BC-14).
        """
        self.train_dataset = self.val_dataset = None
        self.train_dataloader = self.val_dataloader = None

        # total_training_steps from explicit config (mandatory in LiveStore mode).
        total_training_steps = self.config.trainer.total_training_steps
        if total_training_steps is None:
            raise ValueError(
                'trainer.total_training_steps must be set explicitly in LiveStore mode '
                '(no DataLoader means it cannot be derived from dataset length × epochs).'
            )
        self.total_training_steps = int(total_training_steps)
        print(f'Total training steps: {self.total_training_steps}')

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, 'actor_rollout_ref.actor.optim'):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        self.total_training_steps
                    )
                if OmegaConf.select(self.config, 'critic.optim'):
                    self.config.critic.optim.total_training_steps = (
                        self.total_training_steps
                    )
        except Exception as e:
            print(f'Warning: Could not set total_training_steps in config. Error: {e}')

    def _dump_generations(
        self, inputs, outputs, scores, reward_extra_infos_dict, dump_path
    ):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f'{self.global_steps}.jsonl')

        n = len(inputs)
        base_data = {
            'input': inputs,
            'output': outputs,
            'score': scores,
            'step': [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, 'w') as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f'Dumped generations to {filename}')

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(
            self.config.trainer.logger, samples, self.global_steps
        )

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model'
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']

            batch_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
            if self.config.actor_rollout_ref.rollout.get('task_type', None) == 'swegym':
                non_tensor_batch_keys_to_pop = ['instance']
            else:
                non_tensor_batch_keys_to_pop = ['raw_prompt_ids']
            if 'multi_modal_data' in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append('multi_modal_data')
            if 'raw_prompt' in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append('raw_prompt')
            if 'tools_kwargs' in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append('tools_kwargs')
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')
            # pad to be divisible by dp_size
            test_gen_batch, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )

            if self.async_rollout_manager is not None:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch = self.async_rollout_manager.generate_sequences(
                    test_gen_batch, val_mode=True
                )
                self.async_rollout_manager.sleep()
            else:
                test_output_gen_batch = self.actor_rollout_wg.generate_sequences(
                    test_gen_batch
                )

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch, pad_size=pad_size
            )
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [
                [
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    for i in range(self.config.actor_rollout_ref.rollout.val_kwargs.n)
                ]
                for ids in input_ids
            ]
            input_texts = [item for sublist in input_texts for item in sublist]
            sample_inputs.extend(input_texts)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = ray.get(self.val_reward_fn.__call__.remote(test_batch))
            reward_tensor = result['score']
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict['reward'].extend(scores)
            if 'reward_extra_info' in result:
                for key, lst in result['reward_extra_info'].items():
                    reward_extra_infos_dict[key].extend(lst)

            # Log reward_metrics if available during validation
            if 'reward_metrics' in result:
                val_reward_metrics = result['reward_metrics']
                if isinstance(val_reward_metrics, dict):
                    for key, value in val_reward_metrics.items():
                        # Use the same key format as in training
                        metric_key = f'reward_metrics/{key}'
                        if metric_key not in reward_extra_infos_dict:
                            reward_extra_infos_dict[metric_key] = []
                        reward_extra_infos_dict[metric_key].extend(
                            [value] * len(scores)
                        )
                else:
                    print(
                        f'Warning: validation reward_metrics is not a dict, got {type(val_reward_metrics)}'
                    )

            data_source = test_batch.non_tensor_batch.get(
                'data_source', ['unknown'] * reward_tensor.shape[0]
            )
            for idx, x in enumerate(test_batch.non_tensor_batch.get('extra_info', [])):
                if 'name' in x:
                    data_source[idx] = x['name']
            data_source_lst.append(data_source)

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        # dump generations
        val_data_dir = self.config.trainer.get('validation_data_dir', None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (
                f'{key_info}: {len(lst)=}, {len(sample_scores)=}'
            )

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(
            data_sources, sample_inputs, reward_extra_infos_dict
        )
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = 'acc' if 'acc' in var2metric2val else 'reward'
            for var_name, metric2val in var2metric2val.items():
                n_max = max(
                    [
                        int(name.split('@')[-1].split('/')[0])
                        for name in metric2val.keys()
                    ]
                )
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(
                            metric_name.startswith(pfx)
                            for pfx in ['mean', 'maj', 'best']
                        )
                        and (f'@{n_max}' in metric_name)
                    ):
                        metric_sec = 'val-core'
                    else:
                        metric_sec = 'val-aux'
                    pfx = f'{metric_sec}/{data_source}/{var_name}/{metric_name}'
                    metric_dict[pfx] = metric_val

        # ==================== Pass@k Calculation ====================
        # Compute pass@k metrics (if enabled)
        if getattr(self.config.trainer, 'enable_pass_k_evaluation', False):
            print('Computing pass@k metrics...')

            # Get k values list from configuration
            k_values = getattr(self.config.trainer, 'pass_k_values', [1, 4, 8, 16])

            # Build problem statistics information
            problem_results = defaultdict(lambda: {'total': 0, 'correct': 0})

            # Build problem ID identification strategy
            problem_id_strategy = getattr(
                self.config.trainer, 'pass_k_problem_id_strategy', 'input_hash'
            )

            # Assign problem ID for each sample
            for i, (input_text, output_text, score) in enumerate(
                zip(sample_inputs, sample_outputs, sample_scores)
            ):
                # Determine problem ID based on configured strategy
                if problem_id_strategy == 'input_hash':
                    # Use hash of input text as problem identifier
                    problem_id = abs(hash(input_text)) % 10000
                elif problem_id_strategy == 'input_prefix':
                    # Use first 50 characters of input text as problem identifier
                    problem_id = input_text[:50].strip()
                elif problem_id_strategy == 'index_based':
                    # Index-based strategy, assuming every N samples are for the same problem
                    samples_per_problem = getattr(
                        self.config.trainer, 'pass_k_samples_per_problem', 10
                    )
                    problem_id = i // samples_per_problem
                else:
                    # Default to using index
                    problem_id = i

                problem_results[problem_id]['total'] += 1

                # Determine if test passes
                success_threshold = getattr(
                    self.config.trainer, 'pass_k_success_threshold', 0.0
                )
                success_mode = getattr(
                    self.config.trainer, 'pass_k_success_mode', 'threshold'
                )

                is_success = False
                if success_mode == 'threshold':
                    # Judge based on score threshold
                    is_success = score > success_threshold
                elif success_mode == 'positive':
                    # Positive numbers indicate success
                    is_success = score > 0
                elif success_mode == 'percentile':
                    # Judge based on percentile (requires pre-computing all scores)
                    percentile_threshold = getattr(
                        self.config.trainer, 'pass_k_success_percentile', 50
                    )
                    threshold_score = np.percentile(sample_scores, percentile_threshold)
                    is_success = score >= threshold_score
                else:
                    # Default to threshold mode
                    is_success = score > success_threshold

                if is_success:
                    problem_results[problem_id]['correct'] += 1
            # Print problem results
            for problem_id, result in problem_results.items():
                print('=' * 50)
                print('Here is the problem results:')
                print(
                    f'Problem {problem_id}: {result["total"]} total, {result["correct"]} correct'
                )
                print('=' * 50)
            # Compute pass@k metrics
            pass_k_results = compute_pass_k_metrics(dict(problem_results), k_values)

            # Add pass@k results to metric_dict
            for k_metric, k_result in pass_k_results.items():
                metric_dict[f'val-core/pass_k/{k_metric}/score'] = k_result['score']
                metric_dict[f'val-aux/pass_k/{k_metric}/std'] = k_result['std']
                metric_dict[f'val-aux/pass_k/{k_metric}/num_problems'] = k_result[
                    'num_problems'
                ]

            print(
                f'Pass@k evaluation completed. Evaluated {len(problem_results)} unique problems.'
            )

            # Print pass@k results
            print('\n' + '=' * 50)
            print('PASS@K VALIDATION RESULTS')
            print('=' * 50)
            for k in sorted(k_values):
                if f'pass@{k}' in pass_k_results:
                    result = pass_k_results[f'pass@{k}']
                    score = result['score']
                    std = result['std']
                    num_problems = result['num_problems']
                    print(f'pass@{k:3d}: {score:.4f} ± {std:.4f} (n={num_problems})')
            print('=' * 50)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role='actor_rollout',
            )
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = (
                actor_rollout_cls
            )
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role='ref',
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if (
            OmegaConf.select(self.config.trainer, 'ray_wait_register_center_timeout')
            is not None
        ):
            wg_kwargs['ray_wait_register_center_timeout'] = (
                self.config.trainer.ray_wait_register_center_timeout
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        self.async_rollout_manager = None
        if self.config.actor_rollout_ref.rollout.mode == 'async':
            _live_store_socket = os.environ.get('LIVE_STORE_SOCKET', '') or str(
                self.config.replay.get('live_store_socket', '')
                if hasattr(self.config, 'replay')
                else ''
            )
            if _live_store_socket:
                # LiveStore mode (BC-15): all rollout generation is handled
                # externally by RolloutManager. The FSDP actor still computes
                # log_probs via compute_log_prob, but the vLLM engine inside
                # VLLMAsyncRolloutCompat is NEVER initialized
                # (inference_engine stays None). No AsyncLLMServerManager
                # is created, so no GPU memory is consumed by a sleeping vLLM.
                self.async_rollout_mode = True
                # async_rollout_manager deliberately left None.

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f'global_step_{self.global_steps}'
        )

        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f'global_step_{self.global_steps}',
                'actor',
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get(
            'remove_previous_ckpt_in_save', False
        )
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated,'
                + ' set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get('max_actor_ckpt_to_keep', None)
            if not remove_previous_ckpt_in_save
            else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get('max_critic_ckpt_to_keep', None)
            if not remove_previous_ckpt_in_save
            else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f'global_step_{self.global_steps}',
                    'critic',
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        # save dataloader
        from verl.utils.fs import local_mkdir_safe  # noqa: PLC0415

        local_mkdir_safe(local_global_step_folder)
        # In LiveStore mode train_dataloader is None — skip dataloader checkpoint.
        if self.train_dataloader is not None:
            dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, 'latest_checkpointed_iteration.txt'
        )
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _wait_for_publish_headroom(
        self,
        endpoints: list[str],
        max_concurrent_versions: int = 6,
        max_wait_s: float = 30.0,
        poll_interval_s: float = 0.5,
    ) -> dict:
        """Pre-publish backpressure: defer until in-flight LoRA versions drain.

        In path-versioned pinning mode (see scripts/serving/_vllm_child.py),
        publishing a new adapter never evicts in-flight versions — they keep
        serving requests against their pinned LoRARequest. But the vLLM engine
        only has ``--max-loras`` GPU slots; if too many distinct versions are
        in-flight at once, the engine LRU-evicts under pressure and the next
        generate against an evicted slot pays an extra page-in latency.

        Strategy: poll each endpoint's /health, pull
        ``inflight_versions_count``, and if any child reports more than
        ``max_concurrent_versions`` distinct in-flight versions, sleep and
        retry. After ``max_wait_s`` we proceed anyway (deferring further
        would stall training); the metric below records the wait so it's
        observable in WandB.

        Returns metrics for emit-with-publish_lora.
        """
        import time  # noqa: PLC0415

        import requests  # noqa: PLC0415

        t_start = time.monotonic()
        deferred = 0
        last_max = 0
        while True:
            max_inflight_versions = 0
            for ep in endpoints:
                try:
                    r = requests.get(ep.rstrip('/') + '/health', timeout=5)
                    if r.status_code != 200:
                        # Pool health endpoint should always return 200; if it
                        # doesn't we abort the wait and let _publish_lora_adapter
                        # surface the real error.
                        return {
                            'weight_sync/publish_deferred_count': deferred,
                            'weight_sync/publish_deferred_wait_s': round(
                                time.monotonic() - t_start, 3
                            ),
                            'weight_sync/publish_max_inflight_versions': last_max,
                        }
                    body = r.json()
                    n = int(body.get('inflight_versions_count', 0))
                    if n > max_inflight_versions:
                        max_inflight_versions = n
                except (requests.RequestException, ValueError):
                    # Health probe failure: don't block the publish on it.
                    return {
                        'weight_sync/publish_deferred_count': deferred,
                        'weight_sync/publish_deferred_wait_s': round(
                            time.monotonic() - t_start, 3
                        ),
                        'weight_sync/publish_max_inflight_versions': last_max,
                    }

            last_max = max_inflight_versions
            if max_inflight_versions <= max_concurrent_versions:
                return {
                    'weight_sync/publish_deferred_count': deferred,
                    'weight_sync/publish_deferred_wait_s': round(
                        time.monotonic() - t_start, 3
                    ),
                    'weight_sync/publish_max_inflight_versions': last_max,
                }

            if time.monotonic() - t_start >= max_wait_s:
                # Proceeding anyway. The publish will still succeed; the LRU
                # may briefly evict an in-flight slot under pressure.
                print(
                    json.dumps(
                        {
                            'event': 'publish_deferred_capped',
                            'inflight_versions': last_max,
                            'max_concurrent_versions': max_concurrent_versions,
                            'waited_s': round(time.monotonic() - t_start, 3),
                        }
                    ),
                    flush=True,
                )
                return {
                    'weight_sync/publish_deferred_count': deferred,
                    'weight_sync/publish_deferred_wait_s': round(
                        time.monotonic() - t_start, 3
                    ),
                    'weight_sync/publish_max_inflight_versions': last_max,
                }

            deferred += 1
            time.sleep(poll_interval_s)

    def _publish_lora_adapter(self, local_global_step_folder: str) -> None:
        """S4 — delegate LoRA fanout to the PolicyRegistry (single source of truth).

        The PolicyRegistry (slot 5.7, ``policy_registry/server.py``) receives the
        publish RPC, fans out ``/reload_lora`` to every vLLM pool child with the
        §3.3 abort gate (``endpoints_failed > 0`` → ``PublishFailedError`` → trainer
        aborts), writes the S2 JSON manifest for the worker's
        ``FilePollingPolicySubscription``, and notifies any gRPC streaming
        subscribers. The trainer is a single-line caller.

        Path translation: the Docker container sees ``/workspace`` as the repo root;
        the PolicyRegistry service runs on the HOST and needs the host-visible path.
        ``REPO_HOST_PATH`` env var bridges this (set via ``-e`` in
        ``s3_fullasync_docker.sh``; defaults to ``/workspace`` for host-native runs).
        """
        import time  # noqa: PLC0415

        from rollout_fabric.policy_registry.client import (
            PolicyRegistryClient,  # noqa: PLC0415
            PublishFailedError,  # noqa: PLC0415
        )

        adapter_dir = os.path.join(local_global_step_folder, 'actor', 'lora_adapter')
        required = ('adapter_model.safetensors', 'adapter_config.json')
        missing = [
            f for f in required if not os.path.isfile(os.path.join(adapter_dir, f))
        ]
        if missing:
            raise RuntimeError(
                f'PEFT adapter missing from checkpoint at {adapter_dir}: {missing}. '
                "verl's _is_lora save path (fsdp_workers.py) did not emit shards."
            )

        new_version = self.policy_version + 1

        # Path translation: Docker container sees /workspace; PolicyRegistry
        # runs on the HOST and needs the host-visible path to load the adapter.
        # REPO_HOST_PATH is injected via -e in s3_fullasync_docker.sh.
        repo_host_path = os.environ.get('REPO_HOST_PATH', '/workspace')
        host_adapter_dir = adapter_dir.replace('/workspace', repo_host_path, 1)
        adapter_uri = f'file://{host_adapter_dir}'

        _policy_id = str(
            self.config.actor_rollout_ref.model.get('policy_id', 'qwen3-4b-skyrl')
        )
        _socket = str(
            self.config.replay.get(
                'policy_registry_socket', '/tmp/prorl_policy_registry.sock'
            )
        )

        t0 = time.monotonic()
        client = PolicyRegistryClient(_socket)
        try:
            # Single RPC: PolicyRegistry owns fanout, abort gate (§3.3),
            # manifest write, and subscriber notification. BC-9 is preserved:
            # PublishFailedError is raised if endpoints_failed > 0.
            publish_metrics = client.publish_policy_version(
                policy_id=_policy_id,
                version=new_version,
                adapter_uri=adapter_uri,
                trainer_id='trainer-0',
            )
        except PublishFailedError as exc:
            # §3.3 abort gate: any pool child failure → hard abort.
            raise RuntimeError(str(exc)) from exc
        finally:
            client.close()

        publish_latency_s = time.monotonic() - t0

        # Commit only after PolicyRegistry confirmed all endpoints ACKed.
        self.policy_version = new_version
        self._last_publish_step = self.global_steps
        if getattr(self, 'async_rollout_manager', None) is not None:
            self.async_rollout_manager.policy_version = new_version

        self._last_publish_metrics = {
            'weight_sync/policy_version': new_version,
            'weight_sync/publish_latency_s': publish_latency_s,
            **{
                k: v for k, v in publish_metrics.items() if k.startswith('weight_sync/')
            },
        }

        print(
            json.dumps(
                {
                    'event': 'publish_lora_adapter',
                    'policy_version': new_version,
                    'adapter_uri': adapter_uri,
                    'publish_latency_s': round(publish_latency_s, 3),
                    **{
                        k.replace('weight_sync/', ''): v
                        for k, v in publish_metrics.items()
                        if k.startswith('weight_sync/')
                    },
                },
                separators=(',', ':'),
            )
        )

    def _load_checkpoint(self):
        # Sleep and wake up the async rollout manager: https://github.com/volcengine/verl/issues/2613
        # This syncs weights to vllm server. Also release GPU memory.
        # In LiveStore mode async_rollout_manager is None (no local vLLM).
        if self.async_rollout_mode and self.async_rollout_manager is not None:
            self.async_rollout_manager.sleep()

        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = (
                self.config.trainer.default_local_dir
            )  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == 'resume_path':
                assert isinstance(self.config.trainer.resume_from_path, str), (
                    'resume ckpt must be str type'
                )
                assert 'global_step_' in self.config.trainer.resume_from_path, (
                    'resume ckpt must specify the global_steps'
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path) and self.train_dataloader is not None:
            dataloader_state_dict = torch.load(
                dataloader_local_path, weights_only=False
            )
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(
                f'Warning: No dataloader state found at {dataloader_local_path}, will start from scratch'
            )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def _push_and_sample_replay(self, batch: DataProto, metrics: dict) -> DataProto:
        """Push the freshly-generated batch through the replay store and sample.

        This is a no-op when ``config.replay.enable`` is False (store is
        None). When enabled:

        1. Every group (``n`` sibling rollouts per prompt, identified by
           ``uid``) is pushed into the bounded FIFO store tagged with the
           current behavior ``policy_version`` and ``global_steps``.
        2. We immediately sample ``n_groups`` groups back out; with
           ``buffer_size`` >= ``n_groups`` and
           ``staleness_cutoff_k`` >= 1 this is the Cut-2 lockstep pass-
           through mode.
        3. The sampled :class:`SampledMiniBatch` is wrapped in a fresh
           :class:`DataProto`. Downstream stages recompute
           ``response_mask`` / reward / ``old_log_prob`` / advantage on
           the sampled batch, which keeps this seam compatible with Cut-4
           continuous-producer mode where ``rollout_log_probs`` is the
           stored behavior value and ``old_log_prob`` is the current-
           policy value — making the existing TIS ratio at
           ``core_algos.py:586-590`` temporal by construction.

        If the store is warming up and has fewer than ``n_groups`` non-
        stale groups, we fall back to the just-pushed ``batch`` unchanged
        so the trainer never stalls. In Cut 2 lockstep the store always
        holds at least the groups we pushed this step.
        """
        if self.trajectory_store is None:
            return batch

        self.trajectory_store.push_from_dataproto(
            batch,
            behavior_policy_version=self.policy_version,
            current_step=self.global_steps,
        )

        n = int(self.config.actor_rollout_ref.rollout.n)
        n_groups = max(1, len(batch.batch) // n)

        metrics.update(
            self.trajectory_store.metrics(self.global_steps, suffix='_pre_sample')
        )
        try:
            from verl_custom.fabric_adapter.live_store_batch import (
                sample_mini_batch,  # noqa: PLC0415
            )

            sampled = sample_mini_batch(
                self.trajectory_store,
                n_groups=n_groups,
                current_step=self.global_steps,
            )
        except InsufficientTrajectoriesError:
            metrics.update(self.trajectory_store.metrics(self.global_steps))
            metrics.update(
                self.trajectory_store.metrics(self.global_steps, suffix='_post_sample')
            )
            return batch

        new_batch = DataProto.from_dict(
            tensors=sampled.tensors,
            non_tensors=sampled.non_tensors,
            meta_info=sampled.meta_info,
        )
        metrics.update(self.trajectory_store.metrics(self.global_steps))
        metrics.update(
            self.trajectory_store.metrics(self.global_steps, suffix='_post_sample')
        )
        return new_batch

    # ---- Cut 4: continuous-rollout producer --------------------------------

    def _continuous_producer_mode(self) -> bool:
        replay_cfg = self.config.get('replay', None)
        if not replay_cfg or not replay_cfg.get('enable', False):
            return False
        return bool(replay_cfg.get('continuous_producer', False))

    def _build_grpo_producer_generate_fn(self):
        """Closure that turns a dataloader batch_dict into a full merged DataProto.

        Mirrors the classic in-line path (pop prompt cols → generate →
        stamp ``uid`` → repeat by ``n`` → ``union`` responses). Runs in
        the producer thread; returns the same shape the store expects.
        """
        from verl_custom.trainer.ppo.ray_trainer import (
            AdvantageEstimator,  # noqa: PLC0415
        )

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            # REMAX needs the baseline-gen pass on the same ``gen_batch``;
            # it cannot be reconstructed from the store. Fail fast.
            raise RuntimeError(
                'replay.continuous_producer=True is incompatible with '
                'algorithm.adv_estimator=REMAX (requires inline gen_batch).'
            )

        rollout_manager = self.async_rollout_manager
        rollout_cfg = self.config.actor_rollout_ref.rollout
        task_type = rollout_cfg.get('task_type', None)
        n = int(rollout_cfg.n)

        def _generate(batch_dict: dict) -> DataProto:
            full_batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
            if task_type == 'swegym':
                non_tensor_batch_keys_to_pop = ['instance']
            else:
                non_tensor_batch_keys_to_pop = ['raw_prompt_ids']
            if 'multi_modal_data' in full_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append('multi_modal_data')
            if 'raw_prompt' in full_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append('raw_prompt')
            if 'tools_kwargs' in full_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append('tools_kwargs')
            gen_batch = full_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            gen_batch_output = rollout_manager.generate_sequences(gen_batch)
            # Producer-local timings are not propagated into the trainer's
            # per-step ``timing_raw``; drop to avoid leaking the field.
            gen_batch_output.meta_info.pop('timing', None)
            full_batch.non_tensor_batch['uid'] = np.array(
                [str(uuid.uuid4()) for _ in range(len(full_batch.batch))],
                dtype=object,
            )
            full_batch = full_batch.repeat(repeat_times=n, interleave=True)
            full_batch = full_batch.union(gen_batch_output)
            return full_batch

        return _generate

    def _start_continuous_producer_if_needed(self) -> None:
        """Wake the pool and spin up the rollout-producer daemon thread.

        No-op unless ``config.replay.enable`` and
        ``config.replay.continuous_producer`` are both True. Requires
        ``actor_rollout_ref.rollout.mode == 'async'`` — the producer
        relies on ``async_rollout_manager.generate_sequences``.

        Subclasses override :meth:`_make_continuous_producer` to swap
        the ``generate_fn`` (DAPO uses ``generate_sequences_dapo``).
        """
        self._producer = None
        self._step_counter = None
        if not self._continuous_producer_mode():
            return
        raise NotImplementedError(
            'S2 cut: continuous_producer mode removed; use LiveStoreClient only. '
            'Do not set replay.continuous_producer=True.'
        )

    def _make_continuous_producer(self):
        """Build a ``ContinuousRolloutProducer`` for plain GRPO.

        Uses ``generate_sequences`` over a private dataloader iterator.
        Override in subclasses (e.g., ``RayPPOTrainerDAPO``) to wire a
        different ``generate_fn`` / ``prompts_iter_factory``.
        """
        raise NotImplementedError(
            'S2 cut: continuous_producer removed; use LiveStoreClient only.'
        )

    def _stop_continuous_producer_if_needed(self) -> bool:
        """Stop the producer and drop the reference on clean exit.

        Drains unconditionally — ``producer.stop()`` blocks until the
        current ``generate_sequences`` call completes naturally, so the
        publish boundary is aligned to producer-batch completion. No
        finite timeout: a timeout that fires mid-batch would leave an
        orphan thread running across the publish, causing a
        mid-trajectory policy-version switch.

        Always returns ``True`` once stop returns. Genuine wedges
        (vLLM/OpenHands hung) are caught by the trainer's no-progress
        detector at the next ``_acquire_training_batch`` call
        (``replay.no_progress_timeout_s``).
        """
        producer = getattr(self, '_producer', None)
        if producer is None:
            return True
        stopped = producer.stop()
        if stopped:
            self._producer = None
            # Cut 5: drop the eager-push closure so the manager goes
            # back to lockstep/terminal-push semantics in any subsequent
            # classic path. Preserved only while a producer is active.
            if hasattr(self.async_rollout_manager, '_push_fn'):
                self.async_rollout_manager._push_fn = None
        return stopped

    def _acquire_training_batch(
        self, batch_dict: dict, metrics: dict, timing_raw: dict
    ) -> DataProto:
        """Return a fully-merged training batch for the current step.

        In continuous-producer mode, waits for the store to hold
        ``train_batch_size // n`` non-stale groups, then samples. The
        producer owns ``wake_up``/``sleep`` and inline ``generate_sequences``
        so the trainer never blocks on rollout tails. ``batch_dict`` is
        intentionally ignored — the producer iterates its own dataloader.

        In classic mode, performs the existing pop → generate → stamp
        uid → repeat → union → push+sample pipeline.
        """
        if self._producer is not None:
            # S2 cut: this branch is unreachable — _producer is always None.
            raise NotImplementedError(
                'S2 cut: continuous_producer removed; use LiveStoreClient only.'
            )

        # Classic path (original fit() gen block).
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        batch_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
        if self.config.actor_rollout_ref.rollout.get('task_type', None) == 'swegym':
            non_tensor_batch_keys_to_pop = ['instance']
        else:
            non_tensor_batch_keys_to_pop = ['raw_prompt_ids']
        if 'multi_modal_data' in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append('multi_modal_data')
        if 'raw_prompt' in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append('raw_prompt')
        if 'tools_kwargs' in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append('tools_kwargs')
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        with _timer('gen', timing_raw):
            if self.async_rollout_manager is not None:
                self.async_rollout_manager.wake_up()
                gen_batch_output = self.async_rollout_manager.generate_sequences(
                    gen_batch
                )
                self.async_rollout_manager.sleep()
            else:
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            timing_raw.update(gen_batch_output.meta_info['timing'])
            gen_batch_output.meta_info.pop('timing', None)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            with _timer('gen_max', timing_raw):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info['do_sample'] = False
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(
                    gen_baseline_batch
                )
                batch = batch.union(gen_baseline_output)
                reward_baseline_tensor = self.reward_fn(batch)
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                batch.batch['reward_baselines'] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

        batch.non_tensor_batch['uid'] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
            dtype=object,
        )
        batch = batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.n,
            interleave=True,
        )
        batch = batch.union(gen_batch_output)
        batch = self._push_and_sample_replay(batch, metrics)
        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        if self.global_steps > 0:
            # Pool keeps its PV across a trainer restart; align trainer to it so
            # the first post-resume /reload_lora isn't rejected as non-monotonic.
            self.policy_version = self.global_steps
            if self.async_rollout_manager is not None:
                self.async_rollout_manager.policy_version = self.global_steps

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            'val_before_train', True
        ):
            val_metrics = self._validate()
            assert val_metrics, f'{val_metrics=}'
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc='Training Progress',
        )

        # we start from step 1
        self.global_steps += 1

        # Cut 4: spin up the continuous rollout producer when
        # ``replay.continuous_producer=True``. No-op otherwise. Wrapped in
        # try/finally so an exception inside the training loop still stops
        # the daemon thread and releases the pool (``sleep()``).
        self._start_continuous_producer_if_needed()
        try:
            self._run_fit_loop(logger, progress_bar)
        finally:
            self._stop_continuous_producer_if_needed()

    def _run_fit_loop(self, logger, progress_bar):
        last_val_metrics = None
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # Cut 4: gen + push + sample is encapsulated here. In
                    # classic mode this runs inline ``generate_sequences``
                    # and push-then-samples the store. In continuous-
                    # producer mode the daemon thread is already filling
                    # the store and this call just waits + samples.
                    batch = self._acquire_training_batch(
                        batch_dict, metrics, timing_raw
                    )

                    batch.batch['response_mask'] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(
                        batch.batch['attention_mask'], dim=-1
                    ).tolist()

                    with _timer('reward', timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # rule-based rm
                        future_reward = self.reward_fn.__call__.remote(batch)

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch['entropys']
                        response_masks = batch.batch['response_mask']
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_loss = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            'actor/entropy_loss': entropy_loss.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop('entropys')
                        batch = batch.union(old_log_prob)

                        if 'rollout_log_probs' in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch['rollout_log_probs']
                            actor_old_log_probs = batch.batch['old_log_probs']
                            attention_mask = batch.batch['attention_mask']
                            responses = batch.batch['responses']
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(
                                rollout_probs_diff, response_mask.bool()
                            )
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    'training/rollout_probs_diff_max': rollout_probs_diff_max.detach().item(),
                                    'training/rollout_probs_diff_mean': rollout_probs_diff_mean.detach().item(),
                                    'training/rollout_probs_diff_std': rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch
                                )
                            else:
                                ref_log_prob = (
                                    self.actor_rollout_wg.compute_ref_log_prob(batch)
                                )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # we combine with rule-based rm
                        reward_result = ray.get(future_reward)
                        reward_extra_infos_dict = {}

                        batch.batch['token_level_scores'] = reward_result['score']
                        batch.batch['token_level_rewards'] = reward_result['reward']

                        # Log reward_metrics if available
                        if 'reward_metrics' in reward_result:
                            reward_metrics = reward_result['reward_metrics']
                            if isinstance(reward_metrics, dict):
                                for key, value in reward_metrics.items():
                                    metrics[f'reward_metrics/{key}'] = value
                            else:
                                print(
                                    f'Warning: reward_metrics is not a dict, got {type(reward_metrics)}'
                                )

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {
                                    k: np.array(v)
                                    for k, v in reward_extra_infos_dict.items()
                                }
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            'norm_adv_by_std_in_grpo', True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info['metrics']
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            batch.meta_info['multi_turn'] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info['metrics']
                        )
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get('rollout_data_dir', None)
                    if rollout_data_dir:
                        with _timer('dump_rollout_generations', timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(
                                batch.batch['prompts'], skip_special_tokens=True
                            )
                            outputs = self.tokenizer.batch_decode(
                                batch.batch['responses'], skip_special_tokens=True
                            )
                            scores = (
                                batch.batch['token_level_scores'].sum(-1).cpu().tolist()
                            )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    did_save = False
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                        did_save = True
                    elif self.timeout.check_save():
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                        did_save = True

                    if (
                        did_save
                        and self.config.actor_rollout_ref.rollout.get(
                            'publish_on_save', False
                        )
                        and self.config.actor_rollout_ref.model.get('lora_rank', 0) > 0
                    ):
                        local_global_step_folder = os.path.join(
                            self.config.trainer.default_local_dir,
                            f'global_step_{self.global_steps}',
                        )
                        with _timer('publish_lora', timing_raw):
                            self._publish_lora_adapter(local_global_step_folder)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                            or self.timeout.last_saved
                        )
                    ):
                        # Producer + validation share one OpenHands session;
                        # validation's /stop at teardown kills the producer's
                        # in-flight /process. Pause producer around _validate
                        # and resume afterwards. Buffer stays warm across the
                        # pause; K-staleness evicts naturally at next sample.
                        # Gotcha §19: if the producer is mid-asyncio.run the
                        # stop() timeout fires without the thread exiting —
                        # skip validate to avoid concurrent OH dispatch and
                        # let the producer finish its call; retry on the next
                        # save boundary.
                        if self._stop_continuous_producer_if_needed():
                            try:
                                with _timer('testing', timing_raw):
                                    val_metrics: dict = self._validate()
                                    if is_last_step:
                                        last_val_metrics = val_metrics
                                metrics.update(val_metrics)
                            finally:
                                if not is_last_step:
                                    self._start_continuous_producer_if_needed()
                        else:
                            _logger.warning(
                                'step=%d skipping _validate: producer stop '
                                'timed out (still mid-generate_sequences); '
                                'will retry on next save boundary',
                                self.global_steps,
                            )

                # training metrics
                metrics.update(
                    {
                        'training/global_step': self.global_steps,
                        'training/epoch': epoch,
                    }
                )
                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                # LoRA weight-sync metrics: publish keys appear only on steps
                # where _publish_lora_adapter ran; staleness is logged every
                # step as "training steps since last successful publish"
                # (mixing global_steps with policy_version conflates ordinals
                # with cardinals). Before the first publish, staleness equals
                # the current step count.
                if self._last_publish_metrics:
                    metrics.update(self._last_publish_metrics)
                    self._last_publish_metrics = {}
                metrics['rollout/staleness_steps'] = max(
                    0, self.global_steps - self._last_publish_step
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                # Cut 4: keep the producer's step-tag current so records
                # pushed from now on carry the new ``created_at_step``.
                if self._step_counter is not None:
                    self._step_counter.set(self.global_steps)
                if self._producer is not None:
                    self._producer.check_background_error()
                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return
