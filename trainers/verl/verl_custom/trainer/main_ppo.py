# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl_custom.trainer.ppo.ray_trainer_dapo import RayPPOTrainerDAPO


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN',
                    'VLLM_LOGGING_LEVEL': 'WARN',
                    'VLLM_ALLOW_RUNTIME_LORA_UPDATING': 'true',
                    'VLLM_USE_V1': '1',
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get('timeline_json_file', None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get('use_shm', False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get('trust_remote_code', False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )

        if config.actor_rollout_ref.rollout.name in ['vllm']:
            from verl.utils.vllm import is_version_ge

            if config.actor_rollout_ref.model.get('lora_rank', 0) > 0:
                if not is_version_ge(pkg='vllm', minver='0.7.3'):
                    raise NotImplementedError(
                        'PPO LoRA is not supported before vllm 0.7.3'
                    )

        if config.actor_rollout_ref.actor.strategy in ['fsdp', 'fsdp2']:
            assert config.critic.strategy in ['fsdp', 'fsdp2']
            from verl.single_controller.ray import RayWorkerGroup

            # fsdp_workers was renamed to engine_workers in newer verl.
            try:
                from verl.workers.fsdp_workers import (
                    ActorRolloutRefWorker,
                    CriticWorker,
                )
            except (ImportError, ModuleNotFoundError):
                from verl.workers.engine_workers import ActorRolloutRefWorker

                CriticWorker = (
                    ActorRolloutRefWorker  # new verl: critic merged into actor
                )

            from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == 'async'
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            raise NotImplementedError(
                'Megatron strategy not supported in LiveStore mode. Use fsdp or fsdp2.'
            )
        else:
            raise NotImplementedError

        from verl_custom.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ['fsdp', 'fsdp2']:
                try:
                    from verl.workers.fsdp_workers import RewardModelWorker
                except (ImportError, ModuleNotFoundError):
                    from verl.workers.engine_workers import (
                        ActorRolloutRefWorker as RewardModelWorker,
                    )
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            try:
                from verl.workers.fsdp_workers import (
                    ActorRolloutRefWorker as _RefWorker,
                )
            except (ImportError, ModuleNotFoundError):
                from verl.workers.engine_workers import (
                    ActorRolloutRefWorker as _RefWorker,
                )
            role_worker_mapping[Role.RefPolicy] = ray.remote(_RefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_manager.get('type', 'naive')
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager

            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl_custom.nvidia.reward_manager import PrimeRewardManager

            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == 'dapo':
            from verl.workers.reward_manager import DAPORewardManager

            reward_manager_cls = DAPORewardManager
        elif reward_manager_name == 'swebench':
            from verl_custom.nvidia.reward_manager import SWEBenchRewardManager

            reward_manager_cls = SWEBenchRewardManager
        else:
            raise NotImplementedError

        strategy = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=False
        )
        reward_fn = reward_manager_cls.options(scheduling_strategy=strategy).remote(
            tokenizer=tokenizer, compute_score=None, config=config.reward_manager
        )
        val_reward_fn = reward_fn

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # BC-14 + BC-15: trainer is a pure LiveStore consumer. No parquet
        # DataLoader, no ProRL address, no vLLM address. All training data
        # arrives via LiveStoreClient.get_batch().
        trainer = RayPPOTrainerDAPO(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=None,
            val_dataset=None,
            collate_fn=None,
            train_sampler=None,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()
