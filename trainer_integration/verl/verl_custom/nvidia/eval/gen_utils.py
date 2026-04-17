import os
import uuid

import numpy as np
import pandas as pd
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from verl_custom.nvidia.reward_manager import PrimeRewardManager
from verl_custom.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl_custom.workers.fsdp_workers import AsyncActorRolloutRefWorker


def get_generation_results(config, tokenizer):
    if config.actor_rollout_ref.rollout.val_kwargs.temperature == 0.0:
        assert config.actor_rollout_ref.rollout.val_kwargs.n == 1, (
            'When temperature=0, n_samples must be 1.'
        )

    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role='rollout',
    )
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
    )
    wg = RayWorkerGroup(
        resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init
    )
    wg.init_model()

    strategy = NodeAffinitySchedulingStrategy(
        node_id=ray.get_runtime_context().get_node_id(), soft=False
    )
    reward_fn = PrimeRewardManager.options(scheduling_strategy=strategy).remote(
        tokenizer=tokenizer, compute_score=None, config=config.reward_manager
    )

    rl_dataset = RLHFDataset(
        data_files=config.data.train_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=None,
    )

    dataloader = StatefulDataLoader(
        dataset=rl_dataset,
        batch_size=config.data.train_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    reward_tensor_lst = []
    data_source_lst = []
    data_uid_lst = []

    sample_inputs = []
    sample_outputs = []
    sample_scores = []
    for batch_idx, test_data in enumerate(dataloader):
        test_batch = DataProto.from_single_dict(test_data)
        # extend with uid info
        test_batch.non_tensor_batch['uid'] = np.array(
            [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
        )
        # repeat test batch
        test_batch = test_batch.repeat(
            repeat_times=config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
        )

        # Store original inputs
        input_ids = test_batch.batch['input_ids']
        # TODO: Can we keep special tokens except for padding tokens?
        input_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
        ]
        sample_inputs.extend(input_texts)

        batch_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
        non_tensor_batch_keys_to_pop = ['raw_prompt_ids']
        if 'multi_modal_inputs' in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.extend(
                ['multi_modal_data', 'multi_modal_inputs']
            )
        if 'raw_prompt' in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append('raw_prompt')
        if 'tools_kwargs' in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append('tools_kwargs')
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        test_gen_batch.meta_info = {
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': True,
            'validate': True,
        }
        print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

        output_texts_file = config.data.output_path.replace(
            '.parquet', f'_tmp_{batch_idx}.pickle'
        )
        # load the output_texts from tmp file if it exists
        if os.path.exists(output_texts_file):
            print(f'loading generation results from {output_texts_file}')
            test_batch = DataProto.load_from_disk(output_texts_file)
        else:
            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, wg.world_size
            )
            test_output_gen_batch_padded = wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print('validation generation end')
            print(f'finished generation for batch {batch_idx}')

            test_batch = test_batch.union(test_output_gen_batch)
            # save the test_batch to tmp file
            os.makedirs(os.path.dirname(output_texts_file), exist_ok=True)
            test_batch.save_to_disk(output_texts_file)

        # Store generated outputs
        output_ids = test_batch.batch['responses']
        output_texts = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
        ]
        sample_outputs.extend(output_texts)
        # evaluate using reward_function
        reward_tensor = ray.get(reward_fn.__call__.remote(test_batch))['score']

        # Store scores
        scores = reward_tensor.sum(-1).cpu().tolist()
        print('*' * 100)
        print(f'scores: {scores}')
        sample_scores.extend(scores)

        reward_tensor_lst.append(reward_tensor)
        data_source = test_batch.non_tensor_batch.get(
            'data_source', ['unknown'] * reward_tensor.shape[0]
        )
        # set the task name
        if 'reward_model' in test_batch.non_tensor_batch:
            reward_model_info = test_batch.non_tensor_batch['reward_model']
            for idx, x in enumerate(reward_model_info):
                if 'reasoning_task' in x:
                    test_batch.non_tensor_batch['extra_info'][idx]['name'] = x[
                        'reasoning_task'
                    ]
        for idx, x in enumerate(test_batch.non_tensor_batch['extra_info']):
            if 'name' in x:
                data_source[idx] += '/' + x['name']
        data_source_lst.append(data_source)
        data_uid_lst.append(test_batch.non_tensor_batch['uid'])

    reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
    data_sources = np.concatenate(data_source_lst, axis=0)
    data_uids = np.concatenate(data_uid_lst, axis=0)

    # save prompts, responses, scores to parquet
    dataset = pd.DataFrame(
        {'prompt': sample_inputs, 'response': sample_outputs, 'score': sample_scores}
    )
    # make the path if not exists
    os.makedirs(os.path.dirname(config.data.output_path), exist_ok=True)
    dataset.to_parquet(config.data.output_path)

    # evaluate test_score based on data source
    data_source_reward = {}
    for i in range(reward_tensor.shape[0]):
        data_source = data_sources[i]
        data_uid = data_uids[i]
        if data_source not in data_source_reward:
            data_source_reward[data_source] = {}
        if data_uid not in data_source_reward[data_source]:
            data_source_reward[data_source][data_uid] = []
        data_source_reward[data_source][data_uid].append(reward_tensor[i].item())

    metric_dict = {}
    stacked_rewards = []
    for data_source, rewards in data_source_reward.items():
        # shape (bs, n)
        rewards = np.array(list(rewards.values()))
        stacked_rewards.append(rewards)
        # Here we assume reward model generates binary 0/1 rewards
        metric_dict[f'{data_source}/pass@1'] = np.mean(rewards)
        metric_dict[f'{data_source}/pass@1/std'] = np.mean(rewards.std(axis=1))
        metric_dict[
            f'{data_source}/pass@{config.actor_rollout_ref.rollout.val_kwargs.n}'
        ] = np.mean(rewards.max(axis=1))

    if len(list(data_source_reward.keys())) > 1:
        stacked_rewards = np.concatenate(stacked_rewards, axis=0)
        metric_dict['all/pass@1'] = np.mean(stacked_rewards)
        metric_dict['all/pass@1/std'] = np.mean(stacked_rewards.std(axis=1))
        metric_dict[f'all/pass@{config.actor_rollout_ref.rollout.val_kwargs.n}'] = (
            np.mean(stacked_rewards.max(axis=1))
        )

    return metric_dict


def get_agent_generation_results(config, tokenizer):
    if config.actor_rollout_ref.rollout.val_kwargs.temperature == 0.0:
        assert config.actor_rollout_ref.rollout.val_kwargs.n == 1, (
            'When temperature=0, n_samples must be 1.'
        )

    from verl_custom.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    }
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    # see RayPPOTrainer.init_workers()
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {
        pool: {} for pool in resource_pool_manager.resource_pool_dict.values()
    }
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout],
        config=config.actor_rollout_ref,
        role='actor_rollout',
    )
    resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls

    all_wg = {}
    from verl.single_controller.ray.base import create_colocated_worker_cls

    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            device_name='cuda',
        )
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

    wg = all_wg['actor_rollout']
    wg.init_model()

    from verl_custom.nvidia.rollout.async_server import AsyncLLMServerManager

    async_rollout_manager = AsyncLLMServerManager(
        config=config,
        worker_group=wg,
    )
    async_rollout_manager.sleep()

    rl_dataset = RLHFDataset(
        data_files=config.data.train_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=None,
    )

    dataloader = StatefulDataLoader(
        dataset=rl_dataset,
        batch_size=config.data.train_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    data_source_lst = []
    data_uid_lst = []

    sample_outputs = []
    sample_scores = []
    sample_finish = []
    sample_success = []
    sample_instance = []

    for batch_idx, test_data in enumerate(dataloader):
        test_batch = DataProto.from_single_dict(test_data)
        # extend with uid info
        test_batch.non_tensor_batch['uid'] = np.array(
            [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
        )

        batch_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
        non_tensor_batch_keys_to_pop = ['instance']
        if 'multi_modal_inputs' in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.extend(
                ['multi_modal_data', 'multi_modal_inputs']
            )
        if 'raw_prompt' in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append('raw_prompt')
        if 'tools_kwargs' in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append('tools_kwargs')
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        test_gen_batch.meta_info = {
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': True,
            'validate': True,
        }
        print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

        output_texts_file = config.data.output_path.replace(
            '.parquet', f'_tmp_{batch_idx}.pickle'
        )
        # load the output_texts from tmp file if it exists
        if os.path.exists(output_texts_file):
            print(f'loading generation results from {output_texts_file}')
            test_batch = DataProto.load_from_disk(output_texts_file)
        else:
            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, config.trainer.nnodes
            )
            async_rollout_manager.wake_up()
            test_output_gen_batch_padded = async_rollout_manager.generate_sequences(
                test_gen_batch_padded
            )
            async_rollout_manager.sleep()
            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded,
                pad_size=pad_size * config.actor_rollout_ref.rollout.val_kwargs.n,
            )
            print('validation generation end')
            print(f'finished generation for batch {batch_idx}')

            test_batch = test_batch.repeat(
                repeat_times=config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )
            test_batch = test_batch.union(test_output_gen_batch)
            # save the test_batch to tmp file
            os.makedirs(os.path.dirname(output_texts_file), exist_ok=True)
            test_batch.save_to_disk(output_texts_file)

        # Store generated outputs
        output_ids = test_batch.batch['responses'].numpy()
        sample_outputs.extend(output_ids)
        # evaluate using reward_function
        scores = test_batch.non_tensor_batch['resolved']
        finish = test_batch.non_tensor_batch['finish']
        success = test_batch.non_tensor_batch['success']
        instance = test_batch.non_tensor_batch['instance']

        print('*' * 100)
        print(f'scores: {scores}')
        sample_scores.extend(scores)
        sample_finish.extend(finish)
        sample_success.extend(success)
        sample_instance.extend(instance)

        data_source = test_batch.non_tensor_batch.get(
            'data_source', ['unknown'] * len(scores)
        )
        # set the task name
        if 'reward_model' in test_batch.non_tensor_batch:
            reward_model_info = test_batch.non_tensor_batch['reward_model']
            if 'extra_info' in test_batch.non_tensor_batch:
                for idx, x in enumerate(reward_model_info):
                    if 'reasoning_task' in x:
                        test_batch.non_tensor_batch['extra_info'][idx]['name'] = x[
                            'reasoning_task'
                        ]
        if 'extra_info' in test_batch.non_tensor_batch:
            for idx, x in enumerate(test_batch.non_tensor_batch['extra_info']):
                if 'name' in x:
                    data_source[idx] += '/' + x['name']
        data_source_lst.append(data_source)
        data_uid_lst.append(test_batch.non_tensor_batch['uid'])

    data_sources = np.concatenate(data_source_lst, axis=0)
    data_uids = np.concatenate(data_uid_lst, axis=0)

    # save prompts, responses, scores to parquet
    dataset = pd.DataFrame(
        {
            'instance': sample_instance,
            'response': sample_outputs,
            'score': sample_scores,
            'finish': sample_finish,
            'success': sample_success,
        }
    )
    # make the path if not exists
    os.makedirs(os.path.dirname(config.data.output_path), exist_ok=True)
    dataset.to_parquet(config.data.output_path)

    # evaluate test_score based on data source
    data_source_reward = {}
    for i in range(len(sample_scores)):
        data_source = data_sources[i]
        data_uid = data_uids[i]
        if data_source not in data_source_reward:
            data_source_reward[data_source] = {}
        if data_uid not in data_source_reward[data_source]:
            data_source_reward[data_source][data_uid] = []
        data_source_reward[data_source][data_uid].append(sample_scores[i])

    metric_dict = {}
    stacked_rewards = []
    for data_source, rewards in data_source_reward.items():
        # shape (bs, n)
        rewards = np.array(list(rewards.values()))
        stacked_rewards.append(rewards)
        # Here we assume reward model generates binary 0/1 rewards
        metric_dict[f'{data_source}/pass@1'] = np.mean(rewards)
        metric_dict[f'{data_source}/pass@1/std'] = np.mean(rewards.std(axis=1))
        metric_dict[
            f'{data_source}/pass@{config.actor_rollout_ref.rollout.val_kwargs.n}'
        ] = np.mean(rewards.max(axis=1))

    if len(list(data_source_reward.keys())) > 1:
        stacked_rewards = np.concatenate(stacked_rewards, axis=0)
        metric_dict['all/pass@1'] = np.mean(stacked_rewards)
        metric_dict['all/pass@1/std'] = np.mean(stacked_rewards.std(axis=1))
        metric_dict[f'all/pass@{config.actor_rollout_ref.rollout.val_kwargs.n}'] = (
            np.mean(stacked_rewards.max(axis=1))
        )

    return metric_dict
