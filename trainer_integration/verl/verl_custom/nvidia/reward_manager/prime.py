# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import aiohttp
import numpy as np
import ray
import torch
from verl import DataProto

from verl_custom.nvidia.reward_score import (
    _default_compute_score,
    _remote_compute_score_async,
)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch


async def single_compute_score(
    evaluation_func, completion, reference, task, executor, timeout=600, extra_info=None
):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(
                        evaluation_func, task, completion, reference, extra_info
                    ),  # Ensure synchronous
                ),
                timeout=timeout,
            )
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(
            f'Timeout occurred for completion: {completion[:10]}, data source: {task}'
        )
        return None  # Default value for timed-out rows
    except Exception as e:
        print(
            f'Error processing completion: {completion[:10]}, data source: {task}, Error: {e}'
        )
        return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_infos, num_processes=256
):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(
                evaluation_func,
                completion,
                reference,
                task,
                executor,
                timeout=300,
                extra_info=extra_info,
            )
            for completion, reference, task, extra_info in zip(
                completions, references, tasks, extra_infos
            )
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise

    # Process results
    for result, completion, reference, task in zip(
        results, completions, references, tasks
    ):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            print(
                f'Error in parallel_compute_score_async. Completion: {completion[:10]}, data source: {task}, Error: {result}. Setting score to 0.'
            )
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


async def _remote_score_wrapper_with_semaphore(
    task, completion, reference, extra_info, session, semaphore
):
    async with semaphore:
        return await _remote_compute_score_async(
            task, completion, reference, extra_info, session
        )


async def parallel_remote_score_async(
    completions, references, tasks, extra_infos, max_concurrency=4096
):
    # Using client session to avoid spinning multiple sessions
    # We avoid using ProcessPoolExecutor/ThreadPoolExecutor. might clash resource with server and lots of overhead.
    scores = []
    semaphore = asyncio.Semaphore(max_concurrency)
    try:
        timeout = aiohttp.ClientTimeout(
            total=600
        )  # set larger than remote server timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks_async = [
                _remote_score_wrapper_with_semaphore(
                    task, completion, reference, extra_info, session, semaphore
                )
                for completion, reference, task, extra_info in zip(
                    completions, references, tasks, extra_infos
                )
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
    except Exception as e:
        print(
            f'Error in parallel_remote_score_async. All results are set to 0. Error: {e}'
        )
        return [0.0 for _ in range(len(completions))]

    # Process results
    for result, completion, reference, task in zip(
        results, completions, references, tasks
    ):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            print(
                f'Error in parallel_remote_score_async. Completion: {completion[:10]}, data source: {task}, Error: {result}. Setting score to 0.'
            )
            scores.append(0.0)
        else:
            result = float(result)
            if result > 1.0 or result < 0.0:
                print(
                    f'Warning: score {result} is greater than 1.0 or less than 0.0 for completion: {completion[:10]}, data source: {task}. Double check this is as intended.'
                )
            scores.append(result)
    return scores


@ray.remote
class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, config, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.num_examine = (
            self.config.num_examine
        )  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        if isinstance(self.config.server_ip, str):
            self.server_ip = [self.config.server_ip]
        else:
            try:
                self.server_ip = list(self.config.server_ip)
            except:
                raise ValueError(
                    f'server_ip should be a list or a string: {self.config.server_ip}'
                )
        self.use_remote_reward = self.config.use_remote_reward
        if self.use_remote_reward:
            assert self.config.max_concurrency >= len(self.server_ip), (
                f'max_concurrency ({self.config.max_concurrency}) must be greater than or equal to the number of server IPs ({len(self.server_ip)}).'
            )
            self.compute_score = _remote_compute_score_async
            self.max_concurrency = self.config.max_concurrency
        self.binary_score = self.config.get('binary_score', False)
        self.length_penalty = None
        self.stop_properly_penalty = None

    def set_length_penalty(self, length_penalty):
        self.length_penalty = length_penalty

    def set_stop_properly_penalty(self, stop_properly_penalty):
        self.stop_properly_penalty = stop_properly_penalty

    def _apply_stop_properly_penalty(
        self, reward_tensor: torch.Tensor, batch: DataProto
    ):
        # apply stop properly penalty
        # get coefficient from config
        stop_penalty_coef = self.stop_properly_penalty
        # stop_properly is a tensor of shape (bs) float type
        stop_properly = batch.batch['stop_properly'].type(reward_tensor.dtype)

        # if stop_properly is True, the value is 1.0, otherwise, it is self.config.stop_properly_penalty.penalty_coef
        stop_properly_scale = (1.0 - stop_properly) * stop_penalty_coef + stop_properly
        stop_properly_scale = stop_properly_scale.unsqueeze(-1)
        # apply the scale to the reward_tensor
        reward_tensor = reward_tensor * stop_properly_scale
        return reward_tensor

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(
            dim=-1
        )
        sequences = remove_pad_token(
            response_ids, data.batch['attention_mask'][:, prompt_length:]
        )
        sequences_str = self.tokenizer.batch_decode(sequences)

        ground_truth = []
        for data_item in data:
            if 'ground_truth' in data_item.non_tensor_batch['reward_model']:
                ground_truth.append(
                    data_item.non_tensor_batch['reward_model']['ground_truth'].tolist()
                    if isinstance(
                        data_item.non_tensor_batch['reward_model']['ground_truth'],
                        np.ndarray,
                    )
                    else data_item.non_tensor_batch['reward_model']['ground_truth']
                )
            else:
                ground_truth.append(None)

        data_sources = data.non_tensor_batch['data_source']

        # Evenly distribute the data across the server IPs in round-robin manner
        if len(data.batch) % len(self.server_ip) == 0:
            data.non_tensor_batch['server_ip'] = np.array(
                self.server_ip * (len(data.batch) // len(self.server_ip)), dtype=object
            )
        else:
            server_ip_list = self.server_ip * (
                len(data.batch) // len(self.server_ip) + 1
            )
            data.non_tensor_batch['server_ip'] = np.array(
                server_ip_list[: len(data.batch)], dtype=object
            )
        extra_infos = [data_item.non_tensor_batch for data_item in data]

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            if self.use_remote_reward:
                scores = asyncio.run(
                    parallel_remote_score_async(
                        sequences_str,
                        ground_truth,
                        data_sources,
                        extra_infos,
                        self.max_concurrency,
                    )
                )
            else:
                scores = asyncio.run(
                    parallel_compute_score_async(
                        self.compute_score,
                        sequences_str,
                        ground_truth,
                        data_sources,
                        extra_infos,
                        num_processes=64,
                    )
                )
        except asyncio.TimeoutError:
            print('Global timeout in reward computing! Setting all as 0.')
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(
                f'Unexpected error in batched reward computing. Setting all as 0.: {e}'
            )
            scores = [0.0 for _ in range(len(sequences_str))]

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # score_tensor is the unmodulated score of the response
        score_tensor = reward_tensor

        # reward_tensor is the modulated score of the response
        # after applying stop properly penalty and length penalty
        if self.stop_properly_penalty is not None:
            reward_tensor = self._apply_stop_properly_penalty(score_tensor, data)

        if self.length_penalty is not None:
            data.batch['token_level_scores'] = reward_tensor
            reward_tensor = self.length_penalty(data)

        return {'score': score_tensor, 'reward': reward_tensor}
