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

import numpy as np
import ray
import torch
from verl import DataProto

from verl_custom.nvidia.reward_score import (
    _default_compute_score,
    _remote_compute_score,
)


@ray.remote
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, config, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.num_examine = (
            self.config.num_examine
        )  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        if isinstance(self.config.server_ip, str):
            self.server_ip = self.config.server_ip
        else:
            try:
                self.server_ip = list(self.config.server_ip)[0]
                print(
                    f'Detected list of server ips: {self.config.server_ip}. Please use prime reward manager for parallel processing. Currently using naive reward manager with single server ip: {self.server_ip}.'
                )
            except:
                raise ValueError(
                    f'server_ip should be a list or a string: {self.config.server_ip}'
                )
        self.use_remote_reward = self.config.use_remote_reward
        if self.use_remote_reward:
            self.compute_score = _remote_compute_score
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

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode only response
            sequences_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = []
            for data_item in data:
                if 'ground_truth' in data_item.non_tensor_batch['reward_model']:
                    ground_truth.append(
                        data_item.non_tensor_batch['reward_model'][
                            'ground_truth'
                        ].tolist()
                        if isinstance(
                            data_item.non_tensor_batch['reward_model']['ground_truth'],
                            np.ndarray,
                        )
                        else data_item.non_tensor_batch['reward_model']['ground_truth']
                    )
                else:
                    ground_truth.append(None)

            data_source = data_item.non_tensor_batch['data_source']
            data_item.non_tensor_batch['server_ip'] = self.server_ip

            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=data_item.non_tensor_batch,
            )

            if score > 1.0 or score < 0.0:
                print(
                    f'Warning: score {score} is greater than 1.0 or less than 0.0 for completion: {sequences_str[:10]}, data source: {data_source}. Double check this is as intended.'
                )

            reward_tensor[i, valid_response_length - 1] = score

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
