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

import statistics
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import ray
import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

@ray.remote
class SWEBenchRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, config, compute_score=None, num_examine=1) -> None:
        self.data_source = "SWE-Gym/SWE-Gym"

        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.verifier_func = compute_score or _default_compute_score
        self.config = config
    
    def set_length_penalty(self, length_penalty):
        self.length_penalty = length_penalty
    
    def set_stop_properly_penalty(self, stop_properly_penalty):
        self.stop_properly_penalty = stop_properly_penalty
    
    def _apply_stop_properly_penalty(self, reward_tensor: torch.Tensor, batch: DataProto):
        # apply stop properly penalty
        # get coefficient from config
        stop_penalty_coef = self.stop_properly_penalty
        # stop_properly is a tensor of shape (bs) float type
        stop_properly = 1.

        # if stop_properly is True, the value is 1.0, otherwise, it is self.config.stop_properly_penalty.penalty_coef 
        stop_properly_scale = (1.0 - stop_properly) * stop_penalty_coef + stop_properly
        # apply the scale to the reward_tensor
        reward_tensor = reward_tensor * stop_properly_scale
        return reward_tensor
    
    
    
    def verify(self, data):
        resolved = data.non_tensor_batch['resolved']
        error = data.non_tensor_batch['error']
        has_finish_action = data.non_tensor_batch['finish']
        score = [0. for _ in range(len(resolved))]
        for i, r in enumerate(resolved):
            if r:
                score[i] = 1.0

        reward_metrics = {}
        reward_metrics['max_turn_ratio'] = sum("RuntimeError: Agent reached maximum iteration in headless mode" in e for e in error if e) / len(error)
        reward_metrics['finish_action_ratio'] = sum(has_finish_action) / len(has_finish_action)
        reward_metrics['stuck_ratio'] = sum("stuck in a loop" in e for e in error if e) / len(error)

        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch['responses'].device)
        for ability in list(set(data.non_tensor_batch['ability'])):
            score_ = [data.batch['acc'][i].item() for i in range(len(data.batch['acc'])) if
                      data.non_tensor_batch['ability'][i] == ability]
            reward_metrics[f'{ability}'] = statistics.mean(score_)
        reward_metrics['all'] = data.batch['acc'].mean().item()
        
        return score, reward_metrics
    
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        verifier_reward=torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        response_ids = data.batch['responses']
        response_length = response_ids.shape[-1]
        valid_response_length = data.batch['attention_mask'][:, -response_length:].sum(-1)
        
        # if the batch already contains evaluation results, the verification is skipped here.
        if 'acc' in data.batch:
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            # verifier_score, verifier_metrics = self.verify(data)
            # Use ray based concurrency
            verifier_score, verifier_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            verifier_reward[i, valid_response_length[i] - 1] = verifier_score[i]

        reward_tensor_dict['gt_scores'] = verifier_reward
        
        if 'rm_scores' in data.batch.keys():
            reward_tensor_dict['rm_scores'] = data.batch['rm_scores']
            reward_metrics['reward_model']=data.batch['rm_scores'].sum(dim=1).mean().item()
            if self.config.reward_model.rm_coef!=0:
                reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict['rm_scores']

        if self.config.verifier.reward_coef!=0: # TODO: check if this is correct
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        # TODO: log the reward_metrics
        reward_metrics['reward_all'] = reward_tensor.sum(dim=-1).mean(dim=0).item()
        # reward_tensor is the modulated score of the response
        # after applying stop properly penalty and length penalty
        score_tensor=reward_tensor
        data.batch['stop_properly']=torch.ones_like(reward_tensor, dtype=torch.float32) # HACK: this is a hack to make the stop properly penalty work
        if self.stop_properly_penalty is not None:
            reward_tensor = self._apply_stop_properly_penalty(reward_tensor, data)

        if self.length_penalty is not None:
            data.batch['token_level_scores'] = reward_tensor
            reward_tensor = self.length_penalty(data)
        #reward_tensor: [batch_size*NUM_TRAJ, message_length]
        return {'score': score_tensor, 'reward': reward_tensor, 'reward_metrics': reward_metrics}

