# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import heapq
import json
import os
import random
import threading
from typing import cast

import aiohttp
from openai import AsyncOpenAI

from openhands.nvidia.logger import nvidia_logger as logger

from . import llm_judge

# Client pool for LLM judge to avoid repeated client creation
_client_pool = {}
_client_pool_lock = asyncio.Lock()


async def _get_client_from_pool(base_url, api_key):
    """Get or create a client from the pool based on base_url and api_key."""
    async with _client_pool_lock:
        if base_url not in _client_pool:
            _client_pool[base_url] = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return _client_pool[base_url]


class Reward:
    def __init__(self, server_ip: list[str]):
        self.server_ip = server_ip
        self.weighted_addresses = [[0, ip] for ip in server_ip]
        heapq.heapify(self.weighted_addresses)

        self._address_lock = threading.Lock()

    async def get_reward(self, instance, solution_str, session=None):
        with self._address_lock:
            if len(self.weighted_addresses) == 0:
                raise ValueError('No LLM server addresses added')

            ip = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1  # type: ignore
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        data_source = instance['data_source']
        if isinstance(instance['reward_model'], str):
            ground_truth = json.loads(instance['reward_model'])['ground_truth']
        elif isinstance(instance['reward_model'], dict):
            ground_truth = instance['reward_model']['ground_truth']
        else:
            raise ValueError(
                f'Invalid reward_model type: {type(instance["reward_model"])}'
            )
        instance.get('extra_info', None)

        if data_source == 'stem':
            try:
                if isinstance(instance['prompt'], list):
                    question = instance['prompt'][0]['content']
                elif isinstance(instance['prompt'], str):
                    question = json.loads(instance['prompt'])[0]['content']
                else:
                    raise ValueError(f'Invalid prompt type: {type(instance["prompt"])}')
                metadata = {'reference_answer': ground_truth, 'extract_box': True}

                # Get base_url from environment variable, support list format
                base_url_env = os.getenv(
                    'OPENAI_BASE_URL', 'https://integrate.api.nvidia.com/v1'
                )
                # Parse as comma-separated list and randomly select one
                base_url_list = [
                    url.strip() for url in base_url_env.split(',') if url.strip()
                ]
                base_url = (
                    random.choice(base_url_list)
                    if len(base_url_list) > 1
                    else base_url_list[0]
                )

                sampling_params = {
                    'model': os.getenv('OPENAI_MODEL', 'meta/llama-3.3-70b-instruct'),
                    'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.0')),
                    'top_p': float(os.getenv('OPENAI_TOP_P', '0.7')),
                    'top_k': int(os.getenv('OPENAI_TOP_K', '-1')),
                    'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '1024')),
                    'max_model_len': int(os.getenv('OPENAI_MAX_MODEL_LEN', '8192')),
                }
                client = await _get_client_from_pool(
                    base_url, os.getenv('OPENAI_API_KEY', '')
                )
                res = await llm_judge.compute_score(
                    question, solution_str, metadata, sampling_params, client
                )
                logger.debug(
                    f'data_source: {data_source}, ground_truth: {ground_truth}, solution_str: {solution_str}, res: {res}'
                )

            except Exception as e:
                logger.error(f'Error: {e}')
                logger.info(
                    f'Error in reward.py: error: {e}, answer: {solution_str[:10]}, data_source: {data_source}, ground_truth: {ground_truth}'[
                        :100
                    ]
                )
                res = 0
        else:
            # please pass in session, spinning each session is very bad
            if session is None:
                local_session = True
            else:
                local_session = False
            session = session or aiohttp.ClientSession()

            if data_source == 'reasoning_gym':
                try:
                    entry = None
                    task = None
                    reward_model_value = instance['reward_model']
                    rm_dict = cast(dict, reward_model_value)
                    entry = rm_dict['entry']
                    task = rm_dict['reasoning_task']
                    async with session.post(
                        f'http://{ip}:8288/score',
                        json={
                            'answer': solution_str,
                            'entry': entry,
                            'task': task,
                        },
                    ) as response:
                        result = await response.json()
                        res = result['score']
                except Exception as e:
                    logger.error(f'Error: {e}, ip: {ip}')
                    logger.info(
                        f'answer: {solution_str[:10]}, task: {task}, entry: {entry}'[
                            :100
                        ]
                    )
                    res = 0
            else:
                try:
                    async with session.post(
                        f'http://{ip}:8388/compute_score',
                        json={
                            'solution_str': solution_str,
                            'ground_truth': ground_truth,
                            'data_source': data_source,
                        },
                    ) as response:
                        result = await response.json()
                        res = result['score']
                except Exception as e:
                    logger.error(f'Error: {e}, ip: {ip}')
                    logger.info(
                        f'answer: {solution_str[:10]}, data_source: {data_source}, ground_truth: {ground_truth}'[
                            :100
                        ]
                    )
                    res = 0

            if local_session:
                await session.close()

        # TODO: add llm_as_judge reward

        score = float(res)

        return {'resolved': score > 0.99, 'reward': score}
