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

import argparse
import asyncio

import numpy as np
import pandas as pd

from openhands.core.config.llm_config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.nvidia.registry import JobDetails
from openhands.nvidia.reward import Reward
from openhands.nvidia.stem_agent.stem_utils import (
    evaluate_agent,
    initialize_agents,
    run_agent,
)
from openhands.nvidia.timer import PausableTimer


async def run(instance):
    reward_server_ip = ['cpu-0005']
    max_iterations = 35
    sampling_params = {
        'model': 'hosted_vllm/Qwen/Qwen3-4B-Instruct-2507',
        'api_key': 'mykey',
        'modify_params': False,
        'log_completions': True,
        'native_tool_calling': True,
        'temperature': 1.0,
        'top_p': 0.8,
    }
    llm_config = LLMConfig(base_url='http://127.0.0.1:8005/v1', **sampling_params)

    reward = Reward(server_ip=reward_server_ip)

    # test reward server
    test_reward = await reward.get_reward(
        instance,
        '<think> fake thought </think> something something \\boxed{D} \n\n something else',
    )
    logger.info(f'Test reward: {test_reward}')

    job_details = JobDetails(
        job_id='test_job',
        instance=instance,
        llm_config=llm_config,
    )
    job_details.agent_config['max_iterations'] = max_iterations
    job_details.timer = PausableTimer(timeout=500)
    job_details.timer.start()

    # run agent
    runtime, metadata, config = await initialize_agents(
        instance, llm_config=llm_config, agent_config=job_details.agent_config
    )
    job_details.runtime = runtime
    job_details.metadata = metadata
    job_details.config = config

    run_results = await run_agent(job_details)
    logger.info(f'Run Results: {run_results}')
    eval_results = await evaluate_agent(reward, run_results, instance)
    logger.info(f'Eval Results: {eval_results}')
    return eval_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_id', type=int, default=0)
    args = parser.parse_args()
    instance_id = args.instance_id

    dataset = pd.read_parquet(
        '/lustre/fsw/portfolios/nvr/users/sdiao/data/deepscaler/gpqa_diamond.parquet'
    )
    instance = dataset.iloc[instance_id]
    instance = pd.Series(instance)
    instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    instance = instance.to_dict()
    print(f'instance: {instance}')

    # Initialize the agents
    results = asyncio.run(run(instance))

    logger.info(f'Eval Results: {results}')
    logger.info('Agents initialized successfully!')
