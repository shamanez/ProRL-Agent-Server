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

"""
Test script for SWE-Smith dataset, mirroring functionality from test_r2egym.py
"""

import hashlib
import time

import numpy as np
import pandas as pd

from openhands.nvidia.async_server import OpenHandsServer


def pre_process_swesmith_instance(swesmith_instance):
    swesmith_instance = pd.Series(swesmith_instance)
    swesmith_instance = swesmith_instance.apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )

    swesmith_instance['data_source'] = 'swebench'
    swesmith_instance['data_kind'] = 'swesmith'

    if 'instance_id' not in swesmith_instance or pd.isna(
        swesmith_instance['instance_id']
    ):
        # Fallback: create instance_id from repo if available
        if 'repo' in swesmith_instance:
            swesmith_instance['instance_id'] = (
                swesmith_instance['repo'].replace('/', '_').replace(':', '_')
            )
        elif 'image_name' in swesmith_instance:
            swesmith_instance['instance_id'] = (
                swesmith_instance['image_name'].replace('/', '_').replace(':', '_')
            )
        else:
            swesmith_instance['instance_id'] = f'swesmith_instance_{int(time.time())}'

    if 'repo' not in swesmith_instance or pd.isna(swesmith_instance['repo']):
        if 'image_name' in swesmith_instance:
            image_name = swesmith_instance['image_name']
            if '/' in image_name:
                swesmith_instance['repo'] = image_name
            else:
                swesmith_instance['repo'] = image_name
        else:
            swesmith_instance['repo'] = 'unknown/repo'

    if 'version' not in swesmith_instance or pd.isna(swesmith_instance['version']):
        if 'image_name' in swesmith_instance and ':' in swesmith_instance['image_name']:
            _, version_part = swesmith_instance['image_name'].split(':', 1)
            swesmith_instance['version'] = version_part
        else:
            swesmith_instance['version'] = 'latest'

    if 'base_commit' not in swesmith_instance or pd.isna(
        swesmith_instance['base_commit']
    ):
        if 'commit_hash' in swesmith_instance and not pd.isna(
            swesmith_instance['commit_hash']
        ):
            swesmith_instance['base_commit'] = swesmith_instance['commit_hash']
        else:
            swesmith_instance['base_commit'] = swesmith_instance['version']

    return swesmith_instance


def test_server(
    total_jobs: int = 4, max_parallel_jobs: int = 2, allow_skip_eval: bool = False
):
    """
    Test server functionality with SWE-Smith instances.
    Mirrors the structure from test_r2egym.py but adapted for SWE-Smith data.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    # Load SWE-Smith dataset
    swesmith_dataset = pd.read_parquet(
        '/lustre/fsw/portfolios/llmservice/users/shaokunz/project/data/swesmith/SWE-smith/data/train-00000-of-00011.parquet'
    )

    # Select an instance for testing (similar to r2egym test)
    swesmith_instance = swesmith_dataset.iloc[-2]
    swesmith_instance = pre_process_swesmith_instance(swesmith_instance)

    # Create multiple requests for parallel testing
    requests = []
    for i in range(total_jobs):
        cur = swesmith_instance.copy(deep=True)
        cur['trajectory_id'] = i
        requests.append(cur.to_dict())

    # Server configuration (same as r2egym test)
    llm_server_address = 'http://127.0.0.1:8000/v1'
    sampling_params = {
        'model': 'hosted_vllm/Qwen2.5-7B-Instruct',
        'api_key': 'mykey',
        'modify_params': False,
        'log_completions': False,
        'native_tool_calling': False,
        'temperature': 0.6,
        'top_p': 0.9,
        'max_iterations': 5,
    }

    print('Starting server')
    server = OpenHandsServer(
        llm_server_addresses=[llm_server_address, llm_server_address],
        max_init_workers=max_parallel_jobs,
        max_run_workers=max_parallel_jobs,
        allow_skip_eval=allow_skip_eval,
    )
    server.start()
    print('Server started')

    print('Job submission started')

    # Process instances using ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = []
        for inst in requests:
            short_id = hashlib.sha1(
                f'{inst["instance_id"]}_{inst["trajectory_id"]}'.encode()
            ).hexdigest()[:12]
            futures.append(
                executor.submit(server.process, inst, dict(sampling_params), short_id)
            )
        results = [future.result() for future in futures]

    print('Job submission finished')
    server.stop()
    return results


if __name__ == '__main__':
    start = time.time()
    results = test_server(total_jobs=5, max_parallel_jobs=5, allow_skip_eval=False)

    # Validate results (same validation as r2egym test)
    for result in results:
        assert type(result['messages']) is list, (
            f'Result is not a list but of type {type(result["messages"])}.'
        )
        assert result['messages'][-1]['role'] == 'assistant', (
            f'Last message is not assistant but of role {result["messages"][-1]["role"]}.'
        )
        result['messages'] = len(result['messages'])
        result['tools'] = len(result['tools'])

    print(results)
    print(f'Time taken: {time.time() - start}')
    print('All tests passed!')
