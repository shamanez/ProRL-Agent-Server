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

# flake8: noqa: E402
import argparse
import json
import os
from pathlib import Path

from openhands.nvidia.async_server import OpenHandsServer
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.nvidia.utils import get_instance_id

SINGULARITY_IMAGE_PATH = 'Path/to/any/singularity'

DEFAULT_SAMPLING_PARAMS = {
    'model': 'hosted_vllm/Qwen2.5-VL-7B-Instruct',
    'api_key': 'mykey',
    'modify_params': False,
    'log_completions': True,
    'native_tool_calling': False,
    'temperature': 0.6,
    'top_p': 0.9,
    'token_level_generation': True,
    'custom_tokenizer': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'max_iterations': 5,
}


def build_mock_gui_instances(num: int = 3) -> list[dict]:
    """Create a few simple GUI tasks for demo purposes."""
    tasks = [
        {
            'data_source': 'gui',
            'task': 'try to find the homepage of the eagle-2 9b model from Nvidia on huggingface.',
            'start_url': '',
            'trajectory_id': 0,
        },
    ]
    instances: list[dict] = []
    for i, t in enumerate(tasks[:num]):
        s = dict(t)
        s['instance_id'] = get_instance_id(
            {
                'data_source': s['data_source'],
                'extra_info': {'split': 'mock', 'index': i, 'name': 'gui_demo'},
            }
        )
        instances.append(s)
    return instances


def evaluate(args):
    instances = build_mock_gui_instances(args.num_instances or 3)

    params = dict(DEFAULT_SAMPLING_PARAMS)
    if args.sampling_params:
        params.update(json.loads(args.sampling_params))

    server = OpenHandsServer(
        llm_server_addresses=args.llm_addresses,
        max_init_workers=1,
        max_run_workers=1,
        allow_skip_eval=True,
    )
    server.start()

    output_path = Path(args.output)
    try:
        with output_path.open('w', encoding='utf-8') as f:
            for idx, instance in enumerate(instances):
                try:
                    result_obj = server.process(instance, dict(params))
                except Exception as e:
                    logger.warning(f'process error for instance {idx}: {e}')
                    result_obj = {'ok': False, 'error': str(e)}

                record = {
                    'index': idx,
                    'instance_id': instance.get('instance_id'),
                    'result': result_obj,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                logger.info(f'Processed instance {idx + 1}/{len(instances)}')
    finally:
        try:
            server.stop()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(
        'Simple GUI tasks with OpenHands server (sequential, in-process)'
    )
    p.add_argument('--output', default='eval_results_gui.jsonl')
    p.add_argument('--llm-addresses', nargs='+', default=['http://127.0.0.1:8009'])
    p.add_argument('--num-instances', type=int, default=1)
    p.add_argument(
        '--sampling-params',
        default='',
        help='JSON string to merge into default sampling params',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.info(f'Args: {args}')

    os.environ['OH_RUNTIME_SINGULARITY_IMAGE_REPO'] = str(
        Path(SINGULARITY_IMAGE_PATH).parent
    )
    os.environ['SANDBOX_RUNTIME_CONTAINER_IMAGE'] = SINGULARITY_IMAGE_PATH

    evaluate(args)

# Step 1: Set screenshot path
# export OH_SAVE_TRAJECTORY_PATH=/lustre/fsw/portfolios/llmservice/users/shaokunz/project/OpenHands_internal/scripts/sft_data/log
# export OH_SAVE_SCREENSHOTS_IN_TRAJECTORY=true

# Step 2: Specify singularity in current py file

# Step 3: Start vllm server
# python verl_internal/verl/nvidia/rollout/vllm_api_server_vision.py --host 0.0.0.0 --port 8009 --model /path/to/Qwen2.5-VL-7B-Instruct --dtype float16 --max-model-len 65536 &

# Step 4: Set sandbox_config in get_config function on gui_utils.py
# sandbox_config.runtime_container_image = "Path/to/any/singularity"

# Step 5: Run
# python run_gui_vllm_server.py
# Then, you could see the trajectory and token_ids in eval_results_gui.jsonl
