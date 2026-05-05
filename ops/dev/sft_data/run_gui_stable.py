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

SINGULARITY_IMAGE_PATH = '/lustre/fsw/portfolios/llmservice/users/shaokunz/project/OpenHands_internal/singularity_images_oh/oh_v0.40.0_8me96m20iqt6tw9p_t5sffwjb6stny0ze.sif'

DEFAULT_SAMPLING_PARAMS = {
    'model': 'openai/gpt-4.1-mini',
    'api_key': '',
    'modify_params': False,
    'log_completions': False,
    'native_tool_calling': True,
    # 'temperature': 1,
    # 'top_p': 0.9,
    'max_iterations': 5,
}


def build_mock_gui_instances(num: int = 3) -> list[dict]:
    """Create a few simple GUI tasks for demo purposes."""
    tasks = [
        {
            'data_source': 'gui',
            'task': 'nvigate to www.fox.com and scroll down to the bottom of the News page',
            'start_url': 'https://www.fox.com',
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
    p.add_argument('--llm-addresses', nargs='+', default=['https://api.openai.com/v1'])
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

    # Optional: set container runtime env if using Singularity runtime
    os.environ['OH_RUNTIME_SINGULARITY_IMAGE_REPO'] = str(
        Path(SINGULARITY_IMAGE_PATH).parent
    )
    os.environ['SANDBOX_RUNTIME_CONTAINER_IMAGE'] = SINGULARITY_IMAGE_PATH

    evaluate(args)

# export OH_SAVE_TRAJECTORY_PATH=/lustre/fsw/portfolios/llmservice/users/shaokunz/project/OpenHands_internal/scripts/sft_data/log
# export OH_SAVE_SCREENSHOTS_IN_TRAJECTORY=true   # 或 false
# python scripts/sft_data/run_gui_stable.py --num-instances 1
# sandbox_config.runtime_container_image = "/lustre/fsw/portfolios/llmservice/users/shaokunz/project/OpenHands_internal/singularity_images_oh/oh_v0.40.0_8me96m20iqt6tw9p_t5sffwjb6stny0ze.sif"
