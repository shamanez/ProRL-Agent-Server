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
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openhands.nvidia.swe_agent.r2egym_parser import ParsedCommit
from openhands.nvidia.swe_agent.utils import evaluate_agent as _evaluate_agent


def _sif_name_to_instance_id(sif_name: str) -> str:
    # Remove .sif extension
    base_name = sif_name.replace('.sif', '')

    # Remove common prefixes
    if base_name.startswith('docker.io_swebench_'):
        base_name = base_name[len('docker.io_swebench_') :]
    elif base_name.startswith('xingyaoww_'):
        base_name = base_name[len('xingyaoww_') :]

    # Remove sweb.eval.x86_64. prefix if present
    if base_name.startswith('sweb.eval.x86_64.'):
        base_name = base_name[len('sweb.eval.x86_64.') :]

    # Convert _s_ back to __
    instance_id = base_name.replace('_s_', '__')

    if 'monai' in instance_id:
        instance_id = instance_id.replace('monai', 'MONAI')
        instance_id = instance_id.replace('project', 'Project')

    return instance_id


def _sif_to_swesmith_docker(sif_name: str) -> str:
    if sif_name.endswith('.sif'):
        sif_name = sif_name[:-4]
    if '_' in sif_name:
        owner, rest = sif_name.split('_', 1)
        return f'{owner}/{rest}'
    return sif_name


def pre_process_r2egym_instance(r2egym_instance):
    r2egym_instance = pd.Series(r2egym_instance)
    r2egym_instance = r2egym_instance.apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    r2egym_instance['data_source'] = 'swebench'
    r2egym_instance['instance_id'] = (
        r2egym_instance['docker_image'].replace('/', '_').replace(':', '_')
    )
    docker_image = r2egym_instance['docker_image']
    if ':' in docker_image:
        repo_part, version_part = docker_image.split(':', 1)
    else:
        repo_part, version_part = docker_image, 'latest'
    r2egym_instance['repo'] = repo_part
    r2egym_instance['version'] = version_part

    if ('base_commit' not in r2egym_instance) or pd.isna(
        r2egym_instance['base_commit']
    ):
        if 'commit_hash' in r2egym_instance and not pd.isna(
            r2egym_instance['commit_hash']
        ):
            r2egym_instance['base_commit'] = r2egym_instance['commit_hash']
        else:
            r2egym_instance['base_commit'] = version_part
    r2egym_instance['data_kind'] = 'r2egym'
    parsed_commit = ParsedCommit(**json.loads(r2egym_instance['parsed_commit_content']))
    old_commit = parsed_commit.old_commit_hash
    r2egym_instance['old_commit'] = old_commit
    return r2egym_instance


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate ground-truth patches from the SWE-bench dataset using the OpenHands evaluation harness (no agent run).',
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/lustre/fsw/portfolios/llmservice/users/shaokunz/project/data/swe-gym/SkyRL-v0-293-data/train.parquet',
        help="Path to a parquet file that contains the SWE-bench dataset with an 'instance' column.",
    )
    # r2e_gym: /lustre/fs1/portfolios/llmservice/users/shaokunz/Openhands2/OpenHands_internal/data/r2egym/data/train-00003-of-00008.parquet
    parser.add_argument(
        '--output',
        type=str,
        default='swebench_gold_eval_results.jsonl',
        help='Output file to write evaluation results (JSONL).',
    )
    parser.add_argument(
        '--num-instances',
        type=int,
        default=None,
        help='Number of instances to evaluate (default: 1).',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=64,
        help='Maximum number of concurrent evaluations to run (default: 64, matching run_swebench.py).',
    )
    parser.add_argument(
        '--allow-skip-eval',
        action='store_true',
        help='Skip evaluation when the git_patch is empty or None (mirrors utils._evaluate_agent default).',
    )
    parser.add_argument(
        '--sif-name',
        type=str,
        default=None,
        help='Only evaluate instances for the specified SIF filename (e.g., xingyaoww_sweb.eval.x86_64.iterative_s_dvc-1809.sif). The script will convert this back to instance_id for filtering.',
    )
    return parser.parse_args()


async def _evaluate_instances(
    instances: list[dict], concurrency: int, allow_skip: bool
):
    semaphore = asyncio.Semaphore(concurrency)

    async def _evaluate_single(idx: int, inst: dict):
        async with semaphore:
            try:
                patch = inst.get('patch')
                rep = await _evaluate_agent(
                    patch, pd.Series(inst), sid=f'gold_{idx}', allow_skip=allow_skip
                )
                resolved_flag = False
                if isinstance(rep, dict):
                    if 'report' in rep and isinstance(rep['report'], dict):
                        resolved_flag = rep['report'].get('resolved', False)
                    elif 'resolved' in rep:
                        resolved_flag = rep.get('resolved', False)
                return {
                    'instance_id': inst.get('instance_id', idx),
                    'trajectory_id': inst.get('trajectory_id', idx),
                    'resolved': resolved_flag,
                    'evaluation': rep,
                }
            except Exception as e:
                return {
                    'instance_id': inst.get('instance_id', idx),
                    'trajectory_id': inst.get('trajectory_id', idx),
                    'resolved': False,
                    'error': str(e),
                }

    tasks = [_evaluate_single(i, inst) for i, inst in enumerate(instances)]
    return await asyncio.gather(*tasks)


async def _evaluate_r2egym(instances: list[dict], concurrency: int, allow_skip: bool):
    semaphore = asyncio.Semaphore(concurrency)

    async def _evaluate_single(idx: int, inst: dict):
        async with semaphore:
            data_instance = inst
            try:
                parsed_commit = ParsedCommit(
                    **json.loads(data_instance['parsed_commit_content'])
                )
                gt_patch = parsed_commit.get_patch(test_file=False, non_test_file=True)
                rep = await _evaluate_agent(
                    gt_patch, pd.Series(inst), sid=f'gold_{idx}', allow_skip=allow_skip
                )
                resolved_flag = False
                if isinstance(rep, dict):
                    if 'report' in rep and isinstance(rep['report'], dict):
                        resolved_flag = rep['report'].get('resolved', False)
                    elif 'resolved' in rep:
                        resolved_flag = rep.get('resolved', False)
                return {
                    'instance_id': inst.get('instance_id', idx),
                    'trajectory_id': inst.get('trajectory_id', idx),
                    'resolved': resolved_flag,
                    'evaluation': rep,
                }
            except Exception as e:
                return {
                    'instance_id': inst.get('instance_id', idx),
                    'trajectory_id': inst.get('trajectory_id', idx),
                    'resolved': False,
                    'error': str(e),
                }

    tasks = [_evaluate_single(i, inst) for i, inst in enumerate(instances)]
    return await asyncio.gather(*tasks)


def main():
    args = _parse_args()

    dataset_df = pd.read_parquet(args.dataset_path)

    if args.num_instances is not None:
        dataset_df = dataset_df.head(args.num_instances)

    is_r2egym = dataset_df.get('docker_image') is not None
    is_swesmith = (not is_r2egym) and (
        'image_name' in dataset_df.columns or 'repo' in dataset_df.columns
    )

    if is_swesmith and args.sif_name:
        target_docker_image = _sif_to_swesmith_docker(args.sif_name)
        print(f'[swe-smith] Target docker/image_name: {target_docker_image}')
        if 'image_name' in dataset_df.columns:
            mask = dataset_df['image_name'] == target_docker_image
        else:
            mask = dataset_df['repo'] == target_docker_image  # fallback
        dataset_df = dataset_df[mask]
        if len(dataset_df) == 0:
            print(f'[swe-smith] No instance matches docker_image {target_docker_image}')
            return
        dataset_df = dataset_df.head(1)  # only first instance

    if args.sif_name and not is_swesmith:
        target_instance_id = _sif_name_to_instance_id(args.sif_name)
        print(
            f'[evaluate_gold] Filtering for SIF: {args.sif_name} -> instance_id: {target_instance_id}'
        )

        if is_r2egym:
            # For r2egym, we need to check the instance_id in the docker_image field
            def _matches_target(row):
                docker_image = row.get('docker_image', '')
                if not docker_image:
                    return False
                # Extract instance_id from docker_image (reverse of pre_process_r2egym_instance)
                instance_id = docker_image.replace('/', '_').replace(':', '_')
                matches = (
                    target_instance_id in instance_id
                    or instance_id in target_instance_id
                )
                if matches:
                    print(
                        f'[DEBUG] Match found - docker_image: {docker_image}, extracted_id: {instance_id}'
                    )
                return matches

            mask = dataset_df.apply(_matches_target, axis=1)
        else:

            def _extract_instance_id(row):
                if 'instance' in row and isinstance(row['instance'], (dict, pd.Series)):
                    inst = row['instance']
                    if isinstance(inst, dict):
                        return inst.get('instance_id', '')
                    elif hasattr(inst, 'get'):
                        return inst.get('instance_id', '')
                # swe-smith flat structure
                return row.get('instance_id', '')

            instance_ids = dataset_df.apply(_extract_instance_id, axis=1)
            print(f'[DEBUG] Looking for target_instance_id: {target_instance_id}')
            print(
                f'[DEBUG] First 5 instance_ids in dataset: {list(instance_ids.head())}'
            )

            # Try both exact match and substring match
            exact_mask = instance_ids == target_instance_id
            substring_mask = instance_ids.str.contains(
                target_instance_id, na=False
            ) | instance_ids.apply(lambda x: target_instance_id in str(x))

            if exact_mask.any():
                print('[DEBUG] Found exact match(es)')
                mask = exact_mask
            elif substring_mask.any():
                print('[DEBUG] Found substring match(es)')
                mask = substring_mask
            else:
                print(
                    '[DEBUG] No matches found. Checking all instance_ids for partial matches...'
                )
                # Print first few instance IDs for debugging
                for i, iid in enumerate(instance_ids.head(10)):
                    print(f'[DEBUG] instance_ids[{i}]: {iid}')
                mask = pd.Series([False] * len(instance_ids), index=dataset_df.index)

        dataset_df = dataset_df[mask]
        if len(dataset_df) == 0:
            print(
                f'[evaluate_gold] No instances found matching SIF: {args.sif_name} (instance_id: {target_instance_id})'
            )
            return
        else:
            print(
                f'[evaluate_gold] Found {len(dataset_df)} instance(s) matching the SIF file'
            )

    instances: list[dict] = []
    if is_r2egym:
        for idx, row in dataset_df.iterrows():
            inst_series = pre_process_r2egym_instance(row)
            if 'trajectory_id' not in inst_series or pd.isna(
                inst_series.get('trajectory_id')
            ):
                inst_series['trajectory_id'] = idx
            instances.append(inst_series.to_dict())
    elif is_swesmith:
        for idx, row in dataset_df.iterrows():
            inst_series = row.apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
            inst_series['data_kind'] = 'swesmith'
            if 'trajectory_id' not in inst_series or pd.isna(
                inst_series.get('trajectory_id')
            ):
                inst_series['trajectory_id'] = idx
            instances.append(inst_series.to_dict())
    else:
        for idx, row in dataset_df.iterrows():
            inst_series = pd.Series(row['instance'])
            inst_series = inst_series.apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )
            if 'trajectory_id' not in inst_series or pd.isna(
                inst_series.get('trajectory_id')
            ):
                inst_series['trajectory_id'] = idx
            instances.append(inst_series.to_dict())

    start_ts = time.time()
    if is_r2egym:
        results = asyncio.run(
            _evaluate_r2egym(
                instances, concurrency=args.concurrency, allow_skip=args.allow_skip_eval
            )
        )
    else:
        results = asyncio.run(
            _evaluate_instances(
                instances, concurrency=args.concurrency, allow_skip=args.allow_skip_eval
            )
        )
    time.time() - start_ts

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as fp:
        for res in results:
            json.dump(res, fp)
            fp.write('\n')
    print(f'[evaluate_gold] Results written to {output_path}')

    resolved_cnt = sum(1 for r in results if r.get('resolved'))
    print(
        f'[evaluate_gold] Resolved {resolved_cnt}/{len(results)} instances ({resolved_cnt / len(results):.2%})'
    )


if __name__ == '__main__':
    main()
