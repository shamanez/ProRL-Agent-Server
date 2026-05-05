#!/usr/bin/env python3
"""
Standalone SWE-bench validation/testing script.

This script replicates the validation logic from the verl training framework
without depending on any verl project files. It:
1. Loads test data from a parquet file
2. Sends instances to ProRL Agent Server servers for multi-turn agent interaction
3. Computes rewards (resolved, file_iou, block_iou)
4. Computes pass@k metrics
5. Saves results to JSONL

Usage:
    python standalone_swebench_test.py \
        --data_path /path/to/test-transformed-with-prompt.parquet \
        --ProRL_Agent_Server_urls "http://host1:8006+http://host2:8006" \
        --model_name "Qwen/Qwen3-32B" \
        --output_dir ./results
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict

import aiohttp
import numpy as np
import pandas as pd
from scipy.special import comb

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================================
# File / Block IoU computation (from unified diffs)
# =====================================================================

HunkRange = tuple[int, int]


def merge_intervals(intervals: list[HunkRange]) -> list[HunkRange]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[HunkRange] = []
    s, e = intervals[0]
    for cs, ce in intervals[1:]:
        if cs <= e + 1:
            e = max(e, ce)
        else:
            merged.append((s, e))
            s, e = cs, ce
    merged.append((s, e))
    return merged


def intervals_length(intervals: list[HunkRange]) -> int:
    return sum(e - s + 1 for s, e in intervals)


def intervals_intersection_length(a: list[HunkRange], b: list[HunkRange]) -> int:
    i = j = 0
    inter = 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s, e = max(s1, s2), min(e1, e2)
        if s <= e:
            inter += e - s + 1
        if e1 < e2:
            i += 1
        else:
            j += 1
    return inter


def _strip_ab_prefix(path: str) -> str:
    if path.startswith('a/'):
        return path[2:]
    if path.startswith('b/'):
        return path[2:]
    return path


def _set_to_blocks(line_set: set[int]) -> list[HunkRange]:
    if not line_set:
        return []
    seq = sorted(line_set)
    blocks: list[HunkRange] = []
    start = prev = seq[0]
    for v in seq[1:]:
        if v == prev + 1:
            prev = v
        else:
            blocks.append((start, prev))
            start = prev = v
    blocks.append((start, prev))
    return blocks


def parse_unified_diff(patch_text: str):
    """Parse a unified diff and return (files, new_blocks, old_blocks)."""
    files: set[str] = set()
    new_blocks: dict[str, list[HunkRange]] = {}
    old_blocks: dict[str, list[HunkRange]] = {}

    lines = patch_text.splitlines()
    current_file = None
    old_file = None
    new_file = None
    in_hunk = False
    old_line = None
    new_line = None
    file_old_changed: set[int] = set()
    file_new_changed: set[int] = set()

    def finalize_file():
        nonlocal current_file, file_old_changed, file_new_changed
        if current_file is not None:
            files.add(current_file)
            if file_new_changed:
                new_blocks.setdefault(current_file, []).extend(
                    _set_to_blocks(file_new_changed)
                )
            if file_old_changed:
                old_blocks.setdefault(current_file, []).extend(
                    _set_to_blocks(file_old_changed)
                )
        current_file = None

    hunk_re = re.compile(
        r'^@@\s+-([0-9]+)(?:,([0-9]+))?\s+\+([0-9]+)(?:,([0-9]+))?\s+@@'
    )

    for line in lines:
        if line.startswith('diff --git '):
            if current_file is not None:
                finalize_file()
                file_old_changed = set()
                file_new_changed = set()
            in_hunk = False
            old_file = None
            new_file = None
            current_file = None
            old_line = None
            new_line = None
            continue
        if line.startswith('--- '):
            old_file = line[4:].strip()
            continue
        if line.startswith('+++ '):
            new_file = line[4:].strip()
            token_new = (new_file or '').split()[0]
            token_old = (old_file or '').split()[0]
            if token_new and not token_new.endswith('/dev/null'):
                current_file = _strip_ab_prefix(token_new)
            elif token_old and not token_old.endswith('/dev/null'):
                current_file = _strip_ab_prefix(token_old)
            else:
                current_file = token_new or token_old or ''
            continue
        if line.startswith('Binary files '):
            m = re.search(r'and\s+b/(.+?)\s+differ', line)
            if m:
                files.add(m.group(1))
            else:
                m2 = re.search(r'a/(.+?)\s+and', line)
                if m2:
                    files.add(m2.group(1))
            continue
        m = hunk_re.match(line)
        if m:
            in_hunk = True
            old_line = int(m.group(1))
            new_line = int(m.group(3))
            continue
        if in_hunk and current_file is not None:
            if line.startswith(' '):
                old_line += 1
                new_line += 1
            elif line.startswith('-'):
                file_old_changed.add(old_line)
                old_line += 1
            elif line.startswith('+'):
                file_new_changed.add(new_line)
                new_line += 1

    if current_file is not None:
        finalize_file()

    return files, new_blocks, old_blocks


def file_iou(gt_patch: str, pred_patch: str) -> float:
    gt_files, _, _ = parse_unified_diff(gt_patch)
    pred_files, _, _ = parse_unified_diff(pred_patch)
    inter = len(gt_files & pred_files)
    union = len(gt_files | pred_files)
    if union == 0:
        return 1.0
    return inter / union


def block_iou(gt_patch: str, pred_patch: str) -> float:
    gt_files, gt_new, gt_old = parse_unified_diff(gt_patch)
    pred_files, pred_new, pred_old = parse_unified_diff(pred_patch)
    all_files = set(gt_files) | set(pred_files)
    inter_sum = 0
    union_sum = 0
    for side in ('new', 'old'):
        for f in all_files:
            g = merge_intervals((gt_new if side == 'new' else gt_old).get(f, []))
            p = merge_intervals((pred_new if side == 'new' else pred_old).get(f, []))
            if not g and not p:
                continue
            inter_len = intervals_intersection_length(g, p)
            union_len = intervals_length(merge_intervals(g + p))
            inter_sum += inter_len
            union_sum += union_len
    if union_sum == 0:
        return 1.0
    return inter_sum / union_sum


def compute_git_patch_ious(gt_patch: str, pred_patch: str) -> dict[str, float]:
    return {
        'file_iou': file_iou(gt_patch, pred_patch),
        'block_iou': block_iou(gt_patch, pred_patch),
    }


# =====================================================================
# Reward computation
# =====================================================================


def compute_rewards(
    results: list[dict], file_iou_coef: float = 0.0, block_iou_coef: float = 0.0
) -> list[dict]:
    """
    Compute rewards for each result returned by ProRL Agent Server.

    Each result dict should have keys: success, resolved, finish, error, git_patch, messages, instance.
    Returns the same list of dicts with added keys: score, file_iou, block_iou.
    """
    for r in results:
        gt_patch = r.get('instance', {}).get('patch', '') or ''
        pred_patch = r.get('git_patch', '') or ''

        r['resolved'] = bool(r.get('resolved', False))
        r['finish'] = bool(r.get('finish', False))
        r['error'] = r.get('error', None) or ''

        if gt_patch and pred_patch:
            ious = compute_git_patch_ious(gt_patch, pred_patch)
            r['file_iou'] = ious['file_iou']
            r['block_iou'] = ious['block_iou']
        else:
            r['file_iou'] = 0.0
            r['block_iou'] = 0.0

        base_score = 1.0 if r['resolved'] else 0.0
        converted_file_iou = 1.0 if r['resolved'] else r['file_iou']
        converted_block_iou = 1.0 if r['resolved'] else r['block_iou']
        r['score'] = (
            base_score
            + file_iou_coef * converted_file_iou
            + block_iou_coef * converted_block_iou
        )

    return results


# =====================================================================
# Metrics: process_validation_metrics
# =====================================================================


def bootstrap_metric(
    data: list,
    subset_size: int,
    reduce_fns: list,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list:
    np.random.seed(seed)
    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list, vote_key: str, val_key: str) -> float:
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])
    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)
    return vote2vals[maj_vote][0]


def process_validation_metrics(
    data_sources: list,
    sample_inputs: list,
    infos_dict: dict,
    seed: int = 42,
) -> dict:
    """Group by data_source and prompt, then compute mean, best@N, worst@N, maj@N."""
    data_src2prompt2var2vals = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for idx, ds in enumerate(data_sources):
        prompt = sample_inputs[idx]
        var2vals = data_src2prompt2var2vals[ds][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[idx])

    data_src2prompt2var2metric = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for ds, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f'mean@{n_resps}'] = np.mean(var_vals)
                if n_resps > 1:
                    metric[f'std@{n_resps}'] = np.std(var_vals)
                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)
                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals,
                            subset_size=n,
                            reduce_fns=[np.max, np.min],
                            seed=seed,
                        )
                        metric[f'best@{n}/mean'], metric[f'best@{n}/std'] = (
                            bon_mean,
                            bon_std,
                        )
                        metric[f'worst@{n}/mean'], metric[f'worst@{n}/std'] = (
                            won_mean,
                            won_std,
                        )
                data_src2prompt2var2metric[ds][prompt][var_name] = metric

    data_src2var2metric2prompt_vals = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for ds, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[ds][var_name][metric_name].append(
                        metric_val
                    )

    data_src2var2metric2val = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for ds, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[ds][var_name][metric_name] = np.mean(
                    prompt_vals
                )

    return data_src2var2metric2val


# =====================================================================
# Pass@k
# =====================================================================


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(comb(n - c, k)) / float(comb(n, k))


def compute_pass_k_metrics(problem_results: dict, k_values: list = None) -> dict:
    if k_values is None:
        k_values = [1, 4, 8, 16]
    results = {}
    for k in k_values:
        pass_at_k_scores = []
        valid_problems = 0
        for pid, stats in problem_results.items():
            n = stats['total']
            c = stats['correct']
            if n >= k:
                pass_at_k_scores.append(calculate_pass_at_k(n, c, k))
                valid_problems += 1
        if pass_at_k_scores:
            results[f'pass@{k}'] = {
                'score': float(np.mean(pass_at_k_scores)),
                'std': float(np.std(pass_at_k_scores)),
                'num_problems': valid_problems,
            }
        else:
            results[f'pass@{k}'] = {'score': 0.0, 'std': 0.0, 'num_problems': 0}
    return results


# =====================================================================
# Data loading
# =====================================================================


def load_test_data(parquet_path: str) -> list[dict]:
    """
    Load test instances from a parquet file.

    Expected columns:
      - prompt: JSON string of chat messages [{role, content}, ...]
      - instance: dict with instance_id, problem_statement, patch, etc.
      - data_source: str (e.g. "SWE-Gym/SWE-Gym")
      - extra_info: dict with name, index, split, etc.
    """
    df = pd.read_parquet(parquet_path)
    records = []
    for i, row in df.iterrows():
        record = {}

        # Parse instance
        instance = row.get('instance', {})
        if isinstance(instance, str):
            instance = json.loads(instance)
        record['instance'] = instance

        # Data source
        record['data_source'] = row.get('data_source', 'unknown')

        # Extra info
        extra_info = row.get('extra_info', {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        record['extra_info'] = extra_info

        # Build instance_id if not present
        if 'instance_id' not in record['instance']:
            ds = record['data_source']
            name = extra_info.get('name', '')
            split = extra_info.get('split', '')
            index = extra_info.get('index', i)
            record['instance']['instance_id'] = f'{ds}/{name}/{split}/{index}'

        # Prompt text (for input dedup / pass@k grouping)
        prompt = row.get('prompt', '')
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(prompt, list):
            record['prompt_text'] = ' '.join(
                m.get('content', '') for m in prompt if isinstance(m, dict)
            )
        else:
            record['prompt_text'] = str(prompt)

        records.append(record)

    logger.info(f'Loaded {len(records)} test instances from {parquet_path}')
    return records


# =====================================================================
# ProRL Agent Server interaction
# =====================================================================


async def send_to_ProRL_Agent_Server(
    session: aiohttp.ClientSession,
    ProRL_Agent_Server_url: str,
    instance: dict,
    sampling_params: dict,
    timeout_seconds: int = 1500,
) -> dict:
    """Send a single instance to a ProRL Agent Server server's /process endpoint."""
    request_data = {'instance': instance, 'sampling_params': sampling_params}
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with session.post(
            f'{ProRL_Agent_Server_url}/process',
            headers={'Content-Type': 'application/json'},
            json=request_data,
            timeout=timeout,
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                messages = data.get('messages', [])
                if messages and len(messages) > 0:
                    return data
                else:
                    logger.warning(
                        f'Empty messages from {ProRL_Agent_Server_url} for {instance.get("instance_id", "?")}'
                    )
                    return {
                        'success': False,
                        'messages': [],
                        'resolved': False,
                        'finish': False,
                        'error': 'Empty response',
                        'git_patch': '',
                    }
            else:
                error_text = await resp.text()
                logger.error(
                    f'HTTP {resp.status} from {ProRL_Agent_Server_url}: {error_text}'
                )
                return {
                    'success': False,
                    'messages': [],
                    'resolved': False,
                    'finish': False,
                    'error': f'HTTP {resp.status}',
                    'git_patch': '',
                }
    except asyncio.TimeoutError:
        logger.error(f'Timeout for instance {instance.get("instance_id", "?")}')
        return {
            'success': False,
            'messages': [],
            'resolved': False,
            'finish': False,
            'error': 'Timeout',
            'git_patch': '',
        }
    except Exception as e:
        logger.error(f'Error for instance {instance.get("instance_id", "?")}: {e}')
        return {
            'success': False,
            'messages': [],
            'resolved': False,
            'finish': False,
            'error': str(e),
            'git_patch': '',
        }


async def start_ProRL_Agent_Server_server(session: aiohttp.ClientSession, url: str):
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with session.post(f'{url}/start', timeout=timeout) as resp:
            if resp.status == 200:
                logger.info(f'Started ProRL Agent Server server: {url}')
            else:
                logger.warning(f'Failed to start {url}: HTTP {resp.status}')
    except Exception as e:
        logger.warning(f'Failed to start {url}: {e}')


async def stop_ProRL_Agent_Server_server(session: aiohttp.ClientSession, url: str):
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with session.post(f'{url}/stop', timeout=timeout) as resp:
            if resp.status == 200:
                logger.info(f'Stopped ProRL Agent Server server: {url}')
    except Exception as e:
        logger.warning(f'Failed to stop {url}: {e}')


# =====================================================================
# LLM Server Management
# =====================================================================


async def clear_llm_servers_from_ProRL_Agent_Server(
    session: aiohttp.ClientSession, ProRL_Agent_Server_url: str
):
    """Clear existing LLM servers from a ProRL Agent Server."""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with session.post(
            f'{ProRL_Agent_Server_url}/clear_llm_server', timeout=timeout
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                logger.info(
                    f'Cleared LLM servers from {ProRL_Agent_Server_url}: {result}'
                )
            else:
                error_text = await resp.text()
                logger.warning(
                    f'Failed to clear LLM servers from {ProRL_Agent_Server_url}: HTTP {resp.status}: {error_text}'
                )
    except Exception as e:
        logger.warning(
            f'Failed to clear LLM servers from {ProRL_Agent_Server_url}: {e}'
        )


async def add_llm_server_to_ProRL_Agent_Server(
    session: aiohttp.ClientSession, ProRL_Agent_Server_url: str, llm_address: str
):
    """Register a single LLM server address with a ProRL Agent Server server."""
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        payload = {'address': llm_address}
        async with session.post(
            f'{ProRL_Agent_Server_url}/add_llm_server', json=payload, timeout=timeout
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                logger.info(
                    f'Added LLM server {llm_address} to {ProRL_Agent_Server_url}: {result}'
                )
            else:
                error_text = await resp.text()
                logger.error(
                    f'Failed to add LLM server {llm_address} to {ProRL_Agent_Server_url}: HTTP {resp.status}: {error_text}'
                )
    except Exception as e:
        logger.error(
            f'Failed to add LLM server {llm_address} to {ProRL_Agent_Server_url}: {e}'
        )


def _url_to_ip(url: str) -> str:
    """Extract IP address from URL by removing protocol and port."""
    url = url.replace('http://', '').replace('https://', '')
    return url.split(':')[0]


def assign_llm_servers_to_ProRL_Agent_Server(
    ProRL_Agent_Server_urls: list[str], llm_server_urls: list[str]
) -> dict[str, list[str]]:
    """
    Assign LLM servers to ProRL Agent Server servers based on IP locality.

    Phase 1: Assign LLM servers to ProRL Agent Server servers on the same IP.
    Phase 2: Distribute remaining LLM servers evenly across all ProRL Agent Server servers.
    """
    assignments: dict[str, list[str]] = {url: [] for url in ProRL_Agent_Server_urls}
    assigned: set[str] = set()

    # Phase 1: Locality-based assignment
    for llm_url in llm_server_urls:
        llm_ip = _url_to_ip(llm_url)
        for oh_url in ProRL_Agent_Server_urls:
            oh_ip = _url_to_ip(oh_url)
            if llm_ip == oh_ip:
                assignments[oh_url].append(llm_url)
                assigned.add(llm_url)
                break

    # Phase 2: Distribute remaining evenly (round-robin)
    remaining = [url for url in llm_server_urls if url not in assigned]
    if remaining:
        logger.info(
            f'Distributing {len(remaining)} remaining LLM servers across {len(ProRL_Agent_Server_urls)} ProRL Agent Server servers'
        )
        for i, llm_url in enumerate(remaining):
            oh_url = ProRL_Agent_Server_urls[i % len(ProRL_Agent_Server_urls)]
            assignments[oh_url].append(llm_url)

    return assignments


async def wait_for_server_health(urls: list[str], timeout: int = 600):
    """Wait for all servers to become healthy by polling /health endpoint."""
    logger.info(
        f'Waiting for {len(urls)} servers to become healthy (timeout: {timeout}s)...'
    )
    start = time.time()
    async with aiohttp.ClientSession() as session:
        for url in urls:
            healthy = False
            while time.time() - start < timeout:
                try:
                    health_url = f'{url}/health'
                    async with session.get(
                        health_url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            logger.info(f'Server {url} is healthy')
                            healthy = True
                            break
                except Exception:
                    pass
                await asyncio.sleep(5)
            if not healthy:
                raise TimeoutError(
                    f'Server {url} did not become healthy within {timeout}s'
                )
    logger.info('All servers are healthy')


async def setup_llm_servers_with_ProRL_Agent_Server(
    ProRL_Agent_Server_urls: list[str],
    llm_server_urls: list[str],
    token_level_generation: bool = False,
):
    """
    Clear existing LLM servers and register new ones with all ProRL Agent Server servers.

    Steps:
    1. Wait for all LLM servers to be healthy.
    2. Clear existing LLM server registrations from all ProRL Agent Server servers.
    3. Compute locality-based assignment of LLM servers to ProRL Agent Server servers.
    4. Register each LLM server with its assigned ProRL Agent Server server.
    """
    # Step 1: Wait for LLM servers to be healthy
    await wait_for_server_health(llm_server_urls)

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Step 2: Clear existing LLM servers
        logger.info(
            'Clearing existing LLM servers from all ProRL Agent Server servers...'
        )
        clear_tasks = [
            clear_llm_servers_from_ProRL_Agent_Server(session, url)
            for url in ProRL_Agent_Server_urls
        ]
        await asyncio.gather(*clear_tasks)

        # Step 3: Compute assignment
        assignments = assign_llm_servers_to_ProRL_Agent_Server(
            ProRL_Agent_Server_urls, llm_server_urls
        )

        # Step 4: Register LLM servers
        logger.info(
            f'Registering {len(llm_server_urls)} LLM servers with {len(ProRL_Agent_Server_urls)} ProRL Agent Server servers...'
        )
        add_tasks = []
        for oh_url, llm_urls in assignments.items():
            for llm_url in llm_urls:
                # Format address based on token_level_generation mode
                if token_level_generation:
                    address = (
                        llm_url if llm_url.startswith('http') else f'http://{llm_url}'
                    )
                else:
                    base = (
                        llm_url if llm_url.startswith('http') else f'http://{llm_url}'
                    )
                    address = f'{base}/v1' if not base.endswith('/v1') else base
                add_tasks.append(
                    add_llm_server_to_ProRL_Agent_Server(session, oh_url, address)
                )
        await asyncio.gather(*add_tasks)

        # Log summary
        for oh_url, llm_urls in assignments.items():
            logger.info(f'  {oh_url} -> {llm_urls}')
        logger.info('LLM server registration complete')


async def run_ProRL_Agent_Server_evaluation(
    instances: list[dict],
    ProRL_Agent_Server_urls: list[str],
    sampling_params: dict,
    num_trajectories: int = 1,
    num_workers_per_server: int = 64,
    max_retries: int = 2,
    timeout_seconds: int = 1500,
) -> list[dict]:
    """
    Send all instances to ProRL Agent Server servers with load balancing.

    Returns a flat list of result dicts (one per instance * trajectory).
    Each result has the original instance attached.
    """
    # Build job list: expand each instance into num_trajectories copies
    jobs = []
    for inst in instances:
        for traj_id in range(num_trajectories):
            job_instance = inst.copy()
            job_instance['trajectory_id'] = traj_id
            jobs.append(job_instance)

    logger.info(
        f'Total jobs: {len(jobs)} ({len(instances)} instances x {num_trajectories} trajectories)'
    )

    # Build server worker pool
    server_workers = []
    for url in ProRL_Agent_Server_urls:
        for w in range(num_workers_per_server):
            server_workers.append(url)

    results = [None] * len(jobs)
    job_queue = asyncio.Queue()
    for i, job in enumerate(jobs):
        await job_queue.put((i, job, 0))

    server_queue = asyncio.Queue()
    for sw in server_workers:
        await server_queue.put(sw)

    completed = 0
    total = len(jobs)
    processing_start_time = time.time()

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Start all servers
        start_tasks = [
            start_ProRL_Agent_Server_server(session, url)
            for url in ProRL_Agent_Server_urls
        ]
        await asyncio.gather(*start_tasks)
        logger.info('All ProRL Agent Server servers started')

        async def worker():
            nonlocal completed
            while True:
                try:
                    idx, instance, retry_count = await asyncio.wait_for(
                        job_queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    if job_queue.empty():
                        break
                    continue

                server_url = await server_queue.get()
                try:
                    result = await send_to_ProRL_Agent_Server(
                        session, server_url, instance, sampling_params, timeout_seconds
                    )
                    result['instance'] = instance

                    if (
                        not result.get('success', False)
                        and not result.get('messages')
                        and retry_count < max_retries
                    ):
                        await job_queue.put((idx, instance, retry_count + 1))
                        logger.info(f'Retrying job {idx} (attempt {retry_count + 1})')
                    else:
                        results[idx] = result
                        completed += 1
                        if completed % 10 == 0 or completed == total:
                            # Compute resolved/finish ratios from completed results
                            resolved_count = 0
                            finish_count = 0
                            all_results_count = 0
                            resolved_instances = set()
                            for r in results:
                                if r is None:
                                    continue
                                all_results_count += 1
                                inst_id = r.get('instance', {}).get('instance_id', '')
                                if r.get('resolved', False):
                                    resolved_count += 1
                                    resolved_instances.add(inst_id)
                                if r.get('finish', False):
                                    finish_count += 1
                            resolved_ratio = (
                                resolved_count / all_results_count
                                if all_results_count
                                else 0
                            )
                            finish_ratio = (
                                finish_count / all_results_count
                                if all_results_count
                                else 0
                            )
                            total_instances = len(instances)
                            resolved_instance_ratio = (
                                len(resolved_instances) / total_instances
                                if total_instances
                                else 0
                            )
                            elapsed_time = time.time() - processing_start_time
                            logger.info(
                                f'Progress: {completed}/{total} ({completed / total * 100:.1f}%), '
                                f'elapsed: {elapsed_time:.1f}s, '
                                f'resolved: {resolved_ratio:.2f}, finish: {finish_ratio:.2f}, '
                                f'resolved_instance: {resolved_instance_ratio:.2f}'
                            )
                finally:
                    await server_queue.put(server_url)

        # Run workers concurrently
        num_concurrent = len(server_workers)
        workers = [asyncio.create_task(worker()) for _ in range(num_concurrent)]
        await asyncio.gather(*workers)

        # Stop all servers
        stop_tasks = [
            stop_ProRL_Agent_Server_server(session, url)
            for url in ProRL_Agent_Server_urls
        ]
        await asyncio.gather(*stop_tasks)

    # Fill in any None results (jobs that exhausted all retries)
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                'success': False,
                'messages': [],
                'resolved': False,
                'finish': False,
                'error': 'All retries exhausted',
                'git_patch': '',
                'instance': jobs[i],
            }

    return results


# =====================================================================
# Main evaluation logic
# =====================================================================


def save_results_jsonl(results: list[dict], output_path: str):
    """Save results as JSONL."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for r in results:
            record = {
                'instance_id': r.get('instance', {}).get('instance_id', ''),
                'trajectory_id': r.get('instance', {}).get('trajectory_id', 0),
                'score': r.get('score', 0.0),
                'resolved': r.get('resolved', False),
                'finish': r.get('finish', False),
                'error': r.get('error', ''),
                'file_iou': r.get('file_iou', 0.0),
                'block_iou': r.get('block_iou', 0.0),
                'git_patch': r.get('git_patch', ''),
                'num_messages': len(r.get('messages', [])),
            }
            f.write(json.dumps(record) + '\n')
    logger.info(f'Saved {len(results)} results to {output_path}')


def print_summary(
    results: list[dict],
    instances: list[dict],
    num_trajectories: int,
    k_values: list[int],
):
    """Print evaluation summary with metrics."""
    print('\n' + '=' * 60)
    print('EVALUATION SUMMARY')
    print('=' * 60)

    total = len(results)
    resolved_count = sum(1 for r in results if r.get('resolved', False))
    finish_count = sum(1 for r in results if r.get('finish', False))
    error_count = sum(
        1 for r in results if r.get('error') and r['error'] not in ('', None)
    )
    max_turn_count = sum(
        1 for r in results if r.get('error') and 'maximum iteration' in str(r['error'])
    )
    stuck_count = sum(
        1 for r in results if r.get('error') and 'stuck in a loop' in str(r['error'])
    )

    scores = [r.get('score', 0.0) for r in results]
    file_ious = [r.get('file_iou', 0.0) for r in results]
    block_ious = [r.get('block_iou', 0.0) for r in results]

    print(f'Total instances:       {len(instances)}')
    print(f'Trajectories per inst: {num_trajectories}')
    print(f'Total evaluations:     {total}')
    print(
        f'Resolved:              {resolved_count}/{total} ({resolved_count / total * 100:.1f}%)'
    )
    print(
        f'Finish action:         {finish_count}/{total} ({finish_count / total * 100:.1f}%)'
    )
    print(
        f'Max turn reached:      {max_turn_count}/{total} ({max_turn_count / total * 100:.1f}%)'
    )
    print(
        f'Stuck in loop:         {stuck_count}/{total} ({stuck_count / total * 100:.1f}%)'
    )
    print(f'Has error:             {error_count}/{total}')
    print(f'Mean score:            {np.mean(scores):.4f}')
    print(f'Mean file IoU:         {np.mean(file_ious):.4f}')
    print(f'Mean block IoU:        {np.mean(block_ious):.4f}')

    # Group by data source for per-source metrics
    data_sources = [
        r.get('instance', {}).get('data_source', 'unknown') for r in results
    ]
    unique_sources = sorted(set(data_sources))
    if len(unique_sources) > 1:
        print('\nPer data-source breakdown:')
        for src in unique_sources:
            src_scores = [s for s, ds in zip(scores, data_sources) if ds == src]
            src_resolved = sum(
                1
                for r, ds in zip(results, data_sources)
                if ds == src and r.get('resolved')
            )
            print(
                f'  {src}: resolved={src_resolved}/{len(src_scores)} mean_score={np.mean(src_scores):.4f}'
            )

    # Compute process_validation_metrics
    sample_inputs = [
        r.get('instance', {}).get('instance_id', str(i)) for i, r in enumerate(results)
    ]
    infos_dict = {
        'reward': scores,
        'acc': [1.0 if r.get('resolved') else 0.0 for r in results],
    }
    val_metrics = process_validation_metrics(data_sources, sample_inputs, infos_dict)
    print('\nValidation metrics (process_validation_metrics):')
    for ds, var2metric2val in val_metrics.items():
        for var_name, metric2val in var2metric2val.items():
            for metric_name, val in metric2val.items():
                print(f'  {ds}/{var_name}/{metric_name}: {val:.4f}')

    # Compute pass@k
    print(f'\n{"=" * 50}')
    print('PASS@K RESULTS')
    print('=' * 50)

    problem_results = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        problem_id = r.get('instance', {}).get('instance_id', 'unknown')
        problem_results[problem_id]['total'] += 1
        if r.get('resolved', False):
            problem_results[problem_id]['correct'] += 1

    pass_k = compute_pass_k_metrics(dict(problem_results), k_values)
    for k_metric, k_result in sorted(pass_k.items()):
        print(
            f'  {k_metric}: {k_result["score"]:.4f} +/- {k_result["std"]:.4f} (n={k_result["num_problems"]})'
        )

    print('=' * 60)
    return val_metrics, pass_k


def parse_args():
    parser = argparse.ArgumentParser(
        description='Standalone SWE-bench validation script'
    )
    parser.add_argument(
        '--data_path', type=str, required=True, help='Path to test parquet file'
    )
    parser.add_argument(
        '--ProRL_Agent_Server_urls',
        type=str,
        required=True,
        help="ProRL Agent Server URLs separated by '+' (e.g. http://host1:8006+http://host2:8006)",
    )
    parser.add_argument(
        '--llm_server_urls',
        type=str,
        default=None,
        help="vLLM server URLs separated by '+' (e.g. http://host1:8100+http://host2:8100). "
        'If provided, these will be registered with ProRL Agent Server servers before evaluation.',
    )
    parser.add_argument(
        '--model_name', type=str, default='Qwen/Qwen3-32B', help='Model name for vLLM'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./test_results', help='Output directory'
    )
    parser.add_argument(
        '--num_trajectories',
        type=int,
        default=1,
        help='Number of trajectories per instance',
    )
    parser.add_argument(
        '--num_workers_per_server',
        type=int,
        default=64,
        help='Number of concurrent workers per ProRL Agent Server server',
    )
    parser.add_argument(
        '--max_retries', type=int, default=2, help='Max retries per job'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (0 = greedy)',
    )
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling')
    parser.add_argument(
        '--max_iterations', type=int, default=50, help='Max agent iterations'
    )
    parser.add_argument(
        '--max_output_tokens', type=int, default=1536, help='Max output tokens per turn'
    )
    parser.add_argument(
        '--max_model_len', type=int, default=32768, help='Max model context length'
    )
    parser.add_argument(
        '--timeout', type=int, default=1500, help='Timeout per request (seconds)'
    )
    parser.add_argument(
        '--hint_mode', type=str, default='none', help='Hint mode: none, file, old_str'
    )
    parser.add_argument(
        '--file_iou_coef',
        type=float,
        default=0.0,
        help='Coefficient for file IoU in reward',
    )
    parser.add_argument(
        '--block_iou_coef',
        type=float,
        default=0.0,
        help='Coefficient for block IoU in reward',
    )
    parser.add_argument(
        '--k_values',
        type=str,
        default='1,4,8,16',
        help='Comma-separated k values for pass@k',
    )
    parser.add_argument(
        '--token_level_generation',
        action='store_true',
        help='Enable token-level generation',
    )
    parser.add_argument(
        '--custom_tokenizer',
        type=str,
        default=None,
        help='Custom tokenizer path (defaults to model_name)',
    )
    parser.add_argument(
        '--native_tool_calling',
        action='store_true',
        default=True,
        help='Enable native tool calling',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse ProRL Agent Server URLs
    ProRL_Agent_Server_urls = [
        url.strip() for url in args.ProRL_Agent_Server_urls.split('+') if url.strip()
    ]
    logger.info(f'ProRL Agent Server servers: {ProRL_Agent_Server_urls}')

    # Register LLM servers with ProRL Agent Server if provided
    if args.llm_server_urls:
        llm_server_urls = [
            url.strip() for url in args.llm_server_urls.split('+') if url.strip()
        ]
        logger.info(f'LLM servers: {llm_server_urls}')
        asyncio.run(
            setup_llm_servers_with_ProRL_Agent_Server(
                ProRL_Agent_Server_urls, llm_server_urls, args.token_level_generation
            )
        )

    # Parse k values
    k_values = [int(k.strip()) for k in args.k_values.split(',')]

    # Load test data
    instances = load_test_data(args.data_path)

    # Set hint_mode on all instances
    for inst in instances:
        inst['instance']['hint_mode'] = args.hint_mode

    # Build sampling params
    sampling_params = {
        'model': f'hosted_vllm/{args.model_name}',
        'api_key': 'dummy_key',
        'modify_params': False,
        'log_completions': False,
        'native_tool_calling': args.native_tool_calling,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_iterations': args.max_iterations,
        'max_output_tokens': args.max_output_tokens,
        'token_level_generation': args.token_level_generation,
        'custom_tokenizer': args.custom_tokenizer or args.model_name,
        'max_model_len': args.max_model_len,
        'ensure_thinking_end_properly': not args.token_level_generation,
    }
    logger.info(f'Sampling params: {json.dumps(sampling_params, indent=2)}')

    # Run evaluation
    start_time = time.time()
    results = asyncio.run(
        run_ProRL_Agent_Server_evaluation(
            instances=[inst['instance'] for inst in instances],
            ProRL_Agent_Server_urls=ProRL_Agent_Server_urls,
            sampling_params=sampling_params,
            num_trajectories=args.num_trajectories,
            num_workers_per_server=args.num_workers_per_server,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout,
        )
    )
    elapsed = time.time() - start_time
    logger.info(f'Evaluation took {elapsed:.1f}s')

    # Attach data_source back to results
    for r in results:
        inst_id = r.get('instance', {}).get('instance_id', '')
        # Find matching source instance
        for src_inst in instances:
            if src_inst['instance'].get('instance_id') == inst_id.replace(
                f'/{r["instance"].get("trajectory_id", 0)}', ''
            ):
                r['instance']['data_source'] = src_inst.get('data_source', 'unknown')
                break
        if 'data_source' not in r.get('instance', {}):
            r['instance']['data_source'] = 'unknown'

    # Compute rewards
    results = compute_rewards(
        results, file_iou_coef=args.file_iou_coef, block_iou_coef=args.block_iou_coef
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'results.jsonl')
    save_results_jsonl(results, output_path)

    # Print summary
    val_metrics, pass_k = print_summary(
        results, instances, args.num_trajectories, k_values
    )

    # Save metrics as JSON
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    all_metrics = {
        'elapsed_seconds': elapsed,
        'num_instances': len(instances),
        'num_trajectories': args.num_trajectories,
        'total_evaluations': len(results),
        'resolved_count': sum(1 for r in results if r.get('resolved')),
        'resolve_rate': sum(1 for r in results if r.get('resolved')) / len(results)
        if results
        else 0,
        'mean_score': float(np.mean([r.get('score', 0) for r in results])),
        'mean_file_iou': float(np.mean([r.get('file_iou', 0) for r in results])),
        'mean_block_iou': float(np.mean([r.get('block_iou', 0) for r in results])),
        'pass_k': {k: v for k, v in pass_k.items()},
        'sampling_params': sampling_params,
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f'Saved metrics to {metrics_path}')


if __name__ == '__main__':
    main()
