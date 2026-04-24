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

"""
Async LLM Server Manager for VERL Framework

This module provides the AsyncLLMServerManager class that manages multiple async LLM server instances
for distributed inference and rollout generation. It handles:
- Server lifecycle management (start/stop/wake/sleep)
- Request distribution and load balancing
- Integration with OpenHands servers for multi-turn conversations
- Message formatting and tokenization
- Result aggregation and conversion to DataProto format

The manager supports both single-node and multi-node deployments with tensor parallelism.
"""

# Standard library imports for async operations, logging, and networking
import asyncio
import logging
import os
import uuid

import numpy as np

# Ray framework for distributed computing
from omegaconf import DictConfig

# VERL framework imports
from verl.protocol import DataProto  # Data protocol for tensor/non-tensor data exchange
from verl.single_controller.ray.base import (
    RayWorkerGroup,  # Ray worker group management
)

# async_server_class removed in verl v0.8 (replaced by RolloutReplica pattern)
# Logging configuration
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_LOGGING_LEVEL', 'INFO'))

# Chat template management for different model types

# PyTorch and HTTP client imports

import aiohttp

from verl_custom.nvidia.rollout.async_server import AsyncLLMServerManager

from .utils import get_unique_id


class AsyncLLMServerManagerDAPO(AsyncLLMServerManager):
    """
    AsyncLLMServerManagerDAPO
    """

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """
        Initialize the AsyncLLMServerManagerDAPO.
        """
        super().__init__(config, worker_group)
        self.data_loader = None
        self.all_input_batch = None
        self.last_data_index = 0
        self.job_queue = asyncio.PriorityQueue()

    def generate_sequences_dapo(self) -> DataProto:
        import time

        # Cut 4 / gotcha #16: under the continuous-producer, this method is
        # invoked in a loop from a daemon thread. The instance attributes
        # ``all_input_batch`` and ``last_data_index`` are stateful across
        # calls in the classic path (unfinished filter_groups rows carry
        # over into the next call via ``push_remaining_train_data_to_job_queue``).
        # Under producer mode the buffer absorbs the filtered batch, so
        # carrying leftover state would cause ``DataProto.concat`` (line
        # ~225) to fuse prior-call rows into the new call and corrupt the
        # assert at line ~512 (instance-id conservation). Reset here.
        #
        # job_queue leftovers: the result loop (line ~472) breaks as soon as
        # ``num_completed_instances >= requested_batch_size``, with up to N
        # un-dispatched jobs still in ``self.job_queue``. The classic path
        # keeps them — they're the head of the next call's batch. Producer
        # mode can't, because we just reset ``all_input_batch`` above, so
        # those jobs reference forgotten prompts. Additionally, each call
        # runs inside its own ``asyncio.run`` event loop (line ~107), and a
        # PriorityQueue constructed against a now-closed loop can raise
        # from ``put``/``get`` in Python 3.12. Rebuild the queue each call.
        replay_cfg = getattr(self.full_config, 'replay', None)
        producer_mode = bool(
            replay_cfg is not None
            and replay_cfg.get('enable', False)
            and replay_cfg.get('continuous_producer', False)
        )
        if producer_mode:
            self.all_input_batch = None
            self.last_data_index = 0
            leftover_jobs = self.job_queue.qsize()
            self.job_queue = asyncio.PriorityQueue()
            if leftover_jobs:
                logger.info(
                    'DAPO producer-mode: dropped %d leftover jobs from prior '
                    'generate_sequences_dapo call (rebuilt job_queue)',
                    leftover_jobs,
                )

        # Start total timing for performance analysis
        total_start_time = time.time()
        request_start_time = time.time()
        output_messages, input_batch = asyncio.run(self.request_from_openhands_dapo())
        request_end_time = time.time()

        convert_results_start_time = time.time()
        if self.config.rollout.get('token_level_generation', False):
            response = self._convert_results_to_dataproto_token(
                output_messages, input_batch
            )
        else:
            response = self._convert_results_to_dataproto(output_messages)

        convert_results_end_time = time.time()
        total_end_time = time.time()

        # Log detailed timing information for performance monitoring
        logger.info(
            f'convert_results_to_dataproto time: {convert_results_end_time - convert_results_start_time:.3f}s'
        )

        input_batch.non_tensor_batch['uid'] = np.array(
            [str(uuid.uuid4()) for _ in range(len(input_batch.batch))], dtype=object
        )
        input_batch = input_batch.repeat(
            repeat_times=self.config.rollout.n, interleave=True
        )
        out_batch = input_batch.union(response)
        # Add comprehensive timing breakdown
        # Ensure meta_info exists for timing data
        if not hasattr(out_batch, 'meta_info') or out_batch.meta_info is None:
            out_batch.meta_info = {}
        out_batch.meta_info['timing'] = {
            'total': total_end_time - total_start_time,
            'openhands_request': request_end_time - request_start_time,
            'convert_results_to_dataproto': convert_results_end_time
            - convert_results_start_time,
            'generate_sequences': total_end_time
            - total_start_time,  # Main timing metric
        }

        # latencies.md §5 addition #1 — single JSON event summarizing this
        # producer call. Consumers: log-scraper in stages/replay_dynamics.md
        # §9 decision table (effective_tps, groups_drawn/survived/dropped).
        import json as _json  # noqa: PLC0415

        wall_s = total_end_time - total_start_time
        stats = getattr(self, '_last_dapo_call_stats', {}) or {}
        try:
            tokens_out = int(out_batch.batch['responses'].numel())
            trajectories_out = int(out_batch.batch['responses'].shape[0])
        except Exception:  # noqa: BLE001
            tokens_out = 0
            trajectories_out = 0
        effective_tps = float(tokens_out / wall_s) if wall_s > 0 else 0.0
        logger.info(
            'DAPO_PRODUCER_CALL %s',
            _json.dumps(
                {
                    'event': 'dapo_producer_call',
                    'wall_s': round(wall_s, 3),
                    'openhands_s': round(request_end_time - request_start_time, 3),
                    'convert_s': round(
                        convert_results_end_time - convert_results_start_time, 3
                    ),
                    'groups_drawn': stats.get('groups_drawn', 0),
                    'groups_survived': stats.get('groups_survived', 0),
                    'groups_dropped_filter': stats.get('groups_dropped_filter', 0),
                    'trajectories_out': trajectories_out,
                    'tokens_out': tokens_out,
                    'effective_tps': round(effective_tps, 2),
                }
            ),
        )
        return out_batch

    def process_batch(self, batch_dict):
        batch = DataProto.from_single_dict(batch_dict)
        batch_keys_to_pop = ['input_ids', 'attention_mask', 'position_ids']
        non_tensor_batch_keys_to_pop = ['raw_prompt_ids']

        batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        return batch

    async def refill_job_queue(self, data_index):
        """
        Refile the job queue with the remaining messages.
        """
        assert self.data_loader is not None
        next_batch = next(iter(self.data_loader))
        batch = self.process_batch(next_batch)
        messages = self.DataProto2Messages(batch)
        for message in messages:
            await self.job_queue.put((data_index, (data_index, message, 0)))
            data_index += 1
        return batch, data_index

    async def request_from_openhands_dapo(self):
        import time

        requested_batch_size = self.full_config.data.train_batch_size
        # Start total timing for performance analysis
        total_start_time = time.time()

        # Start all OpenHands servers concurrently
        server_start_time = time.time()
        await self.start_servers()
        server_end_time = time.time()
        logger.info(
            f'Starting OpenHands servers took: {server_end_time - server_start_time:.2f} seconds'
        )

        self.existing_ids = set()

        try:
            # Initialize result storage
            all_responses = {}

            if not self.openhands_urls:
                logger.error('No OpenHands base URLs configured')
                raise ValueError(
                    'No OpenHands base URLs configured. Please set actor_rollout_ref.rollout.openhands_base_url.'
                )

            # Thread-safe queue management with asyncio locks
            job_queue_lock = asyncio.Lock()
            results_queue_lock = asyncio.Lock()
            available_servers_queue_lock = asyncio.Lock()

            # for i, message in enumerate(messages):
            #     await job_queue.put((i, message, 0))  # (index, message, retry_count)
            # Create results queue for completed work
            results_queue = asyncio.PriorityQueue()

            # Create server availability queue
            available_servers_queue = asyncio.Queue()

            # Track all active process_job_on_server tasks for cleanup
            active_job_tasks = []
            active_job_tasks_lock = asyncio.Lock()

            # Initialize server pool - all servers start as available
            max_active_tasks = 0
            for server_url in self.openhands_urls:
                for worker_idx in range(self.openhands_num_workers):
                    await available_servers_queue.put(
                        f'{server_url}|worker_{worker_idx}'
                    )
                    max_active_tasks += 1

            logger.info(
                f'Initialized {available_servers_queue.qsize()} available server workers'
            )

            # Job dispatcher - assigns jobs to available servers
            async def job_dispatcher():
                """Continuously dispatch jobs to available servers."""
                data_index = self.last_data_index

                while True:
                    try:
                        # Check for available work and servers
                        async with job_queue_lock:
                            async with available_servers_queue_lock:
                                if available_servers_queue.empty():
                                    await asyncio.sleep(0.1)  # Brief pause before retry
                                    continue
                                if self.job_queue.empty():
                                    (
                                        input_batch,
                                        data_index,
                                    ) = await self.refill_job_queue(data_index)
                                    if self.all_input_batch is None:
                                        self.all_input_batch = input_batch
                                    else:
                                        self.all_input_batch = DataProto.concat(
                                            [self.all_input_batch, input_batch]
                                        )
                                    continue
                                # Get next job and available server
                                job_task = asyncio.create_task(self.job_queue.get())
                                server_task = asyncio.create_task(
                                    available_servers_queue.get()
                                )

                                # Wait for both to be ready
                                done, pending = await asyncio.wait(
                                    [job_task, server_task],
                                    return_when=asyncio.ALL_COMPLETED,
                                )

                                # Extract job and server information
                                priority, (message_index, message, retry_count) = (
                                    job_task.result()
                                )
                                server_worker_id = server_task.result()
                                server_url = server_worker_id.split('|')[0]

                                logger.debug(
                                    f'Dispatching message {message_index} to {server_worker_id}'
                                )
                                logger.debug(
                                    f'Available servers: {available_servers_queue.qsize()}'
                                )

                        # Create async task to process job on assigned server
                        task = asyncio.create_task(
                            process_job_on_server(
                                message_index,
                                message,
                                retry_count,
                                server_url,
                                server_worker_id,
                            )
                        )
                        # Track task for cleanup
                        async with active_job_tasks_lock:
                            active_job_tasks.append(task)

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f'Job dispatcher error: {e}')

            # Process individual job on specific server
            async def process_job_on_server(
                message_index, message, retry_count, server_url, server_worker_id
            ):
                """Process a single job on an assigned server."""
                current_task = asyncio.current_task()
                try:
                    # Send request to OpenHands server
                    (
                        result,
                        should_retry,
                    ) = await self._send_single_message_to_openhands_dapo(
                        message=message,
                        message_index=message_index,
                        openhands_base_url=server_url,
                    )

                    # Handle retry logic
                    if should_retry and retry_count < 2:  # Maximum 3 attempts
                        async with job_queue_lock:
                            await self.job_queue.put(
                                (
                                    message_index,
                                    (message_index, message, retry_count + 1),
                                )
                            )
                        logger.info(
                            f'Retrying message {message_index}, attempt {retry_count + 1}'
                        )
                    else:
                        # Job completed (success or max retries reached)
                        async with results_queue_lock:
                            await results_queue.put(
                                (message_index, (message_index, message, result))
                            )
                        async with job_queue_lock:
                            self.job_queue.task_done()

                    # Return server to available pool
                    async with available_servers_queue_lock:
                        await available_servers_queue.put(server_worker_id)
                        logger.debug(f'Returned server to pool: {server_worker_id}')

                except asyncio.CancelledError:
                    logger.info(f'Job {message_index} processing was cancelled')
                    # Return server to pool even when cancelled
                    async with available_servers_queue_lock:
                        await available_servers_queue.put(server_worker_id)
                    raise  # Re-raise to properly handle cancellation
                except Exception as e:
                    logger.error(f'Error processing job {message_index}: {e}')
                    # Return server to pool even on error
                    async with available_servers_queue_lock:
                        await available_servers_queue.put(server_worker_id)
                finally:
                    # Remove task from tracking list
                    async with active_job_tasks_lock:
                        if current_task in active_job_tasks:
                            active_job_tasks.remove(current_task)

            # Start job dispatcher
            dispatcher_task = asyncio.create_task(job_dispatcher())

            logger.info(
                f'Started job dispatcher with {max_active_tasks} concurrent workers '
            )

            # Collect results as they arrive
            processing_start_time = time.time()
            completed_count = 0
            filtered_instance_ids = []
            while True:
                async with results_queue_lock:
                    if results_queue.empty():
                        await asyncio.sleep(0.1)
                        continue
                    (
                        priority,
                        (message_index, message, result),
                    ) = await results_queue.get()
                # Process and store result
                instance_id = message['instance_id']
                trajectory_id = message['trajectory_id']

                if instance_id not in all_responses:
                    all_responses[instance_id] = {}

                # Store result with proper error handling
                if isinstance(result, Exception):
                    logger.error(f'Error processing message {message_index}: {result}')
                    all_responses[instance_id][trajectory_id] = {
                        'error': str(result),
                        'success': False,
                        'messages': [],
                        'resolved': False,
                        'finish': False,
                    }
                elif result is not None:
                    all_responses[instance_id][trajectory_id] = result
                else:
                    logger.warning(f'Message {message_index} returned empty response')
                    all_responses[instance_id][trajectory_id] = {
                        'error': 'Empty response',
                        'success': False,
                        'messages': [],
                        'resolved': False,
                        'finish': False,
                    }

                completed_count += 1
                num_completed_instances = 0

                all_responses, filtered_instance_ids_tmp = (
                    self.filter_easy_hard_instance(all_responses)
                )
                filtered_instance_ids.extend(filtered_instance_ids_tmp)

                for instance_id in all_responses:
                    if len(all_responses[instance_id]) == self.num_trajectories:
                        num_completed_instances += 1
                logger.error(f'num_completed_instances: {num_completed_instances}')

                # Log progress periodically
                if (
                    completed_count % 4 == 0
                    or num_completed_instances == requested_batch_size
                ):
                    current_time = time.time()
                    elapsed_time = current_time - processing_start_time
                    progress_percent = (
                        num_completed_instances / requested_batch_size
                    ) * 100

                    async with job_queue_lock:
                        pending_jobs = self.job_queue.qsize()
                    async with available_servers_queue_lock:
                        available_servers = available_servers_queue.qsize()

                    active_tasks = max_active_tasks - available_servers
                    # Also log the resolved ratio and finish ratio, use all_responses[instance_id][trajectory_id]['resolved'] to find whether the trajectory of this instance is resolved
                    # You need 2 loops to go through all_responses[instance_id][trajectory_id]['resolved'] to find the resolved ratio and finish ratio
                    resolved_count = 0
                    finish_count = 0
                    all_results_count = 0
                    resolved_instance_count = 0
                    for instance_id in all_responses:
                        resolved_instance_count_tmp = 0
                        for trajectory_id in all_responses[instance_id]:
                            all_results_count += 1
                            if all_responses[instance_id][trajectory_id]['resolved']:
                                resolved_count += 1
                                resolved_instance_count_tmp = 1
                            if all_responses[instance_id][trajectory_id]['finish']:
                                finish_count += 1
                        if resolved_instance_count_tmp == 1:
                            resolved_instance_count += 1
                    resolved_ratio = (
                        resolved_count / all_results_count if all_results_count else 0
                    )
                    finish_ratio = (
                        finish_count / all_results_count if all_results_count else 0
                    )
                    resolved_instance_ratio = (
                        resolved_instance_count / len(all_responses)
                        if all_responses
                        else 0
                    )
                    logger.info(
                        f'Progress: {num_completed_instances}/{requested_batch_size} ({progress_percent:.1f}%), '
                        f'pending: {pending_jobs}, available: {available_servers}, '
                        f'active: {active_tasks}/{max_active_tasks}, elapsed: {elapsed_time:.2f}s, '
                        f'resolved: {resolved_ratio:.2f}, finish: {finish_ratio:.2f}, resolved_instance: {resolved_instance_ratio:.2f}'
                    )

                if num_completed_instances >= requested_batch_size:
                    logger.error(
                        f'Completed {num_completed_instances} instances, requested batch size is {requested_batch_size}'
                    )
                    break

            # Clean up all tasks
            logger.info('Stopping all tasks...')

            # Cancel job dispatcher
            dispatcher_task.cancel()

            # Cancel all active process_job_on_server tasks
            async with active_job_tasks_lock:
                active_tasks_to_cancel = active_job_tasks.copy()

            if active_tasks_to_cancel:
                logger.info(
                    f'Cancelling {len(active_tasks_to_cancel)} active job processing tasks'
                )
                for task in active_tasks_to_cancel:
                    task.cancel()

            # Wait for all tasks to complete or be cancelled
            all_tasks = [dispatcher_task] + active_tasks_to_cancel
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)
                logger.info('All tasks have been stopped')

            processing_end_time = time.time()
            total_end_time = time.time()
            logger.info(
                f'Request processing completed in {processing_end_time - processing_start_time:.2f}s'
            )
            logger.info(f'Total request time: {total_end_time - total_start_time:.2f}s')
            logger.info(f'Processed {len(all_responses)} instances successfully')
            # the set of instance_ids in self.all_input_batch before filtering
            instance_ids_before_filtering = set(
                [
                    instance['instance_id']
                    for instance in self.all_input_batch.non_tensor_batch['instance']
                ]
            )
            # filter the unfinished instances in all_responses and self.all_input_batch and reamin unfinished instances in self.all_input_batch
            all_responses, output_batch = self.filter_uncompleted_instances(
                all_responses, filtered_instance_ids
            )
            # the set of instance_ids in output_batch
            instance_ids_in_output_batch = set(
                [
                    instance['instance_id']
                    for instance in output_batch.non_tensor_batch['instance']
                ]
            )
            # the set of instance_ids in self.all_input_batch after filtering
            instance_ids_after_filtering = set(
                [
                    instance['instance_id']
                    for instance in self.all_input_batch.non_tensor_batch['instance']
                ]
            )
            # the set of instance_ids in filtered_instance_ids
            instance_ids_in_filtered_instance_ids = set(filtered_instance_ids)
            # assert instance_ids_before_filtering is equal to instance_ids_in_all_input_batch_after_filtering + instance_ids_in_output_batch (how to compare two sets?)
            assert instance_ids_before_filtering == instance_ids_after_filtering.union(
                instance_ids_in_output_batch
            ).union(instance_ids_in_filtered_instance_ids), (
                f'The instance_ids before filtering and the instance_ids in the output batch are not the same, {instance_ids_before_filtering} != {instance_ids_after_filtering} + {instance_ids_in_output_batch} + {instance_ids_in_filtered_instance_ids}'
            )
            # put the remaining train data back to the job queue
            await self.push_remaining_train_data_to_job_queue()

            # latencies.md §5 addition #1 — stash filter-group counts for
            # generate_sequences_dapo to emit alongside wall/throughput.
            self._last_dapo_call_stats = {
                'groups_drawn': len(instance_ids_before_filtering),
                'groups_survived': len(instance_ids_in_output_batch),
                'groups_dropped_filter': len(instance_ids_in_filtered_instance_ids),
            }

            return all_responses, output_batch
        except Exception as e:
            logger.error(f'Error in request_from_openhands_dapo: {e}')
            raise

        finally:
            # Always stop servers for cleanup
            stop_start_time = time.time()
            await self.stop_servers()
            stop_end_time = time.time()
            logger.info(f'Server shutdown took: {stop_end_time - stop_start_time:.2f}s')

    async def push_remaining_train_data_to_job_queue(self):
        """
        Push the remaining train data to the job queue
        """
        self.last_data_index = 0
        messages = self.DataProto2Messages(self.all_input_batch)
        for message in messages:
            await self.job_queue.put(
                (self.last_data_index, (self.last_data_index, message, 0))
            )
            self.last_data_index += 1

    def filter_uncompleted_instances(self, all_responses, filtered_instance_ids):
        """
        Filter the uncompleted instances
        """
        instance_id_to_batch_idx = {}
        for i, instance in enumerate(self.all_input_batch.non_tensor_batch['instance']):
            instance_id = instance['instance_id']
            instance_id_to_batch_idx[instance_id] = i
        instance_ids_to_keep = []
        batch_idx_to_keep = []
        instance_ids = list(all_responses.keys())
        for instance_id in instance_ids:
            if len(all_responses[instance_id]) != self.num_trajectories:
                all_responses.pop(instance_id)
            else:
                instance_ids_to_keep.append(instance_id)
                batch_idx_to_keep.append(instance_id_to_batch_idx[instance_id])
        output_batch = self.all_input_batch.select_idxs(batch_idx_to_keep)
        # remain the instances that are not in filtered_instance_ids and not in instance_ids_to_keep
        batch_idx_to_remain = []
        for i, instance in enumerate(self.all_input_batch.non_tensor_batch['instance']):
            if (
                instance['instance_id'] not in filtered_instance_ids
                and instance['instance_id'] not in instance_ids_to_keep
            ):
                batch_idx_to_remain.append(i)
        self.all_input_batch = self.all_input_batch.select_idxs(batch_idx_to_remain)
        assert (
            len(output_batch.non_tensor_batch['instance']) == len(all_responses)
            and len(all_responses) == self.full_config.data.train_batch_size
        ), (
            f'The number of instances in the input batch and the number of instances in the responses are not the same, {len(output_batch.non_tensor_batch["instance"])} != {len(all_responses)} != {self.full_config.data.train_batch_size}'
        )
        return all_responses, output_batch

    def filter_easy_hard_instance(self, all_responses):
        """
        Filter the instance with the resolved ratio
        """
        filtered_instance_ids = []
        keys = list(all_responses.keys())
        for instance_id in keys:
            if len(all_responses[instance_id]) == self.num_trajectories:
                resolved_count = 0
                for trajectory_id in all_responses[instance_id]:
                    if all_responses[instance_id][trajectory_id]['resolved']:
                        resolved_count += 1
                if resolved_count == self.num_trajectories or resolved_count == 0:
                    all_responses.pop(instance_id)
                    filtered_instance_ids.append(instance_id)
                    logger.info(
                        f'Filtered instance {instance_id} with resolved ratio {resolved_count}/{self.num_trajectories}'
                    )
        return all_responses, filtered_instance_ids

    async def _send_single_message_to_openhands_dapo(
        self, message: dict, message_index: int, openhands_base_url: str, val_mode=False
    ):
        """
        Send single message to openhands server.

        This method handles the HTTP communication with OpenHands servers to process
        a single conversation request. It includes comprehensive error handling,
        timeout management, and retry logic.

        Args:
            message (dict): A single message dictionary containing the conversation
                to be processed by OpenHands.
            message_index (int): The index of the message in the input batch.
            openhands_base_url (str): The base URL of the OpenHands server.

        Returns:
            tuple: A tuple (result, should_retry) where:
                - result (dict or Exception): The response from OpenHands or an exception.
                - should_retry (bool): True if the message should be retried, False otherwise.
        """
        try:
            request_data = {
                'job_id': get_unique_id(message, self.existing_ids),
                'instance': message,
                'sampling_params': self.val_sampling_params
                if val_mode
                else self.sampling_params,
            }
            self.existing_ids.add(request_data['job_id'])
            timeout = aiohttp.ClientTimeout(
                total=self.config.rollout.get('openhands_timeout', 1000)
            )
            session = aiohttp.ClientSession(timeout=timeout)

            try:
                async with session.post(
                    url=f'{openhands_base_url}/process',
                    headers={'Content-Type': 'application/json'},
                    json=request_data,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        out_message = data.get('messages', [])
                        if out_message and len(out_message) > 0:
                            return data, False  # No retry needed
                        else:
                            return None, True  # Retry needed
                    else:
                        error_text = await resp.text()
                        logger.error('Failed to process request:', error_text)
                        return None, True  # Retry needed

            finally:
                await session.close()

        except asyncio.CancelledError:
            success = await self.cancel_job(request_data['job_id'], openhands_base_url)
            if not success:
                logger.error(f'Failed to cancel job {request_data["job_id"]}')

            raise  # Re-raise to properly handle cancellation
        except asyncio.TimeoutError as e:
            logger.error(
                f'Timeout error sending message {message_index} to OpenHands: {e}'
            )
            success = await self.cancel_job(request_data['job_id'], openhands_base_url)
            if not success:
                logger.error(f'Failed to cancel job {request_data["job_id"]}')

            return None, True  # Retry needed
        except Exception as e:
            logger.error(f'Error sending message {message_index} to OpenHands: {e}')
            success = await self.cancel_job(request_data['job_id'], openhands_base_url)
            if not success:
                logger.error(f'Failed to cancel job {request_data["job_id"]}')

            return None, True  # Retry needed

    async def cancel_job(self, job_id: str, openhands_base_url: str):
        """
        Cancel a job by job_id
        """
        url = f'{openhands_base_url}/cancel'
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={'job_id': job_id}) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        return False
        except Exception:
            return False
