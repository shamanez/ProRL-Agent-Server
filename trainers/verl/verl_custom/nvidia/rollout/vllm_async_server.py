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
import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from verl.utils.fs import copy_to_local
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.utils import random_uuid
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.v1.worker.worker_base import WorkerWrapperBase

from verl_custom.nvidia.rollout.async_server import AsyncServerBase

logger = logging.getLogger(__file__)


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, (
            'instance_id must be set for external ray actors.'
        )

        fields = self.vllm_config.instance_id.split(':')
        assert len(fields) == 4, (
            f'instance_id: {self.vllm_config.instance_id} must be in the format of <namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>.'
        )
        namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = (
            fields[0],
            fields[1],
            int(fields[2]),
            int(fields[3]),
        )

        # Make sure subprocess in same namespace as parent actor.
        # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
        ray.init(address='auto', namespace=namespace)
        actor_names = [
            actor_name
            for actor_name in ray.util.list_named_actors()
            if actor_name.startswith(f'{wg_prefix}WorkerDict')
        ]

        vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
            f'instance_id: {self.vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: {vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected.'
        )

        def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
            fields = actor_name.split(':')
            assert len(fields) == 2, f'invalid actor name: {actor_name}'
            pg_index, local_rank = int(fields[0].split('_')[-1]), int(fields[1])
            return pg_index, local_rank

        # sort actor names by pg_index and local_rank
        actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
        actor_names = actor_names[
            vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size
        ]
        self.workers: List[WorkerWrapperBase] = [
            ray.get_actor(actor_name) for actor_name in actor_names
        ]
        print(
            f'instance_id: {self.vllm_config.instance_id} initializes with external actors: {actor_names}'
        )

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method='env://',
            is_driver_worker=True,
        )
        self.collective_rpc('init_worker', args=([kwargs],))
        self.collective_rpc('init_device')
        self.collective_rpc('load_model')
        print(f'instance_id: {self.vllm_config.instance_id} initializes finished.')

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        non_block: bool = False,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        # ~3ms overhead per schedule step due to SchedulerOutput/ModelRunnerOutput serialization/deserialization.
        refs = [
            worker.execute_method.remote(sent_method, *args, **(kwargs or {}))
            for worker in self.workers
        ]
        # vLLM 0.18: when non_block=True the engine expects each list element to be a
        # Future whose .result() returns the worker output. The abstract executor's
        # execute_model() does `output = collective_rpc(..., non_block=True); output[0]`
        # then `output[0].result()`.
        if non_block:
            from vllm.v1.executor.ray_utils import FutureWrapper

            return [FutureWrapper(ref) for ref in refs]
        return ray.get(refs, timeout=timeout)

    def check_health(self):
        return


@ray.remote(num_cpus=1)
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout managers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(
        self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str
    ):
        """
        Args:
            config: DictConfig.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__()

        self.config = config.actor_rollout_ref
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None
        self.request_ids = set()

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = '/'.join(model_path.split('/')[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get('trust_remote_code', False)
        config = config.rollout

        tensor_parallel_size = config.get('tensor_model_parallel_size', 1)
        max_num_batched_tokens = config.get('max_num_batched_tokens', 8192)
        max_model_len = (
            config.max_model_len
            if config.max_model_len
            else config.prompt_length + config.response_length
        )
        max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            max_new_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f'override_generation_config: {kwargs}')

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format='auto',
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
            logprobs_mode=config.get('logprobs_mode', 'processed_logprobs'),
        )

        # init async llm engine
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = (
            f'{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}'
        )
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

    async def generate(self, raw_request: Request) -> list[int]:
        request_dict = await raw_request.json()
        # Set logprobs to 0 to only return selected tokens
        request_dict['logprobs'] = 0

        prompt_ids = request_dict.pop('prompt_ids')
        # vLLM 0.18 SamplingParams rejects unknown kwargs (the client sends
        # fields like `stream`, `tools`, `tool_choice` that aren't sampling
        # fields). Filter to only params SamplingParams actually accepts.
        _sp_fields = set(inspect.signature(SamplingParams).parameters)
        sampling_kwargs = {k: v for k, v in request_dict.items() if k in _sp_fields}
        sampling_params = SamplingParams(**sampling_kwargs)
        request_id = random_uuid()
        self.request_ids.add(request_id)
        # Create prompt from token IDs
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        generator = self.engine.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        try:
            async for output in generator:
                final_res = output
        except asyncio.CancelledError:
            return Response(status_code=499)
        finally:
            self.request_ids.remove(request_id)

        assert final_res is not None

        def obtain_logprobs(logprobs):
            if logprobs is None:
                return None
            log_probs = []
            for d in logprobs:
                cur_logprobs = list(d.values())
                assert len(cur_logprobs) == 1, (
                    f'Expected 1 logprob per token when logprobs=0, but got {len(cur_logprobs)}'
                )
                log_probs.append(cur_logprobs[0].logprob)
            return log_probs

        ret = {
            'response_ids': final_res.outputs[0].token_ids,
            'logprobs': obtain_logprobs(final_res.outputs[0].logprobs),
        }
        return JSONResponse(ret)

    async def wake_up(self):
        await self.engine.wake_up()

    async def sleep(self):
        # Abort all unfinished generation requests
        await self._abort_requests()
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()

    async def _abort_requests(self):
        request_states = self.engine.output_processor.request_states
        request_ids = list(request_states.keys())
        print(f'abort {len(request_ids)} request_ids: {request_ids}')

        # abort all unfinished generation requests
        # vLLM 0.18+: abort_requests requires `internal` param
        self.engine.output_processor.abort_requests(request_ids, internal=False)
        await self.engine.engine_core.abort_requests_async(request_ids)

    def get_tokenizer(self):
        """Get the tokenizer from the engine."""
        if self.engine is None:
            raise RuntimeError('Engine not initialized. Call init_engine() first.')

        return self.engine.tokenizer
