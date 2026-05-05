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

from typing import Any, Optional
import json
import httpx
from evaluation.utils.shared import EvalMetadata  # type: ignore
from openhands.core.config import OpenHandsConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.nvidia.registry import AgentHandler, JobDetails
from openhands.runtime.base import Runtime
from openhands.nvidia.utils import (
    initialize_exception,
    run_exception,
    eval_exception,
    final_result as utils_final_result,
)
from openhands.nvidia.logger import nvidia_logger as logger

from openhands.nvidia.reward import Reward
import pandas as pd
import numpy as np
import os

def normalize_prompt(x):
    def convert_numpy_recursive(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_recursive(item) for item in obj]
        else:
            return obj

    x = convert_numpy_recursive(x)

    if isinstance(x, str):
        return x
    elif isinstance(x, (dict, list, tuple)):
        try:
            return json.dumps(x)
        except Exception:
            raise ValueError(f"x is not a string: {x}")
    elif pd.isnull(x):
        raise ValueError(f"x is null: {x}")
    else:
        raise ValueError(f"x is not a string: {x}")

class ProRLHandler(AgentHandler):
    """Handler for Single-turn ProRL integration, using direct token IDs from VERL/ProRL data."""

    def __init__(self):
        """Initialize the ProRLHandler with tokenizer."""
        super().__init__()
        from transformers import AutoTokenizer
        tokenizer_name = os.environ.get('PRORL_TOKENIZER', 'Qwen/Qwen3-8B')
        logger.info(f"ProRLHandler: Using tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @property
    def name(self) -> str:
        """The name identifier for this agent handler."""
        return "prorl"

    async def init(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ):
        """Initialize the ProRL Agent with instance and config validation."""
        instance = job_details.instance
        llm_config = job_details.llm_config

        # Validate configuration
        if not llm_config or not llm_config.base_url:
            raise ValueError("LLM server configuration is required")

        if not instance:
            raise ValueError("Instance is required")

        # Return None values since we don't need runtime for ProRL
        return (None, None, None)

    async def run(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ) -> dict[str, object]:
        """Run the ProRL agent using direct token IDs from VERL/ProRL data with /generate endpoint."""

        try:
            # Get configuration
            instance :pd.Series = job_details.instance
            llm_config = job_details.llm_config

            assert llm_config is not None, "LLM config is required"
            assert self.tokenizer is not None, "Tokenizer is not initialized"

            if isinstance(instance['prompt'], str):
                chat = json.loads(instance.pop('prompt'))
            else:
                chat = instance.pop('prompt')

            tools = None
            if 'tools' in instance and instance['tools'] is not None and instance['tools'] != "":
                tools = json.loads(instance.pop('tools')) # tools from the function-calling training data (tool-access)
            else:
                if 'tools' in instance:
                    instance.pop('tools')

            # Get input token IDs
            raw_prompt = self.tokenizer.apply_chat_template(chat, tools=tools, add_generation_prompt=True, tokenize=False)
            input_token_ids = self.tokenizer(raw_prompt, add_special_tokens=True).pop("input_ids")
            logger.info(f"Successfully tokenized prompt to {len(input_token_ids)} tokens")

            # Prepare request to vLLM /generate endpoint
            # This follows the exact pattern from qwen3.py
            assert llm_config.base_url is not None, "LLM base_url is required"
            base_url = llm_config.base_url.rstrip('/')
            generate_endpoint = f"{base_url}/generate"

            # Calculate max_tokens based on model context length
            max_model_len = getattr(llm_config, 'max_model_len', 8192)  # Default context length
            max_tokens = max_model_len - len(input_token_ids)

            if max_tokens <= 0:
                raise ValueError(f"Input length ({len(input_token_ids)}) exceeds model context limit ({max_model_len})")

            # Prepare the request payload for vLLM generate endpoint
            # This follows the exact format from qwen3.py and matches VERL interface
            request_payload = {
                'prompt_ids': input_token_ids,  # Send token IDs directly from VERL/ProRL
                'max_tokens': max_tokens,
                'top_p': getattr(llm_config, 'top_p', 1.0),
                'temperature': llm_config.temperature or 1.0,
            }

            messages = chat.copy()
            for message in messages:
                message['input_ids'] = None
                message['token_ids'] = None
                message['logprobs'] = None
            messages[-1]['token_ids'] = input_token_ids

            # Add seed if available
            if hasattr(llm_config, 'seed') and llm_config.seed is not None:
                request_payload['seed'] = llm_config.seed

            # Add other optional parameters
            if hasattr(llm_config, 'top_k') and llm_config.top_k is not None:
                request_payload['top_k'] = llm_config.top_k

            logger.info(f"Making request to {generate_endpoint} with {len(input_token_ids)} input tokens from VERL/ProRL")

            # Make the request to the generate endpoint
            async with httpx.AsyncClient(timeout=httpx.Timeout(500.0)) as client:
                headers = {'Content-Type': 'application/json'}

                # Add API key if available
                if llm_config.api_key:
                    headers['Authorization'] = f'Bearer {llm_config.api_key.get_secret_value()}'

                response = await client.post(
                    generate_endpoint,
                    json=request_payload,
                    headers=headers
                )

                if response.status_code != 200:
                    raise Exception(f"LLM server returned status {response.status_code}: {response.text}")

                # Parse the response
                response_data = response.json()

                # Extract token IDs from vLLM response
                # This follows the exact pattern from qwen3.py
                if 'response_ids' not in response_data:
                    raise ValueError(f"Response does not contain response_ids: {response_data}")

                output_token_ids = response_data['response_ids']
                response_input_ids = response_data.get('input_ids', input_token_ids)  # Use response input_ids if available
                logprobs = response_data.get('logprobs', None)

                logger.info(f"Received {len(output_token_ids)} output tokens from LLM server")

                # Create message structure compatible with existing handlers
                # Include both token IDs and minimal text for compatibility
                message = {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': None,
                    'input_ids': response_input_ids,      # Input token IDs from ProRL
                    'token_ids': output_token_ids,        # For compatibility
                    'logprobs': logprobs,
                }
                messages.append(message)

                # Check if generation ended properly
                # end_properly = False if we hit max tokens without EOS token
                end_properly = True

                # Check if we hit max tokens (indicating truncation)
                if len(output_token_ids) >= max_tokens:
                                        # Check if the last token is an EOS token
                    eos_token_id = self.tokenizer.eos_token_id
                    if eos_token_id is not None and output_token_ids[-1] != eos_token_id:
                        # Hit max tokens without EOS token - generation was truncated
                        end_properly = False
                        logger.info(f"Generation hit max tokens ({max_tokens}) without EOS token - marked as incomplete")
                    else:
                        logger.info(f"Generation completed with EOS token after {len(output_token_ids)} tokens")
                else:
                    logger.info(f"Generation completed naturally with {len(output_token_ids)} tokens")

                # Create run_results in the same format as other handlers
                run_results = {
                    'messages': messages,
                    'tools': [],
                    'end_properly': end_properly,  # Based on actual completion status
                }

                return {
                    'success': True,
                    'error': None,
                    'finish': True,
                    **run_results,
                }

        except Exception as e:
            logger.error(f"Error running ProRL agent: {e}")
            return {
                'success': False,
                'error': str(e),
                'finish': False,
                'messages': [],
                'tools': [],
                'end_properly': False,
            }

    async def eval(
            self, job_details: JobDetails,
            sid: str | None = None,
            allow_skip: bool = True,
            reward: Optional[Reward] = None,
        ) -> dict[str, Any]:
        """Evaluate the ProRL agent results using the same pattern as other handlers."""

        if reward is None:
            raise ValueError('Reward is required for evaluation of ProRL problems.')

        try:
            # Extract response from messages like other handlers do
            run_results = job_details.run_results
            if not run_results or 'messages' not in run_results:
                return {'resolved': False, 'reward': 0}

            messages = run_results['messages']
            if not messages:
                return {'resolved': False, 'reward': 0}

            # Get the last message (assistant response)
            last_message = messages[-1]

            # For ProRL, we need to convert output_ids back to text for evaluation
            # Priority: token_ids (convert to text) > content
            if 'token_ids' in last_message and last_message['token_ids']:
                # Decode output token IDs to text
                assert self.tokenizer is not None, "Tokenizer is not initialized"
                response = self.tokenizer.decode(last_message['token_ids'], skip_special_tokens=True)
                logger.info(f"Successfully decoded {len(last_message['token_ids'])} output tokens to text for evaluation")
            else:
                raise ValueError('No output_ids found in last message')

            # Remote server requires <think> tag
            if '<think>' not in response:
                response = '<think>\nfake thought\n</think>\n' + response

            # Send to server for evaluation
            eval_results = await reward.get_reward(job_details.instance, response)
            return eval_results

        except Exception as e:
            logger.error(f"Error evaluating ProRL agent: {e}")
            return {'resolved': False, 'reward': 0}

    def init_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        """Handle exceptions during initialization using utils functions."""
        return initialize_exception(job_details, exception)

    def run_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        """Handle exceptions during run using utils functions."""
        return run_exception(job_details, exception)

    def eval_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        """Handle exceptions during evaluation using utils functions."""
        return eval_exception(job_details, exception)

    def final_result(self, job_details: JobDetails) -> dict[str, Any]:
        """Process final results using utils functions."""
        return utils_final_result(job_details)
