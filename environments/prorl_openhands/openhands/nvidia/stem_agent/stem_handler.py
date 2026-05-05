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

from openhands.nvidia.stem_agent.stem_utils import (  # type: ignore[attr-defined]
    initialize_agents as stem_initialize_agents,
    run_agent as stem_run_agent,
    evaluate_agent as stem_evaluate_agent,
)
from openhands.nvidia.reward import Reward

class STEMHandler(AgentHandler):
    """Handler for SWE Agent integration, reusing functions from utils.py."""

    @property
    def name(self) -> str:
        """The name identifier for this agent handler."""
        return "stem"

    async def init(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
        """Initialize the SWE Agent with instance and config using utils functions."""
        instance = job_details.instance
        llm_config = job_details.llm_config

        return await stem_initialize_agents(
            instance=instance,
            llm_config=llm_config,
            sid=sid,
            agent_config=job_details.agent_config,
        )

    async def run(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ) -> dict[str, object]:
        return await stem_run_agent(
            job_details=job_details,
            sid=sid,
        )

    async def eval(
            self, job_details: JobDetails,
            sid: str | None = None,
            allow_skip: bool = True,
            reward: Optional[Reward] = None,
        ) -> dict[str, Any]:

        if reward is None:
            raise ValueError('Reward is required for evaluation of STEM problems.')

        return await stem_evaluate_agent(
            reward=reward,
            run_results=job_details.run_results,
            instance=job_details.instance,
        )

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