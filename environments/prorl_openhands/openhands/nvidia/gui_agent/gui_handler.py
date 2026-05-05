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

from typing import Any

from evaluation.utils.shared import EvalMetadata  # type: ignore
from openhands.core.config import OpenHandsConfig
from openhands.nvidia.registry import AgentHandler, JobDetails
from openhands.runtime.base import Runtime
from openhands.nvidia.utils import (
    initialize_exception,
    run_exception,
    eval_exception,
    final_result as utils_final_result,
)

from openhands.nvidia.gui_agent.gui_utils import (
    initialize_agents,
    run_agent,
    evaluate_agent,
)


class GuiAgentHandler(AgentHandler):
    """Handler for GUI (visual browsing) tasks using VisualBrowsingAgent."""

    @property
    def name(self) -> str:
        return 'gui'

    async def init(
        self,
        job_details: JobDetails,
        sid: str | None = None,
    ) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
        instance = job_details.instance
        if instance is None:
            raise ValueError('Instance is None in GuiAgentHandler.init')
        llm_config = job_details.llm_config
        return await initialize_agents(
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
        return await run_agent(
            job_details=job_details,
            sid=sid,
        )

    async def eval(
        self,
        job_details: JobDetails,
        sid: str | None = None,
        allow_skip: bool = True,
        reward: None | Any = None,
    ) -> dict[str, Any]:
        return await evaluate_agent(
            run_results=job_details.run_results or {},
            instance=job_details.instance or {},
        )

    def init_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        return initialize_exception(job_details, exception)

    def run_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        return run_exception(job_details, exception)

    def eval_exception(self, job_details: JobDetails, exception: Exception) -> dict[str, Any]:
        return eval_exception(job_details, exception)

    def final_result(self, job_details: JobDetails) -> dict[str, Any]:
        return utils_final_result(job_details)
