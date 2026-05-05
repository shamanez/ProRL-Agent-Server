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

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, cast

try:
    from PIL import Image
except Exception:
    Image = None

from evaluation.utils.shared import (  # type: ignore
    EvalException,
    EvalMetadata,
    get_default_sandbox_config_for_eval,
    update_llm_config_for_completions_logging,
)

from openhands.agenthub.gui_agent.gui_agent import GuiAgent
from openhands.core.config import AgentConfig, OpenHandsConfig
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.logger import openhands_logger as openhands_logger
from openhands.core.main import create_runtime
from openhands.core.setup import create_agent, create_controller
from openhands.controller.state.state import State
from openhands.events.action import MessageAction
from openhands.events.observation import BrowserOutputObservation
from openhands.nvidia.controller import run_controller_with_controller
from openhands.nvidia.logger import nvidia_logger as logger
from openhands.nvidia.registry import JobDetails, _DEFAULT_AGENT_CONFIG
from openhands.nvidia.utils import (
    get_instance_id,
    is_last_action_finish,
)
from openhands.runtime.base import Runtime
from openhands.llm.nvidia.qwen2_5_vl import (
    convert_messages_to_tokens,
    qwen2_5_vl_chat_template_partial,
)
from transformers import AutoTokenizer

def process_messages_from_agent_state_gui(
    agent: GuiAgent,
    state: State,
    job_details: JobDetails | None = None,
) -> dict[str, Any]:
    ########## get config
    if job_details is not None:
        assert job_details.llm_config is not None, (
            'llm_config is required in job_details.'
        )
        token_level_generation = job_details.llm_config.token_level_generation
        if token_level_generation:
            model_name = job_details.llm_config.custom_tokenizer
            if model_name is None:
                model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
            else:
                assert 'qwen2.5-vl' in model_name.lower(), (
                    'token_level_generation is only supported for qwen2.5-vl.'
                )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            chat_template = qwen2_5_vl_chat_template_partial
    else:
        logger.warning(
            'No job_details provided in process_messages_from_agent_state. Assuming token_level_generation is False.'
        )
        token_level_generation = False

    ########## save screenshots and GIFs
    gif_paths: dict[str, str | None] = {'screenshots_gif': None, 'som_gif': None}
    should_save, trajectory_path = _should_save_screenshots()
    assert trajectory_path is not None, (
        'trajectory_path should not be None when should_save is True'
    )
    logger.info(
        f'Saving screenshots and GIFs to trajectory path: {trajectory_path}'
    )
    run_folder = _get_unique_run_folder(trajectory_path, job_details)
    som_dir = os.path.join(run_folder, 'SOM')
    screenshot_dir = os.path.join(run_folder, 'Screenshot')
    logger.debug(f'Created run folder: {run_folder}')
    logger.debug(f'SOM directory: {som_dir}')
    logger.debug(f'Screenshot directory: {screenshot_dir}')
    _write_som_images_from_state(state, som_dir, screenshot_dir)

    gif_paths['screenshots_gif'] = _save_screenshots_gif(
        screenshot_dir,
        run_folder,
        gif_name='browser_run.gif',
        duration_ms=700,
        image_prefix='screenshot_',
    )
    gif_paths['som_gif'] = _save_screenshots_gif(
        som_dir,
        run_folder,
        gif_name='browser_run_som.gif',
        duration_ms=700,
        image_prefix='som_',
    )
    if gif_paths['screenshots_gif']:
        logger.info(f'Screenshots GIF saved: {gif_paths["screenshots_gif"]}')
    if gif_paths['som_gif']:
        logger.info(f'SOM GIF saved: {gif_paths["som_gif"]}')

    ########## get messages
    initial_user_message = agent._get_initial_user_message(state.history)
    raw_messages = agent._get_messages(state.history, initial_user_message)
    if raw_messages[-1].role != 'assistant':
        raw_messages = raw_messages[:-1]

    if is_last_action_finish(state):
        tool_metadata = state.history[-1].tool_call_metadata
        if tool_metadata is not None:
            assistant_msg = getattr(tool_metadata.model_response.choices[0], 'message')
            raw_messages[-1].tool_calls = assistant_msg.tool_calls
    messages = agent.llm.format_messages_for_llm(raw_messages)
    while len(messages) > 0 and messages[-1]['role'] != 'assistant':
        messages = messages[:-1]

    ########## check tools
    from openhands.llm.llm_utils import check_tools
    tools = check_tools(agent.tools, agent.llm.config)

    ########## process messages
    new_messages = []
    for message in messages:
        new_message = {'role': message['role']}
        if isinstance(message['content'], str):
            new_message['content'] = message['content']
        elif isinstance(message['content'], list):
            if len(message['content']) == 0:
                new_message['content'] = ''  # empty string for default
            else:
                new_message['content'] = []
                for content in message['content']:
                    if content['type'] == 'text':
                        new_message['content'].append({'type': 'text', 'text': content['text']})
                    elif content['type'] == 'image_url':
                        new_message['content'].append({'type': 'image_url', 'image_url': content['image_url']})
                    else:
                        raise RuntimeError(f'Unsupported content type: {content["type"]}')
        else:
            raise RuntimeError(
                f'Message content is not a string or list: {message["content"]}'
            )
        if 'tool_calls' in message:
            new_message['tool_calls'] = [
                tool_call['function'] for tool_call in message['tool_calls']
            ]

        # Update token_ids, currently this only works for Qwen2.5-VL models.
        if token_level_generation:
            input_ids = message.get('input_ids', None)
            output_ids = message.get('output_ids', None)
            logprobs = message.get('logprobs', None)
            if output_ids is not None:
                new_message['token_ids'] = output_ids
            else:
                new_message['token_ids'] = convert_messages_to_tokens(
                    [new_message],
                    tokenizer,
                    chat_template=chat_template,
                    add_generation_prompt=False,
                    tools=tools,
                )[0]

            new_message['input_ids'] = input_ids
            new_message['logprobs'] = logprobs

        new_messages.append(new_message)

    result: dict[str, Any] = {
        'messages': new_messages,
        'tools': tools,
    }
    if gif_paths['screenshots_gif'] or gif_paths['som_gif']:
        result['screenshots_gif'] = gif_paths['screenshots_gif']
        result['som_gif'] = gif_paths['som_gif']
    return result

def _should_save_screenshots() -> tuple[bool, str | None]:
    """Check if screenshots should be saved based on environment variables.

    Returns:
        tuple: (should_save, trajectory_path)
    """
    save_screenshots = os.environ.get(
        'OH_SAVE_SCREENSHOTS_IN_TRAJECTORY', 'false'
    ).lower() in ('1', 'true', 'yes')
    trajectory_path = os.environ.get('OH_SAVE_TRAJECTORY_PATH')

    if not save_screenshots:
        return False, None

    if not trajectory_path or not os.path.exists(trajectory_path):
        logger.warning(
            f"OH_SAVE_TRAJECTORY_PATH not set or path doesn't exist: {trajectory_path}"
        )
        return False, None

    return True, trajectory_path

def _get_unique_run_folder(
    base_trajectory_path: str, job_details: JobDetails | None = None
) -> str:
    """Create a unique folder for this run with organized subfolders.

    Returns the path to the unique run folder containing:
    - SOM/ (for SOM screenshots)
    - Screenshot/ (for regular screenshots)
    - GIFs will be saved at the same level as these folders
    """
    # Generate unique identifier for this run
    if job_details and job_details.instance:
        instance_id = job_details.instance.get('instance_id', None)
        if instance_id:
            run_id = f'{instance_id}_{int(time.time())}_{str(uuid.uuid4())[:8]}'
        else:
            run_id = f'run_{int(time.time())}_{str(uuid.uuid4())[:8]}'
    else:
        run_id = f'run_{int(time.time())}_{str(uuid.uuid4())[:8]}'

    run_folder = os.path.join(base_trajectory_path, run_id)

    # Create the folder structure
    try:
        os.makedirs(os.path.join(run_folder, 'SOM'), exist_ok=True)
        os.makedirs(os.path.join(run_folder, 'Screenshot'), exist_ok=True)
    except Exception as e:
        logger.warning(f'Failed to create run folder structure: {e}')
        return run_folder

    return run_folder


def _save_screenshots_gif(
    images_dir: str,
    output_dir: str,
    gif_name: str = 'browser_run.gif',
    duration_ms: int = 700,
    image_prefix: str = 'screenshot_',
) -> str | None:
    """Create a GIF from images in the specified directory.

    Args:
        images_dir: Directory containing the images to create GIF from
        output_dir: Directory to save the GIF file
        gif_name: Name of the output GIF file
        duration_ms: Duration of each frame in milliseconds
        image_prefix: Prefix to filter image files by

    Returns:
        Path to the created GIF file, or None if creation failed
    """
    try:
        if not os.path.isdir(images_dir):
            return None

        all_files = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        files = [f for f in all_files if f.startswith(image_prefix)]
        if not files:
            return None
        files.sort()

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, gif_name)

        if Image is None:
            return None

        frames = []
        for fname in files:
            p = os.path.join(images_dir, fname)
            try:
                img = Image.open(p).convert('RGB')
                frames.append(img)
            except Exception:
                continue
        if not frames:
            return None

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
            quality=85,
            format='GIF',
        )
        return output_path
    except Exception:
        return None


def _write_som_images_from_state(
    state: State, som_dir: str, screenshot_dir: str
) -> None:
    """Write SOM and regular screenshots to their respective directories.

    Args:
        state: The agent state containing browser observations
        som_dir: Directory to save SOM (Set of Marks) images
        screenshot_dir: Directory to save regular screenshots
    """
    try:
        os.makedirs(som_dir, exist_ok=True)
        os.makedirs(screenshot_dir, exist_ok=True)
    except Exception as e:
        logger.warning(
            f'Failed to create directories {som_dir} and {screenshot_dir}: {e}'
        )
        return

    som_idx = 0
    screenshot_idx = 0

    for event in state.history:
        if isinstance(event, BrowserOutputObservation):
            # Handle SOM images
            som = getattr(event, 'set_of_marks', None)
            if som:
                try:
                    screenshot_path = getattr(event, 'screenshot_path', None)
                    if screenshot_path is not None:
                        stem = Path(screenshot_path).stem
                        # Prefer timestamp from screenshot filename if present
                        ts = stem.replace('screenshot_', '')
                    else:
                        ts = time.strftime('%Y%m%d_%H%M%S_%f')
                    fname = f'som_{ts}_{som_idx}.png'
                    som_idx += 1
                    som_path = os.path.join(som_dir, fname)
                    data = som.replace('data:image/png;base64,', '')
                    img_bytes = base64.b64decode(data)
                    with open(som_path, 'wb') as f:
                        f.write(img_bytes)
                except Exception:
                    continue

            # Handle regular screenshots
            screenshot_b64 = getattr(event, 'screenshot', '')
            if screenshot_b64:
                try:
                    screenshot_path2 = getattr(event, 'screenshot_path', None)
                    if screenshot_path2 is not None:
                        stem2 = Path(screenshot_path2).stem
                        ts2 = stem2.replace('screenshot_', '')
                    else:
                        ts2 = time.strftime('%Y%m%d_%H%M%S_%f')
                    fname2 = f'screenshot_{ts2}_{screenshot_idx}.png'
                    screenshot_idx += 1
                    screenshot_path = os.path.join(screenshot_dir, fname2)
                    # Support both data URI and raw base64
                    if screenshot_b64.startswith('data:image'):
                        screenshot_b64 = screenshot_b64.split(',', 1)[1]
                    img_bytes2 = base64.b64decode(screenshot_b64)
                    with open(screenshot_path, 'wb') as f:
                        f.write(img_bytes2)
                except Exception:
                    continue

    logger.debug(f'Saved {som_idx} SOM images to {som_dir}')
    logger.debug(f'Saved {screenshot_idx} regular screenshots to {screenshot_dir}')


def get_config(
    instance: dict,
    metadata: EvalMetadata,
    agent_config: dict = _DEFAULT_AGENT_CONFIG,
) -> OpenHandsConfig:
    """Build OpenHandsConfig for GuiAgent GUI tasks.

    Key differences vs math/code configs:
    - default_agent is GuiAgent
    - browsing is enabled with SoM visual browsing
    - jupyter/cmd/editor disabled
    - use general browsing (no browsergym_eval_env preset)
    """
    sandbox_config = get_default_sandbox_config_for_eval()
    # Ensure general-purpose browsing (no fixed benchmark env)
    sandbox_config.browsergym_eval_env = None
    sandbox_config.runtime_container_image = os.environ.get('SANDBOX_RUNTIME_CONTAINER_IMAGE', None)
    # Browsing often benefits from host networking for external access
    # Keep the default as-is; caller environment may override via env vars

    config = OpenHandsConfig(
        default_agent='GuiAgent',
        run_as_openhands=False,
        max_iterations=metadata.max_iterations,
        runtime='singularity',
        sandbox=sandbox_config,
        workspace_base=None,
        workspace_mount_path=None,
    )
    # Optional: save trajectory and screenshots to the specified path
    # example:
    # step 1: export OH_SAVE_TRAJECTORY_PATH=/lustre/fsw/portfolios/llmservice/users/shaokunz/project/OpenHands_internal/scripts/sft_data/log
    # step 2: export OH_SAVE_SCREENSHOTS_IN_TRAJECTORY=true
    config.save_trajectory_path = os.environ.get("OH_SAVE_TRAJECTORY_PATH")  # 例如 /tmp/oh_trajs 或 /tmp/oh_trajs/run.json
    config.save_screenshots_in_trajectory = os.environ.get(
        "OH_SAVE_SCREENSHOTS_IN_TRAJECTORY", "false"
    ).lower() in ("1", "true", "yes")

    # LLM logging augmentation
    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, get_instance_id(instance)
        )
    )

    agent_cfg = AgentConfig(
        enable_jupyter=False,
        enable_editor=False,
        enable_cmd=False,
        enable_browsing=True,
        enable_llm_editor=False,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_think=False,
        enable_history_truncation=False,
        ensure_thinking_end_properly=agent_config['ensure_thinking_end_properly'],
        action_timeout=30.0,
        strict_loop_detector=agent_config['strict_loop_detector'],
        enable_som_visual_browsing=True,
    )
    config.set_agent_config(agent_cfg)

    # Optional: mount OpenHands source into runtime if provided
    mount_dir_env = os.environ.get('OVERWRITE_OPENHANDS_DIR', '').strip()
    if mount_dir_env:
        try:
            mount_path = Path(mount_dir_env).expanduser().resolve()
            if mount_path.exists() and mount_path.is_dir():
                config.sandbox.volumes = f"{mount_path}:/openhands/code:ro"
            else:
                logger.warning(
                    f"OVERWRITE_OPENHANDS_DIR is not a valid directory: {mount_dir_env}. Skipping mount."
                )
        except Exception as e:
            logger.error(
                f"Failed to set mount from OVERWRITE_OPENHANDS_DIR='{mount_dir_env}': {e}. Skipping mount."
            )

    return config


def get_instruction(instance: dict, metadata: EvalMetadata) -> MessageAction:
    """Construct initial instruction for the GUI/VLM browsing task."""
    task = instance.get('task') or instance.get('goal') or 'Open Google and search for NVIDIA stock price.'
    start_url = instance.get('start_url', None)

    instruction = """
You are a GUI web-browsing assistant with vision. Use the browsing tools to complete the task. Prefer visible, clickable elements identified by bid.
- Think step by step before acting.
- If a page is still loading, briefly wait using noop(ms).
- Interact only with visible elements.

Task:
{task}
{start_url_hint}
""".strip().format(
        task=task,
        start_url_hint=(f"Start from: {start_url}" if start_url else ''),
    )

    return MessageAction(content=instruction)


async def initialize_agents(
    instance: dict,
    llm_config: LLMConfig | None = None,
    sid: str | None = None,
    eval_output_dir: str = '/root',
    git_commit: str = 'unknown',
    dataset: str = 'gui',
    data_split: str = 'mock',
    agent_config: dict = dict(_DEFAULT_AGENT_CONFIG),
) -> tuple[Runtime, EvalMetadata, OpenHandsConfig]:
    if llm_config is None:
        raise ValueError('LLM config is None, cannot initialize.')

    metadata = EvalMetadata(
        agent_class='VisualBrowsingAgent',
        llm_config=llm_config,
        agent_config=None,
        max_iterations=agent_config['max_iterations'],
        eval_output_dir=eval_output_dir,
        start_time=time.strftime('%Y-%m-%d %H:%M:%S'),
        git_commit=git_commit,
        dataset=dataset,
        data_split=data_split,
        details=None,
        condenser_config=NoOpCondenserConfig(),
    )

    config = get_config(instance, metadata, agent_config)
    runtime = create_runtime(config, sid=sid)
    await runtime.connect()
    logger.debug(f"Runtime connected {runtime.sid}")

    try:
        initialize_runtime(runtime, instance, metadata)
    except Exception as e:
        logger.error(f"Error initializing runtime: {e}")
        raise e

    return runtime, metadata, config


def initialize_runtime(runtime: Runtime, instance: dict, metadata: EvalMetadata):
    """Optional runtime initialization for browsing tasks.

    For general web tasks, no special container setup is required beyond connecting.
    """
    openhands_logger.info(f'{"-" * 50} BEGIN GUI Runtime Initialization {"-" * 50}')
    # Intentionally minimal – VisualBrowsingAgent will start with noop to obtain first observation
    openhands_logger.info(f'{"-" * 50} END GUI Runtime Initialization {"-" * 50}')


async def run_agent(
    job_details: JobDetails,
    sid: str | None = None,
) -> dict[str, object]:
    runtime = job_details.runtime
    metadata = job_details.metadata
    config = job_details.config
    instance = job_details.instance

    # Narrow Optional types before use
    if instance is None:
        raise EvalException('Instance is None')
    if metadata is None:
        raise EvalException('Metadata is None')
    if config is None:
        raise EvalException('Config is None')
    if runtime is None:
        raise EvalException('Runtime is None')

    message_action = get_instruction(instance, metadata)

    agent = None
    state: State | None = None

    try:
        agent = create_agent(config)
        # JobDetails.agent is typed as CodeActAgent | None but GUI uses a different agent
        job_details.agent = cast(Any, agent)
        controller, initial_state = create_controller(
            agent=agent,
            runtime=runtime,
            config=config,
            replay_events=None,
        )
        job_details.controller = controller

        state = await run_controller_with_controller(
            config=config,
            initial_user_action=message_action,
            sid=sid,
            runtime=runtime,
            agent=agent,
            fake_user_response_fn=None,
            controller=controller,
            initial_state=initial_state,
        )
        if state is None:
            raise EvalException('Final state is None')

    except Exception as e:
        logger.error(f"Error running GUI agent: {e}")

    # Extract messages from agent state
    try:
        if agent is None:
            raise EvalException('Agent is None')
        if state is None:
            raise EvalException('State is None')
        run_results = process_messages_from_agent_state_gui(cast(GuiAgent, agent), state, job_details)
    except Exception as e:
        logger.error(f"Error while running GUI agent: {e}")
        raise Exception(f"Failed to retrieve agent messages: {str(e)}")

    # For GUI tasks, do not treat headless max-iteration as an error. Preserve messages.
    success = True
    error_msg = None
    finish_flag = is_last_action_finish(state) if state is not None else False
    if state is not None and state.last_error:
        # Detect the specific headless max-iteration pattern
        if 'Agent reached maximum iteration in headless mode' in state.last_error:
            success = True
            error_msg = None
        else:
            success = False
            error_msg = state.last_error

    return {
        'success': success,
        'error': error_msg,
        'finish': finish_flag,
        **run_results,
    }


async def evaluate_agent(run_results: dict, instance: dict) -> dict[str, Any]:
    """Trivial evaluator for GUI tasks.

    Marks resolved True if the agent finished without error; otherwise False.
    """
    success = bool(run_results.get('success', False))
    finish = bool(run_results.get('finish', False))
    return {'resolved': success and finish}
