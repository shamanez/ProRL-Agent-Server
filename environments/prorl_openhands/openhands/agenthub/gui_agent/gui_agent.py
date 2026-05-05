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

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str

from openhands.agenthub.browsing_agent.response_parser import BrowsingResponseParser
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    Action,
    AgentFinishAction,
    BrowseInteractiveAction,
    MessageAction,
)
from openhands.events.event import EventSource, Event
from openhands.events.observation import BrowserOutputObservation
from openhands.events.observation.observation import Observation
from openhands.llm.llm import LLM
from openhands.runtime.plugins import (
    PluginRequirement,
)


def get_error_prefix(obs: BrowserOutputObservation) -> str:
    # temporary fix for OneStopMarket to ignore timeout errors
    if 'timeout' in obs.last_browser_action_error:
        return ''
    return f'## Error from previous action:\n{obs.last_browser_action_error}\n'


def create_goal_prompt(
    goal: str, image_urls: list[str] | None
) -> tuple[str, list[str]]:
    goal_txt: str = f"""\
# Instructions
Review the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

## Goal:
{goal}
"""
    goal_image_urls = []
    if image_urls is not None:
        for idx, url in enumerate(image_urls):
            goal_txt = goal_txt + f'Images: Goal input image ({idx + 1})\n'
            goal_image_urls.append(url)
    goal_txt += '\n'
    return goal_txt, goal_image_urls


def create_observation_prompt(
    axtree_txt: str,
    tabs: str,
    focused_element: str,
    error_prefix: str,
    som_screenshot: str | None,
) -> tuple[str, str | None]:
    # first turn observation is empty
    if len(tabs) == 0 and len(axtree_txt) == 0 and len(focused_element) == 0 and len(error_prefix) == 0:
        txt_observation = ''
        return txt_observation, None
    txt_observation = f"""
# Observation:
{tabs}{focused_element}{error_prefix}
"""
    screenshot_url = None
    if (som_screenshot is not None) and (len(som_screenshot) > 0):
        txt_observation += 'Image: Page screenshot (Note that only visible portion of webpage is present in the screenshot. You may need to scroll to view the remaining portion of the web-page.'
        screenshot_url = som_screenshot
    else:
        logger.info('SOM Screenshot not present in observation!')
    txt_observation += '\n'
    return txt_observation, screenshot_url

def get_tabs(obs: BrowserOutputObservation) -> str:
    prompt_pieces = ['\n## Opened tabs:']
    for page_index, page_url in enumerate(obs.open_pages_urls):
        active_or_not = ' (active tab)' if page_index == obs.active_page_index else ''
        prompt_piece = f"""\
Tab {page_index}{active_or_not}:
URL: {page_url}
"""
        prompt_pieces.append(prompt_piece)
    return '\n'.join(prompt_pieces) + '\n'


def get_axtree(axtree_txt: str) -> str:
    bid_info = """\
Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.

"""
    visible_tag_info = """\
Note: You can only interact with visible elements. If the "visible" tag is not present, the element is not visible on the page.

"""
    return f'\n## AXTree:\n{bid_info}{visible_tag_info}{axtree_txt}\n'


def get_action_prompt(action_set: HighLevelActionSet) -> str:
    action_set_generic_info = """\
Note: This action set allows you to interact with your environment. Most of them are python function executing playwright code. The primary way of referring to elements in the page is through bid which are specified in your observations.

"""
    action_description = action_set.describe(
        with_long_description=False,
        with_examples=False,
    )
    action_prompt = f'# Action space:\n{action_set_generic_info}{action_description}\n'
    return action_prompt


# Build alternating action/observation message history from past events
# Skips the initial noop and excludes the latest observation (handled as current turn)
def build_history_prompts(
    events,
    last_obs: BrowserOutputObservation | None,
) -> list[dict]:
    history_prompts: list[dict] = []
    pending_action: BrowseInteractiveAction | None = None

    for event in events:
        if isinstance(event, BrowseInteractiveAction):
            pending_action = event
        elif isinstance(event, Observation):
            if not isinstance(event, BrowserOutputObservation):
                continue
            if pending_action is None:
                continue
            # exclude the most recent observation; it will be added as current_turn_prompt
            if (last_obs is not None) and (event is last_obs):
                # Include the paired action (unless it's a noop), but skip adding this observation
                if pending_action.browser_actions and pending_action.browser_actions.strip().startswith('noop'):
                    pending_action = None
                    break
                thought = pending_action.thought if hasattr(pending_action, 'thought') and pending_action.thought else ''
                action_text = f"Output thought and action: {thought} ```{pending_action.browser_actions}```"
                history_prompts.append(
                    {
                        'role': 'assistant',
                        'content': [TextContent(type='text', text=action_text)],
                    }
                )
                pending_action = None
                break
            # skip the bootstrap noop
            if pending_action.browser_actions and pending_action.browser_actions.strip().startswith('noop'):
                pending_action = None
                continue

            # Action message (assistant)
            thought = pending_action.thought if hasattr(pending_action, 'thought') and pending_action.thought else ''
            action_text = f"{thought} ```{pending_action.browser_actions}```"
            history_prompts.append(
                {
                    'role': 'assistant',
                    'content': [TextContent(type='text', text=action_text)],
                }
            )

            try:
                error_prefix = get_error_prefix(event) if event.error else ''
                focused_element = '## Focused element:\nNone\n'
                if event.focused_element_bid is not None:
                    focused_element = f"## Focused element:\nbid='{event.focused_element_bid}'\n"
                tabs = get_tabs(event)
                axtree_txt = flatten_axtree_to_str(
                    event.axtree_object,
                    extra_properties=event.extra_element_properties,
                    with_visible=True,
                    with_clickable=True,
                    with_center_coords=False,
                    with_bounding_box_coords=False,
                    filter_visible_only=False,
                    filter_with_bid_only=False,
                    filter_som_only=False,
                )
                axtree_txt = get_axtree(axtree_txt=axtree_txt)
                set_of_marks = event.set_of_marks
                observation_txt, som_screenshot = create_observation_prompt(
                    axtree_txt, tabs, focused_element, error_prefix, set_of_marks
                )
                content = [TextContent(type='text', text=observation_txt)]
                if som_screenshot is not None:
                    content.append(ImageContent(image_urls=[som_screenshot]))
                history_prompts.append({'role': 'user', 'content': content})
            except Exception as e:
                logger.error('Error building history observation: %s', e)
            pending_action = None

    return history_prompts


class GuiAgent(Agent):
    VERSION = '1.0'
    """
    GuiAgent that can uses webpage screenshots during browsing.
    """

    sandbox_plugins: list[PluginRequirement] = []
    response_parser = BrowsingResponseParser()

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        """Initializes a new instance of the GuiAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm, config)
        # define a configurable action space, with chat functionality, web navigation, and webpage grounding using accessibility tree and HTML.
        # see https://github.com/ServiceNow/BrowserGym/blob/main/core/src/browsergym/core/action/highlevel.py for more details
        action_subsets = [
            'chat',
            'bid',
            'nav',
            'tab',
            'infeas',
        ]
        self.action_space = HighLevelActionSet(
            subsets=action_subsets,
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,
        )
        self.action_prompt = get_action_prompt(self.action_space)
        self.abstract_example = f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer:

You must mandatorily think step by step. If you need to make calculations such as coordinates, write them here. Describe the effect that your previous action had on the current content of the page. In summary the next action I will perform is ```{self.action_space.example_action(abstract=True)}```
"""
        self.concrete_example = """
# Concrete Example

Here is a concrete example of how to format your answer. Make sure to generate the action in the correct format ensuring that the action is present inside ``````:

Let's think step-by-step. From previous action I tried to set the value of year to "2022", using select_option, but it doesn't appear to be in the form. It may be a dynamic dropdown, I will try using click with the bid "324" and look at the response from the page. In summary the next action I will perform is ```click('324')```
"""
        self.completion_example = """
# Completion Example

When you have completed the task, return a single action that sends the final answer back to the user:

```send_msg_to_user("Task completed.")```
"""
        self.hints = """
Note:
* Make sure to use bid to identify elements when using commands.
* Interacting with combobox, dropdowns and auto-complete fields can be tricky, sometimes you need to use select_option, while other times you need to use fill or click and wait for the reaction of the page.

"""
        self.reset()

    def reset(self) -> None:
        """Resets the GuiAgent."""
        super().reset()
        self.cost_accumulator = 0
        self.error_accumulator = 0

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        """Get the initial user message from the conversation history.

        Args:
            history: List of events from the conversation

        Returns:
            MessageAction: The initial user message

        Raises:
            ValueError: If no initial user message is found
        """
        initial_user_message = None

        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                initial_user_message = event
                break

        if initial_user_message is None:
            # This should not happen in a valid conversation
            logger.error(
                f'CRITICAL: Could not find the initial user MessageAction in the full {len(history)} events history.'
            )
            raise ValueError(
                'Could not find the initial user MessageAction in the conversation history.'
            )
        return initial_user_message

    def _get_messages(
        self, events: list[Event], initial_user_message: MessageAction
    ) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand, similar to how the step
        method constructs messages but for the full conversation history.

        Args:
            events: The list of events to convert to messages
            initial_user_message: The initial user message action

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption
        """
        messages: list[Message] = []

        # Find the last observation to pass to build_history_prompts
        last_obs: BrowserOutputObservation | None = None
        for event in reversed(events):
            if isinstance(event, BrowserOutputObservation):
                last_obs = event
                break

        # Get goal from initial user message
        goal = initial_user_message.content
        goal_txt, goal_images = create_goal_prompt(goal, initial_user_message.image_urls)

        # System message
        system_msg = """\
You are an agent trying to solve a web task based on the content of the page and user instructions. You can interact with the page and explore, and send messages to the user when you finish the task. Each time you submit an action it will be sent to the browser and you will receive a new page.
""".strip()

        messages.append(Message(role='system', content=[TextContent(text=system_msg)]))

        # Initial user message (with goal and action prompt)
        human_prompt: list[TextContent | ImageContent] = [
            TextContent(type='text', text=goal_txt)
        ]
        if goal_images and len(goal_images) > 0:
            human_prompt.append(ImageContent(image_urls=goal_images))

        remaining_content = f"""
{self.action_prompt}\
{self.hints}\
{self.abstract_example}\
{self.concrete_example}\
{self.completion_example}\
"""
        human_prompt.append(TextContent(type='text', text=remaining_content))
        messages.append(Message(role='user', content=human_prompt))

        # Build history prompts (alternating assistant/user messages)
        history_prompts = build_history_prompts(events, last_obs)
        for history in history_prompts:
            if history["role"] == "user":
                messages.append(Message(role='user', content=history["content"]))
            elif history["role"] == "assistant":
                messages.append(Message(role='assistant', content=history["content"]))

        # Add current turn prompt if there's a last observation
        if last_obs is not None:
            try:
                error_prefix = get_error_prefix(last_obs) if last_obs.error else ''
                focused_element = '## Focused element:\nNone\n'
                if last_obs.focused_element_bid is not None:
                    focused_element = f"## Focused element:\nbid='{last_obs.focused_element_bid}'\n"
                tabs = get_tabs(last_obs)

                cur_axtree_txt = flatten_axtree_to_str(
                    last_obs.axtree_object,
                    extra_properties=last_obs.extra_element_properties,
                    with_visible=True,
                    with_clickable=True,
                    with_center_coords=False,
                    with_bounding_box_coords=False,
                    filter_visible_only=False,
                    filter_with_bid_only=False,
                    filter_som_only=False,
                )
                cur_axtree_txt = get_axtree(axtree_txt=cur_axtree_txt)
                set_of_marks = last_obs.set_of_marks

                observation_txt, som_screenshot = create_observation_prompt(
                    cur_axtree_txt, tabs, focused_element, error_prefix, set_of_marks
                )

                current_turn_prompt: list[TextContent | ImageContent] = [
                    TextContent(type='text', text=observation_txt)
                ]
                if som_screenshot is not None:
                    current_turn_prompt.append(ImageContent(image_urls=[som_screenshot]))

                messages.append(Message(role='user', content=current_turn_prompt))
            except Exception as e:
                logger.error('Error building current turn observation: %s', e)

        return messages

    def step(self, state: State) -> Action:
        """Performs one step using the GuiAgent.

        This includes gathering information on previous steps and prompting the model to make a browsing command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - BrowseInteractiveAction(browsergym_command) - BrowserGym commands to run
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        messages: list[Message] = []
        prev_actions = []
        cur_axtree_txt = ''
        error_prefix = ''
        focused_element = ''
        tabs = ''
        last_obs = None
        last_action = None
        set_of_marks = None  # Initialize set_of_marks to None

        if len(state.view) == 1:
            # for visualwebarena, webarena and miniwob++ eval, we need to retrieve the initial observation already in browser env
            # initialize and retrieve the first observation by issuing an noop OP
            # For non-benchmark browsing, the browser env starts with a blank page, and the agent is expected to first navigate to desired websites
            return BrowseInteractiveAction(browser_actions='noop(1000)')

        for event in state.view:
            if isinstance(event, BrowseInteractiveAction):
                prev_actions.append(event)
                last_action = event
            elif isinstance(event, MessageAction) and event.source == EventSource.AGENT:
                # agent has responded, task finished.
                return AgentFinishAction(outputs={'content': event.content})
            elif isinstance(event, Observation):
                # Only process BrowserOutputObservation and skip other observation types
                if not isinstance(event, BrowserOutputObservation):
                    continue
                last_obs = event

        if len(prev_actions) >= 1:  # ignore noop()
            prev_actions = prev_actions[1:]  # remove the first noop action

        # if the final BrowserInteractiveAction exec BrowserGym's send_msg_to_user,
        # we should also send a message back to the user in OpenHands and call it a day
        if (
            isinstance(last_action, BrowseInteractiveAction)
            and last_action.browsergym_send_msg_to_user
        ):
            return MessageAction(last_action.browsergym_send_msg_to_user)

        # Build structured history (action -> observation pairs) excluding current obs
        history_prompts = build_history_prompts(state.view, last_obs)

        if isinstance(last_obs, BrowserOutputObservation):
            if last_obs.error:
                # add error recovery prompt prefix
                error_prefix = get_error_prefix(last_obs)
                if len(error_prefix) > 0:
                    self.error_accumulator += 1
                    if self.error_accumulator > 5:
                        return MessageAction(
                            'Too many errors encountered. Task failed.'
                        )
            focused_element = '## Focused element:\nNone\n'
            if last_obs.focused_element_bid is not None:
                focused_element = (
                    f"## Focused element:\nbid='{last_obs.focused_element_bid}'\n"
                )
            tabs = get_tabs(last_obs)
            try:
                # IMPORTANT: keep AX Tree of full webpage, add visible and clickable tags
                cur_axtree_txt = flatten_axtree_to_str(
                    last_obs.axtree_object,
                    extra_properties=last_obs.extra_element_properties,
                    with_visible=True,
                    with_clickable=True,
                    with_center_coords=False,
                    with_bounding_box_coords=False,
                    filter_visible_only=False,
                    filter_with_bid_only=False,
                    filter_som_only=False,
                )
                cur_axtree_txt = get_axtree(axtree_txt=cur_axtree_txt)
            except Exception as e:
                logger.error(
                    'Error when trying to process the accessibility tree: %s', e
                )
                return MessageAction('Error encountered when browsing.')
            set_of_marks = last_obs.set_of_marks
        goal, image_urls = state.get_current_user_intent()

        if goal is None:
            goal = state.inputs['task']
        goal_txt, goal_images = create_goal_prompt(goal, image_urls)

        # current turn prompt
        current_turn_prompt: list[TextContent | ImageContent] = []
        observation_txt, som_screenshot = create_observation_prompt(
            cur_axtree_txt, tabs, focused_element, error_prefix, set_of_marks
        )
        current_turn_prompt.append(TextContent(type='text', text=observation_txt))
        if som_screenshot is not None:
            current_turn_prompt.append(ImageContent(image_urls=[som_screenshot]))

        # human prompt
        human_prompt: list[TextContent | ImageContent] = [
            TextContent(type='text', text=goal_txt)
        ]
        if len(goal_images) > 0:
            human_prompt.append(ImageContent(image_urls=goal_images))
        remaining_content = f"""
{self.action_prompt}\
{self.hints}\
{self.abstract_example}\
{self.concrete_example}\
{self.completion_example}\
"""
        human_prompt.append(TextContent(type='text', text=remaining_content))

        system_msg = """\
You are an agent trying to solve a web task based on the content of the page and user instructions. You can interact with the page and explore, and send messages to the user when you finish the task. Each time you submit an action it will be sent to the browser and you will receive a new page.
""".strip()

        messages.append(Message(role='system', content=[TextContent(text=system_msg)]))
        messages.append(Message(role='user', content=human_prompt))
        # TODO: add history prompt
        for history in history_prompts:
            if history["role"] == "user":
                messages.append(Message(role='user', content=history["content"]))
            elif history["role"] == "assistant":
                messages.append(Message(role='assistant', content=history["content"]))
        messages.append(Message(role='user', content=current_turn_prompt))
        flat_messages = self.llm.format_messages_for_llm(messages)
        response = self.llm.completion(
            messages=flat_messages,
            temperature=0.0,
            stop=[')```', ')\n```'],
            extra_body={'metadata': state.to_llm_metadata(agent_name=self.name)},
        )
        return self.response_parser.parse(response)
