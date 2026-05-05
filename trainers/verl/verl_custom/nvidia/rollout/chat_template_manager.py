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
Chat Template Manager for supporting multiple chat templates
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ChatTemplateManager:
    """Manager for different chat templates"""
    
    def __init__(self):
        self._templates: Dict[str, str] = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register default chat templates"""
        # Qwen3 chat template with generation support
        self._templates["qwen3_chat_template_generation"] = (
            "{%- if tools %}"
            "{{'<|im_start|>system\\n' }}"
            "{%- if messages[0].role == 'system' %}"
            "{{ messages[0].content + '\\n\\n' }}"
            "{%- endif %}"
            "{{'# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\n"
            "You are provided with function signatures within <tools></tools> XML tags:\\n<tools>' }}"
            "{%- for tool in tools %}"
            "{{ '\\n' + (tool | tojson) }}"
            "{%- endfor %}"
            "{{ '\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:"
            "\\n<tool_call>\\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\\n</tool_call><|im_end|>\\n' }}"
            "{%- else %}"
            "{%- if messages[0].role == 'system' %}"
            "{{'<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
            "{%- endif %}"
            "{%- endif %}"
            "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}"
            "{%- for message in messages[::-1] %}"
            "{%- set index = (messages|length - 1) - loop.index0 %}"
            "{%- if ns.multi_step_tool and message.role == 'user' and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}"
            "{%- set ns.multi_step_tool = false %}"
            "{%- set ns.last_query_index = index %}"
            "{%- endif %}"
            "{%- endfor %}"
            "{%- for message in messages %}"
            "{%- if message.content is string %}"
            "{%- set content = message.content %}"
            "{%- else %}"
            "{%- set content = '' %}"
            "{%- endif %}"
            "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first) %}"
            "{{'<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}"
            "{%- elif message.role == 'assistant' %}"
            "{{'<|im_start|>' + message['role'] + '\n'}}"
            "{% generation %}"
            "{%- set reasoning_content = '' %}"
            "{%- if message.reasoning_content is string %}"
            "{%- set reasoning_content = message.reasoning_content %}"
            "{%- else %}"
            "{%- if '</think>' in content %}"
            "{%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}"
            "{%- set content = content.split('</think>')[-1].lstrip('\\n') %}"
            "{%- endif %}"
            "{%- endif %}"
            "{%- if loop.index0 > ns.last_query_index %}"
            "{%- if loop.last or (not loop.last and reasoning_content) %}"
            "{{ '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"
            "{%- else %}"
            "{{ content }}"
            "{%- endif %}"
            "{%- else %}"
            "{{ content }}"
            "{%- endif %}"
            "{%- if message.tool_calls %}"
            "{%- for tool_call in message.tool_calls %}"
            "{%- if (loop.first and content) or (not loop.first) %}"
            "{{'\\n' }}"
            "{%- endif %}"
            "{%- if tool_call.function %}"
            "{%- set tool_call = tool_call.function %}"
            "{%- endif %}"
            "{{'<tool_call>\\n{\"name\": \"' + tool_call.name + '\", \"arguments\": ' + (tool_call.arguments if tool_call.arguments is string else (tool_call.arguments | tojson)) + '}\\n</tool_call>' }}"
            "{%- endfor %}"
            "{%- endif %}"
            "{{'<|im_end|>' }}"
            "{% endgeneration %}"
            "{{'\n' }}"
            "{%- elif message.role == 'tool' %}"
            "{%- if loop.first or (messages[loop.index0 - 1].role != 'tool') %}"
            "{{'<|im_start|>user' }}"
            "{%- endif %}"
            "{{'\\n<tool_response>\\n' + content + '\\n</tool_response>' }}"
            "{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}"
            "{{'<|im_end|>\\n' }}"
            "{%- endif %}"
            "{%- endif %}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}"
            "{{'<|im_start|>assistant\\n' }}"
            "{%- if enable_thinking is defined and enable_thinking is false %}"
            "{{'<think>\\n\\n</think>\\n\\n' }}"
            "{%- endif %}"
            "{%- endif %}"
        )
        
        #Some other templates ...
    
    def register_template(self, name: str, template: str):
        """Register a new chat template"""
        self._templates[name] = template
        logger.info(f"Registered chat template: {name}")
    
    def get_template(self, name: str) -> Optional[str]:
        """Get a chat template by name"""
        if name not in self._templates:
            logger.warning(f"Chat template '{name}' not found. Available templates: {list(self._templates.keys())}")
            # Return default template if not found
            return self._templates.get("qwen3_chat_template_generation")
        return self._templates[name]
    
    def list_templates(self) -> list:
        """List all available template names"""
        return list(self._templates.keys())
    
    def has_template(self, name: str) -> bool:
        """Check if a template exists"""
        return name in self._templates


# Global instance
chat_template_manager = ChatTemplateManager()


def get_chat_template(template_name: str) -> str:
    """Get chat template by name"""
    return chat_template_manager.get_template(template_name)


def register_chat_template(template_name: str, template: str):
    """Register a new chat template"""
    chat_template_manager.register_template(template_name, template)


# Backward compatibility
qwen3_chat_template_generation = chat_template_manager.get_template("qwen3_chat_template_generation") 