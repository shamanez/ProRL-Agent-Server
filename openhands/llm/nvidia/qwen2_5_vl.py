import json
import logging
import os
import re
import uuid
from typing import Any

import httpx
from litellm import ContextWindowExceededError, ModelResponse
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
_VLLM_GEN_LOG = os.getenv('OH_VLLM_GEN_LOG', '0') == '1'

qwen2_5_vl_chat_template = (
    '{% set image_count = namespace(value=0) %}'
    '{% set video_count = namespace(value=0) %}'
    '{% for message in messages %}'
    "{% if loop.first and message['role'] != 'system' %}"
    "{{'<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n'}}"
    '{% endif %}'
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% if message['content'] is string %}"
    "{{ message['content'] + '<|im_end|>\\n' }}"
    '{% else %}'
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    '{% set image_count.value = image_count.value + 1 %}'
    "{% if add_vision_id %}{{ 'Picture ' ~ image_count.value ~ ': ' }}{% endif %}"
    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    '{% set video_count.value = video_count.value + 1 %}'
    "{% if add_vision_id %}{{ 'Video ' ~ video_count.value ~ ': ' }}{% endif %}"
    "{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
    "{% elif 'text' in content %}"
    "{{ content['text'] }}"
    '{% endif %}'
    '{% endfor %}'
    "{{ '<|im_end|>\\n' }}"
    '{% endif %}'
    '{% endfor %}'
    '{% if add_generation_prompt %}'
    "{{'<|im_start|>assistant\\n'}}"
    '{% endif %}'
)

qwen2_5_vl_chat_template_partial = (
    '{% set image_count = namespace(value=0) %}'
    '{% set video_count = namespace(value=0) %}'
    '{% for message in messages %}'
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% if message['content'] is string %}"
    "{{ message['content'] + '<|im_end|>\\n' }}"
    '{% else %}'
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    '{% set image_count.value = image_count.value + 1 %}'
    "{% if add_vision_id %}{{ 'Picture ' ~ image_count.value ~ ': ' }}{% endif %}"
    "{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    '{% set video_count.value = video_count.value + 1 %}'
    "{% if add_vision_id %}{{ 'Video ' ~ video_count.value ~ ': ' }}{% endif %}"
    "{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
    "{% elif 'text' in content %}"
    "{{ content['text'] }}"
    '{% endif %}'
    '{% endfor %}'
    "{{ '<|im_end|>\\n' }}"
    '{% endif %}'
    '{% endfor %}'
    '{% if add_generation_prompt %}'
    "{{'<|im_start|>assistant\\n'}}"
    '{% endif %}'
)


def convert_messages_to_tokens(
    messages: list[dict],
    tokenizer: AutoTokenizer,
    chat_template: str | None = None,
    add_generation_prompt: bool = True,
    tools: list[dict] | None = None,
) -> list[list[int]]:
    """
    Convert a list of messages to a list of tokens.
    """
    token_list = []
    for idx, message in enumerate(messages):
        msg = dict(message)
        content = msg.get('content', '')
        if isinstance(content, list):
            normalized = []
            for item in content:
                if isinstance(item, dict):
                    if 'image_url' in item:
                        normalized.append(
                            {'type': 'image_url', 'image_url': item['image_url']}
                        )
                    elif item.get('type') == 'text' and 'text' in item:
                        normalized.append({'type': 'text', 'text': item['text']})
                    elif 'text' in item:
                        normalized.append({'type': 'text', 'text': item['text']})
                elif isinstance(item, str):
                    normalized.append({'type': 'text', 'text': item})
            msg['content'] = normalized

        if msg['role'] == 'system':
            token_ids = tokenizer.apply_chat_template(
                [msg],
                add_generation_prompt=False,
                tokenize=True,
                tools=tools,
                chat_template=chat_template,
            )
        elif idx == len(messages) - 1:
            token_ids = tokenizer.apply_chat_template(
                [msg],
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                chat_template=chat_template,
            )
        else:
            token_ids = tokenizer.apply_chat_template(
                [msg],
                add_generation_prompt=False,
                tokenize=True,
                chat_template=chat_template,
            )
        token_list.append(token_ids)
    return token_list


def parse_response_ids(
    response_ids: list[int], tokenizer: AutoTokenizer
) -> dict[str, Any]:
    if tokenizer.eos_token_id not in response_ids:
        format_error_type = 'length'
    else:
        format_error_type = None
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    split_idx = text.find('<tool_call>')
    # No tool calls
    if split_idx == -1 or format_error_type is not None:
        return {
            'content': text,
            'tool_calls': [],
            'format_error_type': format_error_type,
        }

    content = text[:split_idx]
    tool_text = text[split_idx:]

    if tool_text.count('<tool_call>') != tool_text.count('</tool_call>'):
        format_error_type = 'tool_call'

    try:
        tool_call_strings = re.findall(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_text, re.DOTALL
        )
        tool_calls = [json.loads(tc) for tc in tool_call_strings]
    except Exception:
        tool_calls = []
        format_error_type = 'tool_call'

    format_tool_calls = []
    for tool_call in tool_calls:
        format_tool_calls.append(
            {
                'id': f'{uuid.uuid4()}',
                'type': 'function',
                'function': tool_call,
            }
        )
    return {
        'content': content,
        'tool_calls': format_tool_calls,
        'format_error_type': format_error_type,
    }


def request_response_tokens(
    tokenizer: AutoTokenizer,
    base_url: str | None,
    timeout: int | None,
    top_p: float,
    seed: int | None,
    max_model_len: int,
    **kwargs,
) -> ModelResponse:
    input_ids = []

    messages = kwargs['messages']
    pending_messages = []
    all_images_base64 = []  # Collect all images in order
    processed_messages = []  # Track all messages that contribute to input_ids

    for message in reversed(messages):
        if message['output_ids'] is not None:
            if message['input_ids'] is not None:
                input_ids.extend(message['input_ids'])
            input_ids.extend(message['output_ids'])
            # Add this message to processed messages for image extraction
            processed_messages.append(message)
            # we break here because all messages before this message have been processed
            break
        else:
            new_message = dict(message)
            new_message.pop('input_ids', None)
            new_message.pop('output_ids', None)
            pending_messages.append(new_message)
            # Add this message to processed messages for image extraction
            processed_messages.append(new_message)

    # Extract images from ALL messages that contribute to the final input_ids
    # This includes both processed messages (with output_ids) and pending messages
    def extract_images_from_message(message):
        """Helper function to extract images from a message"""
        if 'content' in message and isinstance(message['content'], list):
            for content in message['content']:
                if isinstance(content, dict) and content.get('type') == 'image_url':
                    image_url = content.get('image_url', '')
                    if isinstance(image_url, dict):
                        # Handle case where image_url is a dict with 'url' key
                        image_b64 = image_url.get('url', '')
                    else:
                        # Handle case where image_url is directly the base64 string
                        image_b64 = image_url

                    # Clean up data URI prefix if present
                    if image_b64.startswith('data:image'):
                        image_b64 = (
                            image_b64.split(',', 1)[1]
                            if ',' in image_b64
                            else image_b64
                        )

                    if image_b64:
                        all_images_base64.append(image_b64)

    # Extract images from all processed messages in the correct order
    # Since we built processed_messages in reverse order, reverse it back to get original order
    for message in reversed(processed_messages):
        extract_images_from_message(message)

    kwargs.pop('chat_template_kwargs', {})

    pending_messages_ids = convert_messages_to_tokens(
        list(reversed(pending_messages)),
        tokenizer,
        chat_template=qwen2_5_vl_chat_template,  # you might want to modify this if you are using a different model
        add_generation_prompt=True,
        tools=kwargs.pop('tools', []),
    )

    for msg_ids in pending_messages_ids:
        input_ids.extend(msg_ids)

    kwargs.pop('messages')
    model = kwargs.pop('model', None)

    extra_body = kwargs.pop('extra_body', {})
    random_id = f'chatcmpl-{uuid.uuid4()}'

    if 'max_completion_tokens' in kwargs:
        kwargs['max_tokens'] = kwargs.pop('max_completion_tokens')
    # We ignore max_completion_tokens and instead use maximum available tokens left.
    kwargs['max_tokens'] = max_model_len - len(input_ids)

    if kwargs['max_tokens'] <= 0:
        raise ContextWindowExceededError(
            message=f'input length and `max_tokens` exceed context limit. input length: {len(input_ids)}, max_tokens: {kwargs["max_tokens"]}, max_model_len: {max_model_len}.',
            model=model,
            llm_provider='hosted_vllm',
        )

    # Prepare request payload with images support
    request_payload = {
        'prompt_ids': input_ids,
        'top_p': top_p,
        'seed': seed,
        **kwargs,
    }

    # Add images if any are present
    if all_images_base64:
        request_payload['images'] = all_images_base64

    # latencies.md §5 addition #3 — gate-controlled raw vLLM throughput log.
    # Off by default (OH_VLLM_GEN_LOG=0). Turn on for a single diagnostic run
    # to measure pool TPS ceiling independent of DAPO filter overhead.
    import time as _time  # noqa: PLC0415

    _gen_start = _time.monotonic() if _VLLM_GEN_LOG else 0.0
    resp = httpx.post(
        url=f'{base_url}/generate',
        json=request_payload,
        timeout=timeout,
    )
    # Raise error if the request is not successful
    resp.raise_for_status()
    resp = resp.json()

    if _VLLM_GEN_LOG:
        _gen_s = _time.monotonic() - _gen_start
        _tokens_out = len(resp.get('response_ids', []) or [])
        _tps = _tokens_out / _gen_s if _gen_s > 0 else 0.0
        logger.info(
            'VLLM_GENERATE %s',
            json.dumps(
                {
                    'event': 'vllm_generate',
                    'base_url': base_url,
                    'tokens_in': len(input_ids),
                    'tokens_out': _tokens_out,
                    'gen_s': round(_gen_s, 3),
                    'tps': round(_tps, 2),
                }
            ),
        )

    if 'response_ids' not in resp:
        raise ValueError(f'Response does not contain response_ids: {resp}')

    try:
        response_ids = resp['response_ids']
        if 'logprobs' in resp:
            logprobs = resp['logprobs']
        else:
            logprobs = None
        result = parse_response_ids(response_ids, tokenizer)
        result = {
            'input_ids': input_ids,
            'output_ids': response_ids,
            'logprobs': logprobs,
            'message': result,
        }
        model_response = ModelResponse(
            model=model,
            id=random_id,
            choices=[result],
            role='assistant',
        )
        return model_response
    except Exception as e:
        raise ValueError(f'Failed to parse response: {e}')
