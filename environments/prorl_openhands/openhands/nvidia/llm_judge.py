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

import logging
import os

from openai import AsyncOpenAI

# Change LLM_JUDGE_LOG_DIR from debug flag to log path
LLM_JUDGE_LOG_DIR = os.getenv('LLM_JUDGE_LOG_DIR', None)

# Setup logging
logger = logging.getLogger(__name__)
# Clear any existing handlers safely
for handler in logger.handlers[
    :
]:  # Create a copy to avoid modification during iteration
    logger.removeHandler(handler)

if LLM_JUDGE_LOG_DIR:
    os.makedirs(os.path.dirname(LLM_JUDGE_LOG_DIR), exist_ok=True)
    logger.addHandler(logging.FileHandler(LLM_JUDGE_LOG_DIR))
    logger.setLevel(logging.INFO)
else:
    # If no log path provided, use NullHandler as the only handler
    logger.addHandler(logging.NullHandler())

# Prevent log propagation to parent loggers to avoid terminal output
logger.propagate = False

# STRICTLY match
STRICTLY_JUDGE_PROMPT_TEMPLATE = """Given a problem, determine whether the final answer in the provided (incomplete) solution process matches the reference answer.

The reference answer may be a single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple questions are involved.

**The reference answer may be in Chinese or another language, but your evaluation should be language-agnostic.**

---

Your task:

1. Carefully compare the **final output** of the solution process with the **reference answer**.

2. Provide a **brief explanation first**, including:
   - Whether the final answer is clearly stated.
   - Whether it matches the reference answer **exactly** (including form, number, symbol, or character).
   - Any mismatch, ambiguity, or incompleteness in the solution process.

3. After the explanation, output only **YES** or **NO**, strictly in the format: `\\boxed{{YES}}` or `\\boxed{{NO}}`.

4. Matching must be **strict**:
   - Any discrepancy in number, character, expression, or format = **NO**.
   - If the solution is ambiguous, missing, or incomplete = **NO**.
   - Only a clear and exact match = **YES**.

---

**Question:**

{question}

**Solution Process (Final Step Only):**

{response}

**Reference Answer:**

{reference}

**Output:**

(Explain first, then give your final answer below in the format `\\boxed{{YES}}` or `\\boxed{{NO}}`.)
"""

# EQUIVALENT match
EQUIVALENT_JUDGE_PROMPT_TEMPLATE = """Given a problem, determine whether the final answer in the provided (incomplete) solution process is equivalent to the reference answer.

The reference answer may be a single option character (e.g., A, B, C, D), a numerical value, an expression, or a list of answers if multiple parts are involved.

**The reference answer may appear in Chinese or other languages. Your evaluation should remain language-agnostic.**

---

Your task:

1. Analyze the final output in the solution process and compare it with the reference answer.

2. If they are **logically or mathematically equivalent**, output **YES**.

3. If they are **not equivalent**, output **NO**.

4. If the final step of the solution is unclear, incomplete, or ambiguous, treat it as incorrect and output **NO**.

---

Your output must be either **'YES'** or **'NO'**, formatted strictly as: `\\boxed{{YES}}` or `\\boxed{{NO}}`.

---

**Question:**
{question}

**Solution Process (Final Step Only):**
{response}

**Reference Answer:**
{reference}

**Output:**

(First, briefly explain your reasoning. Then give your final answer in the required format: `\\boxed{{YES}}` or `\\boxed{{NO}}`.)"""


LLM_JUDGE_PROMPT_TEMPLATE = os.getenv('LLM_JUDGE_PROMPT_TEMPLATE', 'EQUIVALENT')

if LLM_JUDGE_PROMPT_TEMPLATE == 'STRICT':
    LLM_JUDGE_PROMPT_TEMPLATE = STRICTLY_JUDGE_PROMPT_TEMPLATE
elif LLM_JUDGE_PROMPT_TEMPLATE == 'EQUIVALENT':
    LLM_JUDGE_PROMPT_TEMPLATE = EQUIVALENT_JUDGE_PROMPT_TEMPLATE
else:
    print(f'Use env variable LLM_JUDGE_PROMPT_TEMPLATE: {LLM_JUDGE_PROMPT_TEMPLATE}')


def extract_answer_from_box(string):
    """Extract Answer String from \\boxed or \\fbox expression."""
    if not string or not string.strip():
        return None

    # Try to find \\boxed first
    idx = string.rfind('\\boxed')
    box_type = '\\boxed'

    if idx < 0:
        # If not found, try \\fbox
        idx = string.rfind('\\fbox')
        box_type = '\\fbox'
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = box_type + '{'
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == '}'
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


async def compute_score(
    question: str,
    response_to_judge: str,
    metadata: dict,
    sampling_params_dict: dict,
    client: AsyncOpenAI,
) -> float:
    """Judges a single response using the LLM.

    Args:
        question: The question string.
        response_to_judge: The assistant's response string.
        metadata: Metadata containing reference answer and criteria.
        sampling_params_dict: Dictionary containing OpenAI API parameters.
        client: AsyncOpenAI client instance from the client pool.

    Returns:
        Score (float)
    """
    reference = metadata.get('reference_answer', '')
    extract_box = metadata.get('extract_box', False)

    # Check if response has proper format when extract_box is False
    if not extract_box:
        has_think_tag = '</think>' in response_to_judge
        has_answer_tag = '<answer>' in response_to_judge

        # If neither tag exists, return format error reward
        if not has_think_tag and not has_answer_tag:
            logger.warning(
                '[LLM_JUDGE] Format error: response missing both </think> and <answer> tags'
            )
            return 0.0

    # Remove thinking part if using thinking models
    response_to_judge = response_to_judge.split('</think>')[-1].strip()
    response_to_judge = response_to_judge.split('<answer>')[-1].strip()

    response_to_judge = (
        extract_answer_from_box(response_to_judge) if extract_box else response_to_judge
    )
    response_to_judge = 'None' if response_to_judge is None else response_to_judge
    # Prioritize metadata's judge_prompt_template, then default
    # Note that if you want to use a custom judge_prompt_template, you may need to change the verdict extraction logic accordingly
    current_judge_prompt = (
        metadata.get('judge_prompt_template', None) or LLM_JUDGE_PROMPT_TEMPLATE
    )

    prompt = current_judge_prompt.format(
        question=question,
        response=response_to_judge,
        reference=reference,
    )
    logger.info(f'[LLM_JUDGE] Prompt: {prompt}')
    logger.info(f'[LLM_JUDGE] Sampling params dict: {sampling_params_dict}')
    # Extract OpenAI parameters from sampling_params_dict
    model = sampling_params_dict.get('model', 'meta/llama-3.3-70b-instruct')
    temperature = sampling_params_dict.get('temperature', 0.2)
    top_p = sampling_params_dict.get('top_p', 0.7)
    top_k = sampling_params_dict.get('top_k', -1)
    max_tokens = sampling_params_dict.get('max_tokens', 1024)
    max_model_len = sampling_params_dict.get('max_model_len', 8192)
    logger.info(
        f'[LLM_JUDGE] Prompt: {prompt}, model = {model}, temperature = {temperature}, top_p = {top_p}, max_tokens = {max_tokens}, max_model_len = {max_model_len}, top_k = {top_k}'
    )
    try:
        logger.info(f'[LLM_JUDGE] Prompt: {prompt}')
        completion = await client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
            extra_body={
                'chat_template_kwargs': {'enable_thinking': False},
                'truncate_prompt_tokens': max_model_len - max_tokens,
                'top_k': top_k,
            },
        )
        logger.info(f'[LLM_JUDGE] Completion: {completion}')
        generated_text = ''
        async for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                generated_text += chunk.choices[0].delta.content
        logger.info(f'[LLM_JUDGE] Generated text: {generated_text}')
        generated_text = generated_text.strip()
        logger.info(f'[LLM_JUDGE] Generated text: {generated_text}')
    except Exception as e:
        logger.error(f'[LLM_JUDGE] Error calling OpenAI API: {e}')
        return 0.0
    logger.info(f'[LLM_JUDGE] Generated text: {generated_text}')
    score = 0.0
    if generated_text:
        generated_text_lower = generated_text.lower()
        result = extract_answer_from_box(generated_text_lower)
        # to avoid the case that the answer is not in the box
        if result is None:
            has_yes = 'yes' in generated_text_lower
        else:
            has_yes = 'yes' in result
        logger.info(f'[LLM_JUDGE] Has yes: {has_yes}')
        if has_yes:
            score = 1.0
            print(
                f"[LLM_JUDGE] Parsed 'yes'. Score: {score}. Output: '{generated_text}'"
            )
            logger.info(
                f"[LLM_JUDGE] Parsed 'yes'. Score: {score}. Output: '{generated_text}'"
            )
        else:
            score = 0.0
            print(
                f"[LLM_JUDGE] No 'yes' found. Score: {score}. Output: '{generated_text}'"
            )
            logger.info(
                f"[LLM_JUDGE] No 'yes' found. Score: {score}. Output: '{generated_text}'"
            )
    else:
        print('[LLM_JUDGE] No output received from LLM.')
        logger.warning('[LLM_JUDGE] No output received from LLM.')

    return score
