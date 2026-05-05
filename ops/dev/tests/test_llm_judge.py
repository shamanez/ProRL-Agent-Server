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

import asyncio
import os
import random

from openhands.nvidia import llm_judge

os.environ['OPENAI_BASE_URL'] = 'http://pool0-01519:8000/v1'
os.environ['OPENAI_API_KEY'] = 'dummy_key'
os.environ['OPENAI_MODEL'] = (
    '/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/jianh/data/models/Qwen3-30B-A3B-Instruct-2507-FP8'
)
os.environ['OPENAI_TEMPERATURE'] = '0'
os.environ['OPENAI_TOP_P'] = '1.0'
os.environ['OPENAI_TOP_K'] = '-1'
os.environ['OPENAI_MAX_TOKENS'] = '1024'
os.environ['OPENAI_MAX_MODEL_LEN'] = '9216'


# Client pool for LLM judge to avoid repeated client creation
_client_pool = {}
_client_pool_lock = asyncio.Lock()


async def _get_client_from_pool(base_url, api_key):
    from openai import AsyncOpenAI

    """Get or create a client from the pool based on base_url and api_key."""
    async with _client_pool_lock:
        if base_url not in _client_pool:
            _client_pool[base_url] = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return _client_pool[base_url]


# Extract information from ground_truth_info
question = "Consider the pure state\\n\\n\\[\\n|\\psi\\rangle := \\alpha |00\\rangle + \\beta |11\\rangle\\n\\]\\n\\nin the Hilbert space \\(\\mathbb{C}^2 \\otimes \\mathbb{C}^2\\), where \\(\\alpha, \\beta \\in \\mathbb{C}\\) and \\(|\\alpha|^2 + |\\beta|^2 = 1\\). Let \\(\\rho := |\\psi\\rangle \\langle\\psi|\\) be the corresponding density matrix.\\n\\n(i) Find \\(-\\text{tr} \\rho_1 \\log_2 \\rho_1\\) where \\(\\rho_1 := \\text{tr}_{\\mathbb{C}^2} \\rho\\).\\n\\n(ii) Let \\(\\tilde{\\rho}\\) be a density matrix for a disentangled state on \\(\\mathbb{C}^2 \\otimes \\mathbb{C}^2\\). Find the *fidelity* (also called *Uhlmann’s transition probability*)\\n\\n\\[\\n\\mathcal{F}(\\rho, \\tilde{\\rho}) := \\left[ \\text{tr} \\sqrt{\\sqrt{\\tilde{\\rho}} \\rho \\sqrt{\\tilde{\\rho}}} \\right]^2.\\n\\]\\n\\n(iii) Show that the minimum over \\(\\tilde{\\rho}\\) of the *modified Bures metric*\\n\\n\\[\\nD_B(\\rho, \\tilde{\\rho}) := 2 - 2 \\mathcal{F}(\\rho, \\tilde{\\rho})\\n\\]\\n\\nis given by \\(4 |\\alpha|^2 (1 - |\\alpha|^2)\\) at\\n\\n\\[\\n\\sigma := |\\alpha|^2 |00\\rangle \\langle 00| + |\\beta|^2 |11\\rangle \\langle 11|.\\n\\]\\n\\nThe *Bures metric* is defined as\\n\\n\\[\\nD_{\\text{Bures}}(\\rho, \\tilde{\\rho}) := 2 - 2 \\sqrt{\\mathcal{F}(\\rho, \\tilde{\\rho})}.\\n\\]\\n\\n(iv) Compare the result in (iii) with the result from (i). Let's think step by step and output the final answer within \boxed{}."

metadata = {
    'reference_answer': '-|\\alpha|^2 \\log_2 |\\alpha|^2 - |\\beta|^2 \\log_2 |\\beta|^2',
    'extract_box': False,
}

# solution_str = "<answer>-|\\alpha|^2 \\log_2 |\\alpha|^2 - |\\beta|^2 \\log_2 |\\beta|^2</answer>"
solution_str = '<answer>-|α|^2 log2|α|^2 - |β|^2 log2|β|^2</answer>'

# Get base_url from environment variable, support list format
base_url_env = os.getenv('OPENAI_BASE_URL', 'https://integrate.api.nvidia.com/v1')
# Parse as comma-separated list and randomly select one
base_url_list = [url.strip() for url in base_url_env.split(',') if url.strip()]
base_url = random.choice(base_url_list) if len(base_url_list) > 1 else base_url_list[0]

sampling_params = {
    'model': os.getenv('OPENAI_MODEL', 'meta/llama-3.3-70b-instruct'),
    'temperature': float(os.getenv('OPENAI_TEMPERATURE', '0.0')),
    'top_p': float(os.getenv('OPENAI_TOP_P', '0.7')),
    'top_k': int(os.getenv('OPENAI_TOP_K', '-1')),
    'max_tokens': int(os.getenv('OPENAI_MAX_TOKENS', '1024')),
    'max_model_len': int(os.getenv('OPENAI_MAX_MODEL_LEN', '8192')),
}


async def main():
    client = await _get_client_from_pool(base_url, os.getenv('OPENAI_API_KEY', ''))
    res = await llm_judge.compute_score(
        question, solution_str, metadata, sampling_params, client
    )
    print(f'Score: {res}')


if __name__ == '__main__':
    asyncio.run(main())
