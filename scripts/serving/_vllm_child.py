"""Child vLLM /generate server compatible with vLLM 0.18 AsyncLLMEngine.

Serves the exact contract `openhands/llm/nvidia/qwen3.py` speaks:
- POST /generate    { prompt_ids: list[int], <SamplingParams kwargs> }
                 -> { response_ids: list[int], logprobs: list[float] | null }
- GET  /health      200 OK

The `/generate` handler does not tokenize or detokenize text at any point;
it passes token IDs straight into vLLM's TokensPrompt and returns the
output's `token_ids` directly. This preserves the token-level invariant
documented in `openhands/llm/nvidia/README.md` (re-tokenizing across turns
collapses GRPO).

Invoked as a subprocess by `scripts/serving/vllm_launcher.py`. Not intended
for standalone use. The supervisor pins the GPU via `--gpus device=<n>` at
the container boundary, so no CUDA_VISIBLE_DEVICES manipulation is needed
here.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = logging.getLogger('vllm_child')

app = FastAPI()
engine: AsyncLLMEngine | None = None


@app.get('/health')
async def health() -> Response:
    if engine is None:
        return JSONResponse({'detail': 'engine not ready'}, status_code=503)
    return Response(status_code=200)


@app.post('/generate')
async def generate(request: Request) -> Response:
    assert engine is not None, 'engine not initialized'
    body: dict[str, Any] = await request.json()

    prompt_ids = body.pop('prompt_ids', None)
    if prompt_ids is None:
        return JSONResponse(
            {'detail': "request body missing 'prompt_ids'"},
            status_code=400,
        )
    # Request 1 logprob per sampled token so response includes per-token logprobs.
    body.setdefault('logprobs', 0)
    body.pop('stream', None)  # streaming not wired; non-stream only

    try:
        sampling_params = SamplingParams(**body)
    except (TypeError, ValueError) as exc:
        return JSONResponse(
            {'detail': f'invalid sampling params: {exc}'},
            status_code=400,
        )

    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    request_id = random_uuid()

    generator = engine.generate(prompt, sampling_params, request_id)
    final = None
    try:
        async for output in generator:
            final = output
    except asyncio.CancelledError:
        # Client disconnected; abort so vLLM frees GPU blocks for this request
        # instead of continuing to compute a result nobody will read.
        await engine.abort(request_id)
        raise
    assert final is not None

    out = final.outputs[0]
    return JSONResponse(
        {
            'response_ids': list(out.token_ids),
            'logprobs': _flatten_logprobs(out.logprobs),
        }
    )


def _flatten_logprobs(logprobs: Any) -> list[float] | None:
    """Reduce vLLM's per-token dict-of-top-N-Logprob to a flat list[float].

    vLLM returns `list[dict[token_id -> Logprob]]` where each dict has at least
    the sampled token. When `logprobs=0` is requested the dict holds only the
    sampled token; we take its `.logprob`. If the engine returned None
    (logprobs not requested / not supported), propagate None.
    """
    if logprobs is None:
        return None
    out: list[float] = []
    for i, d in enumerate(logprobs):
        if not d:
            # Empty dict mid-sequence is an engine invariant break, not a
            # normal "logprobs not requested" signal. Surface it loudly rather
            # than silently returning None and corrupting the trainer's
            # advantage estimates.
            logger.error('engine returned empty logprobs dict at token %d', i)
            raise RuntimeError(f'empty logprobs dict at token index {i}')
        first = next(iter(d.values()))
        out.append(first.logprob)
    return out


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='vLLM 0.18 /generate child server.')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level='INFO', format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    args = _build_args()
    global engine
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info', access_log=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
