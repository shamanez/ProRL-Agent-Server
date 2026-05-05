# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import asyncio
import ssl
from argparse import Namespace
from typing import Any, Optional

import vllm.envs as envs
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.utils import with_cancellation
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid, set_ulimit
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger('vllm.nvidia.api_server')

app = FastAPI()
engine = None


@app.get('/health')
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post('/generate')
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    return await _generate(request_dict, raw_request=request)


@with_cancellation
async def _generate(request_dict: dict, raw_request: Request) -> list[int]:
    prompt_ids = request_dict.pop('prompt_ids')
    # Set logprobs to 0 to only return selected tokens
    request_dict['logprobs'] = 0
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    # Create prompt from token IDs
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    generator = engine.generate(
        prompt=prompt, sampling_params=sampling_params, request_id=request_id
    )

    # Get final response
    final_res: Optional[RequestOutput] = None
    try:
        async for output in generator:
            final_res = output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_res is not None

    def obtain_logprobs(logprobs):
        if logprobs is None:
            return None
        log_probs = []
        for d in logprobs:
            cur_logprobs = list(d.values())
            assert len(cur_logprobs) == 1, f"Expected 1 logprob per token when logprobs=0, but got {len(cur_logprobs)}"
            log_probs.append(cur_logprobs[0].logprob)
        return log_probs

    ret = {
        'response_ids': final_res.outputs[0].token_ids,
        'logprobs': obtain_logprobs(final_res.outputs[0].logprobs),
    }
    return JSONResponse(ret)


def build_app(args: Namespace) -> FastAPI:
    global app

    app.root_path = args.root_path
    return app


async def init_app(
    args: Namespace,
    llm_engine: Optional[AsyncLLMEngine] = None,
) -> FastAPI:
    app = build_app(args)

    global engine

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = (
        llm_engine
        if llm_engine is not None
        else AsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.API_SERVER
        )
    )
    app.state.engine_client = engine
    return app


async def run_server(
    args: Namespace, llm_engine: Optional[AsyncLLMEngine] = None, **uvicorn_kwargs: Any
) -> None:
    logger.info('vLLM API server version %s', VLLM_VERSION)
    logger.info('args: %s', args)

    set_ulimit()

    app = await init_app(args, llm_engine)
    assert engine is not None

    shutdown_task = await serve_http(
        app,
        sock=None,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=getattr(envs, 'VLLM_HTTP_TIMEOUT_KEEP_ALIVE', 5),
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == '__main__':
    parser = FlexibleArgumentParser()
    parser.add_argument('--host', type=str, default=None)
    parser.add_argument('--port', type=parser.check_port, default=8000)
    parser.add_argument('--ssl-keyfile', type=str, default=None)
    parser.add_argument('--ssl-certfile', type=str, default=None)
    parser.add_argument(
        '--ssl-ca-certs', type=str, default=None, help='The CA certificates file'
    )
    parser.add_argument(
        '--enable-ssl-refresh',
        action='store_true',
        default=False,
        help='Refresh SSL Context when SSL certificate files change',
    )
    parser.add_argument(
        '--ssl-cert-reqs',
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)",
    )
    parser.add_argument(
        '--root-path',
        type=str,
        default=None,
        help='FastAPI root_path when app is behind a path based routing proxy',
    )
    parser.add_argument('--log-level', type=str, default='error')
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
