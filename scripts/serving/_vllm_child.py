"""Child vLLM /generate server compatible with vLLM 0.18 AsyncLLMEngine.

Serves the exact contract `openhands/llm/nvidia/qwen3.py` speaks:
- POST /generate      { prompt_ids: list[int], <SamplingParams kwargs> }
                   -> { response_ids: list[int], logprobs: list[float] | null }
- GET  /health        200 {policy_version} | 503 {detail}
- POST /reload_lora   multipart(adapter: tar.gz, policy_version: int)
                   -> 200 {policy_version, vllm_load_latency_ms, adapter_bytes}

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
import json
import logging
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = logging.getLogger('vllm_child')

app = FastAPI()
engine: AsyncLLMEngine | None = None

# LoRA state. `active_policy_version == 0` is the "no adapter installed"
# sentinel (trainer-authoritative counter starts at 1 on first publish).
# `_swap_lock` serializes concurrent /reload_lora callers; /generate reads
# `active_lora` without the lock via a one-shot snapshot so a mid-swap does
# not split a single request.
#
# `_inflight` counts generates currently executing per lora_int_id (int_id=0
# means base model / no adapter). `/reload_lora` drains the prior adapter's
# count to 0 before calling `engine.remove_lora(prior)` so an in-flight
# generate that snapshotted the prior adapter cannot race its removal.
active_lora: LoRARequest | None = None
active_policy_version: int = 0
_swap_lock: asyncio.Lock | None = None  # created inside event loop in main()
_inflight: dict[int, int] = {}
_inflight_cond: asyncio.Condition | None = None  # created inside event loop
_DRAIN_TIMEOUT_S = 15.0  # cap on drain wait before falling back to "leak old slot,
# return 200 degraded" path. Must stay well below the trainer's 60s HTTP timeout
# so a single straggler never aborts a run. Slot leaks are absorbed by
# --max-loras=8 headroom in _remote_vllm_runner.sh.
CHILD_PORT: int = 0
ADAPTER_STAGING_ROOT = Path('/tmp/lora_adapters')


class _JSONFormatter(logging.Formatter):
    """Emit one-line JSON per log record.

    Structured fields from `logger.info(msg, extra={...})` are merged into the
    JSON object; `msg` lands under "message". Keeps plain-text tails (jq -c)
    parseable for the weight-sync runbook in
    plans-n-solutions/stages/weight_sync_lora.md §5.3.
    """

    _RESERVED = {
        'name',
        'msg',
        'args',
        'levelname',
        'levelno',
        'pathname',
        'filename',
        'module',
        'exc_info',
        'exc_text',
        'stack_info',
        'lineno',
        'funcName',
        'created',
        'msecs',
        'relativeCreated',
        'thread',
        'threadName',
        'processName',
        'process',
        'message',
        'asctime',
        'taskName',
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            'ts': self.formatTime(record, '%Y-%m-%dT%H:%M:%S'),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith('_'):
                payload[k] = v
        if record.exc_info:
            payload['exc'] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


@app.get('/health')
async def health() -> Response:
    if engine is None:
        return JSONResponse({'detail': 'engine not ready'}, status_code=503)
    return JSONResponse({'policy_version': active_policy_version}, status_code=200)


@app.post('/generate')
async def generate(request: Request) -> Response:
    if engine is None or _inflight_cond is None:
        return JSONResponse({'detail': 'engine not ready'}, status_code=503)
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

    # Snapshot active_lora AND register inflight under the same cond lock so a
    # concurrent /reload_lora cannot commit the swap, drain an empty count,
    # and remove_lora(prior) between our snapshot and our increment. Holding
    # the cond for the snapshot guarantees: if we see the prior adapter, our
    # increment is visible to the reloader's drain wait; if we see the new
    # adapter, we're safe by construction.
    async with _inflight_cond:
        current_lora = active_lora
        inflight_key = current_lora.lora_int_id if current_lora is not None else 0
        _inflight[inflight_key] = _inflight.get(inflight_key, 0) + 1

    try:
        generator = engine.generate(
            prompt, sampling_params, request_id, lora_request=current_lora
        )
        final = None
        try:
            async for output in generator:
                final = output
        except asyncio.CancelledError:
            # Client disconnected; abort so vLLM frees GPU blocks for this
            # request instead of continuing to compute a result nobody will read.
            await engine.abort(request_id)
            raise
        if final is None:
            return JSONResponse(
                {'detail': 'engine returned no outputs'}, status_code=500
            )

        out = final.outputs[0]
        return JSONResponse(
            {
                'response_ids': list(out.token_ids),
                'logprobs': _flatten_logprobs(out.logprobs),
            }
        )
    finally:
        async with _inflight_cond:
            _inflight[inflight_key] = _inflight.get(inflight_key, 1) - 1
            if _inflight[inflight_key] <= 0:
                _inflight.pop(inflight_key, None)
                _inflight_cond.notify_all()


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


def _safe_extract_tar(tar_path: Path, dest: Path) -> None:
    """Extract a tar.gz to `dest`, rejecting path-traversal members.

    Members must resolve under `dest`. Absolute paths, symlinks, and '..'
    components raise RuntimeError before any extraction happens.
    """
    dest_resolved = dest.resolve()
    with tarfile.open(tar_path, 'r:gz') as tf:
        members = tf.getmembers()
        for m in members:
            if m.islnk() or m.issym():
                raise RuntimeError(f'tar contains link member: {m.name!r}')
            target = (dest_resolved / m.name).resolve()
            if dest_resolved != target and dest_resolved not in target.parents:
                raise RuntimeError(f'tar escape attempt: {m.name!r}')
        tf.extractall(dest_resolved)  # noqa: S202 — members validated above


@app.post('/reload_lora')
async def reload_lora(
    adapter: UploadFile = File(...),  # noqa: B008 — FastAPI DI uses call-as-default
    policy_version: int = Form(...),  # noqa: B008 — FastAPI DI uses call-as-default
) -> Response:
    """Install a new LoRA adapter in-place and retire the previous one.

    See plans-n-solutions/stages/weight_sync_lora.md §5 and §6.2 for the
    protocol. `policy_version` is trainer-authoritative and monotonic; the
    pool echoes what it installed and never mints a version.
    """
    global active_lora, active_policy_version

    if engine is None:
        return JSONResponse({'detail': 'engine not ready'}, status_code=503)
    if _swap_lock is None:
        return JSONResponse(
            {'detail': 'swap lock not initialized (server not in lifespan)'},
            status_code=503,
        )

    # Read the tarball fully before taking the lock so a slow client cannot
    # hold /reload_lora contention open.
    adapter_bytes = await adapter.read()
    n_bytes = len(adapter_bytes)

    t_start = time.monotonic()
    async with _swap_lock:
        prior_version = active_policy_version
        prior_lora = active_lora

        if policy_version <= prior_version:
            logger.info(
                'reload_lora rejected non-monotonic version',
                extra={
                    'event': 'reload_lora',
                    'port': CHILD_PORT,
                    'policy_version': policy_version,
                    'active_policy_version': prior_version,
                    'ok': False,
                    'reason': 'non-monotonic',
                },
            )
            return JSONResponse(
                {
                    'detail': (
                        f'policy_version {policy_version} <= active '
                        f'{prior_version}; refusing non-monotonic reload'
                    ),
                    'policy_version': prior_version,
                },
                status_code=409,
            )

        extract_dir = ADAPTER_STAGING_ROOT / f'pv{policy_version}'
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        t_extract_start = time.monotonic()
        try:
            with tempfile.NamedTemporaryFile(suffix='.tgz', delete=False) as tmp:
                tmp.write(adapter_bytes)
                tar_path = Path(tmp.name)
            try:
                _safe_extract_tar(tar_path, extract_dir)
            finally:
                tar_path.unlink(missing_ok=True)
        except (tarfile.TarError, RuntimeError, OSError) as exc:
            shutil.rmtree(extract_dir, ignore_errors=True)
            logger.exception(
                'reload_lora tar extraction failed',
                extra={
                    'event': 'reload_lora',
                    'port': CHILD_PORT,
                    'policy_version': policy_version,
                    'ok': False,
                    'reason': 'tar_error',
                },
            )
            return JSONResponse(
                {
                    'detail': f'tar extraction failed: {exc}',
                    'policy_version_before': prior_version,
                    'policy_version_after': prior_version,
                },
                status_code=400,
            )
        extract_ms = int((time.monotonic() - t_extract_start) * 1000)

        missing = [
            f
            for f in ('adapter_model.safetensors', 'adapter_config.json')
            if not (extract_dir / f).exists()
        ]
        if missing:
            shutil.rmtree(extract_dir, ignore_errors=True)
            logger.error(
                'reload_lora tarball missing required files',
                extra={
                    'event': 'reload_lora',
                    'port': CHILD_PORT,
                    'policy_version': policy_version,
                    'ok': False,
                    'reason': 'missing_files',
                    'missing': missing,
                },
            )
            return JSONResponse(
                {
                    'detail': f'adapter tarball missing required files: {missing}',
                    'policy_version_before': prior_version,
                    'policy_version_after': prior_version,
                },
                status_code=400,
            )

        new_request = LoRARequest(
            lora_name=f'pv{policy_version}',
            lora_int_id=policy_version,
            lora_path=str(extract_dir),
        )

        t_add_start = time.monotonic()
        try:
            await engine.add_lora(new_request)
        except Exception as exc:  # noqa: BLE001 — surface all engine errors to caller
            shutil.rmtree(extract_dir, ignore_errors=True)
            logger.exception(
                'reload_lora engine.add_lora failed',
                extra={
                    'event': 'reload_lora',
                    'port': CHILD_PORT,
                    'policy_version': policy_version,
                    'ok': False,
                    'reason': 'add_lora_error',
                },
            )
            return JSONResponse(
                {
                    'detail': f'engine.add_lora failed: {exc}',
                    'policy_version_before': prior_version,
                    'policy_version_after': prior_version,
                },
                status_code=500,
            )
        add_lora_ms = int((time.monotonic() - t_add_start) * 1000)

        # Commit: new adapter is live.
        active_lora = new_request
        active_policy_version = policy_version

        remove_lora_ms = 0
        remove_lora_failed = False
        drain_timed_out = False
        drain_ms = 0
        if prior_lora is not None:
            # Drain in-flight generates that snapshotted the prior adapter
            # before removing it from the engine. Without this, a generate
            # that captured `current_lora = prior_lora` could race ahead and
            # hit the engine after remove_lora was issued (vLLM's behaviour
            # on a removed lora_int_id is undefined — see the hunter finding).
            assert _inflight_cond is not None  # set in startup
            t_drain_start = time.monotonic()
            try:
                async with _inflight_cond:
                    await asyncio.wait_for(
                        _inflight_cond.wait_for(
                            lambda: _inflight.get(prior_lora.lora_int_id, 0) == 0
                        ),
                        timeout=_DRAIN_TIMEOUT_S,
                    )
            except asyncio.TimeoutError:
                drain_timed_out = True
                logger.error(
                    'reload_lora drain timed out',
                    extra={
                        'event': 'reload_lora_drain_timeout',
                        'port': CHILD_PORT,
                        'policy_version': policy_version,
                        'prior_policy_version': prior_version,
                        'inflight_prior': _inflight.get(prior_lora.lora_int_id, 0),
                        'ok': False,
                    },
                )
            drain_ms = int((time.monotonic() - t_drain_start) * 1000)

            # On drain timeout, skipping `remove_lora(prior)` is the correct
            # choice: in-flight generates would otherwise run against a
            # removed lora_int_id (vLLM behaviour undefined). The prior
            # adapter slot leaks until pool restart, but correctness of the
            # in-flight batch is preserved. We still return 5xx so the
            # trainer aborts and an operator intervenes.
            if not drain_timed_out:
                t_remove_start = time.monotonic()
                try:
                    await engine.remove_lora(prior_lora.lora_int_id)
                except Exception:  # noqa: BLE001 — swap already committed
                    remove_lora_failed = True
                    logger.exception(
                        'reload_lora engine.remove_lora(prior) failed',
                        extra={
                            'event': 'reload_lora_remove_failed',
                            'port': CHILD_PORT,
                            'policy_version': policy_version,
                            'prior_policy_version': prior_version,
                            'ok': False,
                        },
                    )
                remove_lora_ms = int((time.monotonic() - t_remove_start) * 1000)

        # Clean up prior adapter dir on disk (best effort; log on failure).
        if prior_version > 0:
            prior_dir = ADAPTER_STAGING_ROOT / f'pv{prior_version}'
            try:
                shutil.rmtree(prior_dir)
            except (FileNotFoundError, OSError) as exc:
                logger.warning(
                    'reload_lora prior adapter dir cleanup failed',
                    extra={
                        'event': 'reload_lora_cleanup_failed',
                        'port': CHILD_PORT,
                        'prior_policy_version': prior_version,
                        'path': str(prior_dir),
                        'err': str(exc),
                        'ok': False,
                    },
                )

    reload_wall_ms = int((time.monotonic() - t_start) * 1000)
    # drain_timed_out alone is benign: the swap already committed at line
    # ~389 (active_lora = new_request) so new generates use the new adapter;
    # the in-flight straggler finishes safely against its snapshot of the
    # old LoRARequest; we only skipped remove_lora(old), which means the
    # old lora_int_id slot leaks until pool restart. max-loras must be
    # sized with headroom (see _remote_vllm_runner.sh). remove_lora_failed
    # is different — that's a real engine error; keep returning 502 so the
    # trainer aborts.
    ok = not remove_lora_failed
    status_code = 200 if ok else 502

    logger.info(
        'reload_lora %s',
        'ok' if ok else 'degraded',
        extra={
            'event': 'reload_lora',
            'port': CHILD_PORT,
            'policy_version': policy_version,
            'prior_policy_version': prior_version,
            'adapter_bytes': n_bytes,
            'extract_ms': extract_ms,
            'add_lora_ms': add_lora_ms,
            'drain_ms': drain_ms,
            'remove_lora_ms': remove_lora_ms,
            'reload_wall_ms': reload_wall_ms,
            'remove_lora_failed': remove_lora_failed,
            'drain_timed_out': drain_timed_out,
            'ok': ok,
        },
    )

    return JSONResponse(
        {
            'policy_version': policy_version,
            # Keep the key named vllm_load_latency_ms for trainer compatibility
            # (design doc §5.2 metric name) but populate it with add_lora_ms
            # — the GPU-side load cost — NOT the all-inclusive wall clock
            # which includes drain + remove + cleanup. Trainer computes
            # transfer_latency_s = publish_latency_s - vllm_load_latency_s;
            # if this field is all-inclusive the diff goes negative.
            'vllm_load_latency_ms': add_lora_ms,
            'adapter_bytes': n_bytes,
            'remove_lora_failed': remove_lora_failed,
            'drain_timed_out': drain_timed_out,
        },
        status_code=status_code,
    )


@app.on_event('startup')
async def _init_async_primitives() -> None:
    """Create asyncio primitives inside the running event loop.

    Creating them at module import binds them to whatever loop imports the
    module, which under uvicorn is *not* the loop that serves requests.
    """
    global _swap_lock, _inflight_cond
    _swap_lock = asyncio.Lock()
    _inflight_cond = asyncio.Condition()
    ADAPTER_STAGING_ROOT.mkdir(parents=True, exist_ok=True)


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='vLLM 0.18 /generate child server.')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


def _configure_json_logging() -> None:
    """Replace the root logger's handlers with a single JSON-formatted stream handler."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def main() -> int:
    _configure_json_logging()
    args = _build_args()
    global engine, CHILD_PORT
    CHILD_PORT = args.port
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info', access_log=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
