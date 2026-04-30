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
# means base model / no adapter).
#
# Two swap protocols are supported (selected at startup via --swap-protocol):
#   * "pinning"  — multi-tenant. Path-versioned /v{N}/generate routes pin a
#                  call to a specific lora_int_id N. /reload_lora installs new
#                  adapters but never calls engine.remove_lora; vLLM's internal
#                  LRU (max_loras / max_cpu_loras) handles GPU/CPU eviction.
#                  Disk pv{N} dirs are kept so an evicted adapter can be
#                  re-added on demand. This is the design that satisfies
#                  per-trajectory and per-group policy consistency.
#   * "quiesce"  — single-tenant fallback. /reload_lora drains in-flight to
#                  zero before calling remove_lora(prior). Returns HTTP 503
#                  on drain timeout (no silent slot leak). Path-versioned
#                  routes accept lora_int_id == active_policy_version only.
active_lora: LoRARequest | None = None
active_policy_version: int = 0
_swap_lock: asyncio.Lock | None = None  # created inside event loop in main()
_inflight: dict[int, int] = {}
_inflight_cond: asyncio.Condition | None = None  # created inside event loop
# Pinning-mode bookkeeping: every LoRARequest we have add_lora'd lives here so
# /v{N}/generate can dispatch against it without re-reading the disk on hot
# paths. vLLM may LRU-evict the GPU/CPU slot underneath us; on the next call
# the engine pages it back from disk via the LoRARequest.lora_path. We never
# drop entries from this dict in pinning mode (the cap is bounded by the
# number of train steps in the run, and disk cleanup is a separate concern).
_resident_loras: dict[int, LoRARequest] = {}
swap_protocol: str = 'pinning'  # set in main() from --swap-protocol
_DRAIN_TIMEOUT_S = 15.0  # quiesce-mode default. Overridden per-request via
# --quiesce-drain-timeout-s for runs where individual sessions can exceed 15s.
# In pinning mode this is unused (no drain).
_QUIESCE_DRAIN_TIMEOUT_S = 600.0  # quiesce-mode max wait; HTTP 503 past this.
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
    # Snapshot inflight without acquiring _inflight_cond — health probes are
    # frequent and we just want a coarse view. CPython dict reads are atomic.
    inflight_snapshot = {int(k): int(v) for k, v in _inflight.items() if v > 0}
    payload: dict[str, Any] = {
        'policy_version': active_policy_version,
        'swap_protocol': swap_protocol,
        # Versions for which the child still has the pv{N} dir on disk + a
        # cached LoRARequest. /v{N}/generate against any of these will succeed
        # (either from the engine's GPU/CPU LoRA cache, or via re-add from
        # disk if vLLM's internal LRU evicted it).
        'pinned_versions': sorted(_resident_loras.keys()),
        # Per-version in-flight count — what the trainer's
        # _publish_lora_adapter backpressure reads. A version with inflight>0
        # cannot be evicted without splitting a trajectory, so the trainer
        # avoids publishing a new adapter while too many distinct versions
        # have in-flight work (to stay within --max-loras headroom).
        'inflight_per_version': inflight_snapshot,
        'inflight_versions_count': len(inflight_snapshot),
    }
    return JSONResponse(payload, status_code=200)


async def _do_generate(
    request: Request,
    *,
    pinned_lora: LoRARequest | None,
    inflight_key: int,
) -> Response:
    """Shared body for /generate and /v{N}/generate.

    `pinned_lora` is the LoRARequest to dispatch against (may be None for
    base-model). `inflight_key` is the lora_int_id used for the inflight
    counter — 0 for base/no-adapter, else N. Caller must already hold a
    valid pre-flight (engine + cond non-None, body parsed)."""
    assert engine is not None
    assert _inflight_cond is not None

    body: dict[str, Any] = await request.json()
    prompt_ids = body.pop('prompt_ids', None)
    if prompt_ids is None:
        return JSONResponse(
            {'detail': "request body missing 'prompt_ids'"},
            status_code=400,
        )
    body.setdefault('logprobs', 0)
    body.pop('stream', None)

    try:
        sampling_params = SamplingParams(**body)
    except (TypeError, ValueError) as exc:
        return JSONResponse(
            {'detail': f'invalid sampling params: {exc}'},
            status_code=400,
        )

    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    request_id = random_uuid()

    async with _inflight_cond:
        _inflight[inflight_key] = _inflight.get(inflight_key, 0) + 1

    try:
        generator = engine.generate(
            prompt, sampling_params, request_id, lora_request=pinned_lora
        )
        final = None
        try:
            async for output in generator:
                final = output
        except asyncio.CancelledError:
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


@app.post('/generate')
async def generate(request: Request) -> Response:
    """Legacy unpinned route. Snapshots whatever adapter is currently active.

    This route is kept for backward compatibility with the eval/validation path
    and any caller that does not yet stamp `policy_version` on its instance.
    For training-time correctness use `/v{N}/generate` (pinning mode) so calls
    cannot be silently mixed across a mid-trajectory swap.
    """
    if engine is None or _inflight_cond is None:
        return JSONResponse({'detail': 'engine not ready'}, status_code=503)

    # Snapshot active_lora under the cond lock so a concurrent /reload_lora
    # (in quiesce mode, where remove_lora can fire) cannot drain past us.
    # In pinning mode this is still correct — we just never call remove_lora
    # so the inflight count is informational only.
    async with _inflight_cond:
        current_lora = active_lora
        inflight_key = current_lora.lora_int_id if current_lora is not None else 0

    # Note: _do_generate increments the inflight counter again under its own
    # cond acquisition, but it does so for the same `inflight_key`. The brief
    # window between the snapshot and the increment is safe because /reload_lora
    # in quiesce mode holds _swap_lock for the whole drain — concurrent
    # snapshots see the same active_lora throughout.
    return await _do_generate(
        request, pinned_lora=current_lora, inflight_key=inflight_key
    )


@app.post('/v{lora_int_id:int}/generate')
async def generate_pinned(lora_int_id: int, request: Request) -> Response:
    """Pinned-version generate: dispatch this call against a specific adapter.

    The trainer-side rollout manager stamps `instance['policy_version']` on
    every trajectory at expansion time (and every sibling of one prompt
    inherits the same version, see refill_job_queue in async_server_dapo.py).
    The ProRL-side OpenHandsServer rewrites the per-job vLLM base_url to
    `<host>:<port>/v{N}` so qwen3.py's downstream `f"{base_url}/generate"`
    lands here.

    Returns HTTP 410 Gone if the requested version is no longer recoverable
    (the pv{N} dir was cleaned up). Caller must abort the trajectory; do NOT
    silently fall back to the active adapter.
    """
    if engine is None or _inflight_cond is None or _swap_lock is None:
        return JSONResponse({'detail': 'engine not ready'}, status_code=503)

    if lora_int_id == 0:
        # Explicit "base model, no adapter" pin. Used by the validation /
        # warmup path when there is no published adapter yet.
        return await _do_generate(request, pinned_lora=None, inflight_key=0)

    pinned_lora = _resident_loras.get(lora_int_id)
    if pinned_lora is None:
        # Slot was never installed (or the process was restarted). Try to
        # re-add from disk; pv{N} dirs persist in pinning mode.
        adapter_dir = ADAPTER_STAGING_ROOT / f'pv{lora_int_id}'
        if not (adapter_dir / 'adapter_model.safetensors').exists():
            logger.warning(
                'pinned generate refused: version not recoverable',
                extra={
                    'event': 'pinned_generate_410',
                    'port': CHILD_PORT,
                    'requested_policy_version': lora_int_id,
                    'active_policy_version': active_policy_version,
                    'reason': 'no_disk_dir',
                },
            )
            return JSONResponse(
                {
                    'detail': (
                        f'pinned policy_version {lora_int_id} no longer resident '
                        f'(active={active_policy_version})'
                    ),
                    'requested_policy_version': lora_int_id,
                    'active_policy_version': active_policy_version,
                },
                status_code=410,
            )
        pinned_lora = LoRARequest(
            lora_name=f'pv{lora_int_id}',
            lora_int_id=lora_int_id,
            lora_path=str(adapter_dir),
        )
        try:
            async with _swap_lock:
                # Re-check under lock (another caller may have added it).
                if lora_int_id not in _resident_loras:
                    await engine.add_lora(pinned_lora)
                    _resident_loras[lora_int_id] = pinned_lora
                else:
                    pinned_lora = _resident_loras[lora_int_id]
        except Exception as exc:  # noqa: BLE001 — surface engine errors
            logger.exception(
                'pinned generate add_lora failed',
                extra={
                    'event': 'pinned_generate_add_lora_failed',
                    'port': CHILD_PORT,
                    'requested_policy_version': lora_int_id,
                    'active_policy_version': active_policy_version,
                },
            )
            return JSONResponse(
                {
                    'detail': f'engine.add_lora({lora_int_id}) failed: {exc}',
                    'requested_policy_version': lora_int_id,
                    'active_policy_version': active_policy_version,
                },
                status_code=500,
            )

    return await _do_generate(
        request, pinned_lora=pinned_lora, inflight_key=lora_int_id
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
        _resident_loras[policy_version] = new_request

        remove_lora_ms = 0
        remove_lora_failed = False
        drain_timed_out = False
        drain_ms = 0
        if swap_protocol == 'pinning':
            # Pinning mode: do NOT drain or call engine.remove_lora(prior).
            # In-flight calls that pinned to the prior version still resolve
            # against `_resident_loras[prior_version]`, vLLM's LRU handles
            # GPU/CPU slot pressure, and we keep pv{prior_version}/ on disk
            # so a re-bind from /v{prior_version}/generate can re-page in.
            # The active_lora swap is purely for the legacy unpinned route
            # (and validation paths). This is what makes per-trajectory and
            # per-group consistency possible at save_freq=1.
            pass
        elif prior_lora is not None:
            # Quiesce mode: drain in-flight that snapshotted the prior adapter
            # before removing it. vLLM's behaviour on a generate against a
            # removed lora_int_id is undefined.
            assert _inflight_cond is not None  # set in startup
            drain_deadline_s = _QUIESCE_DRAIN_TIMEOUT_S
            t_drain_start = time.monotonic()
            try:
                async with _inflight_cond:
                    await asyncio.wait_for(
                        _inflight_cond.wait_for(
                            lambda: _inflight.get(prior_lora.lora_int_id, 0) == 0
                        ),
                        timeout=drain_deadline_s,
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
                        'drain_timeout_s': drain_deadline_s,
                        'ok': False,
                    },
                )
            drain_ms = int((time.monotonic() - t_drain_start) * 1000)

            if not drain_timed_out:
                t_remove_start = time.monotonic()
                try:
                    await engine.remove_lora(prior_lora.lora_int_id)
                    _resident_loras.pop(prior_lora.lora_int_id, None)
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

        # Clean up prior adapter dir on disk only in quiesce mode where we
        # know no future call can refer to prior_version. In pinning mode
        # we keep pv{N} indefinitely so /v{N}/generate can re-page from disk.
        if swap_protocol == 'quiesce' and prior_version > 0 and not drain_timed_out:
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
    # drain_timed_out is fatal in quiesce mode (the contract is "no silent
    # leak"); HTTP 503 so the trainer aborts. remove_lora_failed is also fatal
    # (real engine error); HTTP 502. In pinning mode neither flag fires.
    if drain_timed_out:
        ok = False
        status_code = 503
    elif remove_lora_failed:
        ok = False
        status_code = 502
    else:
        ok = True
        status_code = 200

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
    parser.add_argument(
        '--swap-protocol',
        type=str,
        choices=('pinning', 'quiesce'),
        default='pinning',
        help=(
            'How /reload_lora retires the prior adapter. "pinning" (default): '
            'never call engine.remove_lora; rely on vLLM LRU; serve concurrent '
            'versions via /v{N}/generate. "quiesce": drain in-flight then '
            'remove; HTTP 503 on drain timeout (no silent leak).'
        ),
    )
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
    global engine, CHILD_PORT, swap_protocol
    CHILD_PORT = args.port
    swap_protocol = args.swap_protocol
    logger.info(
        'vllm_child startup',
        extra={
            'event': 'startup',
            'port': args.port,
            'swap_protocol': swap_protocol,
        },
    )
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    uvicorn.run(app, host=args.host, port=args.port, log_level='info', access_log=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
