"""Stage 1 vLLM supervisor.

Runs inside the verlai/verl:vllm018.dev1 container. Spawns a child vLLM AsyncEngine
/generate server as a subprocess pinned to the container's visible GPU, then fronts
it with three HTTP routes on the supervisor port so ProRL can treat the supervisor
as a drop-in vLLM endpoint:

- GET  /health          -> proxies child /health with a startup grace window
- POST /generate        -> thin pass-through to child /generate (no re-tokenization)
- POST /reload_weights  -> 501 stub (Stage 4 territory)

Signal discipline:
- Child is started with prctl(PR_SET_PDEATHSIG, SIGTERM) so the kernel reaps it
  even if the supervisor is SIGKILL'd (OOM killer / orchestrator nuke).
- Clean shutdown path: SIGTERM/SIGINT -> child SIGTERM -> 30 s wait -> SIGKILL.
- PID files: /tmp/vllm-sup-<port>.pid and /tmp/vllm-child-<port>.pid.

Child = scripts/serving/_vllm_child.py. That module speaks the exact
{prompt_ids} -> {response_ids, logprobs} contract ProRL's qwen3.py client
expects, so the supervisor does not translate between OpenAI /v1/completions
and ProRL's /generate. The token-level invariant is preserved by passing
prompt_ids into TokensPrompt without re-tokenization.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger('vllm_launcher')

PR_SET_PDEATHSIG = 1
SIGTERM_GRACE_SECONDS = 30
HEALTH_STARTUP_GRACE_SECONDS = 120  # model load + engine warmup on cold cache
HEALTH_PROXY_TIMEOUT_SECONDS = 2.0
GENERATE_PROXY_TIMEOUT_SECONDS = 1000.0  # aligned with ProRL --timeout 1000
DEFAULT_SUPERVISOR_HOST = '0.0.0.0'
CHILD_HOST = '127.0.0.1'


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGTERM this process when its parent dies.

    Defends against a supervisor that is SIGKILL'd before its signal handlers run.
    Without this the child vLLM would be orphaned and would keep holding GPU memory.
    """
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    rc = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f'prctl(PR_SET_PDEATHSIG) failed: {os.strerror(err)}')


class ChildProcess:
    """Lifecycle wrapper for the child vLLM /generate server subprocess."""

    def __init__(
        self,
        *,
        model: str,
        child_port: int,
        child_log_path: Path,
        child_pid_path: Path,
        vllm_args: list[str],
    ) -> None:
        self.model = model
        self.child_port = child_port
        self.child_log_path = child_log_path
        self.child_pid_path = child_pid_path
        self.vllm_args = vllm_args
        self.proc: subprocess.Popen | None = None
        self.start_monotonic: float = 0.0
        self._log_fh = None
        self._terminated = False

    def start(self) -> None:
        if self.proc is not None:
            raise RuntimeError('child already started')

        cmd: list[str] = [
            sys.executable,
            '-u',
            'scripts/serving/_vllm_child.py',
            '--host',
            CHILD_HOST,
            '--port',
            str(self.child_port),
            '--model',
            self.model,
            *self.vllm_args,
        ]
        logger.info('starting child vLLM: %s', ' '.join(cmd))

        self.child_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = self.child_log_path.open('a', buffering=1)
        self._log_fh.write(
            f'\n=== child launch {time.strftime("%FT%TZ", time.gmtime())} ===\n'
        )
        self._log_fh.flush()

        self.proc = subprocess.Popen(  # noqa: S603 - args are controlled, not user text
            cmd,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=_set_pdeathsig,  # noqa: PLW1509 - intentional, see _set_pdeathsig
        )
        self.start_monotonic = time.monotonic()
        self.child_pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.child_pid_path.write_text(f'{self.proc.pid}\n')
        logger.info(
            'child pid=%d, log=%s, pidfile=%s',
            self.proc.pid,
            self.child_log_path,
            self.child_pid_path,
        )

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def elapsed_since_start(self) -> float:
        return time.monotonic() - self.start_monotonic

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            logger.info('child already exited with code %s', self.proc.returncode)
            self._close_log()
            return

        logger.info('sending SIGTERM to child pid=%d', self.proc.pid)
        try:
            self.proc.terminate()
        except ProcessLookupError:
            self._close_log()
            return

        try:
            self.proc.wait(timeout=SIGTERM_GRACE_SECONDS)
            logger.info('child exited gracefully with code %s', self.proc.returncode)
        except subprocess.TimeoutExpired:
            logger.warning(
                'child did not exit within %ds, sending SIGKILL', SIGTERM_GRACE_SECONDS
            )
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                logger.error('child pid=%d unresponsive to SIGKILL', self.proc.pid)

        self._close_log()
        try:
            self.child_pid_path.unlink()
        except FileNotFoundError:
            pass

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None


def _install_cleanup(child: ChildProcess, sup_pid_path: Path) -> None:
    """Install atexit + signal handlers that guarantee the child does not leak."""

    def _cleanup() -> None:
        child.terminate()
        try:
            sup_pid_path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(_cleanup)

    def _handler(signum: int, _frame: object) -> None:
        # Terminate the child directly from signal context rather than raising
        # SystemExit — raising from a C-call frame is deferred by CPython and
        # can strand the child holding GPU memory. os._exit skips atexit (we
        # already ran the cleanup), which is fine here because terminate() is
        # guarded by _terminated.
        logger.info('supervisor received signal %d, shutting down', signum)
        try:
            child.terminate()
        finally:
            try:
                sup_pid_path.unlink()
            except FileNotFoundError:
                pass
            os._exit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _build_app(child: ChildProcess) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Child is already started in main() before uvicorn binds.
        try:
            yield
        finally:
            # Uvicorn shutdown path also converges here on clean exits.
            child.terminate()

    app = FastAPI(lifespan=lifespan)

    child_base = f'http://{CHILD_HOST}:{child.child_port}'
    health_client = httpx.AsyncClient(
        base_url=child_base, timeout=HEALTH_PROXY_TIMEOUT_SECONDS
    )
    generate_client = httpx.AsyncClient(
        base_url=child_base, timeout=GENERATE_PROXY_TIMEOUT_SECONDS
    )

    @app.get('/health')
    async def health() -> Response:
        if not child.is_alive():
            return JSONResponse(
                {
                    'detail': 'child vLLM exited',
                    'returncode': child.proc.returncode if child.proc else None,
                    'elapsed_seconds': round(child.elapsed_since_start(), 2),
                },
                status_code=503,
            )
        try:
            r = await health_client.get('/health')
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            grace = child.elapsed_since_start() < HEALTH_STARTUP_GRACE_SECONDS
            return JSONResponse(
                {
                    'detail': f'child unreachable{" (startup grace)" if grace else ""}',
                    'error': str(exc),
                    'elapsed_seconds': round(child.elapsed_since_start(), 2),
                },
                status_code=503,
            )
        if r.status_code != 200:
            return JSONResponse(
                {'detail': f'child /health returned {r.status_code}'},
                status_code=503,
            )
        return Response(status_code=200)

    @app.post('/generate')
    async def generate(request: Request) -> Response:
        body = await request.json()
        if 'images' in body:
            return JSONResponse(
                {'detail': 'images not supported in Stage 1 (text-only Qwen3 path)'},
                status_code=400,
            )
        if 'prompt_ids' not in body:
            return JSONResponse(
                {'detail': "request body missing 'prompt_ids'"},
                status_code=400,
            )
        try:
            r = await generate_client.post('/generate', json=body)
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            return JSONResponse(
                {'detail': 'child /generate call failed', 'error': str(exc)},
                status_code=502,
            )
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get('content-type', 'application/json'),
        )

    @app.post('/reload_weights')
    async def reload_weights() -> Response:
        return JSONResponse(
            {'detail': 'Not implemented in Stage 1'},
            status_code=501,
        )

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Stage 1 vLLM supervisor (health + /generate proxy + reload stub).',
    )
    parser.add_argument(
        '--port', type=int, required=True, help='Supervisor listen port (ProRL-facing).'
    )
    parser.add_argument(
        '--child-port',
        type=int,
        default=None,
        help='Child vLLM port. Defaults to supervisor port + 1000.',
    )
    parser.add_argument(
        '--model', type=str, required=True, help='HF model path or repo id for vLLM.'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=DEFAULT_SUPERVISOR_HOST,
        help='Supervisor bind host (default: %(default)s).',
    )
    parser.add_argument(
        '--child-log',
        type=Path,
        default=None,
        help='Path to child vLLM stdout/stderr log. Defaults to /tmp/vllm-child-<supervisor_port>.log.',
    )
    parser.add_argument(
        '--sup-pid-file',
        type=Path,
        default=None,
        help='Supervisor PID file. Defaults to /tmp/vllm-sup-<supervisor_port>.pid.',
    )
    parser.add_argument(
        '--child-pid-file',
        type=Path,
        default=None,
        help='Child PID file. Defaults to /tmp/vllm-child-<supervisor_port>.pid.',
    )
    parser.add_argument(
        'vllm_args',
        nargs=argparse.REMAINDER,
        help='Extra arguments after `--` are forwarded verbatim to scripts/serving/_vllm_child.py.',
    )
    args = parser.parse_args(argv)
    # argparse.REMAINDER keeps the leading `--` if present; strip it.
    if args.vllm_args and args.vllm_args[0] == '--':
        args.vllm_args = args.vllm_args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get('VLLM_LAUNCHER_LOG_LEVEL', 'INFO'),
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    args = _parse_args(argv)

    child_port = args.child_port if args.child_port is not None else args.port + 1000
    if child_port == args.port:
        print('child port must differ from supervisor port', file=sys.stderr)
        return 2

    sup_pid_path = args.sup_pid_file or Path(f'/tmp/vllm-sup-{args.port}.pid')
    child_pid_path = args.child_pid_file or Path(f'/tmp/vllm-child-{args.port}.pid')
    child_log_path = args.child_log or Path(f'/tmp/vllm-child-{args.port}.log')

    sup_pid_path.parent.mkdir(parents=True, exist_ok=True)
    sup_pid_path.write_text(f'{os.getpid()}\n')

    child = ChildProcess(
        model=args.model,
        child_port=child_port,
        child_log_path=child_log_path,
        child_pid_path=child_pid_path,
        vllm_args=list(args.vllm_args),
    )
    _install_cleanup(child, sup_pid_path)
    child.start()

    logger.info(
        'supervisor :%d -> child :%d (pid=%d)',
        args.port,
        child_port,
        child.proc.pid if child.proc else -1,
    )
    app = _build_app(child)

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=os.environ.get('VLLM_LAUNCHER_UVICORN_LOG_LEVEL', 'info'),
            access_log=True,
        )
    finally:
        child.terminate()

    return 0


if __name__ == '__main__':
    sys.exit(main())
