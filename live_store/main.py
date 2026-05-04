"""Service entry point for ``live_store/server.py``.

Run via ``scripts/_internal/s0_5_live_store.sh`` (sources
``/home/ubuntu/.prorl_creds.env`` like every other launcher in this
repo).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from live_store.server import serve

DEFAULT_SOCKET = '/tmp/prorl_live_store.sock'
DEFAULT_MAX_SIZE = 256
DEFAULT_STALENESS_K = 4
DEFAULT_NO_PROGRESS_S = 1800.0
DEFAULT_MAX_WORKERS = 16


def main() -> int:
    p = argparse.ArgumentParser(prog='live_store')
    p.add_argument(
        '--socket',
        default=os.environ.get('LIVE_STORE_SOCKET', DEFAULT_SOCKET),
    )
    p.add_argument(
        '--max-size',
        type=int,
        default=int(os.environ.get('LIVE_STORE_MAX_SIZE', DEFAULT_MAX_SIZE)),
    )
    p.add_argument(
        '--staleness-cutoff-k',
        type=int,
        default=int(os.environ.get('LIVE_STORE_STALENESS_K', DEFAULT_STALENESS_K)),
    )
    p.add_argument(
        '--no-progress-timeout-s',
        type=float,
        default=float(
            os.environ.get('LIVE_STORE_NO_PROGRESS_S', DEFAULT_NO_PROGRESS_S)
        ),
    )
    p.add_argument(
        '--max-workers',
        type=int,
        default=int(os.environ.get('LIVE_STORE_MAX_WORKERS', DEFAULT_MAX_WORKERS)),
    )
    args = p.parse_args()

    logging.basicConfig(
        level=os.environ.get('LIVE_STORE_LOG_LEVEL', 'INFO'),
        format='%(asctime)s %(levelname)s live_store: %(message)s',
    )

    server = serve(
        socket_path=args.socket,
        max_size=args.max_size,
        staleness_cutoff_k=args.staleness_cutoff_k,
        no_progress_timeout_s=args.no_progress_timeout_s,
        max_workers=args.max_workers,
    )

    def _shutdown(_signum, _frame):
        logging.info('shutting down LiveStore (signal %s)', _signum)
        server.stop(grace=5.0)
        try:
            os.unlink(args.socket)
        except FileNotFoundError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.wait_for_termination()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
