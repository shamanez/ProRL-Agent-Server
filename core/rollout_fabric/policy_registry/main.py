"""Service entry point for ``policy_registry/server.py``."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from rollout_fabric.policy_registry.server import serve

DEFAULT_SOCKET = '/tmp/prorl_policy_registry.sock'
DEFAULT_DB = '/tmp/prorl_policy_registry.db'


def main() -> int:
    p = argparse.ArgumentParser(prog='policy_registry')
    p.add_argument(
        '--socket', default=os.environ.get('POLICY_REGISTRY_SOCKET', DEFAULT_SOCKET)
    )
    p.add_argument('--db', default=os.environ.get('POLICY_REGISTRY_DB', DEFAULT_DB))
    p.add_argument(
        '--pool-endpoints',
        default=os.environ.get('POOL_ENDPOINTS', ''),
        help='Comma-separated http://... endpoints for the vLLM pool.',
    )
    args = p.parse_args()

    logging.basicConfig(
        level=os.environ.get('POLICY_REGISTRY_LOG_LEVEL', 'INFO'),
        format='%(asctime)s %(levelname)s policy_registry: %(message)s',
    )

    endpoints = [e.strip() for e in args.pool_endpoints.split(',') if e.strip()]
    if not endpoints:
        raise SystemExit(
            'POOL_ENDPOINTS env var (or --pool-endpoints) is required; '
            'comma-separated http://... URLs of every vLLM child.'
        )

    server = serve(
        socket_path=args.socket,
        db_path=args.db,
        pool_endpoints=endpoints,
    )

    def _shutdown(_signum, _frame):
        logging.info('shutting down PolicyRegistry (signal %s)', _signum)
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
