"""RolloutManager service entry point (slot 5.3).

Startup sequence for this process:
  1. Connect LiveStoreClient (gRPC to live store)
  2. Start PolicyVersionCache + FilePollingPolicySubscription
  3. Create ParquetDataLoader (owns the dataset — §3.8)
  4. Start ReplayArchiveWriter (async tee — S3)
  5. Create ProRLClient (HTTP to EnvironmentProvider)
  6. Start RolloutManagerLoop (the production loop)

**Zero VERL / OpenHands imports (BC-13).**
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

from rollout_fabric.live_store import LiveStoreClient
from rollout_fabric.rollout_manager.dataloader import ParquetDataLoader
from rollout_fabric.rollout_manager.loop import RolloutManagerLoop
from rollout_fabric.rollout_manager.policy_subscription import (
    FilePollingPolicySubscription,
)
from rollout_fabric.rollout_manager.prorl_client import ProRLClient
from rollout_fabric.schemas.policy_version import (
    PolicyVersionCache,
    PolicyVersionSnapshot,
)

DEFAULT_SOCKET = '/tmp/prorl_live_store.sock'
DEFAULT_MANIFEST = '/tmp/prorl_policy_manifest.json'
DEFAULT_ARCHIVE_ROOT = '/tmp/prorl_replay_archive'
DEFAULT_ARCHIVE_DL = '/tmp/prorl_replay_archive_deadletter.jsonl'


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog='rollout_manager')
    p.add_argument(
        '--live-store-socket',
        default=os.environ.get('LIVE_STORE_SOCKET', DEFAULT_SOCKET),
    )
    p.add_argument(
        '--prorl-url',
        default=os.environ.get('PRORL_URL', 'http://localhost:8006'),
        help='Base URL of the EnvironmentProvider (ProRL).',
    )
    p.add_argument(
        '--policy-id',
        default=os.environ.get('POLICY_ID', 'qwen3-4b-skyrl'),
    )
    p.add_argument(
        '--environment-id',
        default=os.environ.get('ENVIRONMENT_ID', 'prorl_default'),
    )
    p.add_argument(
        '--data-files',
        nargs='+',
        default=os.environ.get('DATA_FILES', '').split() or None,
        help='Space-separated parquet file paths (BC-14 — worker owns dataset).',
    )
    p.add_argument(
        '--group-size',
        type=int,
        default=int(os.environ.get('GROUP_SIZE', '16')),
        help='GRPO/DAPO group size (n siblings per task).',
    )
    p.add_argument(
        '--policy-manifest-path',
        default=os.environ.get('POLICY_MANIFEST_PATH', DEFAULT_MANIFEST),
    )
    p.add_argument(
        '--archive-root',
        default=os.environ.get('REPLAY_ARCHIVE_ROOT', DEFAULT_ARCHIVE_ROOT),
    )
    p.add_argument(
        '--archive-dead-letter',
        default=os.environ.get('REPLAY_ARCHIVE_DEAD_LETTER', DEFAULT_ARCHIVE_DL),
    )
    p.add_argument(
        '--archive-disabled',
        action='store_true',
        default=bool(int(os.environ.get('REPLAY_ARCHIVE_DISABLED', '0'))),
    )
    p.add_argument(
        '--filter-zero-variance',
        action='store_true',
        default=bool(int(os.environ.get('FILTER_ZERO_VARIANCE', '0'))),
        help='Drop zero-variance groups (§3.7). Default OFF — enable for production.',
    )
    p.add_argument(
        '--num-parallel-groups',
        type=int,
        default=int(os.environ.get('NUM_PARALLEL_GROUPS', '1')),
        help='Number of groups to dispatch concurrently (fills ProRL worker pool).',
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=os.environ.get('ROLLOUT_WORKER_LOG_LEVEL', 'INFO'),
        format='%(asctime)s %(levelname)s rollout_manager: %(message)s',
    )
    logger = logging.getLogger(__name__)

    if not args.data_files:
        logger.error('--data-files required (BC-14: worker owns the dataset)')
        return 1

    # 1. LiveStoreClient
    live_store = LiveStoreClient(
        socket_path=args.live_store_socket,
        policy_id=args.policy_id,
        environment_id=args.environment_id,
    )

    # 2. PolicyVersionCache + subscription
    cache = PolicyVersionCache(PolicyVersionSnapshot.bootstrap(args.policy_id))
    subscription = FilePollingPolicySubscription(
        cache=cache,
        manifest_path=args.policy_manifest_path,
        poll_interval_s=1.0,
        on_update=lambda snap: live_store.notify_policy_version(
            snap.version, snap.adapter_uri
        ),
    )
    subscription.start()

    # 3. ParquetDataLoader — worker owns the dataset (BC-14 / §3.8)
    dataloader = ParquetDataLoader(data_files=args.data_files)

    # 4. ReplayArchive tee (S3)
    archive_writer = None
    if not args.archive_disabled:
        from rollout_fabric.replay_archive.server import ArchiveServer  # noqa: PLC0415
        from rollout_fabric.replay_archive.writer import (
            ReplayArchiveWriter,  # noqa: PLC0415
        )

        archive_writer = ReplayArchiveWriter(
            server=ArchiveServer(args.archive_root),
            dead_letter_path=args.archive_dead_letter,
        )
        archive_writer.start()
        logger.info('replay archive tee enabled root=%s', args.archive_root)

    # 5. ProRLClient (HTTP, no OpenHands imports — BC-13)
    prorl_client = ProRLClient(base_url=args.prorl_url)

    # 6. RolloutManagerLoop
    loop = RolloutManagerLoop(
        prorl_client=prorl_client,
        live_store_client=live_store,
        policy_cache=cache,
        dataloader=dataloader,
        group_size=args.group_size,
        num_parallel_groups=args.num_parallel_groups,
        created_at_step_fn=lambda: 0,  # updated via StepCounter RPC at S2+
        archive_writer=archive_writer,
        environment_id=args.environment_id,
        filter_zero_variance=args.filter_zero_variance,
    )
    loop.start()

    def _shutdown(_signum, _frame):
        logger.info('shutting down rollout_manager (signal %s)', _signum)
        loop.stop(timeout=30.0)
        subscription.stop(timeout=2.0)
        if archive_writer is not None:
            archive_writer.stop(timeout=5.0)
        live_store.close()
        prorl_client.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        loop.check_error()
        snap = cache.snapshot()
        logger.info(
            'rollout_manager heartbeat: policy_id=%s version=%d store_groups=%d stats=%s',
            snap.policy_id,
            snap.version,
            live_store.num_groups(),
            loop.stats(),
        )
        time.sleep(30.0)


if __name__ == '__main__':
    raise SystemExit(main())
