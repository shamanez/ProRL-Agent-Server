"""Rollout-worker service entry point (slot 5.3).

Wires the four dependencies the worker needs:

1. **LiveStoreClient** — talks to the gRPC LiveStore over UDS.
   Push side; the trainer is the reader.
2. **PolicyVersionCache + FilePollingPolicySubscription** — the
   cleverest primitive. The poller watches the trainer's JSON
   manifest at 1 Hz and feeds strictly-fresher snapshots into the
   cache via atomic-ref-swap. Group dispatch reads the cache once
   per group; every row stamps the same snapshot (§3.2 + §3.5).
3. **ContinuousRolloutProducer** — daemon thread driving the agent
   loop. Lives in this process per §3.8 (data ownership). Reads
   ``policy_version`` from the cache, pushes survivors to the
   LiveStore.
4. **AsyncLLMServerManagerDAPO** — the verl-side DAPO async manager.
   Stays at its current path (heavy verl deps); imported here on
   demand. The eager-push closure (§3.7) goes through the
   :class:`LiveStoreClient.push_from_dataproto` surface — same shape
   as the legacy in-trainer wiring, only the process is different.

Validation flow is **not** wired here per the operating-principle
revision in the implementation plan.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from live_store import LiveStoreClient
from rollout_worker.manager import ContinuousRolloutProducer, StepCounter
from rollout_worker.policy_subscription import FilePollingPolicySubscription
from schemas.policy_version import PolicyVersionCache, PolicyVersionSnapshot

DEFAULT_SOCKET = '/tmp/prorl_live_store.sock'
DEFAULT_MANIFEST = '/tmp/prorl_policy_manifest.json'
DEFAULT_ARCHIVE_ROOT = '/tmp/prorl_replay_archive'
DEFAULT_DEAD_LETTER = '/tmp/prorl_replay_archive_deadletter.jsonl'


def main() -> int:
    p = argparse.ArgumentParser(prog='rollout_worker')
    p.add_argument(
        '--live-store-socket',
        default=os.environ.get('LIVE_STORE_SOCKET', DEFAULT_SOCKET),
    )
    p.add_argument(
        '--policy-manifest-path',
        default=os.environ.get('POLICY_MANIFEST_PATH', DEFAULT_MANIFEST),
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
        '--policy-poll-interval-s',
        type=float,
        default=float(os.environ.get('POLICY_POLL_INTERVAL_S', 1.0)),
    )
    p.add_argument(
        '--archive-root',
        default=os.environ.get('REPLAY_ARCHIVE_ROOT', DEFAULT_ARCHIVE_ROOT),
    )
    p.add_argument(
        '--archive-dead-letter',
        default=os.environ.get('REPLAY_ARCHIVE_DEAD_LETTER', DEFAULT_DEAD_LETTER),
    )
    p.add_argument(
        '--archive-disabled',
        action='store_true',
        default=bool(int(os.environ.get('REPLAY_ARCHIVE_DISABLED', '0'))),
        help='Disable the archive tee. Hot path is unaffected.',
    )
    args = p.parse_args()

    logging.basicConfig(
        level=os.environ.get('ROLLOUT_WORKER_LOG_LEVEL', 'INFO'),
        format='%(asctime)s %(levelname)s rollout_worker: %(message)s',
    )
    logger = logging.getLogger(__name__)

    # 1. LiveStoreClient
    live_store = LiveStoreClient(
        socket_path=args.live_store_socket,
        policy_id=args.policy_id,
        environment_id=args.environment_id,
    )

    # 2. PolicyVersionCache + subscription. Single-writer: only the
    # poller thread updates. Many readers: every group dispatch.
    cache = PolicyVersionCache(PolicyVersionSnapshot.bootstrap(args.policy_id))
    subscription = FilePollingPolicySubscription(
        cache=cache,
        manifest_path=args.policy_manifest_path,
        poll_interval_s=args.policy_poll_interval_s,
        on_update=lambda snap: live_store.notify_policy_version(
            snap.version, snap.adapter_uri
        ),
    )
    subscription.start()

    # 3. ReplayArchive tee (S3). Async-fire-and-forget; archive
    # availability never stalls the producer. Per §7 the tee is
    # producer-side, NOT live-store-side, so it runs even on
    # filtered (zero-variance) groups.
    archive_writer: ReplayArchiveWriter | None = None
    if not args.archive_disabled:
        archive_server = ArchiveServer(args.archive_root)
        archive_writer = ReplayArchiveWriter(
            server=archive_server,
            dead_letter_path=args.archive_dead_letter,
        )
        archive_writer.start()
        logger.info(
            'replay archive tee enabled root=%s dead_letter=%s',
            args.archive_root,
            args.archive_dead_letter,
        )
    else:
        logger.info('replay archive tee disabled (REPLAY_ARCHIVE_DISABLED=1)')

    # 4. The producer + DAPO manager are wired by the verl-side
    # bootstrap (heavy verl/openhands deps). They consume:
    #   * ``live_store`` (push survivors)
    #   * ``cache``     (read PolicyVersionSnapshot per group dispatch)
    #   * ``archive_writer`` (tee EpisodeRecord at episode close, even
    #     when the live-path filter drops the group)
    step_counter = StepCounter(initial=0)

    def _shutdown(_signum, _frame):
        logger.info('shutting down rollout_worker (signal %s)', _signum)
        subscription.stop(timeout=2.0)
        if archive_writer is not None:
            archive_writer.stop(timeout=5.0)
        live_store.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Idle loop: surface the cleverest snapshot + archive stats in
    # INFO logs every 30s so the post-S4 checklist items 11
    # (snapshot read) and 13 (archive count parity) have a heartbeat
    # trace even before the verl-side producer bootstrap kicks in.
    import time  # noqa: PLC0415

    while True:
        snap = cache.snapshot()
        archive_stats = archive_writer.stats() if archive_writer else {}
        logger.info(
            'rollout_worker idle: policy_id=%s version=%d adapter_uri=%s '
            'live_store_size=%d archive=%s',
            snap.policy_id,
            snap.version,
            snap.adapter_uri,
            live_store.num_groups(),
            archive_stats,
        )
        time.sleep(30.0)

    # NOTE: ``ContinuousRolloutProducer`` and ``StepCounter`` are
    # exported here for the verl-bootstrapped run path. They
    # participate in the S2 / S3 verification only when the bootstrap
    # script runs.
    _ = ContinuousRolloutProducer
    _ = step_counter


if __name__ == '__main__':
    raise SystemExit(main())
