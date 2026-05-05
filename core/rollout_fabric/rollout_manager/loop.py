"""RolloutManager production loop.

**Zero VERL / OpenHands imports (BC-13).** This module drives the entire
rollout pipeline using only:
- ``rollout_manager.prorl_client`` (HTTP to ProRL)
- ``rollout_manager.episode_builder`` (TrainingSample construction)
- ``rollout_manager.dataloader`` (ParquetDataLoader)
- ``live_store.client.LiveStoreClient`` (gRPC push)
- ``schemas.policy_version.PolicyVersionCache`` (atomic snapshot)

Group-policy consistency guarantee (BC-0 + §3.4):
    snap = cache.snapshot()          # read ONCE before all N siblings
    episodes = [run_episode(inst, snap.version) for inst in group_batch]
    samples = [build_sample(ep, snap) for ep in episodes]
    # All samples.behavior_policy_version == snap.version

The vLLM pinning swap protocol keeps each episode's turns locked to
``snap.version`` for the full episode — a publish mid-group does not
affect in-flight requests.
"""

from __future__ import annotations

import json as _json
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

from rollout_fabric.rollout_manager.episode_builder import (
    build_group,
    build_training_sample,
    is_zero_variance_group,
)
from rollout_fabric.rollout_manager.prorl_client import ProRLClient, ProRLEpisodeResult
from rollout_fabric.schemas.episode_record import TrustLevel

if TYPE_CHECKING:
    from rollout_fabric.replay_archive.writer import ReplayArchiveWriter
    from rollout_fabric.rollout_manager.dataloader import ParquetDataLoader
    from rollout_fabric.schemas.policy_version import PolicyVersionCache

logger = logging.getLogger(__name__)


class RolloutManagerLoop:
    """Main production loop.

    Parameters
    ----------
    prorl_client:
        HTTP client for the EnvironmentProvider (ProRL :8006).
    live_store_client:
        gRPC client for pushing groups to LiveStore.
    policy_cache:
        Atomic-snapshot cache populated by the subscription thread.
    dataloader:
        Parquet dataloader owned by the worker (§3.8).
    group_size:
        Number of sibling episodes per GRPO/DAPO group (``n`` in
        ``compute_advantage``). All siblings run with the same snapshot.
    created_at_step_fn:
        Callable returning the trainer's current step (via StepCounter).
        Used to stamp ``created_at_step`` on every sample (BC-8).
    archive_writer:
        Optional archive tee. Receives ALL episodes before filtering
        (BC-12 — archive sees pre-filter; LiveStore sees post-filter).
    environment_id, environment_version, verifier_version, split:
        Provenance fields stamped on every TrainingSample.
    filter_zero_variance:
        Drop groups where all rewards are identical (§3.7 zero-variance
        drop). Filtered groups are still teed to the archive.
    poll_interval_s:
        Sleep when the LiveStore is at capacity (producer backpressure).
    """

    def __init__(
        self,
        *,
        prorl_client: ProRLClient,
        live_store_client: Any,  # LiveStoreClient
        policy_cache: PolicyVersionCache,
        dataloader: ParquetDataLoader,
        group_size: int = 16,
        num_parallel_groups: int = 1,
        created_at_step_fn: Any = None,
        archive_writer: ReplayArchiveWriter | None = None,
        environment_id: str = 'prorl',
        environment_version: str = '',
        verifier_version: str = '',
        split: str = 'train',
        filter_zero_variance: bool = True,
        poll_interval_s: float = 0.05,
    ) -> None:
        self._prorl = prorl_client
        self._store = live_store_client
        self._cache = policy_cache
        self._dataloader = dataloader
        self._group_size = int(group_size)
        self._num_parallel_groups = max(1, int(num_parallel_groups))
        self._step_fn = created_at_step_fn or (lambda: 0)
        self._archive = archive_writer
        self._env_id = environment_id
        self._env_version = environment_version
        self._verifier_version = verifier_version
        self._split = split
        self._filter_zero_variance = filter_zero_variance
        self._poll_interval_s = float(poll_interval_s)

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._exception: BaseException | None = None
        self._thread: threading.Thread | None = None
        # Protects next(self._dataloader) across parallel group workers.
        self._dl_lock = threading.Lock()

        # Counters
        self._groups_pushed: int = 0
        self._groups_filtered: int = 0
        self._episodes_total: int = 0

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError('RolloutManagerLoop already started')
        self._stop_event.clear()
        self._pause_event.set()
        self._thread = threading.Thread(
            target=self._run, name='RolloutManagerLoop', daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> bool:
        self._stop_event.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning('RolloutManagerLoop did not exit within timeout')
                return False
        self._thread = None
        if self._exception is not None:
            exc, self._exception = self._exception, None
            raise exc
        return True

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def check_error(self) -> None:
        if self._exception is not None:
            exc, self._exception = self._exception, None
            raise exc

    # ---- main loop --------------------------------------------------------

    def _run(self) -> None:
        try:
            if self._num_parallel_groups <= 1:
                while not self._stop_event.is_set():
                    self._pause_event.wait()
                    if self._stop_event.is_set():
                        break
                    self._run_one_group()
            else:
                # Fan out: num_parallel_groups concurrent group workers, each
                # running their own while-loop. The dataloader is protected by
                # _dl_lock; all other shared state (gRPC clients, counters) is
                # safe under Python's GIL for simple int increments.
                def _worker() -> None:
                    while not self._stop_event.is_set():
                        self._pause_event.wait()
                        if self._stop_event.is_set():
                            break
                        self._run_one_group()

                logger.info(
                    'RolloutManagerLoop starting %d parallel group workers '
                    '(group_size=%d → %d concurrent episodes)',
                    self._num_parallel_groups,
                    self._group_size,
                    self._num_parallel_groups * self._group_size,
                )
                threads = [
                    threading.Thread(
                        target=_worker, name=f'RolloutGroup-{i}', daemon=True
                    )
                    for i in range(self._num_parallel_groups)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                if self._exception is not None:
                    raise self._exception
        except BaseException as exc:  # noqa: BLE001
            logger.exception('RolloutManagerLoop crashed')
            self._exception = exc

    def _run_one_group(self) -> None:
        """Read one task, dispatch group_size episodes, push survivors."""
        # Step 1: read one task instance from the dataloader.
        # The parquet row format is: {prompt, data_source, ability, instance: {...SWE fields}}
        # ProRL expects the INNER instance dict (with instance_id, FAIL_TO_PASS, etc.) plus
        # data_source at the top level for handler routing.
        # If the row has a nested 'instance' key, flatten it here.
        with self._dl_lock:
            row = next(self._dataloader)
        if isinstance(row.get('instance'), dict):
            # Standard SkyRL parquet format: row['instance'] holds the SWE-bench fields.
            instance = dict(row['instance'])
            # Propagate data_source so ProRL can route to the right handler.
            if 'data_source' in row and 'data_source' not in instance:
                instance['data_source'] = row['data_source']
        else:
            instance = row
        task_id = str(instance.get('instance_id', instance.get('trajectory_id', '')))

        # Step 2: take ONE policy snapshot for the whole group (BC-0 / §3.2).
        # All N siblings are dispatched with this same policy_version so no
        # trajectory spans two policies.
        snap = self._cache.snapshot()
        created_at_step = self._step_fn()

        # Step 3: run group_size episodes in parallel — same policy version for all.
        # BC-0: snap captured once above; all futures use the same snap.version.
        group_uid = str(uuid.uuid4())
        raw_episodes: list[ProRLEpisodeResult] = []
        iter_start = time.monotonic()

        ep_lock = threading.Lock()

        def _run_one_episode() -> None:
            if self._stop_event.is_set():
                return
            try:
                ep = self._prorl.run_episode(instance, snap.version)
                with ep_lock:
                    raw_episodes.append(ep)
                    self._episodes_total += 1
            except Exception:  # noqa: BLE001
                logger.exception(
                    'ProRL episode failed task_id=%s pv=%d', task_id, snap.version
                )

        ep_threads = [
            threading.Thread(target=_run_one_episode, daemon=True)
            for _ in range(self._group_size)
        ]
        for t in ep_threads:
            t.start()
        for t in ep_threads:
            t.join()

        if not raw_episodes:
            return

        # Step 4: build TrainingSample objects (no VERL, no DataProto).
        samples_pre_bind = [
            build_training_sample(
                ep,
                snap,
                created_at_step=created_at_step,
                task_id=task_id,
                split=self._split,
                environment_id=self._env_id,
                environment_version=self._env_version,
                verifier_version=self._verifier_version,
            )
            for ep in raw_episodes
        ]
        # Drop samples with no response tokens — these cannot be trained on
        # (response_mask would be all-zeros → AssertionError in actor update).
        valid_pre_bind = [s for s in samples_pre_bind if len(s.response_token_ids) > 0]
        if len(valid_pre_bind) < len(samples_pre_bind):
            logger.warning(
                'Dropped %d/%d samples with empty response_token_ids '
                'group_uid=%s task_id=%s',
                len(samples_pre_bind) - len(valid_pre_bind),
                len(samples_pre_bind),
                group_uid,
                task_id,
            )
        if not valid_pre_bind:
            logger.warning(
                'All samples in group have empty responses — skipping push '
                'group_uid=%s task_id=%s',
                group_uid,
                task_id,
            )
            return

        # Bind all siblings to the shared group_uid (§3.2 group integrity).
        group_samples = build_group(valid_pre_bind, group_uid)

        # Step 5: tee ALL episodes to ReplayArchive BEFORE filter (BC-12).
        if self._archive is not None:
            self._tee_to_archive(raw_episodes, group_samples, snap, created_at_step)

        # Step 6: apply zero-variance filter (§3.7).
        if self._filter_zero_variance and is_zero_variance_group(group_samples):
            self._groups_filtered += 1
            logger.debug(
                'zero-variance group filtered group_uid=%s task_id=%s pv=%d',
                group_uid,
                task_id,
                snap.version,
            )
            return

        # Step 7: push to LiveStore.
        try:
            self._store.push_group(group_samples)
            self._groups_pushed += 1
        except Exception:  # noqa: BLE001
            logger.exception('LiveStore push_group failed group_uid=%s', group_uid)
            return

        logger.info(
            'PRODUCER_ITER %s',
            _json.dumps(
                {
                    'event': 'producer_iter',
                    'wall_s': round(time.monotonic() - iter_start, 3),
                    'policy_id': snap.policy_id,
                    'policy_version': snap.version,
                    'created_at_step': created_at_step,
                    'group_uid': group_uid,
                    'group_size': len(group_samples),
                    'groups_pushed': self._groups_pushed,
                    'groups_filtered': self._groups_filtered,
                }
            ),
        )

    def _tee_to_archive(
        self,
        raw_episodes: list[ProRLEpisodeResult],
        group_samples: list[Any],
        snap: Any,
        created_at_step: int,
    ) -> None:
        """Async-fire-and-forget tee to ReplayArchive (BC-12)."""
        from datetime import datetime, timezone  # noqa: PLC0415

        from rollout_fabric.schemas.episode_record import (  # noqa: PLC0415
            EpisodeRecord,
            Event,
        )

        now = datetime.now(timezone.utc)
        for ep, sample in zip(raw_episodes, group_samples):
            events = tuple(
                Event(
                    turn_index=i,
                    kind='agent_turn'
                    if m.get('role') == 'assistant'
                    else 'tool_result',
                    response_token_ids=tuple(m.get('token_ids') or []),
                    response_loss_mask=None,
                    behavior_log_probs=tuple(m.get('logprobs') or []) or None,
                )
                for i, m in enumerate(ep.messages)
            )
            record = EpisodeRecord(
                episode_uid=sample.episode_uid,
                task_id=sample.task_id,
                split=self._split,
                environment_provider='prorl',
                environment_id=self._env_id,
                environment_version=self._env_version,
                verifier_version=self._verifier_version,
                reward_spec_id='prorl_default',
                policy_id=snap.policy_id,
                policy_version=int(snap.version),
                base_model_id='',
                tokenizer_id='',
                inference_backend='vllm-pinning',
                sampling_params={},
                created_at_step=created_at_step,
                started_at=now,
                finished_at=now,
                termination_reason='done' if ep.finish else 'truncated',
                events=events,
                total_reward=ep.reward,
                prompt_token_ids=sample.prompt_token_ids,
                response_token_ids=sample.response_token_ids,
                response_loss_mask=sample.response_loss_mask,
                behavior_log_probs=sample.behavior_log_probs,
                trust_level=TrustLevel.OWN_FABRIC,
            )
            try:
                self._archive.submit(record)
            except Exception:  # noqa: BLE001
                logger.warning('archive tee failed episode_uid=%s', sample.episode_uid)

    def stats(self) -> dict[str, int]:
        return {
            'groups_pushed': self._groups_pushed,
            'groups_filtered': self._groups_filtered,
            'episodes_total': self._episodes_total,
        }
