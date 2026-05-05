"""Rollout-worker producer loop. Lifted from
``trainer_integration/verl/verl_custom/replay/continuous_producer.py``.

Surgical changes from the legacy file:

1. Store reference is the gRPC :class:`LiveStoreClient`, not the
   in-process ``TrajectoryStore``. The surface (``push_from_dataproto``,
   ``num_groups``) is identical so the loop body is unchanged.
2. ``policy_version`` is read via the cleverest primitive
   (``schemas.policy_version.PolicyVersionCache``): one atomic
   reference load that yields a consistent immutable snapshot.
   Replaces the legacy ``getattr(self._rollout_manager, 'policy_version', 0)``
   GIL-atomic int read.
3. Per-iteration log line tags ``policy_id`` from the snapshot,
   making the post-S4 checklist item 11 ("policy_version snapshot read
   once-per-group dispatch") visible in worker logs without extra
   instrumentation.

DAPO eager-push seam (§3.7) is preserved verbatim — the producer skips
its terminal ``push_from_dataproto`` whenever
``meta_info['eager_pushed_all'] == True``.
"""

from __future__ import annotations

import json as _json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from rollout_fabric.schemas.policy_version import PolicyVersionCache

logger = logging.getLogger(__name__)


class StepCounter:
    """Lock-guarded integer box; trainer writes, worker reads.

    The trainer side is in a separate process at S2+, so this box is
    populated by an RPC (``set_step``) that the trainer fires on each
    iteration. A lock is cheap (one instruction per tick) and eliminates
    cross-thread integer-store reasoning.
    """

    def __init__(self, initial: int = 0) -> None:
        self._value = int(initial)
        self._lock = threading.Lock()

    def set(self, value: int) -> None:
        with self._lock:
            self._value = int(value)

    def get(self) -> int:
        with self._lock:
            return self._value


class ContinuousRolloutProducer:
    """Daemon-thread producer loop in the worker process.

    Per-trajectory and per-group policy consistency contract
    --------------------------------------------------------
    The producer reads :meth:`PolicyVersionCache.snapshot` exactly once
    at end-of-call and uses ``snapshot.version`` only as a *fallback*
    for the per-row ``behavior_policy_version`` stamp. The trustworthy
    stamp lives on each row's ``instance['policy_version']``, set by
    :meth:`AsyncLLMServerManager.DataProto2Messages` at expansion time
    (synchronous, atomic per call), and consumed by
    :meth:`LiveStoreClient.push_from_dataproto`. Reading the cache's
    *current* version at end-of-call would only matter for rows missing
    a per-row stamp — and even then, the snapshot read is itself a
    single GIL-atomic LOAD_ATTR; no torn ``(version, adapter_uri)``
    pair is observable.

    Path-versioned URL routing on the OpenHands side (``/v{N}/generate``)
    ensures every turn of a trajectory — and every sibling of a GRPO
    group — actually runs against the stamped version, regardless of
    whether ``swap_protocol`` is ``pinning`` (multi-tenant; default) or
    ``quiesce`` (drain-and-swap; fallback).

    Parameters
    ----------
    rollout_manager:
        The worker's ``async_rollout_manager`` (talks to ProRL :8006).
        Used for :meth:`wake_up` / :meth:`sleep` only; the policy
        version no longer comes from this object — see ``policy_cache``.
    generate_fn:
        Callable that produces one batch of rollouts (a ``DataProto``).
    store:
        :class:`LiveStoreClient` to push into.
    step_counter:
        :class:`StepCounter` populated by the trainer's ``set_step`` RPC.
    policy_cache:
        :class:`PolicyVersionCache`. The cache is populated by
        :class:`FilePollingPolicySubscription` (S2) or by the gRPC
        streaming subscription at S4. Reads here are lock-free.
    prompts_iter_factory:
        Zero-arg factory returning a fresh prompts iterator. ``None``
        for DAPO, which pulls prompts internally from its own dataloader.
    poll_interval_s:
        Sleep when the LiveStore is at ``max_size`` (gives the trainer
        a chance to drain). Default 0.01s.
    """

    def __init__(
        self,
        *,
        rollout_manager: Any,
        generate_fn: Callable[..., Any],
        store: Any,
        step_counter: StepCounter,
        policy_cache: PolicyVersionCache,
        prompts_iter_factory: Callable[[], Iterator[Any]] | None = None,
        poll_interval_s: float = 0.01,
    ) -> None:
        self._rollout_manager = rollout_manager
        self._generate_fn = generate_fn
        self._store = store
        self._step_counter = step_counter
        self._policy_cache = policy_cache
        self._prompts_iter_factory = prompts_iter_factory
        self._poll_interval_s = float(poll_interval_s)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._exception: BaseException | None = None
        self._pause_event = threading.Event()  # set => running; cleared => paused
        self._pause_event.set()

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError('ContinuousRolloutProducer already started')
        if hasattr(self._rollout_manager, 'wake_up'):
            self._rollout_manager.wake_up()
        self._stop_event.clear()
        self._pause_event.set()
        self._thread = threading.Thread(
            target=self._run, name='ContinuousRolloutProducer', daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> bool:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            if timeout is None:
                thread.join()
            else:
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(
                        'ContinuousRolloutProducer did not exit within %.1fs',
                        timeout,
                    )
                    return False
        self._thread = None
        if hasattr(self._rollout_manager, 'sleep'):
            try:
                self._rollout_manager.sleep()
            except Exception:  # noqa: BLE001
                logger.exception('rollout_manager.sleep() failed during stop()')
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            raise exc
        return True

    def pause(self) -> None:
        """Signal the loop to pause at the next iteration boundary."""
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def check_background_error(self) -> None:
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            raise exc

    # ---- worker -------------------------------------------------------------

    def _run(self) -> None:
        prompts_iter = (
            self._prompts_iter_factory()
            if self._prompts_iter_factory is not None
            else None
        )
        store_full_idles = 0
        try:
            while not self._stop_event.is_set():
                # Honor pause without burning CPU.
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                if self._store_full():
                    store_full_idles += 1
                    if self._stop_event.wait(timeout=self._poll_interval_s):
                        break
                    continue

                iter_start = time.monotonic()
                if prompts_iter is not None:
                    try:
                        gen_batch = next(prompts_iter)
                    except StopIteration:
                        prompts_iter = self._prompts_iter_factory()  # type: ignore[misc]
                        continue
                    batch = self._generate_fn(gen_batch)
                else:
                    batch = self._generate_fn()

                # Cleverest primitive: one LOAD_ATTR yields an immutable
                # snapshot. The whole batch is bound to this snapshot for
                # the fallback stamping path; per-row stamping (already
                # done by AsyncLLMServerManager.DataProto2Messages) takes
                # precedence. Reading once here keeps the §3.5 contract
                # identical to today.
                snap = self._policy_cache.snapshot()
                policy_version = int(snap.version)
                current_step = self._step_counter.get()

                # §3.7 eager-push seam: the DAPO manager already pushed
                # every survivor mid-call; do not double-push here.
                eager_pushed_all = False
                try:
                    eager_pushed_all = bool(
                        batch.meta_info.get('eager_pushed_all', False)
                    )
                except AttributeError:
                    eager_pushed_all = False
                if not eager_pushed_all:
                    self._store.push_from_dataproto(
                        batch,
                        behavior_policy_version=policy_version,
                        current_step=current_step,
                    )
                logger.info(
                    'PRODUCER_ITER %s',
                    _json.dumps(
                        {
                            'event': 'producer_iter',
                            'wall_s': round(time.monotonic() - iter_start, 3),
                            'store_full_idles': store_full_idles,
                            'policy_id': snap.policy_id,
                            'policy_version': policy_version,
                            'adapter_uri': snap.adapter_uri,
                            'current_step': current_step,
                            'store_num_groups': int(self._store.num_groups()),
                            'eager_pushed_all': eager_pushed_all,
                        }
                    ),
                )
                store_full_idles = 0
        except BaseException as exc:  # noqa: BLE001
            logger.exception('ContinuousRolloutProducer worker crashed')
            self._exception = exc

    def _store_full(self) -> bool:
        try:
            return self._store.num_groups() >= self._store_max_size()
        except Exception:  # noqa: BLE001
            logger.warning(
                'store.num_groups() raised inside _store_full; continuing to '
                'generate (FIFO eviction will bound memory)',
                exc_info=True,
            )
            return False

    def _store_max_size(self) -> int:
        # The gRPC client doesn't expose ``_max_size``; we never
        # block-on-full at the producer side past S1 because the server
        # FIFO-evicts. Returning a large sentinel keeps the legacy
        # branch a no-op.
        return int(getattr(self._store, '_max_size', 1 << 31))

    def _is_alive_for_tests(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ------------------------------------------------------------------- helpers


def wait_until(
    predicate: Callable[[], bool], *, timeout: float, interval: float = 0.01
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
