# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Cut 4 — continuous rollout producer for fully-async agentic RL.

The trainer's ``fit()`` loop used to call ``generate_sequences`` inline,
blocking on the tail of the slowest rollout in the batch. Phase 2 moves
rollout generation into a daemon thread that pulls prompts from the
train dataloader in a loop and pushes each completed batch into the
:class:`TrajectoryStore`. The trainer samples from the store on its own
cadence — clock separation between rollout and training.

Scope kept deliberately small:

* No ``asyncio`` task group; ``generate_sequences`` already calls
  ``asyncio.run()`` internally (``async_server.py:1548``), which is
  compatible with running inside a ``threading.Thread``.
* No ``multiprocessing`` — the store is in-process, the producer stays
  in the same Python interpreter.
* ``wake_up()`` / ``sleep()`` lifecycle: wake once at :meth:`start`,
  sleep once at :meth:`stop`. See
  ``plans-n-solutions/stages/full_async.md`` §2 decision table.

The ``current_step`` marker is read back from the trainer via a shared
:class:`StepCounter` (thin ``threading.Lock``-guarded int box) so stored
records carry the right ``created_at_step`` without racy cross-thread
reads of ``trainer.global_steps``. ``policy_version`` is read unlocked
from the rollout manager (single int, atomic under the CPython GIL —
gotcha §20 in handsoff.md).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)


class StepCounter:
    """Lock-guarded integer box shared between trainer and producer.

    The trainer writes ``global_steps`` on each iteration; the producer
    reads it at push time so stored records carry a consistent
    ``created_at_step``. A lock is cheap (one instruction per tick) and
    removes the need to reason about CPython GIL semantics on a
    user-facing invariant.
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
    """Daemon-thread rollout generator for Phase 2 fully-async training.

    Parameters
    ----------
    rollout_manager:
        The trainer's ``async_rollout_manager``. Used for
        :meth:`wake_up`/:meth:`sleep` and ``policy_version`` reads.
    generate_fn:
        Callable that produces one batch of rollouts (a ``DataProto``).
        For plain GRPO: ``partial(rollout_manager.generate_sequences)``
        with the trainer's ``gen_batch`` iterator inside. For DAPO:
        ``rollout_manager.generate_sequences_dapo`` (DAPO pulls prompts
        internally; the producer just invokes it repeatedly).
    store:
        :class:`TrajectoryStore` to push into.
    step_counter:
        :class:`StepCounter` the trainer writes on each loop iteration.
    prompts_iter_factory:
        Zero-arg factory that returns a fresh infinite iterator of
        per-call ``gen_batch`` DataProtos. Not used in DAPO mode (pass
        ``None``).
    poll_interval_s:
        Sleep between iterations when the store is at ``max_size``
        (gives the trainer a chance to drain). Default 0.01s.
    """

    def __init__(
        self,
        *,
        rollout_manager: Any,
        generate_fn: Callable[..., Any],
        store: Any,
        step_counter: StepCounter,
        prompts_iter_factory: Callable[[], Iterator[Any]] | None = None,
        poll_interval_s: float = 0.01,
    ) -> None:
        self._rollout_manager = rollout_manager
        self._generate_fn = generate_fn
        self._store = store
        self._step_counter = step_counter
        self._prompts_iter_factory = prompts_iter_factory
        self._poll_interval_s = float(poll_interval_s)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._exception: BaseException | None = None

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError('ContinuousRolloutProducer already started')
        # Wake the rollout pool once — avoids paying the wake/sleep latency
        # every iteration (Phase 1 lock-step path did wake/sleep per step).
        if hasattr(self._rollout_manager, 'wake_up'):
            self._rollout_manager.wake_up()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name='ContinuousRolloutProducer', daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> bool:
        """Signal the loop to exit and join the thread.

        Returns True if the thread exited cleanly within ``timeout``,
        False if it was still alive at return. On False the ``_thread``
        handle is **retained** and ``rollout_manager.sleep()`` is **not**
        called — the caller must decide whether to (a) retry ``stop()``
        at the next boundary, or (b) skip any work that would contend
        with the still-running producer for the shared OpenHands session
        (see gotcha §19 in plans-n-solutions/handsoff.md).

        A stuck thread is mid-``asyncio.run(generate_sequences)``; the
        event is only checked at the top of the worker loop, so the
        thread resumes the shutdown handshake once the current call
        returns. Thread is daemon, so interpreter shutdown still
        reclaims it.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    'ContinuousRolloutProducer did not exit within %.1fs; '
                    'leaving thread running (mid-generate_sequences); caller '
                    'must skip contention with the producer and retry stop() '
                    'at the next boundary',
                    timeout,
                )
                return False
        self._thread = None
        if hasattr(self._rollout_manager, 'sleep'):
            try:
                self._rollout_manager.sleep()
            except Exception:  # noqa: BLE001
                # Sleep may fail if the pool is already torn down; don't
                # mask the trainer's own exit path with a secondary error.
                logger.exception('rollout_manager.sleep() failed during stop()')
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            raise exc
        return True

    def check_background_error(self) -> None:
        """Re-raise any exception captured inside the worker thread.

        Single-consumer contract: call only from the trainer thread. The
        read-then-clear pattern on ``self._exception`` is non-atomic; it's
        safe here because the worker writes it exactly once (on crash, at
        thread exit) and both this method and :meth:`stop` run on the
        trainer thread. Don't call from a signal handler or from inside
        a background timer.
        """
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            raise exc

    # ---- worker -------------------------------------------------------------

    def _run(self) -> None:
        import json as _json  # noqa: PLC0415

        prompts_iter = (
            self._prompts_iter_factory()
            if self._prompts_iter_factory is not None
            else None
        )
        store_full_idles = 0
        try:
            while not self._stop_event.is_set():
                # Back off when the store is at capacity — no point generating
                # more rollouts the trainer is about to evict FIFO.
                if self._store_full():
                    store_full_idles += 1
                    if self._stop_event.wait(timeout=self._poll_interval_s):
                        break
                    continue

                # Drive one batch. Plain GRPO pulls the next dataloader batch
                # from our iterator; DAPO pulls internally from its own
                # dataloader.
                iter_start = time.monotonic()
                if prompts_iter is not None:
                    try:
                        gen_batch = next(prompts_iter)
                    except StopIteration:
                        # Dataloader exhausted — rebuild iterator for the next
                        # epoch. Matches Phase 1 trainer behavior where the
                        # fit() outer loop resets each epoch.
                        prompts_iter = self._prompts_iter_factory()  # type: ignore[misc]
                        continue
                    batch = self._generate_fn(gen_batch)
                else:
                    batch = self._generate_fn()

                policy_version = int(
                    getattr(self._rollout_manager, 'policy_version', 0)
                )
                current_step = self._step_counter.get()
                # Cut 5: when the DAPO manager has already eagerly pushed
                # every survivor into the store mid-call, skip the terminal
                # push to keep the pop-on-sample invariant (gotcha §15).
                # meta_info may be None on classic paths that never stamp it.
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
                # latencies.md §5 addition #2 — emit one line per producer
                # iteration so the log can reconstruct producer throughput
                # independent of the DAPO-internal metrics.
                logger.info(
                    'PRODUCER_ITER %s',
                    _json.dumps(
                        {
                            'event': 'producer_iter',
                            'wall_s': round(time.monotonic() - iter_start, 3),
                            'store_full_idles': store_full_idles,
                            'policy_version': policy_version,
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
        """Check whether the store is at max capacity.

        Uses ``num_groups()`` (lock-acquiring) rather than peeking the deque
        directly. The call is cheap (O(1) + one lock acquire).
        """
        try:
            return self._store.num_groups() >= self._store_max_size()
        except Exception:  # noqa: BLE001
            # If the store's introspection raises for any reason, err on the
            # side of generating (not crashing the producer). Log it so a
            # real bug isn't masked by the FIFO-evict safety net.
            logger.warning(
                'store.num_groups() raised inside _store_full; continuing to '
                'generate (FIFO eviction will bound memory)',
                exc_info=True,
            )
            return False

    def _store_max_size(self) -> int:
        # Private attr access — the store doesn't expose a max-size getter
        # publicly. Falls back to "never full" (1<<31) if the attr is
        # renamed; the FIFO deque will still evict correctly.
        return int(getattr(self._store, '_max_size', 1 << 31))

    # ---- test hooks --------------------------------------------------------

    def _is_alive_for_tests(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# ------------------------------------------------------------------- helpers


def wait_until(
    predicate: Callable[[], bool], *, timeout: float, interval: float = 0.01
) -> bool:
    """Poll ``predicate`` until it returns truthy or ``timeout`` elapses.

    Used by the trainer loop in Cut 4 to wait for the buffer to warm up.
    Returns True if the predicate became truthy, False on timeout. Lives
    here (not in tests) so the trainer can reuse it.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)
