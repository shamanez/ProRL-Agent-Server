# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Cut 4 tests: :class:`ContinuousRolloutProducer` thread lifecycle.

Host-runnable — the producer has no verl / ray / vLLM coupling. We stub
``rollout_manager`` with a :class:`SimpleNamespace` and feed a stub
``generate_fn`` that returns fake ``DataProto``-shaped payloads directly
to the real :class:`TrajectoryStore`.

Scope (matches ``plans-n-solutions/stages/full_async.md`` §4 Cut 4):

* ``test_producer_start_stop_clean`` — start, push a few batches, stop
  within timeout, no dangling thread.
* ``test_producer_pushes_with_current_policy_version`` — mid-loop
  publish bumps ``policy_version``; the next push carries the new
  version.
* ``test_trainer_sleep_wait_when_buffer_empty`` — :func:`wait_until`
  times out cleanly on an empty store; no exceptions.

Plus a couple of invariants that fall out of the same plumbing:

* ``StepCounter`` is lock-safe under concurrent set/get.
* Exceptions inside the worker are captured and re-raised from
  :meth:`stop`.
* When the store is at capacity the producer backs off (no runaway
  push loop).
* ``wake_up`` / ``sleep`` are invoked exactly once each.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / 'trainer_integration' / 'verl')
)

from verl_custom.replay.continuous_producer import (  # noqa: E402
    ContinuousRolloutProducer,
    StepCounter,
    wait_until,
    wait_until_with_progress,
)
from verl_custom.replay.trajectory_store import TrajectoryStore  # noqa: E402

PAD_ID = 0


# --- stubs ------------------------------------------------------------------


def _stub_dataproto(
    *,
    batch: int = 2,
    prompt_len: int = 3,
    response_len: int = 2,
    uid_prefix: str = 'u',
    uid_offset: int = 0,
):
    """Build a ``DataProto``-shaped stub the store can ingest."""
    import numpy as np

    full_len = prompt_len + response_len
    input_ids = torch.zeros((batch, full_len), dtype=torch.long)
    responses = torch.zeros((batch, response_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones((batch, response_len), dtype=torch.long)
    rollout_log_probs = torch.full((batch, response_len), -0.1)
    is_padded = torch.zeros(batch, dtype=torch.bool)
    error_mask = torch.zeros(batch, dtype=torch.bool)
    uids = np.array(
        [f'{uid_prefix}-{uid_offset + i}' for i in range(batch)],
        dtype=object,
    )
    tensors = {
        'input_ids': input_ids,
        'responses': responses,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
        'rollout_log_probs': rollout_log_probs,
        'is_padded': is_padded,
        'error_mask': error_mask,
    }
    return SimpleNamespace(batch=tensors, non_tensor_batch={'uid': uids})


class _StubRolloutManager:
    """Minimal shape: ``policy_version`` + counted ``wake_up``/``sleep``."""

    def __init__(self, policy_version: int = 0) -> None:
        self.policy_version = int(policy_version)
        self.wake_calls = 0
        self.sleep_calls = 0

    def wake_up(self) -> None:
        self.wake_calls += 1

    def sleep(self) -> None:
        self.sleep_calls += 1


def _store(**kw) -> TrajectoryStore:
    return TrajectoryStore(
        max_size=kw.pop('max_size', 16),
        staleness_cutoff_k=kw.pop('staleness_cutoff_k', 10_000),
        pad_token_id=PAD_ID,
        **kw,
    )


# --- StepCounter ------------------------------------------------------------


class TestStepCounter:
    def test_initial_and_set_roundtrip(self) -> None:
        counter = StepCounter(initial=7)
        assert counter.get() == 7
        counter.set(42)
        assert counter.get() == 42

    def test_concurrent_set_get_no_race(self) -> None:
        """Many threads hammering set/get should never tear the int or
        raise — we only need eventual consistency."""
        counter = StepCounter(initial=0)
        stop = threading.Event()
        errors: list[BaseException] = []

        def writer(target: int) -> None:
            try:
                for i in range(target):
                    if stop.is_set():
                        return
                    counter.set(i)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        def reader() -> None:
            try:
                while not stop.is_set():
                    _ = counter.get()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        writers = [threading.Thread(target=writer, args=(5_000,)) for _ in range(2)]
        readers = [threading.Thread(target=reader) for _ in range(2)]
        for t in writers + readers:
            t.start()
        for t in writers:
            t.join(timeout=5.0)
        stop.set()
        for t in readers:
            t.join(timeout=5.0)
        assert errors == []


# --- lifecycle (start / stop) ----------------------------------------------


class TestProducerLifecycle:
    def test_producer_start_stop_clean(self) -> None:
        """Start, allow a handful of pushes, stop within timeout, no dangling
        thread. ``wake_up`` and ``sleep`` each get called exactly once."""
        store = _store(max_size=8)
        rm = _StubRolloutManager(policy_version=3)
        step_counter = StepCounter(initial=0)

        call_count = {'n': 0}
        call_cv = threading.Condition()

        def gen_fn():
            with call_cv:
                call_count['n'] += 1
                call_cv.notify_all()
            return _stub_dataproto(uid_offset=call_count['n'] * 10)

        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=gen_fn,
            store=store,
            step_counter=step_counter,
            prompts_iter_factory=None,
            poll_interval_s=0.001,
        )

        producer.start()
        assert producer._is_alive_for_tests()
        # Wait until we've observed at least 3 generate calls — deterministic
        # handoff, no wall-clock sleep.
        with call_cv:
            assert call_cv.wait_for(lambda: call_count['n'] >= 3, timeout=5.0), (
                f'producer did not reach 3 generate calls in time (got '
                f'{call_count["n"]})'
            )
        producer.stop(timeout=5.0)

        assert not producer._is_alive_for_tests(), 'thread leaked past stop()'
        assert rm.wake_calls == 1
        assert rm.sleep_calls == 1
        assert store.num_groups() >= 1
        # Every stored record carries the stamped policy_version.
        for group in store._snapshot_groups():
            for rec in group:
                assert rec.behavior_policy_version == 3

    def test_double_start_raises(self) -> None:
        store = _store()
        rm = _StubRolloutManager()
        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=lambda: _stub_dataproto(),
            store=store,
            step_counter=StepCounter(),
            prompts_iter_factory=None,
            poll_interval_s=0.001,
        )
        producer.start()
        try:
            with pytest.raises(RuntimeError, match='already started'):
                producer.start()
        finally:
            producer.stop(timeout=5.0)

    def test_stop_is_idempotent_on_never_started(self) -> None:
        """Trainer's ``finally:`` may call stop() even if start() failed early.
        Must not raise, must not try to sleep() the pool."""
        store = _store()
        rm = _StubRolloutManager()
        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=lambda: _stub_dataproto(),
            store=store,
            step_counter=StepCounter(),
            prompts_iter_factory=None,
            poll_interval_s=0.001,
        )
        # Never started → stop should still be safe.
        producer.stop(timeout=0.5)
        assert rm.wake_calls == 0
        # sleep() is still called — the trainer relies on it for cleanup
        # on the exit path regardless of whether the thread ever ran.
        assert rm.sleep_calls == 1


# --- policy_version freshness ----------------------------------------------


class TestPolicyVersionFreshness:
    def test_producer_pushes_with_current_policy_version(self) -> None:
        """Mid-loop ``policy_version`` bump must land in the next push.

        The producer reads ``rollout_manager.policy_version`` unlocked —
        gotcha §20 in handsoff.md documents reliance on CPython GIL
        atomicity for single-int load/store. This test asserts the
        observable behavior: after a publish, the next batch of stored
        records carries the new version.
        """
        store = _store(max_size=32)
        rm = _StubRolloutManager(policy_version=0)
        step_counter = StepCounter(initial=0)

        gate_after_first = threading.Event()
        first_push_seen = threading.Event()
        bump_seen = threading.Event()

        def gen_fn():
            # After the first push, block until the test bumps the version.
            if first_push_seen.is_set() and not gate_after_first.is_set():
                gate_after_first.wait(timeout=5.0)
            return _stub_dataproto(
                batch=1, uid_offset=int(time.monotonic_ns()) & 0xFFFF
            )

        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=gen_fn,
            store=store,
            step_counter=step_counter,
            prompts_iter_factory=None,
            poll_interval_s=0.001,
        )
        producer.start()
        try:
            # Wait until at least one group landed at version 0.
            assert wait_until(
                lambda: store.num_groups() >= 1, timeout=5.0, interval=0.002
            )
            first_push_seen.set()
            # Bump version, then release the gate so the producer pushes again.
            rm.policy_version = 7
            bump_seen.set()
            gate_after_first.set()
            assert wait_until(
                lambda: any(
                    rec.behavior_policy_version == 7
                    for group in store._snapshot_groups()
                    for rec in group
                ),
                timeout=5.0,
                interval=0.002,
            ), 'no post-bump record observed with policy_version=7'
        finally:
            producer.stop(timeout=5.0)
            assert bump_seen.is_set()

    def test_step_counter_stamped_onto_records(self) -> None:
        """The trainer's ``global_steps`` (via :class:`StepCounter`) is stamped
        on each record's ``created_at_step`` so replay age is accurate."""
        store = _store(max_size=8)
        rm = _StubRolloutManager()
        step_counter = StepCounter(initial=42)

        stored = threading.Event()

        def gen_fn():
            stored.set()
            return _stub_dataproto(batch=1, uid_offset=0)

        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=gen_fn,
            store=store,
            step_counter=step_counter,
            prompts_iter_factory=None,
            poll_interval_s=0.001,
        )
        producer.start()
        try:
            assert stored.wait(timeout=5.0)
            assert wait_until(
                lambda: store.num_groups() >= 1, timeout=5.0, interval=0.002
            )
            groups = store._snapshot_groups()
            assert any(rec.created_at_step == 42 for group in groups for rec in group)
        finally:
            producer.stop(timeout=5.0)


# --- empty-buffer polling (trainer's wait path) ----------------------------


class TestTrainerSleepOnEmptyBuffer:
    def test_wait_until_returns_false_on_timeout(self) -> None:
        """Trainer polls the store via :func:`wait_until`; an empty store must
        time out cleanly (returning False) without raising."""
        store = _store(max_size=8)
        start = time.monotonic()
        result = wait_until(
            lambda: store.num_groups() >= 1,
            timeout=0.05,
            interval=0.005,
        )
        elapsed = time.monotonic() - start
        assert result is False
        # Sanity: we did actually wait approximately the timeout, not longer.
        assert elapsed < 0.5, f'wait_until took {elapsed:.3f}s (>> timeout)'

    def test_wait_until_returns_true_when_buffer_warms(self) -> None:
        """Predicate flips to True during polling → returns True immediately."""
        store = _store(max_size=8)

        def warm_later() -> None:
            time.sleep(0.02)
            store.push_from_dataproto(
                _stub_dataproto(batch=1, uid_offset=0),
                behavior_policy_version=0,
                current_step=0,
            )

        t = threading.Thread(target=warm_later)
        t.start()
        try:
            assert wait_until(
                lambda: store.num_groups() >= 1, timeout=5.0, interval=0.002
            )
        finally:
            t.join(timeout=5.0)


class TestWaitUntilWithProgress:
    """No-progress detector for ``_acquire_training_batch_dapo``.

    Replaces the brittle 7200 s hard-cap that killed prep-100 at
    step 44 once response_length climbed past ~12 k tokens. The new
    helper aborts only when the producer has stopped pushing for
    ``no_progress_timeout`` seconds — a slow-but-healthy producer
    no longer trips the guardrail.
    """

    def test_returns_false_when_no_pushes_arrive(self) -> None:
        store = _store(max_size=8)
        start = time.monotonic()
        result = wait_until_with_progress(
            lambda: store.num_groups() >= 1,
            store.total_pushes,
            no_progress_timeout=0.05,
            interval=0.005,
        )
        elapsed = time.monotonic() - start
        assert result is False
        assert elapsed < 0.5, f'took {elapsed:.3f}s (>> no_progress_timeout)'

    def test_resets_deadline_on_progress(self) -> None:
        """Push interval (0.1 s) > single no-progress timeout window
        (0.15 s) is *not* enough to trip the guardrail because each push
        resets the deadline. Without the reset, total wall (~0.4 s for
        4 pushes) would exceed a single 0.15 s window long before the
        predicate flips."""
        store = _store(max_size=8)
        errors: list[BaseException] = []

        def slow_pusher() -> None:
            try:
                for i in range(4):
                    time.sleep(0.1)
                    store.push_from_dataproto(
                        _stub_dataproto(batch=1, uid_offset=i),
                        behavior_policy_version=0,
                        current_step=0,
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=slow_pusher)
        t.start()
        try:
            result = wait_until_with_progress(
                lambda: store.num_groups() >= 4,
                store.total_pushes,
                no_progress_timeout=0.15,
                interval=0.005,
            )
        finally:
            t.join(timeout=5.0)
        assert errors == [], f'pusher raised: {errors}'
        assert result is True
        assert store.total_pushes() == 4


# --- store-full backoff -----------------------------------------------------


class TestStoreFullBackoff:
    def test_producer_backs_off_when_store_full(self) -> None:
        """When the store hits ``max_size`` the producer should sleep the
        poll interval instead of hammering ``generate_fn``.

        The predicate we assert: after the store fills, the number of
        generate calls in a short window stays near zero.
        """
        store = _store(max_size=2)
        rm = _StubRolloutManager()
        step_counter = StepCounter(initial=0)

        count = {'n': 0}

        def gen_fn():
            count['n'] += 1
            # 1 uid → 1 group → reaches max_size=2 in 2 calls.
            return _stub_dataproto(batch=1, uid_offset=count['n'])

        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=gen_fn,
            store=store,
            step_counter=step_counter,
            prompts_iter_factory=None,
            poll_interval_s=0.02,
        )
        producer.start()
        try:
            # Wait for the store to fill.
            assert wait_until(
                lambda: store.num_groups() >= 2, timeout=5.0, interval=0.002
            )
            calls_at_full = count['n']
            # In a short observation window, backoff should keep the delta
            # bounded. (Not zero — FIFO deque eviction lets one more push
            # land after a pop.) We care that it's not runaway.
            time.sleep(0.2)
            calls_after = count['n']
            assert calls_after - calls_at_full < 50, (
                f'producer did not back off at capacity: '
                f'{calls_after - calls_at_full} extra calls in 0.2s'
            )
        finally:
            producer.stop(timeout=5.0)


# --- exception propagation --------------------------------------------------


class TestWorkerException:
    def test_worker_exception_captured_and_raised_on_stop(self) -> None:
        """A failing ``generate_fn`` should mark the producer dead, capture
        the exception, and re-raise from :meth:`stop`."""
        store = _store()
        rm = _StubRolloutManager()
        step_counter = StepCounter(initial=0)

        class Boom(RuntimeError):
            pass

        def gen_fn():
            raise Boom('rollout exploded')

        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=gen_fn,
            store=store,
            step_counter=step_counter,
            prompts_iter_factory=None,
            poll_interval_s=0.001,
        )
        producer.start()
        # Give the worker a tick to run and die.
        assert wait_until(
            lambda: not producer._is_alive_for_tests(), timeout=5.0, interval=0.002
        )
        with pytest.raises(Boom, match='rollout exploded'):
            producer.stop(timeout=1.0)

    def test_check_background_error_reraises(self) -> None:
        """Trainer can poll ``check_background_error`` between iterations to
        detect a dead producer without shutting down."""
        store = _store()
        rm = _StubRolloutManager()
        step_counter = StepCounter(initial=0)

        def gen_fn():
            raise RuntimeError('kaboom')

        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=gen_fn,
            store=store,
            step_counter=step_counter,
            prompts_iter_factory=None,
            poll_interval_s=0.001,
        )
        producer.start()
        assert wait_until(
            lambda: not producer._is_alive_for_tests(), timeout=5.0, interval=0.002
        )
        with pytest.raises(RuntimeError, match='kaboom'):
            producer.check_background_error()
        # Second call is clean — the exception was consumed.
        producer.check_background_error()
        # Clean up (stop should not re-raise now that the exception was taken).
        producer.stop(timeout=1.0)


# --- prompts_iter path (GRPO) ----------------------------------------------


class TestPromptsIterMode:
    def test_generate_fn_receives_prompts_from_factory(self) -> None:
        """GRPO mode: producer pulls from ``prompts_iter_factory()`` and feeds
        each ``gen_batch`` into ``generate_fn(gen_batch)``."""
        store = _store(max_size=8)
        rm = _StubRolloutManager()
        step_counter = StepCounter(initial=0)

        seen_prompts: list[str] = []
        seen_cv = threading.Condition()

        def factory():
            # Finite iterator rebuilt on StopIteration.
            return iter(['prompt-a', 'prompt-b', 'prompt-c'])

        def gen_fn(prompt_token: str):
            with seen_cv:
                seen_prompts.append(prompt_token)
                seen_cv.notify_all()
            return _stub_dataproto(batch=1, uid_offset=len(seen_prompts))

        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=gen_fn,
            store=store,
            step_counter=step_counter,
            prompts_iter_factory=factory,
            poll_interval_s=0.001,
        )
        producer.start()
        try:
            with seen_cv:
                assert seen_cv.wait_for(lambda: len(seen_prompts) >= 3, timeout=5.0), (
                    f'only saw {len(seen_prompts)} prompts'
                )
        finally:
            producer.stop(timeout=5.0)
        # Factory was exercised — at least one epoch of prompts observed.
        assert seen_prompts[:3] == ['prompt-a', 'prompt-b', 'prompt-c']

    def test_stopiteration_rebuilds_iterator(self) -> None:
        """Dataloader exhausted → factory re-invoked for the next epoch."""
        store = _store(max_size=32)
        rm = _StubRolloutManager()
        step_counter = StepCounter(initial=0)

        factory_calls = {'n': 0}
        saw_two_epochs = threading.Event()

        def factory():
            factory_calls['n'] += 1
            if factory_calls['n'] >= 2:
                saw_two_epochs.set()
            return iter(['only-one'])

        def gen_fn(_token: str):
            return _stub_dataproto(batch=1, uid_offset=factory_calls['n'])

        producer = ContinuousRolloutProducer(
            rollout_manager=rm,
            generate_fn=gen_fn,
            store=store,
            step_counter=step_counter,
            prompts_iter_factory=factory,
            poll_interval_s=0.001,
        )
        producer.start()
        try:
            assert saw_two_epochs.wait(timeout=5.0), 'factory was not re-invoked'
        finally:
            producer.stop(timeout=5.0)
        assert factory_calls['n'] >= 2
