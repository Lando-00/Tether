import asyncio

import pytest
from structlog.testing import capture_logs

from tether.core.logging import reset_logging_for_tests
from tether.runtime.abandoned_tasks import (
    AbandonedTaskTracker,
    get_notebook_abandoned_task_tracker,
)


class _FakeTask:
    def __init__(self) -> None:
        self.callbacks = []
        self.cancel_calls = 0

    def add_done_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def complete(self) -> None:
        for callback in self.callbacks:
            callback(self)

    def cancel(self) -> None:
        self.cancel_calls += 1


def test_thresholds_age_and_latched_error() -> None:
    now = [10.0]
    tracker = AbandonedTaskTracker(
        warn_threshold=2, error_threshold=3, capacity=4, clock=lambda: now[0]
    )
    first, second, third = _FakeTask(), _FakeTask(), _FakeTask()

    tracker.track(first, kind="aclose")
    now[0] = 11.25
    assert tracker.snapshot().oldest_age_ms == 1250
    assert tracker.snapshot().status == "healthy"

    tracker.track(second, kind="anext")
    assert tracker.snapshot().status == "degraded"
    tracker.track(third, kind="anext")
    assert tracker.snapshot().status == "error"
    third.complete()
    assert tracker.snapshot().count == 2
    assert tracker.snapshot().status == "error"


def test_capacity_evicts_oldest_reference_without_cancelling() -> None:
    tracker = AbandonedTaskTracker(warn_threshold=1, error_threshold=2, capacity=2)
    oldest, middle, newest = _FakeTask(), _FakeTask(), _FakeTask()
    tracker.track(oldest, kind="aclose")
    tracker.track(middle, kind="anext")
    tracker.track(newest, kind="anext")

    snapshot = tracker.snapshot()
    assert snapshot.count == 2
    assert snapshot.capacity == 2
    assert snapshot.overflowed is True
    assert snapshot.status == "error"
    assert oldest.cancel_calls == middle.cancel_calls == newest.cancel_calls == 0
    oldest.complete()
    assert tracker.snapshot().count == 2
    middle.complete()
    newest.complete()
    assert tracker.snapshot().count == 0


def test_threshold_telemetry_is_emitted_once_and_reset_for_tests() -> None:
    reset_logging_for_tests()
    tracker = AbandonedTaskTracker(warn_threshold=2, error_threshold=3, capacity=4)
    try:
        with capture_logs() as logs:
            for _ in range(4):
                tracker.track(_FakeTask(), kind="anext")
        assert [
            (entry["event"], entry["log_level"])
            for entry in logs
            if entry["event"].startswith("abandoned_task_tracker_")
        ] == [
            ("abandoned_task_tracker_warn_threshold", "warning"),
            ("abandoned_task_tracker_error_threshold", "error"),
        ]
        assert logs[0]["count"] == 2
        assert logs[0]["warn_threshold"] == 2
        assert logs[1]["count"] == 3
        assert logs[1]["error_threshold"] == 3

        tracker._reset_for_tests()
        with capture_logs() as logs_after_reset:
            for _ in range(3):
                tracker.track(_FakeTask(), kind="anext")
        assert [
            entry["event"]
            for entry in logs_after_reset
            if entry["event"].startswith("abandoned_task_tracker_")
        ] == [
            "abandoned_task_tracker_warn_threshold",
            "abandoned_task_tracker_error_threshold",
        ]
    finally:
        reset_logging_for_tests()


@pytest.mark.anyio
async def test_completion_callback_drains_real_task() -> None:
    tracker = AbandonedTaskTracker()
    gate = asyncio.Event()
    task = asyncio.create_task(gate.wait())
    tracker.track(task, kind="anext")
    assert tracker.snapshot().count == 1
    gate.set()
    await task
    await asyncio.sleep(0)
    assert tracker.snapshot().count == 0


def test_private_test_reset_clears_latches_and_references() -> None:
    tracker = AbandonedTaskTracker(warn_threshold=1, error_threshold=1, capacity=1)
    tracker.track(_FakeTask(), kind="anext")
    assert tracker.snapshot().status == "error"
    tracker._drain_for_tests()
    assert tracker.snapshot().status == "error"
    tracker._reset_for_tests()
    snapshot = tracker.snapshot()
    assert snapshot.count == 0
    assert snapshot.overflowed is False
    assert snapshot.status == "healthy"


def test_process_singleton_reset_isolates_tests() -> None:
    tracker = get_notebook_abandoned_task_tracker()
    tracker._reset_for_tests()
    try:
        tracker.track(_FakeTask(), kind="anext")
        assert tracker.snapshot().count == 1
    finally:
        tracker._reset_for_tests()
    assert get_notebook_abandoned_task_tracker().snapshot().count == 0
