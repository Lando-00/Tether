"""Bounded process-level tracking for tasks abandoned during cleanup."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable

from tether.core.logging import logger


@dataclass(frozen=True)
class AbandonedTaskSnapshot:
    """A point-in-time view of an :class:`AbandonedTaskTracker`."""

    count: int
    oldest_age_ms: int | None
    capacity: int
    overflowed: bool
    status: str

    def to_dict(self) -> dict[str, int | bool | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class _TrackedTask:
    kind: str
    started_at: float


class AbandonedTaskTracker:
    """Retain abandoned tasks just long enough to make their state observable.

    This tracker owns only a bounded strong-reference set.  It never cancels,
    awaits, or otherwise drives the tasks it observes.
    """

    def __init__(
        self,
        *,
        warn_threshold: int = 8,
        error_threshold: int = 16,
        capacity: int = 32,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 < warn_threshold <= error_threshold <= capacity:
            raise ValueError(
                "thresholds must satisfy 0 < warn_threshold <= "
                "error_threshold <= capacity"
            )
        self._warn_threshold = warn_threshold
        self._error_threshold = error_threshold
        self._capacity = capacity
        self._clock = clock
        self._tasks: OrderedDict[Any, _TrackedTask] = OrderedDict()
        self._overflowed = False
        self._error_latched = False
        self._warn_telemetry_emitted = False
        self._error_telemetry_emitted = False

    def track(self, task: Any, *, kind: str) -> None:
        """Track an already-abandoned task until its completion callback fires.

        Re-registering a task is a no-op.  When full, only the oldest strong
        reference is discarded; critically, this does not send another cancel
        request to the task.
        """
        if task in self._tasks:
            return

        if len(self._tasks) >= self._capacity:
            evicted_task, evicted = self._tasks.popitem(last=False)
            del evicted_task
            self._overflowed = True
            self._error_latched = True
            logger.error(
                "abandoned_task_tracker_capacity_eviction",
                capacity=self._capacity,
                evicted_kind=evicted.kind,
            )

        record = _TrackedTask(kind=kind, started_at=self._clock())
        self._tasks[task] = record
        count = len(self._tasks)
        if count >= self._warn_threshold and not self._warn_telemetry_emitted:
            self._warn_telemetry_emitted = True
            logger.warning(
                "abandoned_task_tracker_warn_threshold",
                count=count,
                warn_threshold=self._warn_threshold,
            )
        if count >= self._error_threshold and not self._error_latched:
            self._error_latched = True
        if count >= self._error_threshold and not self._error_telemetry_emitted:
            self._error_telemetry_emitted = True
            logger.error(
                "abandoned_task_tracker_error_threshold",
                count=count,
                error_threshold=self._error_threshold,
            )

        def _completed(_: Any) -> None:
            if self._tasks.get(task) is record:
                self._tasks.pop(task, None)

        task.add_done_callback(_completed)

    def snapshot(self) -> AbandonedTaskSnapshot:
        """Return the current bounded tracker state without mutating tasks."""
        now = self._clock()
        oldest = next(iter(self._tasks.values()), None)
        oldest_age_ms = (
            max(0, int((now - oldest.started_at) * 1000))
            if oldest is not None
            else None
        )
        count = len(self._tasks)
        if self._error_latched or count >= self._error_threshold:
            status = "error"
        elif count >= self._warn_threshold:
            status = "degraded"
        else:
            status = "healthy"
        return AbandonedTaskSnapshot(
            count=count,
            oldest_age_ms=oldest_age_ms,
            capacity=self._capacity,
            overflowed=self._overflowed,
            status=status,
        )

    def _drain_for_tests(self) -> None:
        """Test-only: release all retained task references without cancelling."""
        self._tasks.clear()

    def _tracked_tasks_for_tests(self) -> list[Any]:
        """Test-only: snapshot the retained task references in insertion order."""
        return list(self._tasks)

    def _reset_for_tests(self) -> None:
        """Test-only: restore an empty, unlatched tracker for test isolation."""
        self._drain_for_tests()
        self._overflowed = False
        self._error_latched = False
        self._warn_telemetry_emitted = False
        self._error_telemetry_emitted = False


_NOTEBOOK_ABANDONED_TASK_TRACKER = AbandonedTaskTracker()


def get_notebook_abandoned_task_tracker() -> AbandonedTaskTracker:
    """Return the process singleton used by Notebook cleanup."""
    return _NOTEBOOK_ABANDONED_TASK_TRACKER


__all__ = [
    "AbandonedTaskSnapshot",
    "AbandonedTaskTracker",
    "get_notebook_abandoned_task_tracker",
]
