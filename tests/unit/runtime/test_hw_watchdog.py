"""Tests for ``tether.runtime.hw_watchdog.HardwareWatchdog``.

Synthesis §4 Phase 3 step 30; B6 step 2.

These tests use in-process fakes only (no MLC). The MLC-specific behaviors
will be covered by hardware-marker tests added in p3-mlc-as-hwlifecycle /
p3-lifespan-slim.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List

import pytest

from tether.providers.hw import HwErrorClass, HwHealth
from tether.runtime.hw_watchdog import (
    DEFAULT_SHUTDOWN_BUDGET_SEC,
    HardwareWatchdog,
)
from tether.runtime.watchdog_mode import WatchdogMode


# ---------------------------------------------------------------------------
# Test fakes (defined inline; NOT in tether_service/)
# ---------------------------------------------------------------------------


class FakeHWProvider:
    """Minimal HardwareLifecycle implementation for in-process tests.

    Configurable knobs via constructor: classify result, health to return,
    shutdown side-effects, budgets. Records calls for assertions.
    """

    def __init__(
        self,
        *,
        classify_result: HwErrorClass = HwErrorClass.BENIGN,
        health: HwHealth = HwHealth("healthy"),
        shutdown_budget: float = 1.0,
        per_engine_terminate: float = 0.25,
        shutdown_sleep_sec: float = 0.0,
        raise_in_health: BaseException | None = None,
        raise_in_reset: BaseException | None = None,
    ) -> None:
        self._classify_result = classify_result
        self._health = health
        self._shutdown_budget = shutdown_budget
        self._per_engine_terminate = per_engine_terminate
        self._shutdown_sleep_sec = shutdown_sleep_sec
        self._raise_in_health = raise_in_health
        self._raise_in_reset = raise_in_reset
        self.classify_calls: List[BaseException] = []
        self.reset_calls: List[str] = []
        self.shutdown_called = False

    def hw_classify(self, exc: BaseException) -> HwErrorClass:
        self.classify_calls.append(exc)
        return self._classify_result

    async def hw_reset(self, model_name: str) -> None:
        self.reset_calls.append(model_name)
        if self._raise_in_reset is not None:
            raise self._raise_in_reset

    async def hw_health(self) -> HwHealth:
        if self._raise_in_health is not None:
            raise self._raise_in_health
        return self._health

    @property
    def hw_shutdown_budget_sec(self) -> float:
        return self._shutdown_budget

    @property
    def hw_per_engine_terminate_sec(self) -> float:
        return self._per_engine_terminate

    def shutdown_all(self) -> None:
        self.shutdown_called = True
        if self._shutdown_sleep_sec > 0:
            time.sleep(self._shutdown_sleep_sec)


class FakeHWProviderNoShutdown:
    """A HardwareLifecycle implementation without ``shutdown_all`` (the
    Protocol does NOT require it). The watchdog must feed
    ``daemon_thread_call`` a no-op lambda for this case.
    """

    def hw_classify(self, exc: BaseException) -> HwErrorClass:
        return HwErrorClass.BENIGN

    async def hw_reset(self, model_name: str) -> None:
        return None

    async def hw_health(self) -> HwHealth:
        return HwHealth("healthy")

    @property
    def hw_shutdown_budget_sec(self) -> float:
        return 0.5

    @property
    def hw_per_engine_terminate_sec(self) -> float:
        return 0.1


class _NotAHWProvider:
    """Looks like a regular provider; does NOT implement HardwareLifecycle.

    The watchdog must filter this out at construction.
    """

    def list_models(self) -> List[str]:
        return []


# ---------------------------------------------------------------------------
# Construction / filtering
# ---------------------------------------------------------------------------


def test_watchdog_filters_non_hw_providers():
    """Mixing a HardwareLifecycle impl with a non-HW provider must result
    in only the HW one being registered."""
    hw = FakeHWProvider()
    not_hw = _NotAHWProvider()

    wd = HardwareWatchdog([not_hw, hw, not_hw])

    assert wd.hw_provider_count == 1


def test_watchdog_multi_provider_all_non_hw_is_noop():
    """ADR-0021 P2.A regression: with a multi-provider engine where every
    entry is a non-HW provider (e.g., CopilotProvider + DummyProvider),
    the watchdog's HW filter leaves ``_hw_providers`` empty. ``shutdown_all()``
    is a no-op (no exception). The ``health_summary()`` shape for this
    case is covered by ``test_watchdog_health_summary_no_hw``."""
    wd = HardwareWatchdog([_NotAHWProvider(), _NotAHWProvider()])
    assert wd.hw_provider_count == 0
    wd.shutdown_all()  # no-op, MUST NOT raise


def test_watchdog_no_hw_providers_uses_default_budget():
    """With zero HW providers, the shutdown budget falls back to the
    DEFAULT_SHUTDOWN_BUDGET_SEC constant."""
    wd = HardwareWatchdog([_NotAHWProvider()])

    assert wd.hw_provider_count == 0
    assert wd.shutdown_budget_sec == DEFAULT_SHUTDOWN_BUDGET_SEC


def test_watchdog_budget_is_max_of_providers():
    """With multiple HW providers, the watchdog's total budget = max of
    each provider's hw_shutdown_budget_sec."""
    p1 = FakeHWProvider(shutdown_budget=2.0)
    p2 = FakeHWProvider(shutdown_budget=5.5)
    p3 = FakeHWProvider(shutdown_budget=1.0)

    wd = HardwareWatchdog([p1, p2, p3])

    assert wd.shutdown_budget_sec == 5.5


def test_watchdog_explicit_budget_override():
    """Caller-provided shutdown_budget_sec overrides the computed default."""
    p = FakeHWProvider(shutdown_budget=99.0)
    wd = HardwareWatchdog([p], shutdown_budget_sec=2.0)
    assert wd.shutdown_budget_sec == 2.0


def test_watchdog_mode_default_library():
    wd = HardwareWatchdog([])
    assert wd.mode is WatchdogMode.LIBRARY


def test_watchdog_mode_server():
    wd = HardwareWatchdog([], mode=WatchdogMode.SERVER)
    assert wd.mode is WatchdogMode.SERVER


# ---------------------------------------------------------------------------
# health_summary
# ---------------------------------------------------------------------------


async def test_watchdog_health_summary_no_hw():
    """No HW providers → providers=[], overall='healthy'."""
    wd = HardwareWatchdog([_NotAHWProvider()])
    summary = await wd.health_summary()
    assert summary == {"providers": [], "overall": "healthy"}


async def test_watchdog_health_summary_aggregates_healthy():
    """All providers report healthy → overall=healthy."""
    p1 = FakeHWProvider(health=HwHealth("healthy", {"engines": 1}))
    p2 = FakeHWProvider(health=HwHealth("healthy", {"engines": 2}))
    wd = HardwareWatchdog([p1, p2])

    summary = await wd.health_summary()
    assert summary["overall"] == "healthy"
    assert len(summary["providers"]) == 2
    assert summary["providers"][0]["status"] == "healthy"
    assert summary["providers"][0]["details"] == {"engines": 1}


async def test_watchdog_health_summary_aggregates_worst_error():
    """Mixed: one error + one healthy → overall=error."""
    p1 = FakeHWProvider(health=HwHealth("healthy"))
    p2 = FakeHWProvider(health=HwHealth("error", {"reason": "tvm"}))
    wd = HardwareWatchdog([p1, p2])

    summary = await wd.health_summary()
    assert summary["overall"] == "error"


async def test_watchdog_health_summary_aggregates_worst_degraded():
    """Mixed: degraded + healthy (no error) → overall=degraded."""
    p1 = FakeHWProvider(health=HwHealth("healthy"))
    p2 = FakeHWProvider(health=HwHealth("degraded", {"engines_failed": 1}))
    wd = HardwareWatchdog([p1, p2])

    summary = await wd.health_summary()
    assert summary["overall"] == "degraded"


async def test_watchdog_health_summary_handles_provider_exception():
    """If hw_health() raises, watchdog records error+details and overall
    becomes error (instead of crashing)."""
    p1 = FakeHWProvider(health=HwHealth("healthy"))
    p2 = FakeHWProvider(raise_in_health=RuntimeError("driver dead"))
    wd = HardwareWatchdog([p1, p2])

    summary = await wd.health_summary()
    assert summary["overall"] == "error"
    # The crashing provider's entry has status=error and details with the msg.
    error_entry = next(
        e for e in summary["providers"] if e["status"] == "error"
    )
    assert "driver dead" in error_entry["details"]["error"]


# ---------------------------------------------------------------------------
# reset_after
# ---------------------------------------------------------------------------


async def test_watchdog_reset_after_classifies_fatal_recoverable():
    """When a provider classifies as FATAL_RECOVERABLE, hw_reset is called
    with the model name and the watchdog returns True."""
    p = FakeHWProvider(classify_result=HwErrorClass.FATAL_RECOVERABLE)
    wd = HardwareWatchdog([p])

    exc = RuntimeError("CL_OUT_OF_RESOURCES")
    did_reset = await wd.reset_after(exc, model_name="Qwen3-4B")

    assert did_reset is True
    assert p.classify_calls == [exc]
    assert p.reset_calls == ["Qwen3-4B"]


async def test_watchdog_reset_after_skips_transient():
    """TRANSIENT classification → no hw_reset call, returns False."""
    p = FakeHWProvider(classify_result=HwErrorClass.TRANSIENT)
    wd = HardwareWatchdog([p])

    did_reset = await wd.reset_after(RuntimeError("retry me"), model_name="X")

    assert did_reset is False
    assert p.reset_calls == []


async def test_watchdog_reset_after_skips_benign_and_fatal_unrecoverable():
    """BENIGN and FATAL_UNRECOVERABLE both skip reset and return False."""
    for cls in (HwErrorClass.BENIGN, HwErrorClass.FATAL_UNRECOVERABLE):
        p = FakeHWProvider(classify_result=cls)
        wd = HardwareWatchdog([p])
        assert await wd.reset_after(RuntimeError("x"), model_name="m") is False
        assert p.reset_calls == []


async def test_watchdog_reset_after_no_hw_providers():
    """No HW providers → reset_after returns False, no exceptions."""
    wd = HardwareWatchdog([_NotAHWProvider()])
    assert await wd.reset_after(RuntimeError("x"), model_name="m") is False


async def test_watchdog_reset_after_swallows_reset_exception():
    """If hw_reset itself raises, the watchdog logs and continues; the
    provider's any_reset doesn't flip true (we only return True on success)."""
    p = FakeHWProvider(
        classify_result=HwErrorClass.FATAL_RECOVERABLE,
        raise_in_reset=RuntimeError("reset failed"),
    )
    wd = HardwareWatchdog([p])

    did_reset = await wd.reset_after(RuntimeError("orig"), model_name="m")
    assert did_reset is False
    assert p.reset_calls == ["m"]  # we did try


# ---------------------------------------------------------------------------
# shutdown_all
# ---------------------------------------------------------------------------


def test_watchdog_shutdown_all_no_hw_providers():
    """With no HW providers, shutdown_all is a quick no-op (no daemon
    thread spawned, no error)."""
    wd = HardwareWatchdog([_NotAHWProvider()])
    start = time.time()
    wd.shutdown_all()  # should not raise
    assert time.time() - start < 0.5


def test_watchdog_shutdown_all_calls_provider():
    """A FakeHWProvider's shutdown_all is invoked through daemon_thread_call;
    its side-effect flag flips True; we complete within budget."""
    p = FakeHWProvider(shutdown_budget=2.0, shutdown_sleep_sec=0.05)
    wd = HardwareWatchdog([p])

    start = time.time()
    wd.shutdown_all()
    elapsed = time.time() - start

    assert p.shutdown_called is True
    assert elapsed < 1.0


def test_watchdog_shutdown_all_handles_timeout():
    """If a provider's shutdown_all sleeps longer than its budget, the
    watchdog logs + abandons (does not raise) and returns."""
    p = FakeHWProvider(shutdown_budget=0.1, shutdown_sleep_sec=2.0)
    wd = HardwareWatchdog([p])

    start = time.time()
    wd.shutdown_all()  # MUST NOT raise
    elapsed = time.time() - start

    # Budget was 0.1s; the watchdog should have abandoned around there.
    # Allow generous slack for CI scheduler jitter.
    assert elapsed < 1.0


def test_watchdog_shutdown_all_handles_no_shutdown_all_method():
    """If a HW provider lacks shutdown_all (Protocol does not require it),
    the watchdog feeds daemon_thread_call a no-op lambda — no error."""
    p = FakeHWProviderNoShutdown()
    wd = HardwareWatchdog([p])

    wd.shutdown_all()  # MUST NOT raise


def test_watchdog_shutdown_all_continues_after_provider_failure():
    """If one provider's shutdown raises, the watchdog logs and continues
    to shut down the others (no early return)."""

    class _RaisingProvider(FakeHWProvider):
        def shutdown_all(self) -> None:  # type: ignore[override]
            self.shutdown_called = True
            raise RuntimeError("boom in shutdown")

    p1 = _RaisingProvider()
    p2 = FakeHWProvider()
    wd = HardwareWatchdog([p1, p2])

    wd.shutdown_all()

    assert p1.shutdown_called is True
    assert p2.shutdown_called is True


def test_watchdog_shutdown_all_uses_daemon_thread():
    """Smoke check: the daemon thread used by daemon_thread_call should
    have a name beginning with 'shutdown-' (label per provider type)."""
    captured: Dict[str, str] = {}

    class _CaptureProvider(FakeHWProvider):
        def shutdown_all(self) -> None:  # type: ignore[override]
            captured["thread_name"] = threading.current_thread().name
            super().shutdown_all()

    wd = HardwareWatchdog([_CaptureProvider()])
    wd.shutdown_all()

    assert captured["thread_name"].startswith("shutdown-_CaptureProvider")
    assert captured["thread_name"].endswith("-thread")
