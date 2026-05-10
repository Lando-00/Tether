"""Tests for the Tool lifecycle (Phase 4 step 41) — F1 portion.

Covers:

* :class:`tether_service.tools.base.BaseTool` ``REQUIRED`` ClassVar +
  default no-op ``startup`` / ``shutdown`` coroutines.
* :func:`tether_service.tools.lifecycle.startup_all` — concurrent
  startup with required-vs-optional policy.
* :func:`tether_service.tools.lifecycle.shutdown_all` — best-effort
  concurrent shutdown.

F2 (Engine wiring) tests are appended to this file in the engine-
wiring commit — they live alongside the lifecycle helpers because the
helpers are what the engine integrates.

Synthesis §4 Phase 4 step 41; §13.2 R5 (the "gather completes before
re-raise" semantics).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

import pytest

from tether_service.tools.base import BaseTool
from tether_service.tools.lifecycle import shutdown_all, startup_all


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeOptionalTool(BaseTool):
    """Optional tool with controllable startup/shutdown side effects."""

    def __init__(self, *, startup_delay: float = 0.0,
                 raise_on_startup: bool = False,
                 raise_on_shutdown: bool = False):
        super().__init__()
        self._startup_delay = startup_delay
        self._raise_on_startup = raise_on_startup
        self._raise_on_shutdown = raise_on_shutdown
        self.started = False
        self.stopped = False

    @property
    def schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": "fake", "parameters": {}}}

    async def run(self, **kwargs) -> dict:
        return {}

    async def startup(self) -> None:
        if self._startup_delay:
            await asyncio.sleep(self._startup_delay)
        if self._raise_on_startup:
            raise RuntimeError(f"startup boom: {type(self).__name__}")
        self.started = True

    async def shutdown(self) -> None:
        if self._raise_on_shutdown:
            raise RuntimeError(f"shutdown boom: {type(self).__name__}")
        self.stopped = True


class _FakeRequiredTool(_FakeOptionalTool):
    """Required tool: startup failure must raise."""
    REQUIRED = True


# ---------------------------------------------------------------------------
# F1: BaseTool defaults
# ---------------------------------------------------------------------------


def test_basetool_default_required_false():
    """Subclass without override: REQUIRED defaults to False."""
    class _T(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    assert _T.REQUIRED is False
    assert _T().REQUIRED is False


@pytest.mark.anyio
async def test_basetool_startup_default_noop():
    """Default startup is a no-op coroutine returning None."""
    class _T(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    t = _T()
    result = await t.startup()
    assert result is None


@pytest.mark.anyio
async def test_basetool_shutdown_default_noop():
    """Default shutdown is a no-op coroutine returning None."""
    class _T(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    t = _T()
    result = await t.shutdown()
    assert result is None


# ---------------------------------------------------------------------------
# F1: startup_all / shutdown_all
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_startup_all_runs_concurrent():
    """3 tools each sleeping 0.2s should finish in <0.4s wall time
    (concurrent gather)."""
    tools = {
        "a": _FakeOptionalTool(startup_delay=0.2),
        "b": _FakeOptionalTool(startup_delay=0.2),
        "c": _FakeOptionalTool(startup_delay=0.2),
    }
    t0 = time.monotonic()
    failures = await startup_all(tools)
    elapsed = time.monotonic() - t0

    assert failures == {}
    assert all(t.started for t in tools.values())
    assert elapsed < 0.4, f"startup_all should be concurrent; took {elapsed:.3f}s"


@pytest.mark.anyio
async def test_startup_all_required_failure_raises():
    """REQUIRED tool's startup raises → startup_all raises RuntimeError."""
    tools = {
        "ok": _FakeOptionalTool(),
        "must_have": _FakeRequiredTool(raise_on_startup=True),
    }
    with pytest.raises(RuntimeError) as exc_info:
        await startup_all(tools)
    assert "must_have" in str(exc_info.value)
    # The original exception is chained.
    assert exc_info.value.__cause__ is not None


@pytest.mark.anyio
async def test_startup_all_optional_failure_returned_in_dict():
    """OPTIONAL tool's startup raises → returned in failures dict; no raise."""
    tools = {
        "good": _FakeOptionalTool(),
        "bad": _FakeOptionalTool(raise_on_startup=True),
    }
    failures = await startup_all(tools)
    assert set(failures.keys()) == {"bad"}
    assert isinstance(failures["bad"], RuntimeError)
    assert tools["good"].started is True


@pytest.mark.anyio
async def test_startup_all_required_failure_after_gather_completes():
    """Synthesis §13.2 R5: even when a REQUIRED tool will raise, the gather
    must complete before re-raising — slow optional tools still finish."""
    tools = {
        "slow_optional": _FakeOptionalTool(startup_delay=0.2),
        "must_have": _FakeRequiredTool(raise_on_startup=True),
    }
    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        await startup_all(tools)
    elapsed = time.monotonic() - t0
    assert tools["slow_optional"].started is True
    assert elapsed >= 0.18, f"gather should wait for slow optional; took {elapsed:.3f}s"


@pytest.mark.anyio
async def test_shutdown_all_best_effort():
    """One tool's shutdown raises; others still complete; failures returned
    as dict; no raise."""
    tools = {
        "good": _FakeOptionalTool(),
        "bad": _FakeOptionalTool(raise_on_shutdown=True),
        "good2": _FakeOptionalTool(),
    }
    failures = await shutdown_all(tools)
    assert set(failures.keys()) == {"bad"}
    assert tools["good"].stopped is True
    assert tools["good2"].stopped is True


@pytest.mark.anyio
async def test_startup_all_empty_tools():
    """Empty mapping → empty failure dict, no error."""
    failures = await startup_all({})
    assert failures == {}


@pytest.mark.anyio
async def test_shutdown_all_empty_tools():
    """Empty mapping → empty failure dict, no error."""
    failures = await shutdown_all({})
    assert failures == {}


# ---------------------------------------------------------------------------
# F2: Engine integration (commit 7)
#
# These tests exercise the engine's __aenter__ / aclose wiring against
# the helpers above. They live here (rather than in test_engine.py)
# because the integration is the entire point of the Phase 4 step 41
# work — the tests read more naturally beside the helpers they exercise.
# ---------------------------------------------------------------------------

from typing import AsyncGenerator, List, Optional  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from tether_service.core.interfaces import ModelProvider  # noqa: E402


class _NoOpProvider(ModelProvider):
    """Minimal ModelProvider for engine tests (no HW lifecycle)."""

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "ok"

    def list_models(self) -> List[str]:
        return []

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


def _make_engine(tools: Dict[str, BaseTool]):
    """Build a minimal Engine (direct constructor) for lifecycle tests."""
    from tether_service.engine import Engine

    return Engine(
        provider=_NoOpProvider(),
        parser=MagicMock(),
        session_store=MagicMock(),
        tools=tools,
        system_prompt="",
    )


@pytest.mark.anyio
async def test_engine_aenter_startup_then_aexit_shutdown():
    """async with engine: startup runs once per tool; aclose runs shutdown
    once per tool."""
    a = _FakeOptionalTool()
    b = _FakeOptionalTool()
    eng = _make_engine({"a": a, "b": b})

    async with eng:
        assert a.started is True
        assert b.started is True
        assert a.stopped is False
        assert b.stopped is False

    assert a.stopped is True
    assert b.stopped is True


@pytest.mark.anyio
async def test_engine_aenter_drops_optional_failures():
    """Optional tool failure → dropped from engine.tools AND
    tool_runner.tools; engine still works."""
    good = _FakeOptionalTool()
    bad = _FakeOptionalTool(raise_on_startup=True)
    eng = _make_engine({"good": good, "bad": bad})

    async with eng:
        assert "good" in eng.tools
        assert "bad" not in eng.tools
        assert "bad" not in eng.tool_runner.tools
        assert "good" in eng.tool_runner.tools

    # Good tool received shutdown; bad was dropped before shutdown ran.
    assert good.stopped is True
    assert bad.stopped is False


@pytest.mark.anyio
async def test_engine_aenter_required_failure_raises():
    """Required tool failure → __aenter__ raises; aclose still cleans up
    surviving tools."""
    good = _FakeOptionalTool()
    must_have = _FakeRequiredTool(raise_on_startup=True)
    eng = _make_engine({"good": good, "must_have": must_have})

    with pytest.raises(RuntimeError):
        async with eng:
            pass

    # The good tool was started before the required-tool failure surfaced
    # (gather waits for everyone). aclose() runs from __aexit__ even on
    # error; the _closed flag is its tell.
    assert good.started is True
    assert eng._closed is True


@pytest.mark.anyio
async def test_engine_aclose_runs_tool_shutdown_before_watchdog():
    """Order check: tool shutdown completes before watchdog teardown."""
    from tether_service.runtime.hw_watchdog import HardwareWatchdog

    order: List[str] = []

    class _OrderTrackingTool(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

        async def shutdown(self):
            order.append("tool_shutdown")

    fake_watchdog = MagicMock(spec=HardwareWatchdog)
    fake_watchdog.shutdown_all = MagicMock(
        side_effect=lambda: order.append("watchdog_shutdown")
    )

    from tether_service.engine import Engine
    eng = Engine(
        provider=_NoOpProvider(),
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={"t": _OrderTrackingTool()},
        system_prompt="",
        hw_watchdog=fake_watchdog,
    )
    await eng.aclose()
    assert order == ["tool_shutdown", "watchdog_shutdown"]
