"""Tests for Phase 3 step 36 — orchestrator delegates classification +
recovery to :class:`HardwareWatchdog`.

Verifies F4 of p3-lifespan-slim:

- The substring grep ``"TVMError" in ... or "CLML" in ... or "CL_" in ...``
  is gone from orchestrator.py source.
- ``orchestrate(..., hw_watchdog=watchdog)`` calls
  ``watchdog.reset_after(exc, model_name=...)`` on a stream error.
- ``OrchestratorConfig.auto_reload_on_fatal_error=False`` skips the call.
- ``hw_watchdog=None`` (or omitted) gracefully skips recovery.
- The error wire event still includes ``is_fatal``; True iff a reset fired.

Synthesis §4 Phase 3 step 36; §6 row 13; §11.3 R21.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

import tether.protocol.orchestration.orchestrator as orch_mod
from tests.golden.conftest import MinimalMemoryStore
from tether.core.interfaces import ModelProvider
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.orchestrator import orchestrate
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.sliding import SlidingParser
from tether.runtime.hw_watchdog import HardwareWatchdog


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _config(**overrides) -> OrchestratorConfig:
    defaults = dict(
        max_tool_loops=2,
        auto_reload_on_fatal_error=True,
        save_thinking=False,
        include_thinking_in_history=False,
    )
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)


class _RaisingProvider(ModelProvider):
    """Yields one chunk then raises a stream error. The error message is
    deliberately benign — under the legacy substring-grep this would NOT
    have been classified fatal; under the watchdog it depends on the
    watchdog's classify."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "Hello there, partial reply before failure. "
        raise self._exc

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


# ---------------------------------------------------------------------------
# Source-level assertion: substring grep is gone from orchestrator.py.
# ---------------------------------------------------------------------------


def test_orchestrator_no_substring_grep():
    """The legacy ``"TVMError"/"CLML"/"CL_"`` substring check is replaced
    by HardwareWatchdog.reset_after; none of those literals should appear
    in orchestrator.py source any more.

    Synthesis §11.3 R21.
    """
    src = inspect.getsource(orch_mod)
    forbidden = ['"TVMError"', '"CLML"', '"CL_"']
    for needle in forbidden:
        assert needle not in src, (
            f"orchestrator.py still contains substring-grep token {needle!r}; "
            "Phase 3 step 36 requires HardwareWatchdog.reset_after to own "
            "classification."
        )


# ---------------------------------------------------------------------------
# reset_after gets called on stream error when watchdog is provided.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orchestrator_calls_reset_after_on_error():
    """Mock watchdog.reset_after returns True; orchestrator emits an info
    event mentioning HardwareWatchdog reset, and the error event has
    is_fatal=True."""
    provider = _RaisingProvider(RuntimeError("model boom"))
    watchdog = MagicMock(spec=HardwareWatchdog)
    watchdog.reset_after = AsyncMock(return_value=True)

    events: List[bytes] = []
    async for chunk in orchestrate(
        session_id="sid-reset",
        prompt="hi",
        model_name="my-model",
        provider=provider,
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
        hw_watchdog=watchdog,
    ):
        events.append(chunk)

    watchdog.reset_after.assert_awaited_once()
    call_args = watchdog.reset_after.await_args
    # exc passed as positional, model_name as kwarg
    assert call_args.kwargs.get("model_name") == "my-model"
    # The exception should be the same RuntimeError raised by the provider.
    assert isinstance(call_args.args[0], RuntimeError)

    decoded = [json.loads(c.decode("utf-8").strip()) for c in events if c.strip()]
    info_events = [
        e for e in decoded
        if e.get("type") == "info"
        and "HardwareWatchdog" in e.get("data", {}).get("message", "")
    ]
    assert len(info_events) == 1, f"expected one info reset event, got: {info_events}"

    error_events = [e for e in decoded if e.get("type") == "error"]
    assert len(error_events) == 1
    assert error_events[0]["data"]["is_fatal"] is True


# ---------------------------------------------------------------------------
# auto_reload_on_fatal_error=False short-circuits reset_after.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orchestrator_skips_reset_after_when_disabled():
    """When auto_reload_on_fatal_error is False, reset_after is NOT called
    and the error event has is_fatal=False."""
    provider = _RaisingProvider(RuntimeError("model boom"))
    watchdog = MagicMock(spec=HardwareWatchdog)
    watchdog.reset_after = AsyncMock(return_value=True)

    events: List[bytes] = []
    async for chunk in orchestrate(
        session_id="sid-disabled",
        prompt="hi",
        model_name="my-model",
        provider=provider,
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools={},
        system_prompt="sys",
        config=_config(auto_reload_on_fatal_error=False),
        tool_runner=ToolRunner({}, timeout_sec=5),
        hw_watchdog=watchdog,
    ):
        events.append(chunk)

    watchdog.reset_after.assert_not_called()

    decoded = [json.loads(c.decode("utf-8").strip()) for c in events if c.strip()]
    error_events = [e for e in decoded if e.get("type") == "error"]
    assert len(error_events) == 1
    assert error_events[0]["data"]["is_fatal"] is False


# ---------------------------------------------------------------------------
# hw_watchdog=None: graceful skip, no AttributeError, no hasattr probe.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orchestrator_handles_no_watchdog():
    """orchestrate(..., hw_watchdog=None) gracefully skips recovery and still
    emits the error event with is_fatal=False."""
    provider = _RaisingProvider(RuntimeError("model boom"))

    events: List[bytes] = []
    async for chunk in orchestrate(
        session_id="sid-no-wd",
        prompt="hi",
        model_name="my-model",
        provider=provider,
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
        hw_watchdog=None,
    ):
        events.append(chunk)

    decoded = [json.loads(c.decode("utf-8").strip()) for c in events if c.strip()]
    error_events = [e for e in decoded if e.get("type") == "error"]
    assert len(error_events) == 1
    assert error_events[0]["data"]["is_fatal"] is False
    # No info-event mentioning HardwareWatchdog should appear.
    info_events = [
        e for e in decoded
        if e.get("type") == "info"
        and "HardwareWatchdog" in e.get("data", {}).get("message", "")
    ]
    assert info_events == []


# ---------------------------------------------------------------------------
# Wire-protocol shape preserved: error event always carries is_fatal.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_orchestrator_emits_error_event_with_is_fatal():
    """Wire-protocol contract: every error event from the streaming path has
    an is_fatal field. Two cases — recovered (True) and not (False) — both
    asserted in the same test for completeness."""
    # Case 1: watchdog says recovered → is_fatal=True.
    watchdog_yes = MagicMock(spec=HardwareWatchdog)
    watchdog_yes.reset_after = AsyncMock(return_value=True)
    provider1 = _RaisingProvider(RuntimeError("boom1"))
    events1: List[bytes] = []
    async for chunk in orchestrate(
        session_id="sid-1",
        prompt="hi",
        model_name="m",
        provider=provider1,
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
        hw_watchdog=watchdog_yes,
    ):
        events1.append(chunk)

    # Case 2: watchdog says NOT recovered → is_fatal=False.
    watchdog_no = MagicMock(spec=HardwareWatchdog)
    watchdog_no.reset_after = AsyncMock(return_value=False)
    provider2 = _RaisingProvider(RuntimeError("boom2"))
    events2: List[bytes] = []
    async for chunk in orchestrate(
        session_id="sid-2",
        prompt="hi",
        model_name="m",
        provider=provider2,
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools={},
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner({}, timeout_sec=5),
        hw_watchdog=watchdog_no,
    ):
        events2.append(chunk)

    def _error_payload(events):
        decoded = [
            json.loads(c.decode("utf-8").strip()) for c in events if c.strip()
        ]
        errs = [e for e in decoded if e.get("type") == "error"]
        assert len(errs) == 1, f"expected exactly one error event: {decoded}"
        return errs[0]["data"]

    p1 = _error_payload(events1)
    p2 = _error_payload(events2)
    assert "is_fatal" in p1
    assert "is_fatal" in p2
    assert p1["is_fatal"] is True
    assert p2["is_fatal"] is False
    # Other wire-protocol fields preserved.
    for p in (p1, p2):
        assert "message" in p
        assert "error_type" in p
        assert "recoverable" in p
