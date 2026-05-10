"""Tool spans (tool.start / tool.end / tool.error) emitted by ToolRunner.

Phase 7 step 71. Verifies the structured log events appear with the
right fields across success, timeout, cancellation, and execution-error
paths.

Uses ``structlog.testing.capture_logs()`` which intercepts events at the
BoundLoggerLazyProxy level — cleaner than caplog for structlog unit tests
because bound context (turn_id, request_id) and kwarg fields are captured
as dict keys rather than as LogRecord attributes.
"""
from __future__ import annotations

import asyncio

import pytest
from structlog.testing import capture_logs

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Minimal tool stubs — no real BaseTool needed; ToolRunner only calls .invoke
# ---------------------------------------------------------------------------

class _SuccessTool:
    schema = {"name": "success", "parameters": {"type": "object"}}

    async def invoke(self, args, context=None):
        return {"data": "ok"}


class _SlowTool:
    schema = {"name": "slow", "parameters": {"type": "object"}}

    async def invoke(self, args, context=None):
        await asyncio.sleep(2.0)
        return {"data": "ok"}


class _RaisingTool:
    schema = {"name": "raises", "parameters": {"type": "object"}}

    async def invoke(self, args, context=None):
        raise ValueError("kaboom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_tool_start_and_end_logged_on_success():
    """Successful tool dispatch emits tool.start AND tool.end."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"success": _SuccessTool()}, timeout_sec=5)

    with capture_logs() as cap:
        result = await runner.run("success", {"x": 1})

    assert result == {"data": "ok"}
    events = [log["event"] for log in cap]
    assert "tool.start" in events
    assert "tool.end" in events
    assert "tool.error" not in events


async def test_tool_start_includes_args_redacted():
    """tool.start log includes args_redacted truncated to ≤214 chars."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"success": _SuccessTool()}, timeout_sec=5)

    with capture_logs() as cap:
        await runner.run("success", {"long": "x" * 500})

    start_logs = [log for log in cap if log.get("event") == "tool.start"]
    assert len(start_logs) == 1
    args_redacted = start_logs[0].get("args_redacted")
    assert args_redacted is not None
    # 200 chars + "...[truncated]" (14 chars) = 214 max
    assert len(args_redacted) <= 214


async def test_tool_start_includes_args_size_bytes():
    """tool.start log includes args_size_bytes (positive int)."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"success": _SuccessTool()}, timeout_sec=5)

    with capture_logs() as cap:
        await runner.run("success", {"key": "value"})

    start_logs = [log for log in cap if log.get("event") == "tool.start"]
    assert len(start_logs) == 1
    size = start_logs[0].get("args_size_bytes")
    assert isinstance(size, int) and size > 0


async def test_tool_end_includes_duration_ms():
    """tool.end log includes duration_ms (non-negative integer)."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"success": _SuccessTool()}, timeout_sec=5)

    with capture_logs() as cap:
        await runner.run("success", {})

    end_logs = [log for log in cap if log.get("event") == "tool.end"]
    assert len(end_logs) == 1
    duration_ms = end_logs[0].get("duration_ms")
    assert isinstance(duration_ms, int) and duration_ms >= 0


async def test_tool_end_includes_result_size_bytes():
    """tool.end log includes result_size_bytes."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"success": _SuccessTool()}, timeout_sec=5)

    with capture_logs() as cap:
        await runner.run("success", {})

    end_logs = [log for log in cap if log.get("event") == "tool.end"]
    assert len(end_logs) == 1
    result_size = end_logs[0].get("result_size_bytes")
    assert isinstance(result_size, int) and result_size >= 0


async def test_tool_error_logged_on_exception():
    """When tool raises, tool.error is logged with error_kind='execution'."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"raises": _RaisingTool()}, timeout_sec=5)

    with capture_logs() as cap:
        with pytest.raises(ValueError, match="kaboom"):
            await runner.run("raises", {})

    error_logs = [log for log in cap if log.get("event") == "tool.error"]
    assert len(error_logs) == 1
    log = error_logs[0]
    assert log.get("error_kind") == "execution"
    assert "kaboom" in log.get("error_message", "")
    assert isinstance(log.get("duration_ms"), int)
    assert "tool.end" not in [l["event"] for l in cap]


async def test_tool_error_logged_on_timeout():
    """When tool times out, tool.error is logged with error_kind='timeout'."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"slow": _SlowTool()}, timeout_sec=1)

    with capture_logs() as cap:
        with pytest.raises(asyncio.TimeoutError):
            await runner.run("slow", {})

    error_logs = [log for log in cap if log.get("event") == "tool.error"]
    assert len(error_logs) == 1
    log = error_logs[0]
    assert log.get("error_kind") == "timeout"
    assert isinstance(log.get("duration_ms"), int)


async def test_tool_call_id_propagated_in_spans():
    """tool_call_id kwarg appears in all emitted span events."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"success": _SuccessTool()}, timeout_sec=5)
    test_id = "tcid-abc-123"

    with capture_logs() as cap:
        await runner.run("success", {}, tool_call_id=test_id)

    for log in cap:
        if log.get("event") in ("tool.start", "tool.end"):
            assert log.get("tool_call_id") == test_id


async def test_tool_call_id_none_when_omitted():
    """tool_call_id is None when not provided — no crash."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"success": _SuccessTool()}, timeout_sec=5)

    with capture_logs() as cap:
        await runner.run("success", {})  # no tool_call_id kwarg

    start_logs = [log for log in cap if log.get("event") == "tool.start"]
    assert len(start_logs) == 1
    assert start_logs[0].get("tool_call_id") is None


async def test_args_redacted_scrubs_bearer_tokens():
    """args_redacted redacts Bearer tokens via redact_text()."""
    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner({"success": _SuccessTool()}, timeout_sec=5)

    with capture_logs() as cap:
        await runner.run(
            "success", {"auth": "Bearer secret_token_xxxxxxxxxxxxxxx"}
        )

    start_logs = [log for log in cap if log.get("event") == "tool.start"]
    assert len(start_logs) == 1
    args_redacted = start_logs[0].get("args_redacted", "")
    assert "secret_token_xxxxxxxxxxxxxxx" not in args_redacted


async def test_tool_size_cap_logs_tool_error():
    """When result exceeds size cap, tool.error is logged (not tool.end)."""

    class _OversizeTool:
        schema = {"name": "oversize", "parameters": {"type": "object"}}

        async def invoke(self, args, context=None):
            return {"data": "x" * (300 * 1024)}

    from tether_service.protocol.orchestration.tool_runner import ToolRunner

    runner = ToolRunner(
        {"oversize": _OversizeTool()},
        timeout_sec=5,
        result_max_bytes=10 * 1024,
    )

    with capture_logs() as cap:
        result = await runner.run("oversize", {})

    assert result.get("error") == "tool_result_oversized"

    error_logs = [log for log in cap if log.get("event") == "tool.error"]
    assert len(error_logs) >= 1
    log = error_logs[0]
    assert log.get("error_kind") == "execution"
    assert isinstance(log.get("duration_ms"), int)
    # tool.end must NOT appear when result is oversized
    assert "tool.end" not in [l["event"] for l in cap]
