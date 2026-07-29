"""Tests for ``ToolRunner`` context plumbing.

Verifies that :meth:`ToolRunner.run` forwards a
:class:`ToolExecutionContext` to :meth:`Tool.invoke` and defaults to
``None`` when the orchestrator omits it.

Synthesis §4 Phase 4 step 41a; connector spec §4 footer.
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from tether.core.interfaces import Tool
from tether.core.types import ToolExecutionContext
from tether.protocol.orchestration.tool_runner import ToolRunner

# ---------------------------------------------------------------------------
# Signature inspection
# ---------------------------------------------------------------------------


class TestToolRunnerSignature:
    def test_tool_runner_run_accepts_context(self):
        """``ToolRunner.run`` must declare a keyword-only ``context``
        parameter defaulting to ``None`` (signature is forward-compatible
        with the orchestrator).
        """
        sig = inspect.signature(ToolRunner.run)
        params = sig.parameters
        assert "context" in params, (
            f"'context' not in ToolRunner.run signature: {list(params)}"
        )
        ctx_param = params["context"]
        assert ctx_param.kind is inspect.Parameter.KEYWORD_ONLY
        assert ctx_param.default is None


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


class TestToolRunnerPlumbsContext:
    async def test_tool_runner_passes_context_to_invoke(self):
        """When the caller supplies a context, ToolRunner forwards it
        verbatim to ``tool.invoke``.
        """
        mock_tool = MagicMock(spec=Tool)
        mock_tool.invoke = AsyncMock(return_value={"ok": True})
        runner = ToolRunner({"my_tool": mock_tool}, timeout_sec=5)

        ctx = ToolExecutionContext(
            session_id="sid-7",
            turn_id="turn-7",
            last_user_message="please",
            user_confirmed_send=False,
        )
        result = await runner.run("my_tool", {"x": 1}, context=ctx)

        mock_tool.invoke.assert_called_once_with({"x": 1}, context=ctx)
        assert result == {"ok": True}

    async def test_tool_runner_no_context_passes_none(self):
        """When the caller omits ``context``, ``tool.invoke`` is called
        with ``context=None`` (preserves backward compat).
        """
        mock_tool = MagicMock(spec=Tool)
        mock_tool.invoke = AsyncMock(return_value={"ok": True})
        runner = ToolRunner({"my_tool": mock_tool}, timeout_sec=5)

        await runner.run("my_tool", {"x": 1})

        mock_tool.invoke.assert_called_once_with({"x": 1}, context=None)

    async def test_tool_runner_unknown_tool_raises_before_context_use(self):
        """Unknown tool name still raises ``ValueError`` regardless of
        context."""
        runner = ToolRunner({}, timeout_sec=5)
        ctx = ToolExecutionContext(session_id="s", turn_id="t")
        with pytest.raises(ValueError, match="not found"):
            await runner.run("missing", {}, context=ctx)


# ---------------------------------------------------------------------------
# Backward compat: existing call shape still works
# ---------------------------------------------------------------------------


class TestToolRunnerBackwardCompat:
    async def test_tool_runner_existing_positional_call(self):
        """``runner.run(name, args)`` (no context kwarg) still works for
        all existing call sites.
        """
        mock_tool = MagicMock(spec=Tool)
        mock_tool.invoke = AsyncMock(return_value={"answer": 42})
        runner = ToolRunner({"t": mock_tool}, timeout_sec=5)

        result = await runner.run("t", {"q": "?"})
        assert result == {"answer": 42}
