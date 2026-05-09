"""Tests for ``BaseTool.invoke`` context dispatch.

Verifies the inspect-based dispatcher in :meth:`BaseTool.invoke`:

- Existing tools (no ``context`` param in ``run``) keep working unchanged
- Tools that declare a ``context`` keyword arg on ``run`` receive it
- Tool ABC's ``invoke`` accepts ``context`` as keyword-only with a ``None``
  default

Synthesis §4 Phase 4 step 41a; connector spec §4 footer.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import pytest

from tether_service.core.interfaces import Tool
from tether_service.core.types import ToolExecutionContext
from tether_service.tools.base import BaseTool
from tether_service.tools.time_tool import TimeTool


# ---------------------------------------------------------------------------
# Fixture connector tool — illustrates Phase 4.5+ pattern
# ---------------------------------------------------------------------------


class _CtxAwareTool(BaseTool):
    """Fixture tool whose run() declares a keyword-only ``context`` param.

    Mirrors how a Phase 4.5+ WhatsApp/Gmail connector tool would look:
    the dispatcher in :meth:`BaseTool.invoke` detects ``context`` in the
    signature and passes it through.
    """

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "ctx_aware",
                "description": "fixture",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    async def run(
        self,
        message: str = "",
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Dict[str, Any]:
        return {
            "message": message,
            "got_context": context is not None,
            "session_id": context.session_id if context else None,
            "user_confirmed_send": (
                context.user_confirmed_send if context else None
            ),
        }


# ---------------------------------------------------------------------------
# Existing tools ignore context (backward compat)
# ---------------------------------------------------------------------------


class TestExistingToolsIgnoreContext:
    async def test_invoke_without_context_existing_tool(self):
        """TimeTool's run signature has no ``context`` param — invoke must
        accept ``context=None`` (and omitted) without breaking."""
        tool = TimeTool()

        # Both forms must succeed
        r1 = await tool.invoke({"timezone": "UTC"})
        r2 = await tool.invoke({"timezone": "UTC"}, context=None)
        ctx = ToolExecutionContext(session_id="s", turn_id="t")
        r3 = await tool.invoke({"timezone": "UTC"}, context=ctx)

        for r in (r1, r2, r3):
            assert "time" in r, f"expected 'time' key in {r}"

    async def test_invoke_existing_tool_with_context_does_not_pass_it(self):
        """When ``run`` doesn't declare ``context``, the dispatcher MUST NOT
        pass it through (TypeError would otherwise leak)."""
        tool = TimeTool()
        ctx = ToolExecutionContext(session_id="s", turn_id="t")
        # Just must not raise
        result = await tool.invoke({"timezone": "UTC"}, context=ctx)
        assert "time" in result


# ---------------------------------------------------------------------------
# Context-aware tools receive the context
# ---------------------------------------------------------------------------


class TestContextAwareToolsReceiveContext:
    async def test_invoke_with_context_dispatch(self):
        """A tool whose run() accepts ``context`` receives the value passed
        to invoke()."""
        tool = _CtxAwareTool()
        ctx = ToolExecutionContext(
            session_id="sid-42",
            turn_id="turn-xyz",
            last_user_message="hi there",
            user_confirmed_send=False,
        )
        result = await tool.invoke({"message": "hello"}, context=ctx)
        assert result["got_context"] is True
        assert result["session_id"] == "sid-42"
        assert result["user_confirmed_send"] is False
        assert result["message"] == "hello"

    async def test_invoke_context_aware_with_no_context(self):
        """When invoke is called WITHOUT a context, a context-aware tool
        receives ``context=None`` (the kwarg default on the shim)."""
        tool = _CtxAwareTool()
        result = await tool.invoke({"message": "hi"})
        assert result["got_context"] is False
        assert result["session_id"] is None


# ---------------------------------------------------------------------------
# ABC signature
# ---------------------------------------------------------------------------


class TestToolABCSignature:
    def test_invoke_signature_accepts_optional_context(self):
        """``Tool.invoke`` must declare ``context`` as a keyword-only
        parameter with a default of ``None``."""
        sig = inspect.signature(Tool.invoke)
        params = sig.parameters
        assert "context" in params, (
            f"'context' not in Tool.invoke parameters: {list(params)}"
        )
        ctx_param = params["context"]
        assert ctx_param.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"'context' must be keyword-only, got {ctx_param.kind}"
        )
        assert ctx_param.default is None, (
            f"'context' must default to None, got {ctx_param.default!r}"
        )

    def test_basetool_invoke_signature_accepts_optional_context(self):
        """:meth:`BaseTool.invoke` mirrors the ABC signature."""
        sig = inspect.signature(BaseTool.invoke)
        params = sig.parameters
        assert "context" in params
        ctx_param = params["context"]
        assert ctx_param.kind is inspect.Parameter.KEYWORD_ONLY
        assert ctx_param.default is None
