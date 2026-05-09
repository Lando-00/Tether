"""ToolRunner — execute a tool by name with a hard timeout.

The timeout is injected at construction time (Phase 2 step 23 — config DI;
no more global config reads in business-logic modules). The default of 15s
mirrors ``LimitsSettings.tool_timeout_sec``'s default so that unit tests
can construct ``ToolRunner(tools)`` without plumbing settings.

Phase 4 step 41a: ``run`` accepts an optional ``context``
(:class:`ToolExecutionContext`) keyword arg, threaded from the
orchestrator through to :meth:`BaseTool.invoke`. Connector tools shipping
in Phase 4.5+ consume ``context.user_confirmed_send`` for the
draft+confirm send-safety pattern. Synthesis §4 Phase 4 step 41a;
connector spec §4 footer.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, TYPE_CHECKING

from tether_service.core.interfaces import Tool

if TYPE_CHECKING:
    from tether_service.core.types import ToolExecutionContext


class ToolRunner:
    """Execute tools with a hard timeout."""

    def __init__(self, tools: Dict[str, Tool], *, timeout_sec: int = 15):
        self.tools = tools
        self.timeout = timeout_sec

    async def run(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        context: Optional["ToolExecutionContext"] = None,
    ) -> Any:
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        # Synthesis §6 row 4: orchestrator-facing API is invoke(args).
        # BaseTool.invoke unpacks the dict into kwargs for the typed run() method.
        # Phase 4 step 41a: forward the per-tool-call ToolExecutionContext.
        return await asyncio.wait_for(
            tool.invoke(args, context=context),
            timeout=self.timeout,
        )
