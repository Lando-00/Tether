"""ToolRunner — execute a tool by name with a hard timeout.

The timeout is injected at construction time (Phase 2 step 23 — config DI;
no more global config reads in business-logic modules). The default of 15s
mirrors ``LimitsSettings.tool_timeout_sec``'s default so that unit tests
can construct ``ToolRunner(tools)`` without plumbing settings.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from tether_service.core.interfaces import Tool


class ToolRunner:
    """Execute tools with a hard timeout."""

    def __init__(self, tools: Dict[str, Tool], *, timeout_sec: int = 15):
        self.tools = tools
        self.timeout = timeout_sec

    async def run(self, name: str, args: Dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        # Synthesis §6 row 4: orchestrator-facing API is invoke(args).
        # BaseTool.invoke unpacks the dict into kwargs for the typed run() method.
        return await asyncio.wait_for(tool.invoke(args), timeout=self.timeout)
