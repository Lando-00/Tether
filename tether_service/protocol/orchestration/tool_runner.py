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

Phase 7 step 77: ``result_max_bytes`` cap (default 256 KB). Results that
serialise to more bytes are replaced with a structured error dict so the
model can react and retry with smaller args. No exception raised; the
orchestrator's FEED_BACK_TO_MODEL policy handles it.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, TYPE_CHECKING

from tether_service.core.interfaces import Tool

if TYPE_CHECKING:
    from tether_service.core.types import ToolExecutionContext


class ToolRunner:
    """Execute tools with a hard timeout and a result-size cap."""

    def __init__(
        self,
        tools: Dict[str, Tool],
        *,
        timeout_sec: int = 15,
        result_max_bytes: int = 256 * 1024,
    ):
        self.tools = tools
        self.timeout = timeout_sec
        self.result_max_bytes = result_max_bytes

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
        result = await asyncio.wait_for(
            tool.invoke(args, context=context),
            timeout=self.timeout,
        )

        # Phase 7 step 77: reject oversized results before they hit persistence
        # and history reconstruction. Graceful — returns a structured error dict
        # that the model can read and react to (FEED_BACK_TO_MODEL policy).
        size_bytes = self._measure_result_size(result)
        if size_bytes > self.result_max_bytes:
            return {
                "error": "tool_result_oversized",
                "tool_name": name,
                "size_bytes": size_bytes,
                "limit_bytes": self.result_max_bytes,
            }

        return result

    @staticmethod
    def _measure_result_size(result: Any) -> int:
        """Approximate byte size of result via JSON serialisation.

        Tool results are persisted as JSON; this is the canonical measurement.
        Falls back to ``repr`` length for non-serialisable objects.
        """
        try:
            return len(json.dumps(result, default=str))
        except (TypeError, ValueError):
            return len(repr(result))
