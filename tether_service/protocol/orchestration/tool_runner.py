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

Phase 7 step 71: structured log spans (tool.start / tool.end / tool.error)
emitted via structlog for every tool dispatch. Fields include args_redacted
(redacted + truncated repr), args_size_bytes, duration_ms, result_size_bytes,
error_kind, and error_message. Synthesis §3 (observability).

Phase 7 step 77: ``result_max_bytes`` cap (default 256 KB). Results that
serialise to more bytes are replaced with a structured error dict so the
model can react and retry with smaller args. No exception raised; the
orchestrator's FEED_BACK_TO_MODEL policy handles it.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

import structlog

from tether_service.core.interfaces import Tool
from tether_service.core.redact import redact_text

if TYPE_CHECKING:
    from tether_service.core.types import ToolExecutionContext

_ARGS_REDACTED_MAX_CHARS = 200
_ERROR_MESSAGE_MAX_CHARS = 500


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
        tool_call_id: Optional[str] = None,
    ) -> Any:
        # Get a fresh logger proxy each call so structlog.testing.capture_logs()
        # can intercept events even when cache_logger_on_first_use=True is set
        # by configure_logging() in other tests. Phase 7 step 71.
        log = structlog.get_logger(__name__)
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        # Phase 7 step 71: build args_redacted before invoking so we always
        # log what was sent, even if the tool raises immediately.
        args_repr_raw = repr(args)
        args_redacted = redact_text(args_repr_raw)
        if len(args_redacted) > _ARGS_REDACTED_MAX_CHARS:
            args_redacted = args_redacted[:_ARGS_REDACTED_MAX_CHARS] + "...[truncated]"
        args_size_bytes = len(args_repr_raw.encode("utf-8"))

        log.info(
            "tool.start",
            tool_name=name,
            tool_call_id=tool_call_id,
            args_size_bytes=args_size_bytes,
            args_redacted=args_redacted,
        )

        start = time.monotonic()

        try:
            # Synthesis §6 row 4: orchestrator-facing API is invoke(args).
            # BaseTool.invoke unpacks the dict into kwargs for the typed run() method.
            # Phase 4 step 41a: forward the per-tool-call ToolExecutionContext.
            result = await asyncio.wait_for(
                tool.invoke(args, context=context),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.error(
                "tool.error",
                tool_name=name,
                tool_call_id=tool_call_id,
                error_kind="timeout",
                error_message=f"timed out after {self.timeout}s",
                duration_ms=duration_ms,
            )
            raise
        except asyncio.CancelledError:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.warning(
                "tool.error",
                tool_name=name,
                tool_call_id=tool_call_id,
                error_kind="cancelled",
                error_message="task cancelled",
                duration_ms=duration_ms,
            )
            raise
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.error(
                "tool.error",
                tool_name=name,
                tool_call_id=tool_call_id,
                error_kind="execution",
                error_message=str(exc)[:_ERROR_MESSAGE_MAX_CHARS],
                duration_ms=duration_ms,
            )
            raise

        duration_ms = int((time.monotonic() - start) * 1000)

        # Phase 7 step 77: reject oversized results before they hit persistence
        # and history reconstruction. Graceful — returns a structured error dict
        # that the model can read and react to (FEED_BACK_TO_MODEL policy).
        size_bytes = self._measure_result_size(result)
        if size_bytes > self.result_max_bytes:
            log.warning(
                "tool.error",
                tool_name=name,
                tool_call_id=tool_call_id,
                error_kind="execution",
                error_message="tool_result_oversized",
                duration_ms=duration_ms,
                result_size_bytes=size_bytes,
                result_limit_bytes=self.result_max_bytes,
            )
            return {
                "error": "tool_result_oversized",
                "tool_name": name,
                "size_bytes": size_bytes,
                "limit_bytes": self.result_max_bytes,
            }

        log.info(
            "tool.end",
            tool_name=name,
            tool_call_id=tool_call_id,
            duration_ms=duration_ms,
            result_size_bytes=size_bytes,
        )

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
