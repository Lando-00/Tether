"""Loop-limit and tool-error policy enums.

Synthesis §3.5 (orchestrator contract); user-ratified defaults:

  - :class:`LoopLimitPolicy` defaults to :attr:`EMIT_LIMIT_EVENT`.
  - :class:`ToolErrorPolicy` defaults to :attr:`FEED_BACK_TO_MODEL`
    (synthesis §3.5; A5 P2).

The tool-error policy change is wire-visible: under
``FEED_BACK_TO_MODEL`` a tool error no longer breaks the loop — the
orchestrator persists the error as a ``tool_result`` row and continues so
the model can react. Old ``BREAK_LOOP`` behaviour stays available for
deterministic single-turn tests.

Loop-limit policy:

  - :attr:`RAISE` — orchestrator raises
    :class:`tether.core.errors.LoopLimitReached`. Caller decides.
  - :attr:`EMIT_LIMIT_EVENT` — emit a ``LoopLimitReached`` wire event
    plus a ``MessageStop(stop_reason="tool_loop_exhausted")`` and exit
    cleanly.
"""
from __future__ import annotations

from enum import StrEnum


class LoopLimitPolicy(StrEnum):
    """What the orchestrator does when ``max_tool_loops`` is reached."""

    RAISE = "raise"
    EMIT_LIMIT_EVENT = "emit_limit_event"


class ToolErrorPolicy(StrEnum):
    """What the orchestrator does when a tool raises during execution."""

    BREAK_LOOP = "break_loop"
    FEED_BACK_TO_MODEL = "feed_back_to_model"


__all__ = ["LoopLimitPolicy", "ToolErrorPolicy"]
