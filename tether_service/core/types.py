from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict, Dict, Any, Optional

if TYPE_CHECKING:
    from tether_service.config.settings import Settings


class StreamEvent(StrEnum):
    TEXT = "text"
    THINK = "think"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETE = "tool_complete"
    ERROR = "error"
    DONE = "done"


class Event(TypedDict, total=False):
    type: str  # "text" | "think_stream" | "tool_started" | "tool_complete" | "error" | "done"
    session_id: str
    data: Dict[str, Any]
    ts: str


@dataclass(frozen=True)
class OrchestratorConfig:
    """Typed slice of Settings consumed by the orchestrator.

    Built once (typically by ``Engine.from_settings``) and passed to
    ``orchestrate()`` via dependency injection. Frozen so the orchestrator
    cannot mutate it. Per _synthesis.md §4 Phase 2 step 23 (kill legacy
    config dict reads from business logic).
    """

    max_tool_loops: int
    auto_reload_on_fatal_error: bool
    save_thinking: bool
    include_thinking_in_history: bool

    @classmethod
    def from_settings(cls, settings: "Settings") -> "OrchestratorConfig":
        return cls(
            max_tool_loops=settings.limits.max_tool_loops,
            auto_reload_on_fatal_error=settings.limits.auto_reload_on_fatal_error,
            save_thinking=settings.context.save_thinking,
            include_thinking_in_history=settings.context.include_thinking_in_history,
        )


@dataclass(frozen=True)
class ToolExecutionContext:
    """Per-tool-call context threaded from the orchestrator through
    :class:`ToolRunner` to :meth:`BaseTool.invoke`.

    Allows tools (especially Phase 4.5+ connector tools — WhatsApp, Gmail)
    to inspect surrounding turn state without storing it on ``self``.

    Fields:
        session_id: The chat session driving this tool call. Lets tools
            tag artifacts (audit logs, persisted state) with the right
            session.
        turn_id: A short identifier for the current ``orchestrate()`` call.
            Stable across all tool-loop iterations within one user→
            assistant exchange. Useful for de-duplicating retries.
        last_user_message: The most recent user-facing message text, if
            any. Connectors use this to verify "the user just asked for
            this" before performing destructive operations.
        user_confirmed_send: Whether the user has explicitly confirmed an
            outbound action (draft+confirm pattern). **Always ``False``
            in this refactor**; the regex classifier that flips it ships
            with the WhatsApp/Gmail connectors in subsequent sessions.
            Synthesis §10.8 #4; connector spec §4 footer.

    Synthesis §4 Phase 4 step 41a; connector spec §4.
    """

    session_id: str
    turn_id: str
    last_user_message: Optional[str] = None
    user_confirmed_send: bool = False