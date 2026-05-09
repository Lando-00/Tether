from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict, Dict, Any

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
    cannot mutate it. Per _synthesis.md §4 Phase 2 step 23 (kill
    ``load_settings_legacy()`` from business logic).
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