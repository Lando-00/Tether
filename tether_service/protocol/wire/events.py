"""Wire-protocol typed events for Tether's NDJSON streaming API.

Phase 5 foundation. Existing v0 wire vocabulary (``text``, ``tool_started``,
``tool_completed``, ``done``, etc.) continues to ship via the legacy emitter
during the cutover. ``p5-cutover-a`` (dual-emit) will introduce content
negotiation; ``p5-cutover-c`` flips the default to v2.

``PROTOCOL_VERSION = "1.0"`` is the version handshake exposed via the
``Accept`` header during the dual-emit phase.

Synthesis §4 Phase 5 step 49; §11.3 R1 (``stop_reason`` includes
``cancelled`` + ``client_disconnect``); §11.3 R6 (capabilities-denied uses
``ToolResult`` with ``error_kind="permission"`` + ``missing_capabilities``).
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


PROTOCOL_VERSION = "1.0"


class _Base(BaseModel):
    """Common envelope fields for every :data:`WireEvent`.

    All v2 events share these. The ``type`` discriminator is on each
    concrete subclass.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = PROTOCOL_VERSION
    session_id: str
    turn_id: str
    seq: int = Field(ge=0, description="Per-turn monotonically-increasing sequence number")
    ts: datetime = Field(description="Server-side emission timestamp (UTC)")


class ToolDescriptor(BaseModel):
    """Tool descriptor used in :class:`MessageStart` ``available_tools``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


# --- WireEvent variants -----------------------------------------------------


class MessageStart(_Base):
    """Start of an assistant turn. Emitted before any text/tool events."""

    type: Literal["message_start"] = "message_start"
    available_tools: List[ToolDescriptor] = Field(default_factory=list)


class MessageStop(_Base):
    """End of an assistant turn. Always emitted last (one per turn).

    ``stop_reason`` values per synthesis §11.3 R1:

    ``complete``
        Model produced a final answer.
    ``tool_loop_exhausted``
        ``max_tool_loops`` hit (``LoopLimitPolicy``).
    ``cancelled``
        Orchestrator's ``cancel_event`` was set (or the library user
        called ``engine.aclose`` mid-stream).
    ``client_disconnect``
        HTTP client closed the connection.
    ``error``
        Recoverable error during the turn.
    """

    type: Literal["message_stop"] = "message_stop"
    stop_reason: Literal[
        "complete",
        "tool_loop_exhausted",
        "cancelled",
        "client_disconnect",
        "error",
    ]


class TextDelta(_Base):
    """Assistant text token (incremental)."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(_Base):
    """Reasoning channel token (incremental). Used by models that emit
    thinking separately (e.g., R1, future Tether-supported reasoning
    models)."""

    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class ToolCall(_Base):
    """Model decides to call a tool. Followed by :class:`ToolResult` after
    execution."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(_Base):
    """Result of a tool call. Always paired with a preceding
    :class:`ToolCall`.

    ``error_kind`` values per synthesis §11.3 R6:

    ``permission``
        Capability denied (``missing_capabilities`` populated).
    ``execution``
        Tool raised during run.
    ``timeout``
        Tool exceeded timeout budget.
    ``cancelled``
        Tool task was cancelled mid-run.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["ok", "error"]
    error_kind: Optional[Literal["permission", "execution", "timeout", "cancelled"]] = None
    missing_capabilities: List[str] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Error(_Base):
    """Non-recoverable error during the turn."""

    type: Literal["error"] = "error"
    message: str
    error_type: str
    is_fatal: bool = False


class LoopLimitReached(_Base):
    """Tool loop hit ``max_tool_loops``. Emitted by
    ``LoopLimitPolicy.EMIT_LIMIT_EVENT`` (the default per the ratified
    plan)."""

    type: Literal["loop_limit_reached"] = "loop_limit_reached"
    loops: int


class HwReset(_Base):
    """``HardwareWatchdog`` performed an ``hw_reset`` on the model.
    Informational; follow-up retry is the orchestrator's choice."""

    type: Literal["hw_reset"] = "hw_reset"
    model_name: str


WireEvent = Annotated[
    Union[
        MessageStart,
        MessageStop,
        TextDelta,
        ThinkingDelta,
        ToolCall,
        ToolResult,
        Error,
        LoopLimitReached,
        HwReset,
    ],
    Field(discriminator="type"),
]


__all__ = [
    "PROTOCOL_VERSION",
    "_Base",
    "ToolDescriptor",
    "MessageStart",
    "MessageStop",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "Error",
    "LoopLimitReached",
    "HwReset",
    "WireEvent",
]
