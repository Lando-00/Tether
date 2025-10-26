from enum import StrEnum
from typing import TypedDict, Dict, Any


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