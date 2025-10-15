# llm_service/protocol/service/helpers/event_logger.py

from __future__ import annotations
from typing import Tuple, Dict, Any

class EventLogger:
    """
    Tiny helper that logs stream events and returns booleans indicating
    whether the current event is a tool_start or tool_end.
    It keeps NO state; the caller accumulates flags across the stream.
    """

    def __init__(self, logger):
        self.logger = logger

    def log(self, event: Dict[str, Any], count: int) -> Tuple[bool, bool]:
        """
        Log the event and return (is_tool_start, is_tool_end).

        Never raises on missing fields. Never mutates external state.
        """
        t = (event or {}).get("type", "unknown")
        if t == "tool_start":
            # optional details for debugging
            name = event.get("name")
            eid = event.get("id")
            self.logger.info(f"[{count}] tool_start name={name!r} id={eid!r}")
            return True, False

        if t == "tool_end":
            name = event.get("name")
            eid = event.get("id")
            self.logger.info(f"[{count}] tool_end   name={name!r} id={eid!r}")
            return False, True

        # keep non-tool events at debug to avoid log spam
        self.logger.debug(f"[{count}] event type={t}")
        return False, False
