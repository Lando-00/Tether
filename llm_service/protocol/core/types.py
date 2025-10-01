from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenDelta:
    token: Optional[str]
    finish_reason: Optional[str]  # "stop", "length", "tool_calls", etc.

@dataclass
class RoutedChunk:
    """
    Represents a token routed to detector, UI visible stream, and UI hidden stream.
    """
    to_detector: str
    to_ui_visible: str
    to_ui_hidden: str
