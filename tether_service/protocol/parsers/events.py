"""Parser-level events (internal to the parser; NOT wire-facing).

These are produced by ``SlidingParser`` (the Phase 5
``p5-parser-typed-events`` step will refactor ``sliding.py`` to return
these). The orchestrator (``p5-orchestrator-wire-policies``) consumes
them and translates to :data:`tether_service.protocol.wire.events.WireEvent`.

Frozen dataclasses (NOT Pydantic) — these are tight, internal types
that don't need validation or JSON serialization.

Synthesis §4 Phase 5 steps 49 + 51.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PText:
    """Text delta from parser."""

    text: str


@dataclass(frozen=True)
class PThink:
    """Thinking-channel delta from parser."""

    text: str


@dataclass(frozen=True)
class PToolCallDetected:
    """Parser spotted the ``<<function_call>>`` marker; payload incoming.

    Emitted before :class:`PToolCallParsed` so the orchestrator can emit
    a placeholder if needed.
    """


@dataclass(frozen=True)
class PToolCallParsed:
    """Parser fully parsed the JSON payload after ``<<function_call>>``."""

    tool_call_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PParseError:
    """Parser hit malformed JSON or other parse-time error."""

    message: str
    raw: Optional[str] = None


@dataclass(frozen=True)
class PStreamEnd:
    """Provider stream ended. Parser is being finalized."""


__all__ = [
    "PText",
    "PThink",
    "PToolCallDetected",
    "PToolCallParsed",
    "PParseError",
    "PStreamEnd",
]
