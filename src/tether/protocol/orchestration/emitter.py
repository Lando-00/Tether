"""WireEvent ↔ NDJSON-bytes serializers + legacy dict emitter shim.

Two paths:

  - :func:`v0_compat_serialize` — Translates a typed
    :data:`tether.protocol.wire.events.WireEvent` into the
    legacy v0 NDJSON dict bytes (e.g., ``MessageStop`` →
    ``{"type": "done", ...}``, ``TextDelta`` → ``{"type": "text", ...}``).
    This keeps the bytes wire UNCHANGED while the orchestrator switches
    to typed yield. ``p5-cutover-c`` drops it.
  - :class:`NdjsonEmitter` — legacy dict→bytes shim (back-compat for
    the still-extant ``orchestrate()`` function during the Phase 5
    transition). Once :class:`Orchestrator` is the only caller, this
    class is replaced with a 4-line WireEvent wrapper (the v2 path used
    by :meth:`Engine.chat` consumers and by the SSE transport in
    ``p5-cutover-a-dual-emit``).

Synthesis §3.4 (streaming), §11.3 R1 (stop_reason vocabulary), §3.5
(orchestrator contract).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from tether.protocol.wire.events import (
    Error,
    HwReset,
    LoopLimitReached,
    MessageStart,
    MessageStop,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    WireEvent,
)


class NdjsonEmitter:
    """Legacy dict-shape NDJSON emitter (transitional).

    Used by the still-extant ``orchestrate()`` function during the Phase
    5 transition. After the orchestrator is fully migrated to the
    :class:`Orchestrator` class (which yields :class:`WireEvent` directly
    and routes through :func:`v0_compat_serialize`), this shim is no
    longer needed and ``p5-cutover-c`` drops the dict path entirely.
    """

    def emit(self, event: Dict[str, Any]) -> bytes:
        # A4 PAIN-3: timezone-aware datetime so the ISO 8601 string is
        # unambiguous (+00:00 suffix); naive UTC is deprecated in 3.12.
        out = {
            "type": event.get("type", ""),
            "session_id": event.get("session_id", ""),
            "data": event.get("data", {}),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        return (json.dumps(out) + "\n").encode("utf-8")


def v0_compat_serialize(wire_event: WireEvent) -> bytes:
    """Translate a typed :class:`WireEvent` into legacy v0 NDJSON dict bytes.

    Preserves the exact wire vocabulary the existing tests assert
    against (modulo timestamps). Mapping:

      ``MessageStart``      → ``b""`` (legacy didn't emit this; absorbed)
      ``MessageStop``       → ``{"type": "done", "data": {}}`` regardless of
                              ``stop_reason`` (legacy was reason-agnostic).
      ``TextDelta``         → ``{"type": "text", "data": {"delta": text}}``
      ``ThinkingDelta``     → ``{"type": "think", "data": {"delta": text}}``
      ``ToolCall``          → ``{"type": "tool_started",
                                 "data": {"tool_name": name, "tool_args": arguments}}``
      ``ToolResult`` (ok)   → ``{"type": "tool_completed",
                                 "data": {"tool_name": name, "tool_result": result}}``
      ``ToolResult`` (err)  → ``{"type": "tool_error",
                                 "data": {"tool_name": name, "error": error}}``
      ``Error``             → ``{"type": "error",
                                 "data": {"message": message,
                                          "error_type": error_type,
                                          "is_fatal": is_fatal,
                                          "recoverable": False}}``
      ``LoopLimitReached``  → ``{"type": "loop_limit_reached",
                                 "data": {"loops": loops}}`` (legacy was silent;
                              new policy default emits this so clients have a
                              signal — existing tests don't cover loop limit).
      ``HwReset``           → ``{"type": "info",
                                 "data": {"message": "Model '<name>' was reset by
                                          HardwareWatchdog after fatal error"}}``

    The legacy ``tool_marker_detected`` v0 event has NO :class:`WireEvent`
    equivalent — :class:`PToolCallDetected` (parser-internal) is consumed by
    the orchestrator without emitting a wire event. Existing tests / golden
    fixtures that asserted it must be updated.

    All output ends with ``b"\\n"`` (NDJSON line terminator). Returns
    ``b""`` for events that don't have a v0 equivalent so the caller
    can ``if bytes_out: yield bytes_out``.

    Synthesis §3.4 streaming, §11.3 R1.
    """
    if isinstance(wire_event, MessageStart):
        # Legacy had no message_start event; emit nothing.
        return b""

    payload: Dict[str, Any] = {
        "type": "",
        "session_id": wire_event.session_id,
        "data": {},
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    if isinstance(wire_event, MessageStop):
        payload["type"] = "done"
        # stop_reason is absorbed — legacy emitted unconditional done {}.

    elif isinstance(wire_event, TextDelta):
        payload["type"] = "text"
        payload["data"] = {"delta": wire_event.text}

    elif isinstance(wire_event, ThinkingDelta):
        payload["type"] = "think"
        payload["data"] = {"delta": wire_event.text}

    elif isinstance(wire_event, ToolCall):
        payload["type"] = "tool_started"
        payload["data"] = {
            "tool_name": wire_event.name,
            "tool_args": dict(wire_event.arguments),
        }

    elif isinstance(wire_event, ToolResult):
        if wire_event.status == "ok":
            payload["type"] = "tool_completed"
            payload["data"] = {
                "tool_name": wire_event.name,
                "tool_result": (
                    wire_event.result if wire_event.result is not None else {}
                ),
            }
        else:
            payload["type"] = "tool_error"
            payload["data"] = {
                "tool_name": wire_event.name,
                "error": wire_event.error or "unknown",
            }

    elif isinstance(wire_event, Error):
        payload["type"] = "error"
        payload["data"] = {
            "message": wire_event.message,
            "error_type": wire_event.error_type,
            "is_fatal": wire_event.is_fatal,
            "recoverable": False,
        }

    elif isinstance(wire_event, LoopLimitReached):
        payload["type"] = "loop_limit_reached"
        payload["data"] = {"loops": wire_event.loops}

    elif isinstance(wire_event, HwReset):
        payload["type"] = "info"
        payload["data"] = {
            "message": (
                f"Model '{wire_event.model_name}' was reset by "
                f"HardwareWatchdog after fatal error"
            )
        }

    else:
        payload["type"] = "error"
        payload["data"] = {
            "message": f"unknown_wire_event: {type(wire_event).__name__}"
        }

    return (json.dumps(payload) + "\n").encode("utf-8")


__all__ = ["NdjsonEmitter", "v0_compat_serialize"]