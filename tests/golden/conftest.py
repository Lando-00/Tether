"""
Shared helpers and fixtures for golden-stream regression tests.
Synthesis §6 (free-wins / regression net), B2 lines 160-190 (normalize_event design).
"""
import copy
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether_service.core.interfaces import ModelProvider, SessionStore


# ---------------------------------------------------------------------------
# ScriptedProvider — lives here, NOT in tether_service/providers/.
# Drives the orchestrator with pre-canned chunks so the golden test is
# deterministic without touching any MLC hardware. (R6: test fake only)
# ---------------------------------------------------------------------------

class ScriptedProvider(ModelProvider):
    """Yields scripted string chunks, one script per call to stream().

    ``scripts`` is a list of lists-of-strings. The n-th call to stream()
    yields the n-th script's chunks.
    """

    def __init__(self, scripts: List[List[str]]):
        self._scripts = list(scripts)
        self._call_index = 0

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        if self._call_index >= len(self._scripts):
            raise RuntimeError(
                f"ScriptedProvider exhausted after {self._call_index} calls"
            )
        chunks = self._scripts[self._call_index]
        self._call_index += 1
        for chunk in chunks:
            yield chunk

    def list_models(self) -> List[str]:
        return ["scripted-model"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


# ---------------------------------------------------------------------------
# MinimalMemoryStore — minimal SessionStore implementation for golden tests.
# Does not hit SQLite. Implements only what orchestrate() calls.
# ---------------------------------------------------------------------------

class MinimalMemoryStore(SessionStore):
    """In-memory SessionStore covering only the methods orchestrate() uses."""

    def __init__(self):
        self._sessions: Dict[str, List[Dict]] = {}

    # --- required abstract methods (unused by orchestrate, provided for ABC) ---
    async def create_session(self, session_id: str, created_at: int) -> None:
        self._sessions.setdefault(session_id, [])

    async def list_sessions(self) -> List[Dict[str, Any]]:
        return []

    async def delete_session(self, session_id: str) -> bool:
        return bool(self._sessions.pop(session_id, None))

    async def delete_all_sessions(self) -> int:
        count = len(self._sessions)
        self._sessions.clear()
        return count

    # --- methods called by orchestrate() ---
    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        history = self._sessions.setdefault(session_id, [])
        if not history:
            history.insert(0, {"role": "system", "content": prompt})

    async def add_user(self, session_id: str, text: str, *, turn_id=None, seq_start=None) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": "user", "content": text}
        )

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
        *,
        turn_id=None,
        seq_start=None,
    ) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": "assistant", "content": text}
        )

    async def add_assistant_toolcall(
        self, session_id: str, tool_name: str, args: Dict[str, Any],
        *, turn_id=None, tool_call_id=None, seq_start=None,
    ) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": "tool", "tool": tool_name, "args": args}
        )

    async def add_tool_result(
        self, session_id: str, tool_name: str, result: Any,
        *, turn_id=None, tool_call_id=None, seq_start=None,
        status="ok", error=None, duration_ms=None,
    ) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": "tool_result", "tool": tool_name, "result": result}
        )

    async def get_history(
        self, session_id: str, include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        return list(self._sessions.get(session_id, []))

    # --- v2 lifecycle stubs (no-op; test fakes don't need v2 state) ---
    async def start_turn(self, session_id: str, turn_id: str, *, model_name=None) -> None:
        pass

    async def complete_turn(self, turn_id: str, *, status="completed", stop_reason=None, error_json=None) -> None:
        pass

    async def record_raw_event(self, session_id, turn_id, seq, event_type, payload, *, tool_call_id=None) -> None:
        pass


# ---------------------------------------------------------------------------
# normalize_event — scrubs volatile / non-deterministic fields.
# Synthesis §6, B2 lines 160-190: scrub ts, session_id, turn_id,
# tool_call_id, and time-like values.
#
# Updated in p5-cutover-b-clients to cover v2 envelope fields (turn_id,
# tool_call_id) that have no v0 equivalent. Synthesis §11.3 R18.
# ---------------------------------------------------------------------------

def normalize_event(event: dict) -> dict:
    """Return a copy of *event* with volatile fields replaced by sentinels.

    Fields scrubbed:
    - ``ts``           → removed (wall-clock timestamp)
    - ``session_id``   → "<sid>"
    - ``turn_id``      → "<tid>"  (v2 envelope; absent in v0 events)
    - ``tool_call_id`` → "<tcid>" (v2 ToolCall / ToolResult; absent in v0)
    - ``result.time``  → "<time>" (TimeTool return value)
    """
    e = copy.deepcopy(event)
    e["session_id"] = "<sid>"
    e.pop("ts", None)
    # v2 envelope fields
    if "turn_id" in e:
        e["turn_id"] = "<tid>"
    if "tool_call_id" in e:
        e["tool_call_id"] = "<tcid>"
    # TimeTool result normalization (v2: top-level result dict)
    result = e.get("result")
    if isinstance(result, dict) and "time" in result:
        result["time"] = "<time>"
    # Legacy v0 path (keep for backward-compat in case any v0 test calls this)
    data = e.get("data", {})
    if isinstance(data.get("tool_result"), dict):
        tr = data["tool_result"]
        if "time" in tr:
            tr["time"] = "<time>"
    return e
