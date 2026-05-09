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

    async def add_user(self, session_id: str, text: str) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": "user", "content": text}
        )

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
    ) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": "assistant", "content": text}
        )

    async def add_assistant_toolcall(
        self, session_id: str, tool_name: str, args: Dict[str, Any]
    ) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": "tool", "tool": tool_name, "args": args}
        )

    async def add_tool_result(
        self, session_id: str, tool_name: str, result: Any
    ) -> None:
        self._sessions.setdefault(session_id, []).append(
            {"role": "tool_result", "tool": tool_name, "result": result}
        )

    async def get_history(
        self, session_id: str, include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        return list(self._sessions.get(session_id, []))


# ---------------------------------------------------------------------------
# normalize_event — scrubs volatile / non-deterministic fields.
# Synthesis §6, B2 lines 160-190: scrub ts, session_id, time-like values.
# ---------------------------------------------------------------------------

def normalize_event(event: dict) -> dict:
    """Return a copy of *event* with volatile fields replaced by sentinels.

    Fields scrubbed:
    - ``ts``         → removed (wall-clock timestamp)
    - ``session_id`` → "<sid>"
    - ``data.tool_result.time`` → "<time>" (TimeTool return value)
    """
    e = copy.deepcopy(event)
    e["session_id"] = "<sid>"
    e.pop("ts", None)
    data = e.get("data", {})
    if isinstance(data.get("tool_result"), dict):
        tr = data["tool_result"]
        if "time" in tr:
            tr["time"] = "<time>"
    return e
